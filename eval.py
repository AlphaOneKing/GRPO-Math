"""在 GSM8K test 上评测（贪心解码）。

  python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --tag base            # 训练前
  python eval.py --model outputs/baseline/final --tag baseline            # LoRA 检查点自动合并
  python eval.py --model outputs/baseline/final --tag baseline --backend hf  # 没装 vLLM 时

指标:
  acc          严格准确率：从 <answer> 标签中取答案（和训练奖励一致）
  acc_lenient  宽松准确率：没有标签时取全文最后一个数字（衡量“会不会做”，排除格式因素）
  format_rate  满足严格格式的比例
  mean_tokens  平均回答长度；trunc_rate 被截断比例
结果追加到 results/results.csv，逐条样本写入 results/<tag>.jsonl（Day6 做 case study 用）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data import PROMPT_STYLE, load_gsm8k, question_of  # noqa: E402
from rewards import (  # noqa: E402
    _STRICT_FORMAT_RE, answer_num_count, extract_answer, extract_last_number, is_equal,
)


def maybe_merge_lora(path: str, base_model: str | None) -> str:
    """LoRA 检查点 -> 合并成完整模型目录（vLLM 直接加载）。"""
    cfg_file = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_file):
        return path
    merged = path.rstrip("/") + "_merged"
    if os.path.exists(os.path.join(merged, "config.json")):
        return merged
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = base_model or json.load(open(cfg_file))["base_model_name_or_path"]
    print(f"merging LoRA {path} onto {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    return merged


def generate_vllm(model_path, prompts, max_new_tokens, gpu_mem=0.5):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_mem)  # 1.5B 用不了 85%
    # 显式写出所有采样参数，不继承模型 generation_config（Qwen 自带 repetition_penalty=1.1）
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, repetition_penalty=1.0)
    outs = llm.chat(prompts, sp)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


def generate_hf(model_path, prompts, max_new_tokens, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    res = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            # 必须显式 repetition_penalty=1.0：HF 会继承 Qwen generation_config 里的 1.1，
            # 而且惩罚范围包括 prompt，会把格式说明里出现过的 <think>/<answer> 压下去（Day 3 发现）
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, top_k=None, top_p=None, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            ids = row[row != tok.pad_token_id].tolist()
            res.append((tok.decode(ids, skip_special_tokens=True), len(ids)))
        print(f"{min(i + batch_size, len(prompts))}/{len(prompts)}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=None, help="只评前 n 条（调试用），默认全部 1319 条")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gpu_mem", type=float, default=0.5, help="vLLM 显存占比")
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()

    ds = load_gsm8k("test")
    if a.n:
        ds = ds.select(range(a.n))
    prompts, answers = list(ds["prompt"]), list(ds["answer"])

    looks_local = a.model.startswith((".", "/", "~", "outputs"))
    if looks_local and not os.path.exists(os.path.expanduser(a.model)):
        raise SystemExit(f"找不到模型目录 {a.model}，请确认训练是否成功完成")
    model_path = maybe_merge_lora(a.model, a.base_model)
    if a.backend == "vllm":
        gens = generate_vllm(model_path, prompts, a.max_new_tokens, a.gpu_mem)
    else:
        gens = generate_hf(model_path, prompts, a.max_new_tokens, a.batch_size)

    os.makedirs(a.out_dir, exist_ok=True)
    n = len(gens)
    acc = lenient = fmt = tokens = trunc = ambig = 0
    with open(os.path.join(a.out_dir, f"{a.tag}.jsonl"), "w") as f:
        for prompt, gt, (text, ntok) in zip(prompts, answers, gens):
            pred = extract_answer(text)
            ok = is_equal(pred, gt)
            ok_l = ok or is_equal(extract_last_number(text), gt)
            good_fmt = bool(_STRICT_FORMAT_RE.match(text))
            n_num = answer_num_count(text)
            ambig += n_num > 1
            acc += ok; lenient += ok_l; fmt += good_fmt
            tokens += ntok; trunc += ntok >= a.max_new_tokens
            f.write(json.dumps(dict(question=question_of(prompt), gt=gt, pred=pred,
                                    correct=ok, correct_lenient=ok_l, format=good_fmt,
                                    nums_in_answer=n_num, tokens=ntok,
                                    output=text), ensure_ascii=False) + "\n")

    summary = dict(tag=a.tag, model=a.model, n=n, acc=round(acc / n, 4),
                   acc_lenient=round(lenient / n, 4), format_rate=round(fmt / n, 4),
                   mean_tokens=round(tokens / n, 1), trunc_rate=round(trunc / n, 4))
    print(f"prompt_style={PROMPT_STYLE}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"<answer> 里有多个数字（解析可能有歧义）: {ambig}/{n}")
    csv_path = os.path.join(a.out_dir, "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
