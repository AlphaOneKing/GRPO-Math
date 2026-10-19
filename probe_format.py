"""Day 2 诊断：正式训练前，测一下 GRPO 能不能拿到学习信号。

冒烟实验里 0.5B 完全不按 <think>/<answer> 格式输出，导致奖励全 0、frac_reward_zero_std=1。
这个脚本用和训练一样的采样设置（T=1.0、top_p=1、每题 G 条）在训练集上采样，统计：
  - 格式遵守率（严格 / 只有 <answer>）、\\boxed 使用率、截断率
  - 正确率：严格解析（训练奖励用的）vs 宽松解析（<answer> → \\boxed → 最后一个数字）
  - 不同奖励方案下“组内奖励全相同”的比例 —— 这些组优势为 0，不产生梯度
  - 两种 prompt 写法：格式要求放 system（当前 data.py）vs 放 user 消息末尾

  python probe_format.py --model Qwen/Qwen2.5-1.5B-Instruct --styles system user
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data import build_prompt, load_gsm8k, question_of  # noqa: E402
from rewards import (  # noqa: E402
    _ANSWER_RE, _BOXED_RE, _NUM_RE, _STRICT_FORMAT_RE, _normalize,
    answer_num_count, extract_answer, extract_last_number, is_equal,
)

def lenient_pred(text: str) -> str | None:
    p = extract_answer(text)
    if p is not None:
        return p
    boxed = _BOXED_RE.findall(_normalize(text))
    if boxed:
        nums = _NUM_RE.findall(boxed[-1])
        if nums:
            return nums[-1]
    return extract_last_number(text)


def frac_zero_std(groups: list[list[float]]) -> float:
    return sum(len(set(g)) == 1 for g in groups) / len(groups)


def sample(model, tok, prompts, G, max_new_tokens, temperature, batch_size, device):
    import torch

    per_batch = max(1, batch_size // G)
    res = []
    for i in range(0, len(prompts), per_batch):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + per_batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                num_return_sequences=G, pad_token_id=tok.pad_token_id,
            )
        for row in out[:, enc["input_ids"].shape[1]:]:
            ids = row[row != tok.pad_token_id].tolist()
            res.append((tok.decode(ids, skip_special_tokens=True), len(ids)))
        print(f"  {min(i + per_batch, len(prompts))}/{len(prompts)} prompts", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--styles", nargs="+", default=["system", "user"])
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32, help="一次生成的序列数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device).eval()
    print(f"model={a.model} device={device} dtype={dtype}")

    ds = load_gsm8k("train", a.n_prompts, seed=a.seed)
    questions, answers = [question_of(p) for p in ds["prompt"]], list(ds["answer"])
    os.makedirs(a.out_dir, exist_ok=True)

    summaries = []
    for style in a.styles:
        print(f"\n=== prompt style: {style} ===")
        torch.manual_seed(a.seed)
        t0 = time.time()
        gens = sample(model, tok, [build_prompt(q, style) for q in questions], a.G,
                      a.max_new_tokens, a.temperature, a.batch_size, device)
        rows, groups = [], {"baseline": [], "baseline+soft": [], "lenient+strict_fmt": []}
        with open(os.path.join(a.out_dir, f"probe_{style}.jsonl"), "w") as f:
            for qi, (q, gt) in enumerate(zip(questions, answers)):
                g_rewards = {k: [] for k in groups}
                for text, ntok in gens[qi * a.G:(qi + 1) * a.G]:
                    corr = float(is_equal(extract_answer(text), gt))
                    corr_l = float(is_equal(lenient_pred(text), gt))
                    strict = 0.5 if _STRICT_FORMAT_RE.match(text) else 0.0
                    soft = 0.2 if _ANSWER_RE.search(text) else 0.0
                    g_rewards["baseline"].append(corr + strict)
                    g_rewards["baseline+soft"].append(corr + strict + soft)
                    g_rewards["lenient+strict_fmt"].append(corr_l + strict)
                    r = dict(qid=qi, gt=gt, pred=extract_answer(text), pred_lenient=lenient_pred(text),
                             correct=corr, correct_lenient=corr_l, strict_format=strict > 0,
                             has_answer_tag=soft > 0, has_boxed="\\boxed" in text,
                             nums_in_answer=answer_num_count(text), tokens=ntok,
                             truncated=ntok >= a.max_new_tokens, output=text)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                for k in groups:
                    groups[k].append(g_rewards[k])

        n = len(rows)
        mean = lambda key: round(sum(r[key] for r in rows) / n, 4)  # noqa: E731
        s = dict(
            style=style, model=a.model, n_prompts=a.n_prompts, G=a.G,
            max_new_tokens=a.max_new_tokens, temperature=a.temperature,
            strict_format_rate=mean("strict_format"), answer_tag_rate=mean("has_answer_tag"),
            boxed_rate=mean("has_boxed"), acc_strict=mean("correct"),
            acc_lenient=mean("correct_lenient"), trunc_rate=mean("truncated"),
            mean_tokens=round(statistics.mean(r["tokens"] for r in rows), 1),
            ambiguous_answer_rate=round(sum(r["nums_in_answer"] > 1 for r in rows) / n, 4),
            frac_zero_std={k: round(frac_zero_std(v), 4) for k, v in groups.items()},
            minutes=round((time.time() - t0) / 60, 1),
        )
        summaries.append(s)
        print(json.dumps(s, indent=2, ensure_ascii=False))

    with open(os.path.join(a.out_dir, "probe_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\n================ 汇总 ================")
    keys = ["strict_format_rate", "answer_tag_rate", "boxed_rate", "acc_strict",
            "acc_lenient", "trunc_rate", "mean_tokens", "ambiguous_answer_rate"]
    print(f"{'指标':28s}" + "".join(f"{s['style']:>12s}" for s in summaries))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{s[k]:>12}" for s in summaries))
    for k in groups:
        print(f"{'zero_std: ' + k:28s}" + "".join(f"{s['frac_zero_std'][k]:>12}" for s in summaries))
    print("\n解读：zero_std 越接近 1，GRPO 越没有梯度。baseline 这一行 > ~0.8 就不该直接开训。")


if __name__ == "__main__":
    main()
