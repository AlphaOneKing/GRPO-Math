#!/usr/bin/env bash
# Day 2：奖励函数测试 + 1.5B 学习信号诊断 + base 模型评测（在 Mac 终端里运行）
#   用法：conda activate grpo && bash day2.sh
#   只跑诊断、跳过完整评测：     SKIP_EVAL=1 bash day2.sh
#   诊断已跑过、只跑评测：       SKIP_PROBE=1 bash day2.sh
#   先评测前 200 题看个大概：    EVAL_N=200 bash day2.sh
#   全部输出保存到 logs/day2.log，20 条人工检查样本在 logs/day2_samples.txt
cd "$(dirname "$0")"
mkdir -p logs results
exec > >(tee logs/day2.log) 2>&1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

step() { echo; echo "==================== $1 ===================="; }

step "0. 环境信息"
date; uname -a; which python; python --version

step "1. 单元测试"
python -m pytest tests/ -q; echo "PYTEST_EXIT=$?"

step "2. 学习信号诊断（16 题 × 8 采样 × 2 种 prompt，T=1.0，512 token）"
if [ -n "$SKIP_PROBE" ]; then
  echo "跳过（SKIP_PROBE=1），已有结果见 results/probe_summary.json"
else
  python probe_format.py --model "$MODEL" --n_prompts 16 --G 8 --styles system user
  echo "PROBE_EXIT=$?"
fi

step "3. base 模型评测（贪心，512 token）"
TAG=base
if [ -n "$SKIP_EVAL" ]; then
  echo "跳过（SKIP_EVAL=1）"
else
  if [ -n "$EVAL_N" ]; then TAG="base_n$EVAL_N"; N_ARG="--n $EVAL_N"; fi
  python eval.py --model "$MODEL" --tag "$TAG" --backend hf --batch_size 32 \
    --max_new_tokens 512 $N_ARG
  echo "EVAL_EXIT=$?"
fi

step "4. 抽 20 条输出供人工检查 -> logs/day2_samples.txt"
TAG="$TAG" python - <<'PY'
import json, os, random
src = f"results/{os.environ['TAG']}.jsonl"
if not os.path.exists(src):
    src = "results/probe_system.jsonl"   # 没跑评测时用诊断采样
rows = [json.loads(l) for l in open(src)]
random.Random(0).shuffle(rows)
with open("logs/day2_samples.txt", "w") as f:
    f.write(f"source: {src}\n")
    for i, r in enumerate(rows[:20]):
        meta = {k: v for k, v in r.items() if k not in ("output", "question")}
        f.write(f"\n########## {i} ##########\n{json.dumps(meta, ensure_ascii=False)}\n{r['output']}\n")
print(f"已写入 logs/day2_samples.txt（来自 {src}）")
PY

step "完成"
echo "把这句话发给 Claude：Day2 脚本跑完了"
