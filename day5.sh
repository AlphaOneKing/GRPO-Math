#!/usr/bin/env bash
# Day 5：DAPO / Dr. GRPO 对比（在服务器上运行）
#   cd /root/autodl-tmp/grpo-math && nohup bash day5.sh > logs/day5.log 2>&1 &
#   看进度：grep -E "====|FAILED|CHECK" logs/day5.log
#
# 实验分两组（每组内每次只加一个改动）：
#   μ=1：baseline(已有) → dapo_token（token 级损失）；baseline → dr_grpo（常数归一化 + 不除 std）
#   μ=2：dapo_token_mu2 → dapo_clip_higher（ε_high=0.28）→ dapo_full（+超长惩罚 +截断 mask）
# μ=2 是因为 μ=1 时重要性比率恒为 1，裁剪永不触发（Day 4 发现），Clip-Higher 会是空操作。
cd "$(dirname "$0")"
mkdir -p logs figs
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}
if python -c "import vllm" 2>/dev/null; then VLLM_ARG="--use_vllm"; else VLLM_ARG=""; fi

echo "==== $(date) 0. 冒烟：μ=2 + ε_high=0.28 训练 10 步，确认裁剪真的会触发"
rm -rf outputs/smoke_mu2
python train.py --exp dapo_clip_higher --model "$MODEL" --num_train 64 --max_steps 10 \
  --run_name smoke_mu2 $VLLM_ARG || { echo "SMOKE FAILED"; exit 1; }
python - <<'PY' || { echo "CHECK FAILED：μ=2 下裁剪仍然是 0，停止，先排查"; exit 1; }
import json
cfg = json.load(open("outputs/smoke_mu2/run_config.json"))["cfg"]
H = json.load(open("outputs/smoke_mu2/trainer_state.json"))["log_history"]
clip = [h.get("clip_ratio/region_mean", 0) for h in H if "entropy" in h]
print("CHECK num_iterations =", cfg["num_iterations"], "epsilon_high =", cfg["epsilon_high"])
print("CHECK clip_ratio/region_mean 每 5 步:", [round(c, 5) for c in clip])
raise SystemExit(0 if any(c > 0 for c in clip) else 1)
PY
echo "CHECK PASSED"

for e in dapo_token dr_grpo dapo_token_mu2 dapo_clip_higher dapo_full; do
  EXP=$e MODEL="$MODEL" bash day3.sh || echo "$e FAILED"
done

echo "==== $(date) 画图和汇总"
python plot_curves.py --runs baseline dapo_token dr_grpo --out figs/dapo_mu1.png
python plot_curves.py --runs dapo_token_mu2 dapo_clip_higher dapo_full --out figs/dapo_mu2.png
{
  echo "## μ=1：相对 baseline"; python analyze_runs.py --ref baseline --runs baseline dapo_token dr_grpo
  echo; echo "## μ=2：相对 dapo_token_mu2"; python analyze_runs.py --ref dapo_token_mu2 --runs dapo_token_mu2 dapo_clip_higher dapo_full
} | tee logs/day5_summary.md
echo "==== $(date) Day5 全部完成"
