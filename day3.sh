#!/usr/bin/env bash
# Day 3：在服务器上训练 GRPO baseline，然后在 GSM8K test 上评测。
# 用 nohup 放到后台，断开 SSH 也会继续跑：
#   cd /root/autodl-tmp/grpo-math && nohup bash day3.sh > logs/day3.log 2>&1 &
#   看进度：tail -f logs/day3.log      （Ctrl+C 只是退出查看，不会停训练）
# 跑别的实验：EXP=g16 nohup bash day3.sh > logs/g16.log 2>&1 &
cd "$(dirname "$0")"
mkdir -p logs
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export HF_HUB_DISABLE_XET=1   # hf-mirror 不支持 Xet 协议，开着会 401
export TOKENIZERS_PARALLELISM=false
# 模型从 ModelScope 下载到数据盘（国内比 hf-mirror 快得多，权重与 HF 上的完全相同）
MODEL_DIR=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
if [ -z "$MODEL" ] && [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "从 ModelScope 下载模型到 $MODEL_DIR ..."
  python -m pip install -q modelscope && modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir "$MODEL_DIR" \
    || echo "!!! ModelScope 下载失败，退回 HF 镜像"
fi
if [ -z "$MODEL" ]; then
  if [ -f "$MODEL_DIR/config.json" ]; then MODEL="$MODEL_DIR"; else MODEL=Qwen/Qwen2.5-1.5B-Instruct; fi
fi
export MODEL
echo "MODEL=$MODEL"
EXP=${EXP:-baseline}
STEPS=${STEPS:-300}
if python -c "import vllm" 2>/dev/null; then VLLM_ARG="--use_vllm"; BACKEND=vllm; else VLLM_ARG=""; BACKEND=hf; fi
[ -n "$WANDB" ] && VLLM_ARG="$VLLM_ARG --wandb"

# 还没有有效的 base 评测时，先评测训练前的模型（vLLM 全量 1319 题只要几分钟）
if ! grep -q "^base," results/results.csv 2>/dev/null; then
  echo "==== $(date) 先评测 base 模型"
  python eval.py --model "$MODEL" --tag base --backend $BACKEND || { echo "BASE_EVAL_FAILED"; exit 1; }
fi

echo "==== $(date) 开始训练 EXP=$EXP STEPS=$STEPS $VLLM_ARG"
python train.py --exp "$EXP" --model "$MODEL" --max_steps "$STEPS" $VLLM_ARG || { echo "TRAIN_FAILED"; exit 1; }
echo "==== $(date) 训练完成，开始评测（$BACKEND）"
python eval.py --model "outputs/$EXP/final" --tag "$EXP" --backend $BACKEND || { echo "EVAL_FAILED"; exit 1; }
# 合并出来的完整模型约 3GB，评测完就删（LoRA 权重在 final/ 里，需要时一分钟就能重新合并）。
# 不删的话 11 个实验会把 50GB 数据盘写满（Day 5 的 dapo_full 就是这样评测失败的）
rm -rf "outputs/$EXP/final_merged"
echo "==== $(date) $EXP 全部完成"
