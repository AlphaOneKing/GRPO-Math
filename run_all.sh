#!/usr/bin/env bash
# 一键跑完整实验矩阵（按天分组，可以只复制其中一段执行）
# 单张 24GB 卡 + Qwen2.5-1.5B-Instruct + LoRA，300 步每个实验约 2~4 小时（开 vLLM 更快）
set -e
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
STEPS=${STEPS:-300}
EXTRA=${EXTRA:-"--use_vllm --wandb"}

# Day 2: 训练前基线
python eval.py --model $MODEL --tag base

# Day 3: GRPO baseline
python train.py --exp baseline --model $MODEL --max_steps $STEPS $EXTRA
python eval.py --model outputs/baseline/final --tag baseline

# Day 4: 消融
for e in g4 g16 kl0.04 no_format; do
  python train.py --exp $e --model $MODEL --max_steps $STEPS $EXTRA
  python eval.py --model outputs/$e/final --tag $e
done
python plot_curves.py --runs baseline g4 g16 kl0.04 no_format --out figs/ablation.png

# Day 5: DAPO / Dr. GRPO
for e in dapo_token dapo_clip_higher dapo_full dr_grpo; do
  python train.py --exp $e --model $MODEL --max_steps $STEPS $EXTRA
  python eval.py --model outputs/$e/final --tag $e
done
python plot_curves.py --runs baseline dapo_token dapo_clip_higher dapo_full dr_grpo --out figs/dapo.png

# 选做: reward hacking
# python train.py --exp hack_length --model $MODEL --max_steps $STEPS $EXTRA
# python eval.py --model outputs/hack_length/final --tag hack_length
