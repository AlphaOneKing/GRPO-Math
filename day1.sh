#!/usr/bin/env bash
# Day 1：环境安装 + 单元测试 + 冒烟训练/评测（在 Mac 终端里运行）
#   用法：先激活 Python 环境，再执行  bash day1.sh
#   已经装过依赖时：SKIP_INSTALL=1 bash day1.sh
#   全部输出会保存到 logs/day1.log，Claude 可以直接读取
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee logs/day1.log) 2>&1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

step() { echo; echo "==================== $1 ===================="; }

step "0. 环境信息"
date; uname -a; which python; python --version

step "1. 安装依赖"
if [ -z "$SKIP_INSTALL" ]; then
  python -m pip install -U pip
  python -m pip install -U torch transformers trl peft datasets accelerate matplotlib pytest \
    || { echo "!!! 安装失败"; exit 1; }
  python -m pip freeze > logs/pip_freeze.txt
else
  echo "跳过安装（SKIP_INSTALL=1）"
fi
python -c "import torch,transformers,trl,peft,datasets; print('torch',torch.__version__,'mps',torch.backends.mps.is_available()); print('transformers',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__,'datasets',datasets.__version__)"

step "2. 单元测试"
python -m pytest tests/ -q; echo "PYTEST_EXIT=$?"

step "3. 冒烟训练（0.5B，16 条数据，2 步）"
python train.py --exp baseline --model Qwen/Qwen2.5-0.5B-Instruct \
  --num_train 16 --max_steps 2 --max_completion_length 128 \
  --per_device_batch 8 --grad_accum 1 --run_name smoke
echo "TRAIN_EXIT=$?"

step "4. 冒烟评测（10 道题）"
python eval.py --model outputs/smoke/final --tag smoke \
  --n 10 --backend hf --batch_size 5 --max_new_tokens 256
echo "EVAL_EXIT=$?"

step "完成"
echo "把这句话发给 Claude：Day1 脚本跑完了"
