#!/usr/bin/env bash
# 在 GPU 服务器上运行（不是 Mac）：装环境 + 单元测试 + GPU 冒烟训练 + 评测一致性检查
#   cd /root/autodl-tmp/grpo-math && bash server_setup.sh
#   已经装过依赖：SKIP_INSTALL=1 bash server_setup.sh
#   无卡模式开机时会自动只做：装依赖 + 单元测试 + 预下载模型和数据集，然后停下
#   全部输出在 logs/server_setup.log
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee logs/server_setup.log) 2>&1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}   # 模型缓存放数据盘，系统盘只有 30GB
export HF_HUB_DISABLE_XET=1   # hf-mirror 不支持 Xet 协议，开着会 401
export TOKENIZERS_PARALLELISM=false

step() { echo; echo "==================== $1 ===================="; }

step "0. 机器信息"
date; uname -a; nvidia-smi; df -h /root /root/autodl-tmp 2>/dev/null; free -g; which python; python --version

step "1. 安装依赖"
if [ -z "$SKIP_INSTALL" ]; then
  python -m pip install -U pip
  # trl[vllm] 会装一个和当前 TRL 兼容的 vLLM（以及它锁定的 torch）
  if python -m pip install -U "trl[vllm]" transformers peft datasets accelerate matplotlib pytest; then
    echo "INSTALL_WITH_VLLM=ok"
  else
    echo "!!! 带 vLLM 安装失败，改装不带 vLLM 的版本（训练会慢一些）"
    python -m pip install -U torch trl transformers peft datasets accelerate matplotlib pytest \
      || { echo "!!! 安装失败"; exit 1; }
  fi
  python -m pip freeze > logs/pip_freeze_server.txt
else
  echo "跳过安装（SKIP_INSTALL=1）"
fi
python - <<'PY'
import torch, transformers, trl, peft, datasets
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", ok,
      torch.cuda.get_device_name(0) if ok else "",
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB" if ok else "")
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__,
      "datasets", datasets.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm 不可用:", e)
PY
if python -c "import vllm" 2>/dev/null; then VLLM_ARG="--use_vllm"; BACKEND=vllm; else VLLM_ARG=""; BACKEND=hf; fi
echo "VLLM_ARG=$VLLM_ARG BACKEND=$BACKEND"
# 装了但导入失败的 vLLM 会让 TRL 的 GRPOTrainer 也导入失败，这里提前拦住
python -c "from trl import GRPOTrainer" || { echo "!!! TRL GRPOTrainer 导入失败（通常是 vLLM/torch 不匹配），先修环境再继续"; exit 1; }

step "2. 单元测试"
python -m pytest tests/ -q; echo "PYTEST_EXIT=$?"

step "预下载模型和数据集（模型放 /root/autodl-tmp/models，数据集放 $HF_HOME）"
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
python - <<'PY'
import os
from datasets import load_dataset
m = os.environ["MODEL"]
if not os.path.isdir(m):
    from huggingface_hub import snapshot_download
    m = snapshot_download(m)
print("model ->", m, sorted(os.listdir(m)))
ds = load_dataset("openai/gsm8k", "main"); print({k: len(v) for k, v in ds.items()})
PY
PREFETCH_EXIT=$?; echo "PREFETCH_EXIT=$PREFETCH_EXIT"
[ "$PREFETCH_EXIT" = 0 ] || { echo "!!! 模型/数据集下载失败，先解决下载问题"; exit 1; }

if ! nvidia-smi -L >/dev/null 2>&1; then
  step "无卡模式：到此为止"
  df -h /root /root/autodl-tmp 2>/dev/null
  echo "切回有卡模式开机后运行：cd /root/autodl-tmp/grpo-math && SKIP_INSTALL=1 bash server_setup.sh"
  exit 0
fi

step "3. GPU 冒烟训练（1.5B，64 条，10 步，其余为正式配置）"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > logs/smoke_gpu_mem.txt &
MON=$!
python train.py --exp baseline --model "$MODEL" --num_train 64 --max_steps 10 --run_name smoke_gpu $VLLM_ARG
echo "TRAIN_EXIT=$?"
kill $MON 2>/dev/null
python - <<'PY'
import json, os
mem = [int(x) for x in open("logs/smoke_gpu_mem.txt").read().split() if x.strip().isdigit()]
print(f"显存峰值（nvidia-smi）: {max(mem)/1024:.1f} GB" if mem else "没有显存记录")
p = "outputs/smoke_gpu/trainer_state.json"
if os.path.exists(p):
    hist = json.load(open(p))["log_history"]
    rt = [h for h in hist if "train_runtime" in h]
    if rt:
        sec = rt[-1]["train_runtime"] / 10
        print(f"每步约 {sec:.1f} 秒 -> 300 步约 {sec*300/3600:.1f} 小时")
    for h in hist:
        keys = [k for k in h if "reward" in k or k in ("step", "completions/mean_length", "entropy")]
        print({k: (round(h[k], 3) if isinstance(h[k], float) else h[k]) for k in keys})
PY

step "4. 评测一致性检查（服务器 $BACKEND vs Mac hf，前 200 题，base 模型）"
python eval.py --model "$MODEL" --tag base_server_n200 --n 200 --backend $BACKEND
echo "EVAL_EXIT=$?"
python - <<'PY'
import json, os
if os.path.exists("results/base.jsonl"):
    rows = [json.loads(l) for l in open("results/base.jsonl")][:200]
    n = len(rows)
    print(f"Mac 前 200 题: acc={sum(r['correct'] for r in rows)/n:.3f} "
          f"acc_lenient={sum(r['correct_lenient'] for r in rows)/n:.3f} "
          f"format_rate={sum(r['format'] for r in rows)/n:.3f}")
    print("两边相差几个点以内算正常（vLLM 和 HF 的贪心解码有数值差异）")
PY

step "完成"
echo "下一步：nohup bash day3.sh > logs/day3.log 2>&1 &"
