"""GRPO 训练入口。

用法示例:
  python train.py --exp baseline
  python train.py --exp dapo_full --use_vllm --wandb
  python train.py --exp baseline --num_generations 16 --run_name g16   # 任意覆盖

实验预设见 EXPERIMENTS；所有参数会写入 <output_dir>/run_config.json，方便复现和写报告。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data import PROMPT_STYLE, load_gsm8k  # noqa: E402
from rewards import REWARD_REGISTRY, make_overlong_penalty  # noqa: E402
from mac_compat import apply_mac_patches  # noqa: E402

# ----------------------------------------------------------------- 实验预设
# Day3 baseline，Day4 消融，Day5 DAPO/Dr.GRPO，选做 reward hacking
BASE = dict(
    rewards=["correctness", "strict_format"],
    loss_type="grpo",
    scale_rewards="group",
    beta=0.0,
    epsilon=0.2,
    epsilon_high=None,       # None = 对称裁剪
    num_generations=8,
    overlong_penalty=False,
    mask_truncated_completions=False,
    num_iterations=1,        # μ：每批样本做几次策略更新。=1 时完全 on-policy，重要性比率恒为 1，裁剪永不触发
)
EXPERIMENTS = {
    # ---- Day 3
    "baseline": {},
    # ---- Day 4 消融
    "g4": dict(num_generations=4),
    "g16": dict(num_generations=16),
    "kl0.04": dict(beta=0.04),
    "no_format": dict(rewards=["correctness"]),
    # ---- Day 5 DAPO / Dr. GRPO（每次只加一个改动，便于归因）
    # μ=1 组：和 baseline 比（损失归一化方式的影响，不涉及裁剪）
    "dapo_token": dict(loss_type="dapo"),
    "dr_grpo": dict(loss_type="dr_grpo", scale_rewards="none"),
    # μ=2 组：裁剪只有在 μ>1 时才起作用（Day 4 发现 μ=1 时 clip_ratio 全程为 0），
    # 所以 Clip-Higher 链条整体放在 μ=2 下，参照组是 dapo_token_mu2
    "dapo_token_mu2": dict(loss_type="dapo", num_iterations=2),
    "dapo_clip_higher": dict(loss_type="dapo", epsilon_high=0.28, num_iterations=2),
    "dapo_full": dict(
        loss_type="dapo", epsilon_high=0.28, num_iterations=2,
        overlong_penalty=True, mask_truncated_completions=True,
    ),
    # ---- 选做：reward hacking
    "hack_length": dict(rewards=["correctness", "hackable_length"]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="baseline", choices=list(EXPERIMENTS))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--run_name", default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--num_train", type=int, default=None, help="训练集子采样数量")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--full_finetune", action="store_true", help="默认用 LoRA")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # 任意覆盖预设
    p.add_argument("--num_generations", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--loss_type", default=None)
    p.add_argument("--epsilon_high", type=float, default=None)
    p.add_argument("--num_iterations", type=int, default=None, help="μ，每批样本的更新次数")
    return p.parse_args()


def build_cfg(args):
    cfg = {**BASE, **EXPERIMENTS[args.exp]}
    for k in ["num_generations", "beta", "loss_type", "epsilon_high", "num_iterations"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return cfg


def make_grpo_config(args, cfg, output_dir):
    """只传入当前 TRL 版本支持的字段，避免版本差异导致报错。"""
    fields = {f.name: f for f in dataclasses.fields(GRPOConfig)}
    lr = args.lr or (1e-6 if args.full_finetune else 1e-5)

    kw = dict(
        output_dir=output_dir,
        run_name=args.run_name or args.exp,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=max(1, round(0.03 * args.max_steps)),  # 新版 transformers 移除了 warmup_ratio
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=cfg["num_generations"],
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        loss_type=cfg["loss_type"],
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        num_iterations=cfg["num_iterations"],
        mask_truncated_completions=cfg["mask_truncated_completions"],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        log_completions=True,
        report_to=["wandb"] if args.wandb else ["none"],
        seed=args.seed,
        use_vllm=args.use_vllm,
    )
    if args.use_vllm:
        kw.update(vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
    if cfg["epsilon_high"] is not None:
        kw["epsilon_high"] = cfg["epsilon_high"]

    # scale_rewards: 新版 TRL 接受 "group"/"batch"/"none"，旧版是 bool
    sr = cfg["scale_rewards"]
    if "scale_rewards" in fields and isinstance(fields["scale_rewards"].default, bool):
        if sr == "batch":
            print("[warn] 当前 TRL 版本不支持 scale_rewards='batch'，退回 True")
        sr = sr != "none"
    kw["scale_rewards"] = sr

    dropped = [k for k in kw if k not in fields]
    if dropped:
        print(f"[warn] 当前 TRL 版本不支持这些参数，已忽略: {dropped}")
        needed = {"epsilon_high", "loss_type", "num_iterations"} & set(dropped)
        if needed:
            raise SystemExit(f"该实验依赖 {sorted(needed)}，请升级 TRL: pip install -U trl")
    # 保证一次生成的补全数能被 group size 整除
    total = args.per_device_batch * args.grad_accum
    if total % cfg["num_generations"]:
        raise SystemExit(
            f"per_device_batch*grad_accum={total} 必须能被 num_generations="
            f"{cfg['num_generations']} 整除（单卡情况下）"
        )
    return GRPOConfig(**{k: v for k, v in kw.items() if k in fields})


def main():
    args = parse_args()
    apply_mac_patches()
    cfg = build_cfg(args)
    run_name = args.run_name or args.exp
    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    reward_funcs = [REWARD_REGISTRY[r] for r in cfg["rewards"]]
    if cfg["overlong_penalty"]:
        # 最后 1/4 长度作为缓冲区，线性惩罚
        reward_funcs.append(
            make_overlong_penalty(args.max_completion_length, args.max_completion_length // 4)
        )

    training_args = make_grpo_config(args, cfg, output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"exp": args.exp, "prompt_style": PROMPT_STYLE, "cli": vars(args), "cfg": cfg}, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # 较老的 transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    train_ds = load_gsm8k("train", args.num_train, seed=args.seed)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_state()  # 写出 trainer_state.json（含完整 log_history，plot_curves.py 读它）
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"done -> {output_dir}/final")


if __name__ == "__main__":
    main()
