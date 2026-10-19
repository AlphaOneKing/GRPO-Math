"""把多个实验的训练曲线画在一起（不依赖 wandb）。

  python plot_curves.py --runs baseline g4 g16 kl0.04 no_format --out figs/ablation.png
  python plot_curves.py --runs baseline dapo_token dapo_clip_higher dapo_full dr_grpo --out figs/dapo.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

# (TRL 日志键, 图标题)；不存在的键会自动跳过
METRICS = [
    ("rewards/correctness_reward/mean", "train accuracy (correctness reward)"),
    ("reward", "total reward"),
    ("completions/mean_length", "mean completion length (tokens)"),
    ("entropy", "token entropy"),
    ("frac_reward_zero_std", "frac groups with zero reward std"),
    ("clip_ratio/high_mean", "clip ratio (high)"),
    ("completions/clipped_ratio", "truncated ratio"),
    ("kl", "KL to reference"),
]


def load_history(run_dir: str):
    cands = [os.path.join(run_dir, "trainer_state.json")]
    cands += sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(p.split("checkpoint-")[-1].split(os.sep)[0]))[::-1]
    for c in cands:
        if os.path.exists(c):
            return json.load(open(c))["log_history"]
    raise FileNotFoundError(f"no trainer_state.json under {run_dir}")


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--root", default="outputs")
    p.add_argument("--out", default="figs/curves.png")
    p.add_argument("--smooth", type=int, default=5)
    a = p.parse_args()

    hist = {r: load_history(os.path.join(a.root, r)) for r in a.runs}
    present = [m for m in METRICS if any(m[0] in h for H in hist.values() for h in H)]
    cols = 2
    rows = (len(present) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.4 * rows), squeeze=False)
    # 每个实验固定一个颜色：否则只有部分实验有的指标（如 KL）会重新从第一个颜色开始，和图例对不上
    colors = {run: f"C{i}" for i, run in enumerate(hist)}
    for ax, (key, title) in zip(axes.flat, present):
        for run, H in hist.items():
            pts = [(h["step"], h[key]) for h in H if key in h]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, smooth(list(ys), a.smooth), label=run, linewidth=1.6, color=colors[run])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(present):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
