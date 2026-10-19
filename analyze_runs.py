"""把多个实验的评测结果和训练末段指标汇总成一张表（Day 5 起每次跑完都用它）。

  python analyze_runs.py --ref baseline --runs baseline dapo_token dr_grpo
  python analyze_runs.py --ref dapo_token_mu2 --runs dapo_token_mu2 dapo_clip_higher dapo_full

列说明：
  acc / lenient    严格 / 宽松准确率（测试集）
  len_ok / len_bad 答对 / 答错回答的平均 token 数；gap = 错 - 对，用来看长度偏差（Dr. GRPO 声称能消掉）
  +/- , p          相对 --ref 逐题：ref 错它对 / ref 对它错 的题数，精确 McNemar 双侧 p 值
  ent / zstd / clip_hi / clip_all   训练最后 30 步的平均：token 熵、零方差组比例、上界裁剪比例、总裁剪比例
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else None


def mcnemar_p(b01: int, b10: int) -> float:
    n, k = b01 + b10, min(b01, b10)
    if n == 0:
        return 1.0
    pk = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-15))


def train_tail(run_dir: str, last: int = 6):
    p = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(p):
        return {}
    rows = [h for h in json.load(open(p))["log_history"] if "entropy" in h][-last:]

    def avg(k):
        v = [h[k] for h in rows if k in h]
        return sum(v) / len(v) if v else None

    return {"ent": avg("entropy"), "zstd": avg("frac_reward_zero_std"),
            "clip_hi": avg("clip_ratio/high_mean"), "clip_all": avg("clip_ratio/region_mean")}


def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--results", default="results")
    ap.add_argument("--root", default="outputs")
    a = ap.parse_args()

    ref = load_jsonl(os.path.join(a.results, f"{a.ref}.jsonl"))
    head = "| run | acc | lenient | len_ok | len_bad | gap | +/- vs " + a.ref + " | p | ent | zstd | clip_hi | clip_all |"
    print(head)
    print("|" + " --- |" * (head.count("|") - 1))
    for run in a.runs:
        rows = load_jsonl(os.path.join(a.results, f"{run}.jsonl"))
        if rows is None:
            print(f"| {run} | 没有 results/{run}.jsonl |")
            continue
        n = len(rows)
        ok = [r["tokens"] for r in rows if r["correct"]]
        bad = [r["tokens"] for r in rows if not r["correct"]]
        len_ok = sum(ok) / len(ok) if ok else None
        len_bad = sum(bad) / len(bad) if bad else None
        gap = len_bad - len_ok if ok and bad else None
        cmp_, p = "—", None
        if ref is not None and run != a.ref and len(ref) == n:
            b01 = sum((not x["correct"]) and y["correct"] for x, y in zip(ref, rows))
            b10 = sum(x["correct"] and not y["correct"] for x, y in zip(ref, rows))
            cmp_, p = f"+{b01} / -{b10}", mcnemar_p(b01, b10)
        t = train_tail(os.path.join(a.root, run))
        print(f"| {run} | {sum(r['correct'] for r in rows) / n:.1%} "
              f"| {sum(r.get('correct_lenient', r['correct']) for r in rows) / n:.1%} "
              f"| {fmt(len_ok, '.0f')} | {fmt(len_bad, '.0f')} | {fmt(gap, '+.0f')} | {cmp_} | {fmt(p, '.2f')} "
              f"| {fmt(t.get('ent'), '.3f')} | {fmt(t.get('zstd'), '.2f')} "
              f"| {fmt(t.get('clip_hi'), '.4f')} | {fmt(t.get('clip_all'), '.4f')} |")


if __name__ == "__main__":
    main()
