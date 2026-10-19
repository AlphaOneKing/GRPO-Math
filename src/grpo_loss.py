"""手写 GRPO / DAPO / Dr. GRPO 损失（项目 3 的核心，面试重点）。

实验时用 TRL 的同名开关保证正确性和效率；这个文件是你自己实现的版本，
配合 tests/test_grpo_loss.py 证明你理解每个改动在数学上做了什么。

记号:
  B = prompt 数, G = 每个 prompt 的采样数 (group size), T = 补全最大长度
  logp / old_logp / ref_logp: [B*G, T] 每个 token 的对数概率
  mask: [B*G, T] 有效 token 为 1，padding 为 0
  rewards: [B*G]，同一个 prompt 的 G 个样本连续排列
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ------------------------------------------------------------------ 优势函数
def group_advantages(
    rewards: torch.Tensor, group_size: int, scale: str = "group", eps: float = 1e-4
):
    """组内相对优势: A_i = (r_i - mean_group) / std。

    scale:
      "group"  原版 GRPO，除以组内 std        → Dr. GRPO 指出会带来“题目难度偏差”
      "batch"  均值按组算，std 按整个 batch 算 (Lite PPO 的做法)
      "none"   不除 std (Dr. GRPO)
    返回 (advantages [B*G], zero_std_mask [B*G])；zero_std_mask=1 表示该组奖励全相同
    （全对或全错），这些样本优势为 0，不提供梯度 —— DAPO 的动态采样就是要把它们过滤掉并补采。
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    adv = r - mean
    if scale == "group":
        adv = adv / (std + eps)
    elif scale == "batch":
        adv = adv / (rewards.std(unbiased=False) + eps)
    elif scale != "none":
        raise ValueError(f"unknown scale: {scale}")
    zero_std = (std < 1e-8).expand_as(r).float()
    return adv.reshape(-1), zero_std.reshape(-1)


# ------------------------------------------------------------------ 损失
@dataclass
class LossOutput:
    loss: torch.Tensor
    clip_frac_low: float
    clip_frac_high: float
    kl: float


def policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "grpo",
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    max_len: int | None = None,
    ref_logp: torch.Tensor | None = None,
    beta: float = 0.0,
    seq_weight: torch.Tensor | None = None,
) -> LossOutput:
    """PPO-clip 风格的 token 级损失，再按 loss_type 聚合。

    loss_type:
      "grpo"    先对每条序列按自身长度求平均，再对序列平均   -> 长度偏差
                （正确的长回答每个 token 拿到的梯度更小；错误的长回答受罚更轻）
      "dapo"    所有 token 求和后除以总 token 数（token-level）
      "dr_grpo" 求和后除以常数 (序列数 * max_len)，彻底去掉长度归一化
    eps_low / eps_high: 非对称裁剪，DAPO 的 Clip-Higher 用 eps_high=0.28 放宽上界，
      让低概率 token 的概率能被更大幅度提升，缓解熵坍缩。
    seq_weight: [B*G] 序列权重，比如传入 1 - zero_std_mask 模拟动态采样的过滤。
    """
    mask = mask.float()
    if seq_weight is not None:
        mask = mask * seq_weight.unsqueeze(1)

    A = advantages.unsqueeze(1)  # [N,1] 广播到每个 token（结果奖励，序列内 token 共享优势）
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token = -torch.min(ratio * A, clipped * A)

    kl_mean = 0.0
    if ref_logp is not None and beta > 0:
        # Schulman k3 估计: exp(ref-logp) - (ref-logp) - 1，恒 >= 0 且无偏
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1
        per_token = per_token + beta * kl
        kl_mean = ((kl * mask).sum() / mask.sum().clamp(min=1)).item()

    if loss_type == "grpo":
        seq_len = mask.sum(dim=1).clamp(min=1)
        seq_loss = (per_token * mask).sum(dim=1) / seq_len
        n_valid = (mask.sum(dim=1) > 0).float().sum().clamp(min=1)
        loss = seq_loss.sum() / n_valid
    elif loss_type == "dapo":
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    elif loss_type == "dr_grpo":
        assert max_len is not None, "dr_grpo 需要 max_len"
        loss = (per_token * mask).sum() / (logp.size(0) * max_len)
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    with torch.no_grad():
        denom = mask.sum().clamp(min=1)
        low = ((ratio < 1 - eps_low) & (A < 0)).float()   # 负优势下比率被压到下界
        high = ((ratio > 1 + eps_high) & (A > 0)).float()  # 正优势下比率超过上界
        clip_low = ((low * mask).sum() / denom).item()
        clip_high = ((high * mask).sum() / denom).item()

    return LossOutput(loss, clip_low, clip_high, kl_mean)


def token_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """平均 token 熵，用来观察熵坍缩。logits: [N, T, V]"""
    logp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * mask).sum() / mask.sum().clamp(min=1)
