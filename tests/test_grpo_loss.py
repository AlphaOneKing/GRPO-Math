"""这些测试同时是面试素材：每个测试对应一个“为什么要这样改”的问题。"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grpo_loss import group_advantages, policy_loss  # noqa: E402


def test_group_advantage_zero_mean_and_zero_std_groups():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0,   1.0, 1.0, 1.0, 1.0])
    adv, zero = group_advantages(r, group_size=4)
    assert torch.allclose(adv[:4].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.all(adv[4:] == 0)          # 全对的组没有梯度信号
    assert zero.tolist() == [0] * 4 + [1] * 4


def test_std_scaling_difficulty_bias():
    """Dr. GRPO：除以 std 会放大“几乎全对/全错”的题（std 小）的优势。"""
    easy = torch.tensor([1.0, 1.0, 1.0, 0.0])   # 3/4 对，std 小
    mid = torch.tensor([1.0, 1.0, 0.0, 0.0])    # 2/4 对，std 大
    a_easy, _ = group_advantages(easy, 4, "group")
    a_mid, _ = group_advantages(mid, 4, "group")
    # 同样是答错的那个样本，easy 题里受到的惩罚绝对值更大
    assert a_easy[3].abs() > a_mid[3].abs()
    n_easy, _ = group_advantages(easy, 4, "none")
    assert torch.allclose(n_easy, easy - easy.mean())


def _setup(lengths, T=8):
    N = len(lengths)
    logp = torch.zeros(N, T, requires_grad=True)
    mask = torch.zeros(N, T)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1
    return logp, mask


def test_grpo_length_bias_vs_dapo_and_dr_grpo():
    """同为正优势：GRPO 下长序列每个 token 的梯度更小（被 1/|o| 稀释），DAPO / Dr.GRPO 下相同。"""
    adv = torch.tensor([1.0, 1.0])
    for loss_type, expect_equal in [("grpo", False), ("dapo", True), ("dr_grpo", True)]:
        logp, mask = _setup([2, 8])
        out = policy_loss(logp, logp.detach(), adv, mask, loss_type=loss_type, max_len=8)
        out.loss.backward()
        g_short, g_long = logp.grad[0, 0].item(), logp.grad[1, 0].item()
        assert (abs(g_short - g_long) < 1e-6) == expect_equal, loss_type
        if loss_type == "grpo":
            assert abs(g_short) == abs(g_long) * 4   # 1/2 vs 1/8


def test_clip_higher_allows_larger_increase():
    """ratio=1.25 时：对称裁剪 eps=0.2 已截断（梯度为 0），clip-higher eps_high=0.28 仍有梯度。"""
    adv = torch.tensor([1.0])
    old = torch.zeros(1, 4)
    for eps_high, has_grad in [(0.2, False), (0.28, True)]:
        logp = torch.full((1, 4), float(torch.log(torch.tensor(1.25))), requires_grad=True)
        out = policy_loss(logp, old, adv, torch.ones(1, 4), "dapo", eps_high=eps_high)
        out.loss.backward()
        assert (logp.grad.abs().sum() > 0).item() == has_grad
        assert (out.clip_frac_high > 0) == (not has_grad)


def test_negative_advantage_lower_clip():
    adv = torch.tensor([-1.0])
    logp = torch.full((1, 4), float(torch.log(torch.tensor(0.7))), requires_grad=True)
    out = policy_loss(logp, torch.zeros(1, 4), adv, torch.ones(1, 4), "dapo")
    out.loss.backward()
    assert out.clip_frac_low == 1.0 and logp.grad.abs().sum() == 0


def test_kl_k3_nonnegative_and_zero_at_ref():
    logp = torch.randn(3, 5)
    adv = torch.zeros(3)
    out = policy_loss(logp, logp, adv, torch.ones(3, 5), "dapo", ref_logp=logp, beta=0.1)
    assert abs(out.kl) < 1e-6
    out2 = policy_loss(logp, logp, adv, torch.ones(3, 5), "dapo",
                       ref_logp=logp + torch.randn(3, 5), beta=0.1)
    assert out2.kl > 0


def test_dynamic_sampling_weight_removes_groups():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0])
    adv, zero = group_advantages(r, 2)
    logp, mask = _setup([4, 4, 4, 4])
    out = policy_loss(logp, logp.detach(), adv, mask, "dapo", seq_weight=1 - zero)
    out.loss.backward()
    assert logp.grad[2:].abs().sum() == 0
