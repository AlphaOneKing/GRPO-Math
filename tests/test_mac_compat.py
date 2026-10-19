"""验证 Mac 兼容实现与直接计算 log_softmax 的结果一致，且梯度可以回传。"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mac_compat import TorchChunkedLogProb  # noqa: E402


def test_matches_reference_and_backprops():
    torch.manual_seed(0)
    N, H, V, T = 6, 8, 20, 0.7
    hidden = torch.randn(N, H, requires_grad=True)
    weight = torch.randn(V, H)
    targets = torch.randint(0, V, (N,))
    lp, ent, lss, ml, top1 = TorchChunkedLogProb.apply(hidden, weight, None, targets, T, 0, None, 1.0)

    ref = torch.log_softmax(hidden @ weight.T / T, -1)
    assert torch.allclose(lp, ref.gather(-1, targets[:, None]).squeeze(-1), atol=1e-5)
    assert torch.allclose(ent, -(ref.exp() * ref).sum(-1), atol=1e-5)
    assert torch.equal(top1, ref.argmax(-1) == targets)
    lp.sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_optional_outputs_none():
    out = TorchChunkedLogProb.apply(torch.randn(2, 4), torch.randn(5, 4), None,
                                    torch.tensor([0, 1]), 1.0, 0, None, 1.0, ())
    assert out[0] is not None and all(x is None for x in out[1:])
