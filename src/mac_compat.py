"""Mac（无 Triton）兼容补丁。

问题：TRL 1.15 的 GRPOTrainer 用 `fused_lm_head` 计算每个 token 的 log 概率，内部调用
`trl.kernels.ChunkedLogProbFunction`。它依赖 Triton，而 Triton 只随 Linux 版 PyTorch 提供，
所以在 macOS 上这个对象是 None，训练第一步就报 `'NoneType' object has no attribute 'apply'`。

解决：在 Triton 不可用时，换成一个纯 PyTorch 实现（直接算完整 logits 再 log_softmax）。
它更占内存、更慢，但数学上等价，足够在 Mac 上做冒烟测试。Linux + CUDA 服务器上
Triton 可用，本补丁不会生效，训练仍走 TRL 原生的高效内核。
"""
from __future__ import annotations

import torch


class TorchChunkedLogProb:
    """与 trl.kernels.ChunkedLogProbFunction.apply 相同的输入输出约定。

    输入:  hidden [N,H], weight [V,H], bias [V]|None, targets [N], temperature,
           chunk_size(此处不用), final_logit_softcapping, logit_scale, outputs
    输出:  (log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1)，未请求的项为 None
    logit 变换顺序与 TRL 内核一致: z * logit_scale -> softcap -> / temperature
    """

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size=None,
              final_logit_softcapping=None, logit_scale=1.0,
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        z = torch.nn.functional.linear(hidden, weight, bias).float()
        z = z * logit_scale
        if final_logit_softcapping is not None:
            z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
        z = z / temperature

        logp_all = torch.log_softmax(z, dim=-1)
        log_probs = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        entropy = log_sum_sq = mean_logits = is_top1 = None
        if "entropy" in outputs:
            entropy = -(logp_all.exp() * logp_all).sum(-1)
        with torch.no_grad():
            if "log_sum_sq_probs" in outputs:
                log_sum_sq = torch.logsumexp(2 * logp_all, dim=-1)
            if "mean_logits" in outputs:
                mean_logits = z.mean(-1)
            if "is_top1" in outputs:
                is_top1 = z.argmax(-1) == targets
        return log_probs, entropy, log_sum_sq, mean_logits, is_top1


def apply_mac_patches(verbose: bool = True) -> bool:
    """Triton 不可用时打补丁，返回是否打了补丁。"""
    try:
        import trl.trainer.utils as tu
    except ImportError:
        return False
    if getattr(tu, "_ChunkedLogProbFunction", "missing") is None:
        tu._ChunkedLogProbFunction = TorchChunkedLogProb
        if verbose:
            print("[mac_compat] Triton 不可用，已使用纯 PyTorch 的 log 概率实现（仅适合冒烟测试）")
        return True
    return False
