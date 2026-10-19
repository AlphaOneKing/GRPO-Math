"""奖励函数（TRL GRPOTrainer 接口）。

接口约定（TRL）:
  - 以关键字参数接收 prompts / completions / completion_ids，以及数据集中除 prompt 外的所有列
    （本项目是 answer）。用 **kwargs 兜住其他参数。
  - 对话格式下，completions[i] 是 [{"role": "assistant", "content": "..."}]。
  - 返回 list[float]，与 completions 一一对应。

奖励设计（可在实验中组合 / 消融）:
  correctness_reward      答案正确 +1.0，否则 0.0           —— 主奖励
  strict_format_reward    完整 <think>...</think><answer>...</answer> 结构 +0.5
  soft_format_reward      只要出现 <answer>...</answer> +0.2   —— 前期给更密的信号
  make_overlong_penalty   DAPO 的 soft overlong punishment  —— 惩罚接近长度上限的回答
  hackable_length_reward  故意设计的“有漏洞”奖励，只用于 reward hacking 分析实验
"""
from __future__ import annotations

import re

_NUM_RE = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")   # 也识别 .5
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FRAC_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$")       # 整个答案是 a/b
_EXPONENT_RE = re.compile(r"\^\{?\d+\}?|[\u00b2\u00b3]")   # m^2、cm³ 里的指数不算答案
# 严格格式：恰好一个 think 块 + 一个 answer 块，块内不能再嵌套任何标签
_NO_TAG = r"(?:(?!</?(?:think|answer)>).)+"
_STRICT_FORMAT_RE = re.compile(
    rf"^\s*<think>{_NO_TAG}</think>\s*<answer>{_NO_TAG}</answer>\s*$", re.DOTALL
)


# ---------------------------------------------------------------- 工具函数
def completion_text(completion) -> str:
    """同时兼容标准格式（str）和对话格式（list[dict]）。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "") or ""
    return ""


def _normalize(s: str) -> str:
    """去掉千分位逗号和货币符号，Unicode 减号统一成 '-'。"""
    return s.replace(",", "").replace("$", "").replace("\u2212", "-")


def answer_content(text: str) -> str | None:
    """最后一个 <answer> 块里规范化后的内容（\\boxed{} 会被拆开，指数会被去掉）。"""
    m = _ANSWER_RE.findall(text)
    if not m:
        return None
    content = _normalize(m[-1])
    boxed = _BOXED_RE.findall(content)
    if boxed:
        content = boxed[-1]
    return _EXPONENT_RE.sub("", content)


def extract_answer(text: str) -> str | None:
    """优先取 <answer> 标签里的内容；标签里若有多个数字取最后一个。
    已知局限：'18 (9 eggs x 2)' 这种答案后带解释的会取到 2，
    用 answer_num_count() 统计这种情况有多常见（Day 2 诊断）。"""
    content = answer_content(text)
    if content is None:
        return None
    frac = _FRAC_RE.match(content)
    if frac and int(frac.group(2)) != 0:
        return repr(int(frac.group(1)) / int(frac.group(2)))
    nums = _NUM_RE.findall(content)
    return nums[-1] if nums else content.strip() or None


def answer_num_count(text: str) -> int:
    """<answer> 里有几个数字。>1 说明答案有歧义，取最后一个可能误判。没有标签返回 0。"""
    content = answer_content(text)
    return len(_NUM_RE.findall(content)) if content is not None else 0


def extract_last_number(text: str) -> str | None:
    """不带标签时的兜底：取全文最后一个数字（只在评测 baseline 时用）。"""
    nums = _NUM_RE.findall(_normalize(text))
    return nums[-1] if nums else None


def is_equal(pred: str | None, gt: str, tol: float = 1e-4) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt)) <= tol
    except ValueError:
        return pred.strip() == gt.strip()


# ---------------------------------------------------------------- 奖励函数
def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [
        1.0 if is_equal(extract_answer(completion_text(c)), gt) else 0.0
        for c, gt in zip(completions, answer)
    ]


def strict_format_reward(completions, **kwargs) -> list[float]:
    return [
        0.5 if _STRICT_FORMAT_RE.match(completion_text(c)) else 0.0
        for c in completions
    ]


def soft_format_reward(completions, **kwargs) -> list[float]:
    return [
        0.2 if _ANSWER_RE.search(completion_text(c)) else 0.0 for c in completions
    ]


def make_overlong_penalty(max_len: int, cache_len: int):
    """DAPO 的 Soft Overlong Punishment。

    长度 L（token 数）:
      L <= max_len - cache_len           -> 0
      max_len - cache_len < L <= max_len -> 线性从 0 降到 -1
      L > max_len                        -> -1
    """
    assert 0 < cache_len < max_len

    def overlong_penalty(completion_ids, **kwargs) -> list[float]:
        out = []
        for ids in completion_ids:
            L = len(ids)
            if L <= max_len - cache_len:
                out.append(0.0)
            elif L <= max_len:
                out.append((max_len - cache_len - L) / cache_len)
            else:
                out.append(-1.0)
        return out

    overlong_penalty.__name__ = "overlong_penalty"
    return overlong_penalty


def hackable_length_reward(completion_ids, **kwargs) -> list[float]:
    """反面教材：奖励更长的思考过程（每 100 token +0.1，最多 +0.5）。
    用于 reward hacking 实验：观察模型是否开始灌水/重复而准确率不升反降。"""
    return [min(len(ids) / 1000.0, 0.5) for ids in completion_ids]


REWARD_REGISTRY = {
    "correctness": correctness_reward,
    "strict_format": strict_format_reward,
    "soft_format": soft_format_reward,
    "hackable_length": hackable_length_reward,
}
