"""GSM8K 数据处理：把原始数据转成 TRL GRPOTrainer 需要的对话格式。

每条样本输出:
    prompt: [{"role": "user", "content": 题目 + 格式说明}]（style="system" 时是旧的 system+user 写法）
    answer: 标准答案（纯数字字符串），会作为关键字参数传给奖励函数
"""
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "First think about the problem step by step inside <think> </think> tags, "
    "then give ONLY the final numeric answer inside <answer> </answer> tags.\n"
    "Format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
)


def extract_gsm8k_answer(solution: str) -> str:
    """GSM8K 的标准答案在 '####' 之后，例如 '... #### 72'。"""
    ans = solution.split("####")[-1].strip()
    return ans.replace(",", "").replace("$", "").strip()


# 格式要求放在 user 消息末尾（Day 2 诊断：1.5B 的格式遵守率 system 写法 52% → user 写法 62%，
# 宽松准确率不变）。所有评测和训练都用同一种写法，RL 的提升才是和“最好的 prompt”比。
FORMAT_HINT = (
    "Think about the problem step by step inside <think> </think> tags, "
    "then give ONLY the final numeric answer inside <answer> </answer> tags.\n"
    "Format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
)
PROMPT_STYLE = "user"  # "user"（默认）或 "system"（Day 1 的旧写法，只用于对比）


def build_prompt(question: str, style: str = PROMPT_STYLE):
    if style == "user":
        return [{"role": "user", "content": f"{question}\n\n{FORMAT_HINT}"}]
    if style == "system":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    raise ValueError(f"unknown prompt style: {style}")


def question_of(prompt) -> str:
    """从 prompt 里还原原始题目（去掉 user 风格追加的格式说明）。"""
    return prompt[-1]["content"].split(f"\n\n{FORMAT_HINT}")[0]


def load_gsm8k(split: str = "train", num_samples: int | None = None, seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if num_samples is not None and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    ds = ds.map(
        lambda x: {
            "prompt": build_prompt(x["question"]),
            "answer": extract_gsm8k_answer(x["answer"]),
        },
        remove_columns=ds.column_names,
    )
    return ds
