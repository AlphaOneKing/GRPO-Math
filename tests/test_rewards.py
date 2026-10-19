import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rewards import (  # noqa: E402
    correctness_reward, extract_answer, hackable_length_reward,
    make_overlong_penalty, soft_format_reward, strict_format_reward,
)


def conv(text):
    return [{"role": "assistant", "content": text}]


GOOD = "<think>\n3 apples + 4 apples = 7\n</think>\n<answer>\n7\n</answer>"
NO_TAG = "3 + 4 = 7. The answer is 7."
WRONG = "<think>guess</think><answer>8</answer>"


def test_extract_answer():
    assert extract_answer(GOOD) == "7"
    assert extract_answer("<answer>$1,250.00</answer>") == "1250.00"
    assert extract_answer("<answer>The answer is 42 dollars</answer>") == "42"
    assert extract_answer(NO_TAG) is None


def test_correctness_conversational_and_numeric_equivalence():
    comps = [conv(GOOD), conv(NO_TAG), conv(WRONG), conv("<answer>7.0</answer>")]
    assert correctness_reward(comps, answer=["7"] * 4) == [1.0, 0.0, 0.0, 1.0]


def test_correctness_standard_format():
    assert correctness_reward([GOOD], answer=["7"]) == [1.0]


def test_format_rewards():
    comps = [conv(GOOD), conv(NO_TAG), conv("<answer>7</answer>")]
    assert strict_format_reward(comps) == [0.5, 0.0, 0.0]
    assert soft_format_reward(comps) == [0.2, 0.0, 0.2]


def test_overlong_penalty():
    f = make_overlong_penalty(max_len=100, cache_len=20)
    r = f(completion_ids=[[0] * 50, [0] * 80, [0] * 90, [0] * 100, [0] * 120])
    assert r == [0.0, 0.0, -0.5, -1.0, -1.0]


def test_hackable_length_capped():
    assert hackable_length_reward(completion_ids=[[0] * 100, [0] * 10_000]) == [0.1, 0.5]


# ---------------------------------------------------------------- Day 2：边界用例
def test_extract_answer_edge_cases():
    assert extract_answer("<answer>−5</answer>") == "-5"          # Unicode 减号
    assert float(extract_answer("<answer>.5</answer>")) == 0.5           # 省略前导 0
    assert float(extract_answer("<answer>3/4</answer>")) == 0.75       # 分数
    assert extract_answer("<answer>540 m^2</answer>") == "540"          # 指数不是答案
    assert extract_answer("<answer>540 cm²</answer>") == "540"
    assert extract_answer("<answer>\\boxed{18}</answer>") == "18"
    assert extract_answer("<answer>18.</answer>") == "18"
    assert extract_answer("<answer>16-3-4=9, 9*2=18</answer>") == "18"
    assert extract_answer("<answer></answer>") is None


def test_correctness_edge_cases():
    comps = ["<answer>−5</answer>", "<answer>.5</answer>", "<answer>$1,000.50</answer>"]
    assert correctness_reward(comps, answer=["-5", "0.5", "1000.5"]) == [1.0, 1.0, 1.0]
    # 不带标签的 \boxed 不给正确性奖励（训练奖励和格式绑定，这是设计选择）
    assert correctness_reward(["the answer is \\boxed{18}"], answer=["18"]) == [0.0]


def test_strict_format_rejects_duplicate_and_nested_tags():
    bad = [
        "<think>a</think><answer>1</answer><answer>2</answer>",          # 两个 answer
        "<think>x <answer>3</answer></think><answer>7</answer>",          # answer 嵌在 think 里
        "<think>a</think><think>b</think><answer>7</answer>",             # 两个 think
        "<think></think><answer>7</answer>",                              # 空 think
        "<think>a</think><answer>7</answer> trailing text",               # 结尾有多余文字
    ]
    assert strict_format_reward(bad) == [0.0] * len(bad)
    assert strict_format_reward([GOOD]) == [0.5]


def test_answer_num_count_flags_ambiguous_answers():
    from rewards import answer_num_count
    assert answer_num_count("<answer>18</answer>") == 1
    assert answer_num_count("<answer>18 (9 eggs x $2)</answer>") == 3
    assert answer_num_count("no tags 18") == 0
