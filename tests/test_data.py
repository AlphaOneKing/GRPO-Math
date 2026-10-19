import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import (  # noqa: E402
    FORMAT_HINT, PROMPT_STYLE, build_prompt, extract_gsm8k_answer, question_of,
)

Q = "Janet has 3 apples and buys 4 more. How many apples does she have?"


def test_default_prompt_is_user_style():
    assert PROMPT_STYLE == "user"
    p = build_prompt(Q)
    assert [m["role"] for m in p] == ["user"]          # 不传 system，用模型默认的
    assert p[0]["content"].startswith(Q) and p[0]["content"].endswith(FORMAT_HINT)
    assert "<think>" in FORMAT_HINT and "<answer>" in FORMAT_HINT


def test_system_style_still_available():
    p = build_prompt(Q, style="system")
    assert [m["role"] for m in p] == ["system", "user"] and p[1]["content"] == Q


def test_question_of_roundtrip():
    assert question_of(build_prompt(Q)) == Q
    assert question_of(build_prompt(Q, style="system")) == Q


def test_extract_gsm8k_answer():
    assert extract_gsm8k_answer("16 - 3 - 4 = 9\n9 * 2 = 18\n#### 18") == "18"
    assert extract_gsm8k_answer("... #### 1,250") == "1250"
