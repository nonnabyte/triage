import asyncio
import json

import pandas as pd
import pytest

from triage.config import REPO_ROOT, ModelEntry
from triage.eval import graders as g
from triage.eval.graders import CHECK_TYPES, JudgeVerdict, grade
from triage.eval.sandbox import run_python

DATA = pd.read_csv(REPO_ROOT / "data" / "triage_eval_v2.csv", keep_default_na=False)


def run(item: dict, output: str, judge=None) -> g.Grade:
    return asyncio.run(grade(item, output, judge))


# --- every dataset item is gradable, and its own expected answer passes ----------------------------------


def test_every_check_type_in_the_data_has_a_grader():
    assert set(DATA.check_type) <= set(CHECK_TYPES)


def _gold_as_output(row) -> str:
    gold = json.loads(row.answer) if row.check_type != "sql_exec" else row.answer
    return {
        "exact_match": lambda: gold[0],
        "numeric": lambda: str(gold[0]),
        "json_match": lambda: json.dumps(gold[0]),
        "span_set": lambda: json.dumps(gold),
        "sql_exec": lambda: gold,
    }[row.check_type]()


@pytest.mark.parametrize("check_type", ["exact_match", "numeric", "json_match", "span_set", "sql_exec"])
def test_gold_answers_pass_their_own_grader(check_type):
    rows = DATA[DATA.check_type == check_type]
    failures = [r.id for r in rows.itertuples() if not run(r._asdict(), _gold_as_output(r)).passed]
    assert not failures, f"{check_type}: gold answer fails for {failures}"


# --- exact_match ---------------------------------------------------------------------------------------


def test_exact_match_normalises_case_punctuation_and_articles():
    item = {"check_type": "exact_match", "answer": '["The Ossachite village"]', "prompt": ""}
    assert run(item, "ossachite village.").passed
    assert run(item, "Answer: Ossachite Village").passed


def test_exact_match_any_accepted_answer():
    item = {"check_type": "exact_match", "answer": '["86 km long,", "86 km"]', "prompt": ""}
    assert run(item, "86 km").passed


def test_exact_match_extra_words_fail_but_flag_review():
    item = {"check_type": "exact_match", "answer": '["86 km"]', "prompt": ""}
    r = run(item, "The section is 86 km long")
    assert not r.passed and r.needs_review and 0 < r.score < 1


def test_exact_match_answer_quoting_the_gold_with_a_word_or_two_passes():
    """Real baseline answers that were right but graded wrong."""
    item = {"check_type": "exact_match", "answer": '["27.7 million tons"]', "prompt": ""}
    assert run(item, "27.7 million tons (15%)").passed
    assert run({**item, "answer": '["86 km"]'}, "nearly 86 km").passed
    assert run({**item, "answer": '["marginal"]'}, "marginal significance").passed
    gold = '["by calling the vendor on a number taken from its website, never from the invoice"]'
    assert run(
        {**item, "answer": gold},
        "calling the vendor on a number taken from its website, never from the invoice",
    ).passed
    assert not run({**item, "answer": '["Florida East Coast Railway"]'}, "Railway").passed  # too little of it


def test_exact_match_not_in_document():
    item = {"check_type": "exact_match", "answer": '["NOT_IN_DOCUMENT"]', "prompt": ""}
    assert run(item, "NOT_IN_DOCUMENT").passed
    assert not run(item, "Paris").passed


# --- numeric -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("output", ["2125", "2,125", "$2,125.00", "The answer is 2125."])
def test_numeric_formats(output):
    assert run({"check_type": "numeric", "answer": "[2125]", "prompt": ""}, output).passed


def test_numeric_uses_the_final_number():
    item = {"check_type": "numeric", "answer": "[300]", "prompt": ""}
    assert run(item, "250 + 120 = 370, capped at 300").passed
    r = run(item, "300 is the cap (Section 3)")
    assert not r.passed and r.needs_review


def test_numeric_prefers_the_bolded_number_over_a_later_aside():
    """Real QA01 answer: the bolded 3 is the answer, the 30 is a note about exceptions."""
    item = {"check_type": "numeric", "answer": "[3]", "prompt": ""}
    out = "Employees may work remotely **up to 3 days per week**.\n\n(Note: a director may allow 30 days.)"
    assert run(item, out).passed
    assert not run(item, "**30 days**, though normally 3").passed


def test_numeric_no_number():
    assert not run({"check_type": "numeric", "answer": "[4]", "prompt": ""}, "four").passed


# --- json_match ----------------------------------------------------------------------------------------


def test_json_match_fences_prose_and_key_order():
    item = {"check_type": "json_match", "answer": '[{"net_days": 45, "late_interest_monthly_pct": 1.5}]'}
    out = 'Here you go:\n```json\n{"late_interest_monthly_pct": 1.50, "net_days": 45}\n```'
    assert run({**item, "prompt": ""}, out).passed


def test_json_match_lists_are_order_insensitive_and_strings_case_insensitive():
    item = {"check_type": "json_match", "prompt": "", "answer": '[{"x": ["Gross negligence", "Breach"]}]'}
    assert run(item, '{"x": ["breach", "gross  negligence"]}').passed


def test_json_match_wrong_value_scores_partial():
    item = {"check_type": "json_match", "prompt": "", "answer": '[{"a": 1, "b": 2, "c": 3, "d": 4}]'}
    r = run(item, '{"a": 1, "b": 2, "c": 3, "d": 5}')
    assert not r.passed and r.score == 0.75 and r.needs_review


def test_json_match_not_json():
    item = {"check_type": "json_match", "prompt": "", "answer": '[{"a": 1}]'}
    assert not run(item, "a is one").passed


# --- span_set ------------------------------------------------------------------------------------------

SPANS = [
    "The Franchisee shall not assign this Agreement without prior written consent.",
    "Notice is required.",
]


def test_span_set_exact_and_near_matches():
    item = {"check_type": "span_set", "prompt": "", "answer": json.dumps(SPANS)}
    assert run(item, json.dumps(SPANS[::-1])).passed
    near = [
        "the franchisee shall not assign this agreement without prior written consent",
        "notice is required",
    ]
    assert run(item, json.dumps(near)).passed


def test_span_set_mirrored_clause_fails_and_is_flagged():
    """Real cx-001 case: the model returned the Franchisor's right instead of the Franchisee's restriction."""
    gold = (
        "The Franchisee shall not be entitled to assign, transfer, encumber or dispose of any of its rights "
        "and or obligations under this Agreement, including to an affiliate, without the prior written "
        "consent of the Franchisor."
    )
    mirrored = (
        "10.11 Assignment. The Franchisor shall be entitled to assign, transfer, encumber or dispose of any "
        "of its rights and or obligations under this Agreement, including to an affiliate, without the prior "
        "written consent of the Franchisee."
    )
    r = run({"check_type": "span_set", "prompt": "", "answer": json.dumps([gold])}, json.dumps([mirrored]))
    assert not r.passed and r.needs_review and "close to a gold span" in r.detail


def test_span_set_heading_prefix_and_partial_quote():
    gold = (
        "AJR shall not take any action which could result in an adverse impact on XC or its Licensed "
        "Materials."
    )
    item = {"check_type": "span_set", "prompt": "", "answer": json.dumps([gold])}
    assert run(item, json.dumps(["7. Reputation. " + gold])).passed  # clause heading kept
    assert run(item, json.dumps([gold.rsplit(" ", 1)[0]])).passed  # last word dropped, >= 80% kept
    assert not run(item, json.dumps(["AJR shall not take any action"])).passed  # too little of it
    whole_section = " ".join([gold] * 3)
    buried = run(item, json.dumps([whole_section]))  # gold buried in a much longer quote
    assert not buried.passed and buried.needs_review and "much longer or shorter" in buried.detail


def test_span_set_missing_or_extra_span_fails_with_f1():
    item = {"check_type": "span_set", "prompt": "", "answer": json.dumps(SPANS)}
    missing = run(item, json.dumps(SPANS[:1]))
    extra = run(item, json.dumps([*SPANS, "Governing law: Ontario."]))
    assert not missing.passed and not extra.passed
    assert 0 < missing.score < 1 and 0 < extra.score < 1


def test_span_set_one_span_covering_two_gold_spans_passes():
    """CUAD split one sentence into two spans; the model quoted it whole."""
    gold = [
        "If other services are required by Party B, Party B shall first give Party A the requirements.",
        "If Party A refuses to provide such services, Party B may turn to third parties.",
    ]
    item = {"check_type": "span_set", "prompt": "", "answer": json.dumps(gold)}
    assert run(item, json.dumps(["2.4 " + " ".join(gold)])).passed
    assert not run(item, json.dumps([gold[0] + " " + " ".join(["Unrelated text."] * 20)])).passed


def test_span_set_duplicate_gold_and_repeated_span():
    """Real cx-034: CUAD lists the clause twice; models return it once, or twice (with a heading)."""
    clause = "The term of the Agreement will be for 12 months unless terminated sooner"
    item = {"check_type": "span_set", "prompt": "", "answer": json.dumps([clause + ".", clause])}
    assert run(item, json.dumps([clause + "."])).passed
    assert run(item, json.dumps([clause + ".", "Term: " + clause + "."])).passed
    assert not run(item, json.dumps([clause + ".", "Governing law: Ontario."])).passed


def test_span_set_empty_gold():
    item = {"check_type": "span_set", "prompt": "", "answer": "[]"}
    assert run(item, "[]").passed
    assert not run(item, '["Some clause."]').passed


# --- sql_exec ------------------------------------------------------------------------------------------

SQL_ITEM = {
    "check_type": "sql_exec",
    "prompt": "",
    "answer": "SELECT * FROM products WHERE price > 50 ORDER BY price DESC;",
}


def test_sql_equivalent_query_passes():
    assert run(
        SQL_ITEM, "```sql\nSELECT id, name, price FROM products WHERE price > 50.0 ORDER BY price DESC\n```"
    ).passed


def test_sql_wrong_boundary_or_order_fails():
    assert not run(SQL_ITEM, "SELECT * FROM products WHERE price >= 50 ORDER BY price DESC").passed
    assert not run(SQL_ITEM, "SELECT * FROM products WHERE price > 50 ORDER BY price").passed


def test_sql_cannot_modify_fixture():
    r = run(SQL_ITEM, "DELETE FROM products")
    assert not r.passed and "failed" in r.detail


# --- unit_tests / sandbox ------------------------------------------------------------------------------

UT_ITEM = {
    "check_type": "unit_tests",
    "prompt": "",
    "answer": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
}


def test_unit_tests_pass_and_fail():
    assert run(UT_ITEM, "```python\ndef add(a, b):\n    return a + b\n```").passed
    r = run(UT_ITEM, "def add(a, b):\n    return a - b")
    assert not r.passed and "AssertionError" in r.detail


def test_unit_tests_humaneval_check_style():
    item = {**UT_ITEM, "answer": "def check(candidate):\n    assert candidate(2) == 4\n\ncheck(double)"}
    assert run(item, "def double(x):\n    return 2 * x").passed


def test_sandbox_limits_apply_and_code_runs_as_main():
    """Limits are set inside the child interpreter (not preexec_fn, which can deadlock a threaded harness)."""
    code = (
        "import resource, sys\n"
        "assert resource.getrlimit(resource.RLIMIT_CPU) == (10, 10)\n"
        "assert __name__ == '__main__' and sys.argv[0].endswith('candidate.py')\n"
    )
    assert run_python(code).ok
    assert "ValueError" in run_python("raise ValueError('boom')").detail


def test_sandbox_timeout_and_no_stdin():
    assert run_python("while True:\n    pass", timeout_s=1).timed_out
    assert not run_python("input()").ok


def test_sandbox_has_no_secrets_in_env():
    assert run_python("import os\nassert 'GOOGLE_API_KEY' not in os.environ").ok


# --- rubric --------------------------------------------------------------------------------------------


class FakeJudge:
    entry = ModelEntry(key="fake-judge", provider="fake", model="frontier")

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, task, reference, answer):
        self.calls.append((task, reference, answer))
        return JudgeVerdict(verdict=self.verdict, reason="because")


def test_rubric_uses_the_judge():
    item = {"check_type": "rubric", "prompt": "Q?", "answer": "Reference."}
    judge = FakeJudge("pass")
    r = run(item, "My answer.", judge)
    assert r.passed and judge.calls == [("Q?", "Reference.", "My answer.")]
    assert not run(item, "My answer.", FakeJudge("fail")).passed


def test_rubric_without_judge_is_an_error():
    with pytest.raises(ValueError):
        run({"check_type": "rubric", "prompt": "", "answer": ""}, "x")
