"""The v2 checks judge an answer from the prompt and the answer alone (no answer key)."""

import asyncio

from triage import checks
from triage.profile import build_profile
from triage.schemas import Call, Task, Tier

QA = (
    "Answer using only the passage below. Return only the shortest answer span.\n\n"
    "PASSAGE:\nThe Rhine section near Chur is 86 km long.\n\nQUESTION:\nHow long is the section near Chur?"
)
CONTRACT = (
    "Read the contract and perform the extraction task. Return only a JSON array of all matching text "
    "spans, copied verbatim from the contract. Return [] if no matching text is present.\n\nCONTRACT:\n"
    "1. The Franchisee shall not assign this Agreement without prior written consent.\n"
    "2. This Agreement is governed by the laws of Ontario.\n\nDetails: Is consent required to assign?"
)
FIELDS = 'CONTRACT TEXT: \'Net 45 days.\' Return JSON like {"net_days": 0, "late_interest_monthly_pct": 0}.'
CODE = (
    "Complete the Python function below so that it satisfies its docstring. Return the complete code, as "
    'plain Python without Markdown fences.\n\ndef loyalty_points(amount_cents):\n    """Points: 3 per full '
    '$10.\n\n    loyalty_points(2500) == 6\n    loyalty_points(999) == 0\n    """\n'
)
LABELS = (
    "Classify the customer banking query into exactly one of the intent labels below. Return only the "
    "label, exactly as written.\n\nLABELS:\ncard_arrival\nexchange_rate\n\nQUERY:\nWhen does my card come?"
)
MATH = "Solve the problem. Return only the final numeric answer, with no units.\n\nPROBLEM:\n2 apples + 3?"


def ctx(prompt, output, second=None):
    c = checks.CheckContext(
        task=Task(user=prompt),
        profile=build_profile(Task(user=prompt)),
        output=output,
        client=None,
        next_seq=lambda: 2,
        tier=Tier.cheap,
    )
    if second is not None:

        async def ask_model(prompt, system=None, tier=Tier.cheap):  # the second cheap sample
            call = Call(
                seq=2, purpose="judge", tier=tier, provider="fake", model="fake", latency_ms=1, output=second
            )
            c.extra_calls.append(call)
            return call

        c.ask_model = ask_model
        c.ask_verifier = lambda prompt, system=None: ask_model(prompt, system)
    return c


def run(name, *args):
    return asyncio.run(checks.CHECKS[name](ctx(*args)))


def test_json_format_span_list_fields_label_number_and_code():
    assert run("json_format", CONTRACT, '["The Franchisee shall not assign"]').passed
    assert not run("json_format", CONTRACT, "The Franchisee shall not assign").passed
    assert run("json_format", FIELDS, '{"net_days": 45, "late_interest_monthly_pct": 1.5}').passed
    assert "missing keys" in run("json_format", FIELDS, '{"net_days": 45}').detail
    assert run("json_format", LABELS, "card_arrival").passed
    assert not run("json_format", LABELS, "card arrival time").passed
    assert run("json_format", MATH, "5").passed and not run("json_format", MATH, "about 5 apples").passed
    assert run("json_format", CODE, "def loyalty_points(c):\n    return c // 1000 * 3").passed
    assert "not defined" in run("json_format", CODE, "def points(c):\n    return 0").detail
    assert "does not parse" in run("json_format", CODE, "def loyalty_points(c)\n    return 0").detail


def test_grounding_rejects_spans_not_in_the_contract():
    assert run("grounding", CONTRACT, '["The Franchisee shall not assign this Agreement"]').passed
    assert run("grounding", CONTRACT, "[]").passed
    r = run("grounding", CONTRACT, '["The Franchisor may assign this Agreement freely"]')
    assert not r.passed and "not in the contract" in r.detail
    assert run("grounding", QA, "anything").passed  # only applies to span lists


def test_examples_runs_the_prompts_own_examples():
    assert run("examples", CODE, "def loyalty_points(c):\n    return c // 1000 * 3").passed
    r = run("examples", CODE, "def loyalty_points(c):\n    return round(c / 1000 * 3)")
    assert not r.passed and "loyalty_points" in r.detail
    assert run("examples", LABELS, "card_arrival").passed  # no examples: nothing to run


def test_agreement_compares_a_second_answer():
    assert run("agreement", QA, "86 km", "86 km.").passed  # normalised
    assert not run("agreement", QA, "86 km", "90 km").passed
    assert run("agreement", CONTRACT, '["A", "B"]', '["b", "a"]').passed  # spans compared as a set
    assert not run("agreement", CONTRACT, '["A"]', '["A", "B"]').passed
    assert run("agreement", MATH, "5", "$5").passed and not run("agreement", MATH, "5", "6").passed
    good = "def loyalty_points(c):\n    return c // 1000 * 3"
    same = "def loyalty_points(amount_cents):\n    return 3 * (amount_cents // 1000)"
    other = "def loyalty_points(c):\n    return c // 100"
    assert run("agreement", CODE, good, same).passed  # same behaviour, different code
    assert not run("agreement", CODE, good, other).passed


def test_agreement_logs_the_second_call_and_skips_open_ended_text():
    c = ctx(QA, "86 km", "86 km")
    asyncio.run(checks.agreement(c))
    assert [x.purpose for x in c.extra_calls] == ["judge"]
    open_ended = ctx("Plan a 3-day trip to Rome.", "Day 1...", "Something else")
    assert asyncio.run(checks.agreement(open_ended)).passed and not open_ended.extra_calls


def test_checks_stop_at_the_first_failure():
    c = ctx(CONTRACT, "not json", "[]")
    outcome = asyncio.run(checks.run_checks(["non_empty", "json_format", "agreement"], c))
    assert not outcome.passed and [r.name for r in outcome.results] == ["non_empty", "json_format"]
    assert not c.extra_calls  # the second (paid) answer was never requested


def test_refusal_only_counts_an_answer_that_opens_with_one():
    assert not run("refusal", QA, "I'm sorry, but I can't help with that.").passed
    assert not run("refusal", QA, "I cannot provide that information.").passed
    assert run("refusal", QA, "86 km").passed
    assert run("refusal", CONTRACT, '["The Franchisee cannot assign this Agreement"]').passed


def test_span_count_escalates_an_extraction_that_returns_several_spans():
    assert run("span_count", CONTRACT, "[]").passed
    assert run("span_count", CONTRACT, '["The Franchisee shall not assign this Agreement"]').passed
    r = run("span_count", CONTRACT, '["The Franchisee shall not assign", "governed by the laws of Ontario"]')
    assert not r.passed and "2 spans" in r.detail
    assert run(
        "span_count", FIELDS, '{"net_days": 45, "late_interest_monthly_pct": 1.5}'
    ).passed  # not a span list
    assert run("span_count", QA, "86 km").passed


SQL = "Write a SQL query that returns all rows from the products table where price is greater than 50."


def test_json_format_checks_that_sql_parses_without_knowing_the_schema():
    assert run("json_format", SQL, "SELECT * FROM products WHERE price > 50 ORDER BY price DESC;").passed
    r = run("json_format", SQL, "SELEC * FROM products WHERE price > 50")
    assert not r.passed and "SQL does not parse" in r.detail
    assert not run("json_format", SQL, "SELECT * FROM products WHERE (price > 50").passed


def test_cross_agreement_compares_with_the_second_opinion_model():
    assert run("cross_agreement", QA, "86 km", "The section is 86 km.").detail == "the two answers differ"
    assert run("cross_agreement", QA, "86 km", "86 km.").passed
    assert run("cross_agreement", MATH, "5", "The answer is 5").passed
    unavailable = run("cross_agreement", QA, "86 km", "")
    assert unavailable.passed and "not compared" in unavailable.detail


def test_span_lists_agree_when_they_quote_the_same_clause_with_a_little_more_or_less():
    short = '["The Franchisee shall not assign this Agreement"]'
    longer = '["1. The Franchisee shall not assign this Agreement without prior written consent."]'
    other = '["This Agreement is governed by the laws of Ontario."]'
    assert run("cross_agreement", CONTRACT, short, longer).passed
    assert not run("cross_agreement", CONTRACT, short, other).passed
    assert not run("cross_agreement", CONTRACT, short, "[]").passed
    assert run("cross_agreement", CONTRACT, "[]", "[]").passed


def test_labels_are_read_from_any_layout():
    bullets = "Message: when does my card come?\n\nWhich category is it?\n- card_arrival\n- exchange_rate"
    inline = 'Pick the intent. Intents: card_arrival, exchange_rate\n\nUtterance: "when does my card come"'
    sentence = (
        "Classify this ticket as Billing, Technical or Other: 'refund my order'. Return only the label."
    )
    assert checks._labels(LABELS) == ["card_arrival", "exchange_rate"]
    assert checks._labels(bullets) == ["card_arrival", "exchange_rate"]
    assert checks._labels(inline) == ["card_arrival", "exchange_rate"]
    assert checks._labels(sentence) == ["Billing", "Technical", "Other"]
