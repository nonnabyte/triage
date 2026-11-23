"""Each version's path through the cascade, on the fake models."""

from triage import checks
from triage.providers import FAKE_FAIL
from triage.schemas import CheckResult


def tiers(r):
    return [c["tier"] for c in r.json()["triage"]["calls"]]


def test_baseline_calls_frontier_only(make_client, ask):
    r = ask(make_client("baseline"), "hi")
    assert tiers(r) == ["frontier"]
    assert r.json()["triage"]["calls"][0]["check"] is None


def test_v2_accepts_good_cheap_answer(make_client, ask):
    r = ask(make_client("v2"), "hi")
    block = r.json()["triage"]
    assert tiers(r) == ["cheap"]
    assert block["calls"][0]["check"]["passed"] is True
    assert block["totals"]["wasted_cost_usd"] == 0


def test_v2_escalates_failed_cheap_answer(make_client, ask):
    r = ask(make_client("v2"), f"hi {FAKE_FAIL}")
    block = r.json()["triage"]
    assert tiers(r) == ["cheap", "frontier"]
    assert block["escalation_path"] == ["cheap", "frontier"]
    assert block["final_tier"] == "frontier"
    assert block["calls"][0]["check"]["passed"] is False
    assert block["calls"][1]["check"] is None  # frontier is never checked
    assert block["totals"]["wasted_cost_usd"] == block["calls"][0]["cost_usd"] > 0
    assert block["totals"]["cost_usd"] == sum(c["cost_usd"] for c in block["calls"])
    assert block["totals"]["cost_complete"] is True
    assert r.json()["choices"][0]["message"]["content"].startswith("[frontier]")


def test_check_model_calls_are_logged_as_judge_overhead(make_client, ask, monkeypatch):
    """A check that calls a model (like the judge will) shows up in the call log and in overhead cost."""

    async def judging_check(ctx: checks.CheckContext) -> CheckResult:
        call = await ctx.ask_model(f"Is this a good answer? {ctx.output}")
        return CheckResult(name="non_empty", passed=bool(call.output))

    monkeypatch.setitem(checks.CHECKS, "non_empty", judging_check)
    block = ask(make_client("v2"), "hi").json()["triage"]
    calls = block["calls"]
    assert [(c["purpose"], c["tier"]) for c in calls] == [("answer", "cheap"), ("judge", "cheap")]
    assert [c["seq"] for c in calls] == [1, 2]
    assert block["escalation_path"] == ["cheap"]  # judge calls aren't answer hops
    assert block["totals"]["overhead_cost_usd"] == calls[1]["cost_usd"] > 0
