"""v3: the classifier weighs a cheap answer's signals against the cost of escalating."""

import asyncio
import json

from triage import checks, escalation
from triage.config import ClassifierConfig, Price
from triage.profile import build_profile
from triage.schemas import Call, Task, Tier

CONTRACT = (
    "Document:\n1. The Franchisee shall not assign this Agreement without prior written consent. "
    + "The parties agree as follows. " * 30
    + "\n\nRequest: return a JSON array of every clause about assignment, quoted exactly."
)
ONE = '["The Franchisee shall not assign this Agreement without prior written consent."]'
MANY = '["The Franchisee shall not assign", "The parties agree as follows.", "prior written consent"]'


def ctx(prompt, output, second=None, artifact=None, max_usd_per_fix=None):
    c = checks.CheckContext(
        task=Task(user=prompt),
        profile=build_profile(Task(user=prompt)),
        output=output,
        client=None,
        next_seq=lambda: 2,
        tier=Tier.cheap,
        classifier=ClassifierConfig(artifact=artifact, max_usd_per_fix=max_usd_per_fix) if artifact else None,
    )

    async def ask_model(prompt, system=None, tier=Tier.cheap):
        call = Call(
            seq=2, purpose="judge", tier=tier, provider="fake", model="fake", latency_ms=1, output=second
        )
        c.extra_calls.append(call)
        return call

    c.ask_model = ask_model
    return c


def signals(prompt, output, second=None):
    return asyncio.run(escalation.signals(ctx(prompt, output), second))


def test_signals_describe_the_answer_without_an_answer_key():
    f = signals(CONTRACT, ONE, ONE)
    assert f["task=contract_extraction"] == 1 and f["spans=1"] == 1 and f["agrees"] == 1
    assert f["fails=span_count"] == 0 and f["fails=grounding"] == 0
    f = signals(CONTRACT, MANY, ONE)
    assert f["spans=2+"] == 1 and f["fails=span_count"] == 1 and f["agrees"] == 0
    assert signals(CONTRACT, ONE, None)["agrees"] == 0  # no second answer is not agreement
    assert set(f) == set(escalation.FEATURES)


def model(**weights):
    coef = [weights.get(name, 0.0) for name in escalation.FEATURES]
    n = len(coef)
    return {
        "features": escalation.FEATURES,
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "coef": coef,
        "intercept": 0.0,
        "output_tokens": {"default": 100},
        "max_usd_per_fix": 0.05,
    }


def test_probability_rises_with_risk_signals_and_falls_with_agreement():
    m = model(agrees=-2.0, **{"spans=2+": 2.0})
    agree = escalation.probability(m, signals(CONTRACT, ONE, ONE))
    many = escalation.probability(m, signals(CONTRACT, MANY, ONE))
    assert agree < 0.5 < many


def test_escalation_is_weighed_against_its_cost():
    assert escalation.should_escalate(p=0.5, cost=0.01, max_usd_per_fix=0.05)  # $0.02 per fix
    assert not escalation.should_escalate(p=0.1, cost=0.01, max_usd_per_fix=0.05)  # $0.10 per fix
    assert not escalation.should_escalate(p=0.5, cost=0.10, max_usd_per_fix=0.05)  # a costly long document
    assert escalation.should_escalate(p=0.01, cost=None, max_usd_per_fix=0.05)  # unknown price: escalate
    price = Price(input=2.0, output=10.0)
    short = escalation.frontier_cost("x" * 400, "document_qa", price, model())
    long = escalation.frontier_cost("x" * 40000, "document_qa", price, model())
    assert short < long


def test_the_check_escalates_when_untrained_and_reports_its_reasoning(tmp_path):
    missing = asyncio.run(
        escalation.escalation_model(ctx(CONTRACT, ONE, ONE, artifact=tmp_path / "none.json"))
    )
    assert not missing.passed and "no trained classifier" in missing.detail
    path = tmp_path / "escalation.json"
    path.write_text(json.dumps(model(agrees=-4.0, **{"spans=2+": 4.0})))
    escalation.load_model.cache_clear()
    kept = asyncio.run(escalation.escalation_model(ctx(CONTRACT, ONE, ONE, artifact=path)))
    assert "P(escalating fixes it)" in kept.detail
    assert "escalation_model" in checks.CHECKS
