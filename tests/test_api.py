import re

import pytest
from fastapi.testclient import TestClient

from triage.api import create_app
from triage.config import Settings
from triage.providers import FAKE_CUT, FAKE_DOWN, FAKE_ERROR


def test_healthz(make_client):
    r = make_client().get("/healthz")
    assert r.status_code == 200
    assert r.json()["tiers"] == ["cheap", "frontier"]


def test_response_has_openai_fields_and_triage_block(make_client, ask):
    r = ask(make_client("v2"), "What is 2+2?")
    assert r.status_code == 200
    body = r.json()
    usage = body["usage"]
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert body["triage"]["strategy"] == "v2"
    assert body["triage"]["calls"][0]["output"] is None  # answers hidden outside eval mode


def test_model_field_value_is_ignored(make_client, ask):
    assert ask(make_client(), "hi", model="gpt-4o").status_code == 200


def test_system_message_allowed(make_client):
    body = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}
    assert make_client().post("/v1/chat/completions", json=body).status_code == 200


def test_rejects_multi_turn(make_client):
    body = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    }
    r = make_client().post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_request"


def test_rejects_stream_and_tools(make_client, ask):
    client = make_client()
    assert ask(client, "hi", stream=True).status_code == 400
    assert ask(client, "hi", tools=[{"type": "function"}]).status_code == 400


def test_rejects_missing_messages(make_client):
    r = make_client().post("/v1/chat/completions", json={"model": "triage"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_strategy_header_ignored_outside_eval_mode(make_client, ask):
    r = ask(make_client("v2", eval_mode=False), "hi", headers={"X-Triage-Strategy": "baseline"})
    assert r.json()["triage"]["strategy"] == "v2"


def test_strategy_header_used_in_eval_mode(make_client, ask):
    r = ask(make_client("v2", eval_mode=True), "hi", headers={"X-Triage-Strategy": "baseline"})
    block = r.json()["triage"]
    assert block["strategy"] == "baseline"
    assert block["calls"][0]["output"]  # answers visible in eval mode


def test_v1_routes_with_its_rules(make_client, ask):
    block = ask(make_client("v1"), "hi").json()["triage"]  # unrecognised: the safe tier
    assert block["decision"]["start_tier"] == "frontier"
    assert block["decision"]["reasons"] == ["U0 unknown task type"]
    block = ask(make_client("v1"), "Classify this ticket as Billing or Other: refund my order").json()[
        "triage"
    ]
    assert block["decision"]["start_tier"] == "cheap" and block["decision"]["reasons"][0].startswith("S")
    assert block["escalation_path"] == ["cheap"]  # v1 never checks or escalates


def test_v3_without_a_trained_classifier_escalates_instead_of_failing(ask, tmp_path):
    strategies = tmp_path / "strategies.yaml"
    real = Settings().strategies_config.read_text()
    strategies.write_text(re.sub(r"(?m)^  artifact: .*$", f"  artifact: {tmp_path / 'missing.json'}", real))
    client = TestClient(create_app(Settings(backend="fake", strategy="v3", strategies_config=strategies)))
    r = ask(client, "hi")
    assert r.status_code == 200
    block = r.json()["triage"]
    assert block["escalation_path"] == ["cheap", "frontier"]
    assert "no trained classifier" in block["calls"][0]["check"]["results"][-1]["detail"]


def test_cheap_error_escalates_then_frontier_answers(make_client, ask):
    r = ask(make_client("v2"), f"hello {FAKE_ERROR}")
    assert r.status_code == 200
    calls = r.json()["triage"]["calls"]
    assert [c["tier"] for c in calls] == ["cheap", "frontier"]
    assert calls[0]["error"] and calls[0]["check"]["passed"] is False


def test_all_tiers_failing_returns_502_with_spend(make_client, ask):
    r = ask(make_client("v2"), f"hello {FAKE_DOWN}")
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == "model_failed"
    assert [c["tier"] for c in body["triage"]["calls"]] == ["cheap", "frontier"]


def test_version_needing_an_unassigned_tier_is_a_clear_503(ask, tmp_path):
    # Real backend with only frontier assigned: v2 starts at cheap, which has no model. No call is made.
    real = Settings().models_config.read_text()
    models = tmp_path / "models.yaml"
    models.write_text(re.sub(r"(?m)^  cheap: .*$", "  cheap: null", real))
    settings = Settings(backend="real", strategy="v2", reasoning="low", models_config=models)
    client = TestClient(create_app(settings))
    r = ask(client, "hi")
    assert r.status_code == 503 and r.json()["error"]["code"] == "tier_not_configured"


def test_reasoning_model_without_a_fixed_level_refuses_to_start():
    # Frontier is Gemini 4 Argon, a reasoning model: it may not run with its own default effort.
    with pytest.raises(ValueError, match="TRIAGE_REASONING"):
        create_app(Settings(backend="real", strategy="baseline"))


def test_truncated_answer_is_flagged(make_client, ask):
    r = ask(make_client("baseline", eval_mode=True), f"hi {FAKE_CUT}")
    assert r.json()["triage"]["calls"][0]["truncated"] is True
