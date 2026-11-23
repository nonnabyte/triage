"""Eval runs reuse the baseline's saved frontier answer instead of calling frontier again."""

import asyncio
import json

from triage.config import Settings, load_models
from triage.models import ModelClient
from triage.schemas import Task, Tier


def baseline(tmp_path, model):
    call = {
        "seq": 1, "purpose": "answer", "tier": "frontier", "provider": "fake", "model": model,
        "reasoning": None, "input_tokens": 100, "output_tokens": 10, "cost_usd": 0.5, "latency_ms": 4000.0,
        "retries": 0, "timed_out": False, "truncated": False, "error": None, "check": None,
        "output": "the saved answer",
    }  # fmt: skip
    row = {"id": "qa-001", "run": 2, "status": 200, "triage": {"calls": [call]}}
    (tmp_path / "calls.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path


def client(tmp_path, **settings):
    plain = ModelClient(Settings(backend="fake"), load_models(Settings().models_config))
    run = baseline(tmp_path, plain.entries[Tier.frontier].model)
    return ModelClient(
        Settings(backend="fake", eval_mode=True, reuse_frontier=run, **settings),
        load_models(Settings().models_config),
    )


def test_a_frontier_answer_call_for_a_saved_prompt_and_repeat_is_reused(tmp_path):
    c = client(tmp_path)
    call = asyncio.run(c.complete(Tier.frontier, Task(user="q", eval_key=("qa-001", 2)), seq=3))
    assert call.reused and call.output == "the saved answer" and call.cost_usd == 0.5 and call.seq == 3
    assert call.latency_ms == 4000.0


def test_other_repeats_other_tiers_and_check_calls_are_made_fresh(tmp_path):
    c = client(tmp_path)
    other_repeat = asyncio.run(c.complete(Tier.frontier, Task(user="q", eval_key=("qa-001", 1)), seq=1))
    cheap = asyncio.run(c.complete(Tier.cheap, Task(user="q", eval_key=("qa-001", 2)), seq=1))
    judge = asyncio.run(
        c.complete(Tier.frontier, Task(user="q", eval_key=("qa-001", 2)), seq=2, purpose="judge")
    )
    assert not other_repeat.reused and not cheap.reused and not judge.reused
