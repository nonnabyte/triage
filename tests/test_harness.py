"""The eval harness on the fake models: runs, resume, cost cap, failures, regrade."""

import asyncio
import csv
import json

import pytest

from triage.config import ModelEntry, Settings
from triage.eval import harness
from triage.eval.graders import JudgeVerdict
from triage.eval.harness import RunConfig, load_items, read_jsonl, run_eval, sample
from triage.providers import FAKE_DOWN

FIELDS = ["id", "use_case", "check_type", "prompt", "answer", "source"]


class FakeJudge:
    entry = ModelEntry(key="fake-judge", provider="fake", model="frontier")

    def __init__(self):
        self.calls = 0

    async def __call__(self, task, reference, answer):
        self.calls += 1
        return JudgeVerdict(verdict="pass", reason="fine")


def write_dataset(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(dict(zip(FIELDS, r, strict=True)) for r in rows)
    return path


@pytest.fixture
def dataset(tmp_path):
    # The fake frontier answers "[frontier] answer to: <prompt>".
    return write_dataset(
        tmp_path / "data.csv",
        [
            ("A1", "document_qa", "exact_match", "hi", '["frontier answer to hi"]', "squad"),
            ("A2", "document_qa", "exact_match", "hello", '["something else"]', "synthetic"),
            ("B1", "misc_planning", "rubric", "plan a trip", "A reasonable plan.", "handwritten"),
            ("C1", "coding", "exact_match", f"boom {FAKE_DOWN}", '["x"]', "mbpp"),
        ],
    )


def config(tmp_path, dataset, **kw):
    settings = Settings(backend="fake", eval_mode=True, strategy="baseline")
    return RunConfig(strategy="baseline", out_dir=tmp_path / "run", settings=settings, dataset=dataset, **kw)


def go(cfg, judge=None):
    return asyncio.run(run_eval(cfg, judge=judge or FakeJudge()))


def test_run_writes_calls_grades_manifest_summary(tmp_path, dataset):
    out = go(config(tmp_path, dataset, runs=2))
    calls, grades = read_jsonl(out / "calls.jsonl"), read_jsonl(out / "grades.jsonl")
    assert len(calls) == len(grades) == 8

    assert grades[("A1", 1)]["passed"] and not grades[("A2", 1)]["passed"]
    assert grades[("B1", 2)]["grader"] == "rubric" and grades[("B1", 2)]["passed"]
    # A request the service couldn't answer is kept and graded wrong.
    assert calls[("C1", 1)]["status"] == 502 and calls[("C1", 1)]["triage"]["calls"]
    assert grades[("C1", 1)]["grader"] == "no_answer" and not grades[("C1", 1)]["passed"]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["strategy"] == "baseline" and manifest["tiers"]["frontier"]["key"] == "fake-frontier"
    assert manifest["dataset_sha256"] and manifest["git"]["commit"]

    summary = json.loads((out / "summary.json").read_text())["overall"]
    assert summary["n"] == 8 and summary["accuracy"] == 0.5 and summary["failed_requests"] == 2
    assert summary["final_tier"] == {"frontier": 8} and summary["cost_usd"] > 0

    by_set = json.loads((out / "summary.json").read_text())["by_set"]
    assert by_set["synthetic"]["n"] == 2 and by_set["synthetic"]["accuracy"] == 0
    assert by_set["public"]["n"] == 6


def test_resume_skips_answered_and_retries_failed(tmp_path, dataset):
    cfg = config(tmp_path, dataset, runs=1)
    out = go(cfg)
    lines = (out / "calls.jsonl").read_text().count("\n")
    go(cfg)
    # Only the failed item (C1) is sent again.
    assert (out / "calls.jsonl").read_text().count("\n") == lines + 1


def test_more_runs_extend_an_existing_run(tmp_path, dataset):
    go(config(tmp_path, dataset, runs=1))
    out = go(config(tmp_path, dataset, runs=2))
    assert {run for _, run in read_jsonl(out / "calls.jsonl")} == {1, 2}


def test_resume_refuses_a_different_version(tmp_path, dataset):
    go(config(tmp_path, dataset, runs=1))
    cfg = config(tmp_path, dataset, runs=1)
    cfg.strategy = "v2"
    with pytest.raises(SystemExit, match="strategy"):
        go(cfg)


def test_max_cost_stops_sending(tmp_path, dataset):
    out = go(config(tmp_path, dataset, runs=3, max_cost=1e-9, concurrency=1))
    assert len(read_jsonl(out / "calls.jsonl")) == 1


def test_rubric_without_judge_is_left_ungraded_then_regraded(tmp_path, dataset):
    cfg = config(tmp_path, dataset, runs=1)
    out = asyncio.run(run_eval(cfg, judge=None))
    assert "error" in read_jsonl(out / "grades.jsonl")[("B1", 1)]

    judge = FakeJudge()
    summary = asyncio.run(harness.regrade(out, judge, dataset=dataset))
    assert judge.calls == 1 and summary["overall"]["grading_errors"] == 0
    assert read_jsonl(out / "grades.jsonl")[("B1", 1)]["passed"]


def test_pilot_sample_covers_every_use_case():
    items = load_items()
    picked = sample(items, 10)
    assert len(picked) == 10
    assert {i["use_case"] for i in picked} == {i["use_case"] for i in items} or len(picked) < 7


def test_latency_uses_run_1_only(tmp_path, dataset):
    out = go(config(tmp_path, dataset, runs=2))
    rows = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    for row in rows:  # make run 2 look much faster, as a cache hit would
        if row["run"] == 2 and row["triage"]:
            row["triage"]["totals"]["latency_ms"] = 0.001
    (out / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert harness.summarize(out)["overall"]["latency_p50_ms"] > 0.01
