"""The synthetic set: regenerates identically, is balanced, and every answer is provably correct."""

import asyncio
import csv
import json
from collections import Counter

import pytest

from triage.eval import build_dataset, synthetic
from triage.eval.graders import grade
from triage.eval.sandbox import run_python

ROWS = synthetic.build()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_committed_files_match_the_generator():
    assert read(synthetic.SYNTHETIC) == ROWS, "rerun: uv run python -m triage.eval.build_dataset"
    rows, dropped = build_dataset.build()
    assert read(build_dataset.COMBINED) == rows
    assert build_dataset.DROPPED.read_text().split() == dropped


def test_size_balance_and_ids():
    assert len(ROWS) == 40
    assert Counter(r["label"] for r in ROWS) == {"easy": len(ROWS) // 2, "hard": len(ROWS) // 2}
    for case in ("document_qa", "contract_extraction", "coding"):
        labels = Counter(r["label"] for r in ROWS if r["use_case"] == case)
        assert labels["easy"] == labels["hard"], case
    ids = [r["id"] for r in read(build_dataset.COMBINED)]
    assert len(ids) == len(set(ids))


def test_prompts_use_the_public_instruction_wording():
    public = read(build_dataset.PUBLIC)
    for row in ROWS:
        # Same instruction, new domain: the classification items use an invented helpdesk label set.
        opening = row["prompt"].replace("staff helpdesk query", "customer banking query").split("\n")[0][:40]
        if row["check_type"] == "json_match":
            opening = "CONTRACT TEXT: '"
        assert any(p["prompt"].startswith(opening) for p in public), row["id"]


@pytest.mark.parametrize("row", [r for r in ROWS if r["check_type"] != "unit_tests"], ids=lambda r: r["id"])
def test_gold_answer_passes_its_grader(row):
    gold = json.loads(row["answer"])
    output = (
        json.dumps(gold)
        if row["check_type"] == "span_set"
        else (json.dumps(gold[0]) if row["check_type"] == "json_match" else str(gold[0]))
    )
    assert asyncio.run(grade(row, output)).passed


@pytest.mark.parametrize("task", synthetic.CODE_TASKS, ids=lambda t: t.name)
def test_reference_solution_passes_and_stub_fails(task):
    tests = synthetic.code_tests(task)
    assert run_python(task.reference + "\n\n" + tests).ok, "reference solution fails its own tests"
    assert not run_python(task.stub + "    pass\n\n" + tests).ok, "tests pass without an implementation"


def test_unanswerable_and_absent_cases_exist():
    answers = [r["answer"] for r in ROWS]
    assert '["NOT_IN_DOCUMENT"]' in answers and "[]" in answers


def test_eval_set_is_300_with_40_synthetic_and_the_same_shape():
    rows = read(build_dataset.COMBINED)
    public = read(build_dataset.PUBLIC)
    assert len(rows) == 300 and sum(r["source"] == "synthetic" for r in rows) == 40
    groups = Counter(build_dataset.group(r) for r in rows)
    assert groups == {"document_qa": 90, "contract_extraction": 90, "coding": 90, "misc": 30}
    # The team's handwritten prompts are all kept; only public dataset items were dropped.
    kept = {r["id"] for r in rows}
    assert all(r["id"] in kept for r in public if r["source"] == "handwritten")
    coding = Counter(r["label"] for r in rows if r["use_case"] == "coding" and r["source"] != "synthetic")
    assert coding["easy"] == coding["hard"] == 39
