"""Offline routing analysis on saved answers: no model calls.

A routing decision is replayed by taking each prompt's saved answer, grade and cost from the cheap-only run
or from the frontier baseline. This is how rules, checks and the classifier are designed and validated before
a live run; the live run then confirms the estimate.

  uv run python -m triage.eval.offline            prints the reference points (frontier, cheap, hindsight)
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triage.config import REPO_ROOT
from triage.eval.harness import DATASET, read_jsonl
from triage.profile import build_profile
from triage.schemas import Task

CHEAP_RUN = REPO_ROOT / "results" / "cheap-gpt-oss-20b-low"
FRONTIER_RUN = REPO_ROOT / "results" / "baseline-low"
RUNS = (1, 2, 3)


def document_group(prompt: str, task_type: str) -> str:
    """An id for the source document, so prompts about the same passage or contract share a fold."""
    body = prompt
    if task_type == "document_qa" and (m := re.search(r"PASSAGE:\s*(.*?)\s*QUESTION:", prompt, re.S)):
        body = m.group(1)
    elif task_type == "contract_extraction" and (
        m := re.search(r"CONTRACT(?: TEXT)?:\s*(.*?)(?:\nTASK:|$)", prompt, re.S)
    ):
        body = m.group(1)
    return hashlib.md5(body[:2000].encode()).hexdigest()


@dataclass
class Saved:
    """Per-prompt outcomes of the cheap and frontier models (arrays aligned with `items`)."""

    items: list[dict[str, str]]
    task_types: list[str]
    groups: np.ndarray  # source-document id per prompt
    cheap_pass: np.ndarray  # runs passed, 0-3
    frontier_pass: np.ndarray
    cheap_cost: np.ndarray  # summed over the runs
    frontier_cost: np.ndarray
    runs: int = 3

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.items]

    @property
    def loss(self) -> np.ndarray:
        """1 where the cheap model passed fewer runs than frontier."""
        return (self.frontier_pass > self.cheap_pass).astype(float)

    def score(self, go_cheap: np.ndarray, extra_cost: float = 0.0) -> dict[str, float]:
        """Accuracy, cost and the two router errors when `go_cheap` prompts are answered by cheap."""
        go = np.asarray(go_cheap, dtype=bool)
        n = self.runs * len(self.items)
        cost = self.cheap_cost[go].sum() + self.frontier_cost[~go].sum() + extra_cost
        hard = self.cheap_pass < self.runs
        return {
            "accuracy": 100 * (self.cheap_pass[go].sum() + self.frontier_pass[~go].sum()) / n,
            "frontier_accuracy": 100 * self.frontier_pass.sum() / n,
            "cost": float(cost),
            "saved": 100 * (1 - cost / self.frontier_cost.sum()),
            "to_cheap": 100 * go.mean(),
            "hard_to_cheap": int((go & hard).sum()),
            "easy_to_frontier": int((~go & ~hard).sum()),
        }


def load_saved(
    cheap_run: Path = CHEAP_RUN,
    frontier_run: Path = FRONTIER_RUN,
    dataset: Path = DATASET,
    runs: tuple[int, ...] = RUNS,
) -> Saved:
    with dataset.open(newline="") as f:
        items = list(csv.DictReader(f))
    ids = [r["id"] for r in items]

    def run(path: Path) -> tuple[np.ndarray, np.ndarray]:
        grades = read_jsonl(path / "grades.jsonl")
        calls = read_jsonl(path / "calls.jsonl")
        passed = np.array([sum(bool(grades[(i, r)]["passed"]) for r in runs) for i in ids])
        cost = np.array([sum(calls[(i, r)]["triage"]["totals"]["cost_usd"] for r in runs) for i in ids])
        return passed, cost

    task_types = [build_profile(Task(user=r["prompt"])).task_type for r in items]
    cheap_pass, cheap_cost = run(cheap_run)
    frontier_pass, frontier_cost = run(frontier_run)
    groups = np.array([document_group(r["prompt"], t) for r, t in zip(items, task_types, strict=True)])
    return Saved(items, task_types, groups, cheap_pass, frontier_pass, cheap_cost, frontier_cost, len(runs))


def grouped_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """k train/test splits in which no source document appears on both sides."""
    unique = np.array(sorted(set(groups)))
    np.random.RandomState(seed).shuffle(unique)
    fold_of = {g: n % k for n, g in enumerate(unique)}
    fold = np.array([fold_of[g] for g in groups])
    return [(np.where(fold != n)[0], np.where(fold == n)[0]) for n in range(k)]


def main() -> None:
    s = load_saved()
    n = len(s.items)
    rows = [
        ("frontier only", np.zeros(n, bool)),
        ("cheap only", np.ones(n, bool)),
        ("hindsight (cheap wherever it is no worse)", s.cheap_pass >= s.frontier_pass),
    ]
    print(f"{'policy':44s} {'acc':>6s} {'saved':>6s} {'to cheap':>8s}")
    for name, go in rows:
        r = s.score(go)
        print(f"{name:44s} {r['accuracy']:6.1f} {r['saved']:5.0f}% {r['to_cheap']:7.0f}%")


if __name__ == "__main__":
    main()
