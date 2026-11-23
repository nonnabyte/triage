"""Score candidate v2 check lists on saved answers: no model calls.

  uv run python -m triage.eval.checks_report
  uv run python -m triage.eval.checks_report --external

v2 answers on cheap, checks the answer and escalates to frontier when a check fails. Here each check list
is replayed on the cheap model's saved answers. A second opinion (agreement) is taken from saved answers
too: the cheap model's next run for self-agreement, another model's run for cross-agreement, and its saved
cost is charged. Per task type the report shows, for every candidate list, the accuracy, the saving, how
many answers escalate and how many kept answers are still wrong. The list in configs/strategies.yaml was
chosen from this table and is re-scored on held-out documents at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triage.checks import CHECKS, CheckContext, _same_answer
from triage.config import REPO_ROOT, Settings, load_strategies
from triage.eval.harness import DATASET, read_jsonl
from triage.eval.offline import document_group, grouped_folds
from triage.profile import build_profile
from triage.schemas import Task

RESULTS = REPO_ROOT / "results"
VERIFIERS = {
    "gemma-3-12b": "cheap-gemma-3-12b",
    "gemma-3-27b": "cheap-gemma-3-27b",
    "gpt-oss-120b": "cheap-gpt-oss-120b-low",
    "qwen3-32b": "cheap-qwen3-32b",
    "qwen3-coder-30b": "cheap-qwen3-coder-30b",
}
FREE = ["non_empty", "refusal", "json_format"]
SECOND = ("agreement", "cross_agreement")


def candidates(task_type: str, verifiers: list[str]) -> dict[str, list[str]]:
    """Check lists worth comparing for a task type. `cross:<model>` is cross_agreement with that verifier."""
    base = FREE + {"contract_extraction": ["grounding"], "coding": ["examples"]}.get(task_type, [])
    out = {"no checks": [], "free checks": base}
    if task_type == "contract_extraction":
        out["free + span_count"] = [*base, "span_count"]
    out["free + self-agreement"] = [*base, "agreement"]
    for v in verifiers:
        out[f"free + cross:{v}"] = [*base, f"cross:{v}"]
        if task_type == "contract_extraction":
            out[f"free + span_count + cross:{v}"] = [*base, "span_count", f"cross:{v}"]
    return out


@dataclass
class Replay:
    items: list[dict[str, str]]
    runs: tuple[int, ...]
    cheap: dict
    cheap_grade: dict
    frontier: dict
    frontier_grade: dict
    second: dict[str, dict]  # verifier name -> its saved calls; "self" is the cheap run itself

    def __post_init__(self) -> None:
        self.by_id = {r["id"]: r for r in self.items}
        self.profile = {r["id"]: build_profile(Task(user=r["prompt"])) for r in self.items}
        self.keys = [(r["id"], run) for r in self.items for run in self.runs]
        self._check: dict[tuple, bool] = {}

    def partner_run(self, key: tuple) -> int:
        """Another saved run of the cheap model on the same prompt, for self-agreement."""
        saved = sorted(run for (i, run), c in self.cheap.items() if i == key[0] and c.get("triage"))
        others = [run for run in saved if run != key[1]]
        later = [run for run in others if run > key[1]]
        return (later or others or [key[1]])[0]

    @property
    def has_second_sample(self) -> bool:
        return "self" in self.second or all(self.partner_run(k) != k[1] for k in self.keys)

    def cost(self, calls: dict, key: tuple) -> float:
        return calls[key]["triage"]["totals"]["cost_usd"] or 0.0

    async def passes(self, key: tuple, name: str) -> tuple[bool, float]:
        """Whether the cheap answer for `key` passes check `name`, and what the check cost."""
        item, profile = self.by_id[key[0]], self.profile[key[0]]
        answer = self.cheap[key].get("answer") or ""
        extra = 0.0
        if name == "agreement" or name.startswith("cross:"):
            if profile.expected_output == "text" and profile.task_type == "unknown":
                return True, 0.0
            if name == "agreement" and "self" in self.second:  # a separate second run of the cheap model
                other_key, calls = key, self.second["self"]
            elif name == "agreement":  # the same model's next saved run
                other_key, calls = (key[0], self.partner_run(key)), self.cheap
            else:
                other_key, calls = key, self.second[name.split(":", 1)[1]]
            other = calls[other_key].get("answer") or ""
            extra = self.cost(calls, other_key)
            cache = (key, name)
            if cache not in self._check:
                ctx = CheckContext(Task(user=item["prompt"]), profile, answer, None, lambda: 0)
                same = True if not other.strip() else await asyncio.to_thread(_same_answer, ctx, other)
                self._check[cache] = same if isinstance(same, bool) else same[0]
            return self._check[cache], extra
        cache = (key, name)
        if cache not in self._check:
            ctx = CheckContext(Task(user=item["prompt"]), profile, answer, None, lambda: 0)
            self._check[cache] = (await CHECKS[name](ctx)).passed
        return self._check[cache], 0.0

    async def route(self, key: tuple, names: list[str]) -> tuple[bool, float]:
        """(kept on cheap, total cost) for one answer under a check list; checks stop at the first failure."""
        cost = self.cost(self.cheap, key)
        for name in names:
            ok, extra = await self.passes(key, name)
            cost += extra
            if not ok:
                return False, cost + self.cost(self.frontier, key)
        return True, cost

    async def score(self, keys: list[tuple], names_for: dict[str, list[str]]) -> dict[str, float]:
        sem = asyncio.Semaphore(8)

        async def one(key: tuple) -> tuple[bool, float]:
            async with sem:
                return await self.route(key, names_for[self.profile[key[0]].task_type])

        routed = await asyncio.gather(*(one(k) for k in keys))
        kept = [k for k, (keep, _) in zip(keys, routed, strict=True) if keep]
        passed = sum(
            bool((self.cheap_grade if keep else self.frontier_grade)[k]["passed"])
            for k, (keep, _) in zip(keys, routed, strict=True)
        )
        frontier_cost = sum(self.cost(self.frontier, k) for k in keys)
        return {
            "accuracy": 100 * passed / len(keys),
            "frontier": 100 * sum(bool(self.frontier_grade[k]["passed"]) for k in keys) / len(keys),
            "saved": 100 * (1 - sum(c for _, c in routed) / frontier_cost),
            "escalated": 100 * (1 - len(kept) / len(keys)),
            "kept_wrong": 100 * sum(not self.cheap_grade[k]["passed"] for k in kept) / max(len(kept), 1),
        }


def load(
    dataset: Path, cheap: str, frontier: str, runs: tuple[int, ...], verifiers: dict[str, str]
) -> Replay:
    with dataset.open(newline="") as f:
        items = list(csv.DictReader(f))
    need = {(r["id"], run) for r in items for run in runs}
    second = {}
    for name, run_dir in verifiers.items():
        calls = read_jsonl(RESULTS / run_dir / "calls.jsonl")
        if need <= {k for k, c in calls.items() if c.get("triage")}:
            second[name] = calls
    return Replay(
        items,
        runs,
        read_jsonl(RESULTS / cheap / "calls.jsonl"),
        read_jsonl(RESULTS / cheap / "grades.jsonl"),
        read_jsonl(RESULTS / frontier / "calls.jsonl"),
        read_jsonl(RESULTS / frontier / "grades.jsonl"),
        second,
    )


def row(name: str, r: dict[str, float]) -> str:
    gap = r["accuracy"] - r["frontier"]
    return (
        f"  {name:44s} acc {r['accuracy']:5.1f} ({gap:+5.1f})  saved {r['saved']:4.0f}%  "
        f"escalated {r['escalated']:3.0f}%  kept answers wrong {r['kept_wrong']:3.0f}%"
    )


def configured(strategy: str, verifier: str | None) -> dict[str, list[str]]:
    """The check list per task type from configs/strategies.yaml, with cross_agreement bound to a verifier."""
    cfg = load_strategies(Settings().strategies_config).versions[strategy]

    class Lists(dict):
        def __missing__(self, task_type: str) -> list[str]:
            names = cfg.checks_for(task_type)
            return [f"cross:{verifier}" if n == "cross_agreement" else n for n in names]

    return Lists()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--external", action="store_true", help="score on the external test set instead")
    ap.add_argument("--verifier", default="gemma-3-12b", help="model bound to cross_agreement in the config")
    args = ap.parse_args()
    if args.external:
        verifiers = {k: f"ext-{v}" for k, v in VERIFIERS.items()} | {
            "self": "ext-cheap-gpt-oss-20b-low-second"
        }
        replay = load(
            REPO_ROOT / "data" / "triage_external_v1.csv",
            "ext-cheap-gpt-oss-20b-low",
            "ext-baseline-low-api",
            (1,),
            verifiers,
        )
    else:
        replay = load(DATASET, "cheap-gpt-oss-20b-low", "baseline-low", (1, 2, 3), VERIFIERS)
    print(f"second opinions with saved answers: {sorted(replay.second) or 'none'}")
    by_type: dict[str, list[tuple]] = defaultdict(list)
    for key in replay.keys:
        by_type[replay.profile[key[0]].task_type].append(key)
    for task_type, keys in sorted(by_type.items()):
        if len(keys) < 9:
            continue
        print(f"\n{task_type} ({len(keys)} answers)")
        for name, names in candidates(task_type, sorted(set(replay.second) - {"self"})).items():
            if "agreement" in names and not replay.has_second_sample:
                continue  # self-agreement needs a second saved run of the same model
            print(row(name, await replay.score(keys, defaultdict(lambda names=names: names))))
    verifier = args.verifier if args.verifier in replay.second else None
    lists = configured("v2", verifier)
    if any(n.startswith("cross:None") for t in by_type for n in lists[t]):
        print(f"\nconfigured v2 not scored: no saved answers for verifier {args.verifier}")
        return
    if not replay.has_second_sample:
        print("\nconfigured v2 not scored: self-agreement needs a second saved run of the cheap model")
        return
    print(f"\nv2 as configured (cross_agreement = {verifier})")
    print(row("all task types", await replay.score(replay.keys, lists)))
    if len(replay.runs) > 1:
        groups = np.array(
            [document_group(r["prompt"], replay.profile[r["id"]].task_type) for r in replay.items]
        )
        ids = [r["id"] for r in replay.items]
        public = [k for k in replay.keys if replay.by_id[k[0]]["source"] != "synthetic"]
        print(row("public prompts", await replay.score(public, lists)))
        synthetic = [k for k in replay.keys if replay.by_id[k[0]]["source"] == "synthetic"]
        print(row("synthetic prompts", await replay.score(synthetic, lists)))
        folds = []
        for _, test in grouped_folds(groups, 5, 0):
            held = {ids[n] for n in test}
            folds.append((await replay.score([k for k in replay.keys if k[0] in held], lists))["accuracy"])
        print(f"  accuracy across 5 document folds: {min(folds):.1f} to {max(folds):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
