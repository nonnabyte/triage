"""Score the v1 rules on saved answers: no model calls.

  uv run python -m triage.eval.rules_report

Four tests, from the least to the most independent:
  1. the eval set, per rule and overall, with the cost threshold chosen on held-out documents;
  2. a shuffled control: the same number of prompts per task type sent to cheap, picked at random;
  3. re-worded eval prompts (triage.eval.retemplate): the decision must not depend on the template;
  4. the external test set (triage.eval.external), if its two runs are in results/: other datasets, other
     wording, never used to choose a rule or a threshold.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from triage.config import REPO_ROOT, Settings, load_models, load_strategies
from triage.eval.offline import Saved, grouped_folds, load_saved
from triage.eval.retemplate import variants
from triage.policies import input_saving, rules
from triage.profile import build_profile
from triage.schemas import Task, Tier

EXTERNAL = REPO_ROOT / "data" / "triage_external_v1.csv"
EXTERNAL_CHEAP = REPO_ROOT / "results" / "ext-cheap-gpt-oss-20b-low"
EXTERNAL_FRONTIER = REPO_ROOT / "results" / "ext-baseline-low-api"
THRESHOLDS = (float("inf"), 0.03, 0.025, 0.02, 0.0175, 0.015, 0.0125, 0.01, 0.0075, 0.005, 0.0025, 0.0)
MAX_DROP = 3.0  # accuracy points below frontier that the cost rule may give up


class Router:
    def __init__(self) -> None:
        settings = Settings()
        models = load_models(settings.models_config)
        self.cheap = models.catalog[models.tiers[Tier.cheap]]
        self.frontier = models.catalog[models.tiers[Tier.frontier]]
        self.threshold = load_strategies(settings.strategies_config).rules.min_saving_usd

    def decide(self, prompt: str, threshold: float | None = None) -> tuple[bool, list[str], str]:
        task = Task(user=prompt)
        profile = build_profile(task)
        tier, why = rules(
            task,
            profile,
            cheap_reasons=bool(self.cheap.reasoning_levels),
            saving_usd=input_saving(task, self.cheap.price, self.frontier.price),
            min_saving_usd=self.threshold if threshold is None else threshold,
        )
        return tier == Tier.cheap, why, profile.task_type

    def go_cheap(self, s: Saved, threshold: float | None = None) -> np.ndarray:
        return np.array([self.decide(r["prompt"], threshold)[0] for r in s.items])


def line(name: str, r: dict[str, float]) -> str:
    return (
        f"  {name:34s} acc {r['accuracy']:5.1f} ({r['accuracy'] - r['frontier_accuracy']:+5.1f})  "
        f"saved {r['saved']:3.0f}%  to cheap {r['to_cheap']:3.0f}%  "
        f"hard->cheap {r['hard_to_cheap']:3d}  easy->frontier {r['easy_to_frontier']:3d}"
    )


def subset(s: Saved, keep: np.ndarray) -> Saved:
    return Saved(
        [r for r, k in zip(s.items, keep, strict=True) if k],
        [t for t, k in zip(s.task_types, keep, strict=True) if k],
        s.groups[keep],
        s.cheap_pass[keep],
        s.frontier_pass[keep],
        s.cheap_cost[keep],
        s.frontier_cost[keep],
        s.runs,
    )


def overall(s: Saved, go: np.ndarray, slices: dict[str, np.ndarray]) -> None:
    print(line("frontier only", s.score(np.zeros(len(go), bool))))
    print(line("cheap only", s.score(np.ones(len(go), bool))))
    print(line("v1 rules", s.score(go)))
    for name, keep in slices.items():
        if keep.any():
            print(line(f"  v1 on {name} (n={keep.sum()})", subset(s, keep).score(go[keep])))


def per_rule(s: Saved, router: Router) -> None:
    by_rule: dict[tuple[str, str], list[int]] = defaultdict(list)
    for n, r in enumerate(s.items):
        cheap, why, _ = router.decide(r["prompt"])
        by_rule[("cheap" if cheap else "frontier", why[0])].append(n)
    print(f"  {'deciding rule':62s} {'n':>4s} {'cheap':>6s} {'front':>6s}")
    for (tier, rule), idx in sorted(by_rule.items()):
        k = np.array(idx)
        c = 100 * s.cheap_pass[k].sum() / (s.runs * len(k))
        f = 100 * s.frontier_pass[k].sum() / (s.runs * len(k))
        print(f"  {tier:8s} {rule[:53]:53s} {len(k):4d} {c:6.1f} {f:6.1f}")


def shuffled_control(s: Saved, go: np.ndarray, draws: int = 2000, seed: int = 0) -> None:
    """The same count of prompts per task type on cheap, chosen at random: what the rules must beat."""
    rng = np.random.RandomState(seed)
    types = np.array(s.task_types)
    acc, saved = [], []
    for _ in range(draws):
        pick = np.zeros(len(go), bool)
        for t in set(types):
            idx = np.where(types == t)[0]
            pick[rng.choice(idx, int(go[idx].sum()), replace=False)] = True
        r = s.score(pick)
        acc.append(r["accuracy"])
        saved.append(r["saved"])
    lo, hi = np.percentile(acc, [2.5, 97.5])
    r = s.score(go)
    beat = 100 * np.mean(np.array(acc) < r["accuracy"])
    print(
        f"  v1 acc {r['accuracy']:.1f} at {r['saved']:.0f}% saved | shuffled: acc {np.mean(acc):.1f} "
        f"(95% of draws {lo:.1f} to {hi:.1f}) at {np.mean(saved):.0f}% saved | v1 beats {beat:.0f}% of draws"
    )


def threshold_curve(s: Saved, router: Router) -> dict[float, np.ndarray]:
    table = {t: router.go_cheap(s, t) for t in THRESHOLDS}
    for t, go in table.items():
        print(line(f"min_saving_usd = {t:g}", s.score(go)))
    return table


def held_out_threshold(s: Saved, table: dict[float, np.ndarray], seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Choose the threshold on training documents (largest saving within MAX_DROP points of frontier),
    apply it to the held-out documents, and score the pooled held-out decisions."""
    chosen: Counter[float] = Counter()
    results = []
    for seed in seeds:
        go = np.zeros(len(s.items), bool)
        for train, test in grouped_folds(s.groups, 5, seed):
            keep = np.zeros(len(s.items), bool)
            keep[train] = True
            tr = subset(s, keep)
            ok = [
                (tr.score(table[t][train])["saved"], t)
                for t in THRESHOLDS
                if (r := tr.score(table[t][train]))["accuracy"] >= r["frontier_accuracy"] - MAX_DROP
            ]
            best = max(ok)[1]
            chosen[best] += 1
            go[test] = table[best][test]
        results.append(s.score(go))
    mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    mean["hard_to_cheap"], mean["easy_to_frontier"] = (
        int(mean["hard_to_cheap"]),
        int(mean["easy_to_frontier"]),
    )
    print(line("held-out (5 folds x 3 seeds)", mean))
    print(f"  thresholds chosen on the training folds: {dict(sorted(chosen.items()))}")


def invariance(items: list[dict[str, str]], router: Router) -> None:
    total, type_flips, decision_flips = Counter(), Counter(), Counter()
    for r in items:
        cheap0, _, type0 = router.decide(r["prompt"])
        for _, prompt in variants(r["use_case"], r["prompt"]):
            cheap1, _, type1 = router.decide(prompt)
            total[r["use_case"]] += 1
            type_flips[r["use_case"]] += type1 != type0
            decision_flips[r["use_case"]] += cheap1 != cheap0
    for use_case in sorted(total):
        print(
            f"  {use_case:22s} {total[use_case]:4d} re-worded prompts: task type changed on "
            f"{type_flips[use_case]}, decision changed on {decision_flips[use_case]}"
        )
    print(
        f"  {'all':22s} {sum(total.values()):4d} re-worded prompts: task type changed on "
        f"{sum(type_flips.values())}, decision changed on {sum(decision_flips.values())}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--external", type=Path, default=EXTERNAL)
    args = ap.parse_args()
    router = Router()
    s = load_saved()
    source = np.array([r["source"] for r in s.items])
    slices = {
        "public": ~np.isin(source, ["synthetic", "handwritten"]),
        "handwritten": source == "handwritten",
        "synthetic": source == "synthetic",
    }
    go = router.go_cheap(s)
    print(
        f"cheap = {router.cheap.key}, frontier = {router.frontier.key}, min_saving_usd = {router.threshold:g}"
    )
    print("\n1. Eval set (300 prompts x 3 runs, replayed on saved answers)")
    overall(s, go, slices)
    print("\n   By rule: accuracy of each model on the prompts the rule decides")
    per_rule(s, router)
    print("\n   The cost threshold (S4) is the accuracy/cost dial")
    table = threshold_curve(s, router)
    held_out_threshold(s, table)
    print("\n2. Shuffled control (same number of prompts per task type on cheap, at random)")
    shuffled_control(s, go)
    print("\n3. Re-worded eval prompts")
    invariance(s.items, router)
    print("\n4. External test set")
    if not (args.external.exists() and EXTERNAL_CHEAP.exists() and EXTERNAL_FRONTIER.exists()):
        print(
            "  not run: build it with triage.eval.external, then run the cheap model and the baseline on it"
        )
        return
    ext = load_saved(EXTERNAL_CHEAP, EXTERNAL_FRONTIER, args.external, runs=(1,))
    with args.external.open(newline="") as f:
        expected = [r["use_case"] for r in csv.DictReader(f)]
    right = sum(t == e for t, e in zip(ext.task_types, expected, strict=True))
    print(f"  task type recognised on {right} of {len(expected)} prompts")
    ext_go = router.go_cheap(ext)
    use_case = np.array(expected)
    overall(ext, ext_go, {u: use_case == u for u in sorted(set(expected))})
    print("\n   By rule")
    per_rule(ext, router)
    print("\n   Shuffled control")
    shuffled_control(ext, ext_go)


if __name__ == "__main__":
    main()
