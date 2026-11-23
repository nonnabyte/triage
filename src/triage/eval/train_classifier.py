"""Train the v3 classifier: P(escalating fixes the answer), from the cheap answer's own signals.

  uv run python -m triage.eval.train_classifier
  uv run python -m triage.eval.train_classifier --cheap-run results/cheap-<model> --max-drop 2

Every saved cheap answer becomes one training row: the signals of triage.escalation (the same function the
service runs), with the cheap model's next saved run as the second answer. The label is 1 when escalating
would fix the answer: the cheap answer failed and frontier's passed. The model is a logistic regression.

The service escalates when estimated frontier cost / P <= max_usd_per_fix. That limit is chosen here on
held-out documents (5 folds grouped by source document, 3 seeds): the largest saving whose accuracy stays
within --max-drop points of frontier, picked on the training folds and scored on the held-out fold. The
score counts every call: the cheap answer, the second cheap answer, and frontier when escalated.

Writes artifacts/escalation.json: coefficients, feature scaling, the limit and what it was trained on. If the
external test set has been run, the saved model is scored on it too; nothing is fitted there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from triage import escalation
from triage.checks import CheckContext
from triage.config import REPO_ROOT, Settings, load_models
from triage.eval.checks_report import RESULTS, Replay, load
from triage.eval.harness import DATASET
from triage.eval.offline import document_group, grouped_folds
from triage.schemas import Task

DEFAULT_OUT = REPO_ROOT / "artifacts" / "escalation.json"
EXTERNAL = REPO_ROOT / "data" / "triage_external_v1.csv"
EXTERNAL_RUNS = ("ext-cheap-gpt-oss-20b-low", "ext-baseline-low-api", "ext-cheap-gpt-oss-20b-low-second")
LIMITS = (0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 1e9)
C = 0.3  # inverse regularisation strength; small data, so on the strong side


class Rows:
    """One row per saved cheap answer: its signals, what escalating would do, and what each call cost."""

    def __init__(self, replay: Replay, frontier_price, output_tokens: dict | None = None):
        self.replay, self.keys = replay, replay.keys
        self.x = np.array(asyncio.run(self._signals()))
        cheap_ok = np.array([bool(replay.cheap_grade[k]["passed"]) for k in self.keys])
        self.frontier_ok = np.array([bool(replay.frontier_grade[k]["passed"]) for k in self.keys])
        self.cheap_ok = cheap_ok
        self.fixes = (~cheap_ok & self.frontier_ok).astype(float)
        self.cheap_cost = np.array([replay.cost(replay.cheap, k) for k in self.keys])
        self.second_cost = np.array([self._second(k)[1] for k in self.keys])
        self.frontier_cost = np.array([replay.cost(replay.frontier, k) for k in self.keys])
        self.task_types = [replay.profile[k[0]].task_type for k in self.keys]
        self.output_tokens = output_tokens or self._output_medians()
        model = {"output_tokens": self.output_tokens}
        self.estimate = np.array(
            [
                escalation.frontier_cost(replay.by_id[k[0]]["prompt"], t, frontier_price, model) or 0.0
                for k, t in zip(self.keys, self.task_types, strict=True)
            ]
        )
        ids = [r["id"] for r in replay.items]
        groups = np.array(
            [document_group(r["prompt"], replay.profile[r["id"]].task_type) for r in replay.items]
        )
        self.groups, self.prompt = groups, np.array([ids.index(k[0]) for k in self.keys])

    def _second(self, key: tuple) -> tuple[str, float]:
        r = self.replay
        calls, other = (
            (r.second["self"], key) if "self" in r.second else (r.cheap, (key[0], r.partner_run(key)))
        )
        return calls[other].get("answer") or "", r.cost(calls, other)

    async def _signals(self) -> list[list[float]]:
        rows = []
        for key in self.keys:
            item, profile = self.replay.by_id[key[0]], self.replay.profile[key[0]]
            answer = self.replay.cheap[key].get("answer") or ""
            ctx = CheckContext(Task(user=item["prompt"]), profile, answer, None, lambda: 0)
            f = await escalation.signals(ctx, self._second(key)[0])
            rows.append([f[name] for name in escalation.FEATURES])
        return rows

    def _output_medians(self) -> dict[str, float]:
        out: dict[str, list[int]] = {"default": []}
        for key, t in zip(self.keys, self.task_types, strict=True):
            tokens = self.replay.frontier[key]["triage"]["calls"][0]["output_tokens"]
            out.setdefault(t, []).append(tokens)
            out["default"].append(tokens)
        return {t: float(np.median(v)) for t, v in out.items()}

    def score(self, escalate: np.ndarray, keep: np.ndarray | None = None) -> dict[str, float]:
        keep = np.ones(len(self.keys), bool) if keep is None else keep
        e = escalate[keep]
        cost = self.cheap_cost[keep] + self.second_cost[keep] + np.where(e, self.frontier_cost[keep], 0)
        return {
            "accuracy": 100 * np.where(e, self.frontier_ok[keep], self.cheap_ok[keep]).mean(),
            "frontier": 100 * self.frontier_ok[keep].mean(),
            "saved": 100 * (1 - cost.sum() / self.frontier_cost[keep].sum()),
            "escalated": 100 * e.mean(),
        }

    def escalate(self, p: np.ndarray, limit: float) -> np.ndarray:
        return self.estimate <= limit * p


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    lr = LogisticRegression(C=C, max_iter=5000).fit((x - scaler.mean_) / scale, y)
    return {
        "features": escalation.FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scale.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"]) + model[
        "intercept"
    ]
    return 1 / (1 + np.exp(-z))


def best_limit(rows: Rows, p: np.ndarray, keep: np.ndarray, max_drop: float) -> float:
    """The limit with the largest saving whose accuracy is within max_drop points of frontier."""
    ok = []
    for limit in LIMITS:
        r = rows.score(rows.escalate(p, limit), keep)
        if r["accuracy"] >= r["frontier"] - max_drop:
            ok.append((r["saved"], limit))
    return max(ok)[1] if ok else LIMITS[-1]


def line(name: str, r: dict[str, float]) -> str:
    gap = r["accuracy"] - r["frontier"]
    return (
        f"  {name:34s} acc {r['accuracy']:5.1f} ({gap:+5.1f})  saved {r['saved']:4.0f}%  "
        f"escalated {r['escalated']:3.0f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cheap-run", type=Path, default=RESULTS / "cheap-gpt-oss-20b-low")
    ap.add_argument("--frontier-run", type=Path, default=RESULTS / "baseline-low")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--max-drop", type=float, default=3.0, help="accuracy points below frontier allowed")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    replay = load(args.dataset, args.cheap_run.name, args.frontier_run.name, (1, 2, 3), {})
    cheap_key = json.loads((args.cheap_run / "manifest.json").read_text())["tiers"]["frontier"]["key"]
    frontier_key = json.loads((args.frontier_run / "manifest.json").read_text())["tiers"]["frontier"]["key"]
    frontier_price = load_models(Settings().models_config).catalog[frontier_key].price
    rows = Rows(replay, frontier_price)
    n = len(rows.keys)
    print(
        f"cheap {cheap_key} vs frontier {frontier_key}: {n} answers, escalating would fix "
        f"{int(rows.fixes.sum())}; frontier {100 * rows.frontier_ok.mean():.1f}%, "
        f"cheap {100 * rows.cheap_ok.mean():.1f}%"
    )

    # Held-out: fit and choose the limit on the training documents, score the held-out documents.
    seeds, held, aucs, chosen = (0, 1, 2), [], [], []
    p_out = np.zeros((len(seeds), n))
    for s, seed in enumerate(seeds):
        escalate = np.zeros(n, bool)
        for train, test in grouped_folds(rows.groups, 5, seed):
            tr, te = np.isin(rows.prompt, train), np.isin(rows.prompt, test)
            model = fit(rows.x[tr], rows.fixes[tr])
            limit = best_limit(rows, predict(model, rows.x), tr, args.max_drop)
            chosen.append(limit)
            p_out[s, te] = predict(model, rows.x[te])
            escalate[te] = rows.escalate(predict(model, rows.x), limit)[te]
        held.append(rows.score(escalate))
        aucs.append(roc_auc_score(rows.fixes, p_out[s]))
    mean = {k: float(np.mean([h[k] for h in held])) for k in held[0]}
    print(f"\nheld-out documents (5 folds x {len(seeds)} seeds), AUC {np.mean(aucs):.3f}")
    print(line("never escalate (cheap + second)", rows.score(np.zeros(n, bool))))
    print(line("v3, limit chosen on training folds", mean))
    print(f"  limits chosen: {sorted(set(chosen))}")
    print("\n  held-out predictions at each limit (max_usd_per_fix)")
    for limit in LIMITS:
        r = {
            k: float(np.mean([rows.score(rows.escalate(p_out[s], limit))[k] for s in range(len(seeds))]))
            for k in mean
        }
        print(line(f"limit ${limit:g}", r))

    model = fit(rows.x, rows.fixes)
    limit = float(np.median(chosen))
    model |= {
        "output_tokens": rows.output_tokens,
        "max_usd_per_fix": limit,
        "trained_on": {
            "cheap_run": args.cheap_run.name,
            "frontier_run": args.frontier_run.name,
            "cheap_model": cheap_key,
            "frontier_model": frontier_key,
            "dataset": args.dataset.name,
            "answers": n,
        },
        "held_out": {**mean, "auc": float(np.mean(aucs)), "max_drop": args.max_drop},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2) + "\n")
    print(f"\nsaved {args.out} (max_usd_per_fix = {limit:g})")
    print(
        "  largest weights: "
        + ", ".join(
            f"{name} {coef:+.2f}"
            for coef, name in sorted(
                zip(model["coef"], model["features"], strict=True), key=lambda c: -abs(c[0])
            )[:8]
        )
    )

    if EXTERNAL.exists() and all((RESULTS / r).exists() for r in EXTERNAL_RUNS):
        ext = load(EXTERNAL, EXTERNAL_RUNS[0], EXTERNAL_RUNS[1], (1,), {"self": EXTERNAL_RUNS[2]})
        ext_rows = Rows(ext, frontier_price, rows.output_tokens)
        p = predict(model, ext_rows.x)
        print(
            f"\nexternal test set ({len(ext_rows.keys)} answers, nothing fitted there), "
            f"AUC {roc_auc_score(ext_rows.fixes, p):.3f}"
        )
        print(line("never escalate (cheap + second)", ext_rows.score(np.zeros(len(p), bool))))
        print(line(f"v3 at the saved limit ${limit:g}", ext_rows.score(ext_rows.escalate(p, limit))))


if __name__ == "__main__":
    main()
