"""Build the evaluation set: 260 public prompts + 40 synthetic = 300.

  uv run python -m triage.eval.build_dataset

Keeps 90 prompts per main use case and 30 misc. To make room for the synthetic prompts, 40 public prompts
are dropped (12 per main use case, 4 misc), chosen at random with a fixed seed:
  - the team's handwritten prompts are never dropped;
  - within a use case, drops are split across easy/hard in proportion (coding: equally, to stay 39/39),
    then across sources in proportion, so the mix of each slice is unchanged.
Writes data/triage_eval_v3.csv and data/triage_eval_v3_dropped.txt (the dropped ids, for audit).
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict

from triage.config import REPO_ROOT
from triage.eval import synthetic

SEED = 2701
PUBLIC = REPO_ROOT / "data" / "triage_eval_v2.csv"
COMBINED = REPO_ROOT / "data" / "triage_eval_v3.csv"
DROPPED = REPO_ROOT / "data" / "triage_eval_v3_dropped.txt"
DROP = {"document_qa": 12, "contract_extraction": 12, "coding": 12, "misc": 4}
KEEP_SOURCES = {"handwritten"}


def group(row: dict[str, str]) -> str:
    return "misc" if row["use_case"].startswith("misc") else row["use_case"]


def split(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """Share `total` across keys in proportion to sizes (largest remainder; ties by key)."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def choose_drops(public: list[dict[str, str]], seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    strata: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in public:
        if row["source"] not in KEEP_SOURCES:
            strata[group(row)][row["label"]][row["source"]].append(row["id"])
    dropped = []
    for use_case, k in DROP.items():
        by_label = strata[use_case]
        sizes = {label: sum(map(len, sources.values())) for label, sources in by_label.items()}
        per_label = {label: k // len(sizes) for label in sizes} if use_case == "coding" else split(k, sizes)
        for label, n_label in sorted(per_label.items()):
            sources = by_label[label]
            for source, n in sorted(split(n_label, {s: len(ids) for s, ids in sources.items()}).items()):
                dropped += rng.sample(sorted(sources[source]), n)
    return sorted(dropped)


def build() -> tuple[list[dict[str, str]], list[str]]:
    with PUBLIC.open(newline="") as f:
        public = list(csv.DictReader(f))
    dropped = set(choose_drops(public))
    rows = [r for r in public if r["id"] not in dropped] + synthetic.build()
    return rows, sorted(dropped)


def main() -> None:
    synthetic.main()
    rows, dropped = build()
    synthetic.write(COMBINED, rows)
    DROPPED.write_text("\n".join(dropped) + "\n")
    print(f"{len(rows)} rows -> {COMBINED.name} ({len(dropped)} public prompts dropped -> {DROPPED.name})")


if __name__ == "__main__":
    main()
