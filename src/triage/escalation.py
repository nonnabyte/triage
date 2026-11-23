"""v3: a classifier decides, from the cheap answer's own signals, whether escalating is worth its cost.

The request is answered on cheap and asked once more. A small logistic model then estimates the
probability that sending the request to frontier would fix a wrong answer, from signals no answer key is
needed for: the task type and length, what the free checks found, how many spans an extraction returned,
and whether the two cheap answers agree. The request escalates when the estimated frontier cost per fixed
answer is within `classifier.max_usd_per_fix`. A long contract costs much more to escalate than a short
question, so it needs a higher probability to be worth it.

Why not decide before the call: prompt-only predictors (surface features, embeddings, nearest neighbours)
reach an AUC of about 0.62 on held-out documents here and in published results; adding the answer's
signals reaches about 0.78. Embeddings added nothing once the answer signals were in, so there are none.

Train with `uv run python -m triage.eval.train_classifier`; the model is a small JSON file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from triage.checks import CHECKS, CheckContext, _json, _open_ended, _same_answer, _wants_span_list
from triage.config import Price
from triage.features import estimated_tokens
from triage.schemas import TASK_TYPES, CheckResult, Tier

log = logging.getLogger("triage.escalation")

FREE_CHECKS = ("json_format", "grounding", "span_count", "examples")
_DECLINES = re.compile(
    r"NOT_IN_DOCUMENT|\b(not (stated|mentioned|specified|provided|found|addressed)|no (answer|information)|"
    r"cannot be determined|unanswerable)\b",
    re.IGNORECASE,
)
FEATURES = [
    *(f"task={t}" for t in (*TASK_TYPES, "unknown")),
    "log_tokens",
    *(f"fails={name}" for name in FREE_CHECKS),
    "spans=0",
    "spans=1",
    "spans=2+",
    "declines",
    "log_answer_chars",
    "agrees",
]


async def signals(ctx: CheckContext, second: str | None) -> dict[str, float]:
    """What is known about a cheap answer without an answer key. `second` is another answer from the same
    model, or None when there is none to compare (counted as no agreement)."""
    text = "\n".join(p for p in (ctx.task.system, ctx.task.user) if p)
    f = {name: 0.0 for name in FEATURES}
    f[f"task={ctx.profile.task_type}"] = 1.0
    f["log_tokens"] = math.log1p(estimated_tokens(text))
    for name in FREE_CHECKS:
        f[f"fails={name}"] = float(not (await CHECKS[name](ctx)).passed)
    if ctx.profile.expected_output == "json" and _wants_span_list(ctx.task.user):
        try:
            spans = _json(ctx.output)
            n = len(spans) if isinstance(spans, list) else -1
        except ValueError:
            n = -1
        f["spans=0"], f["spans=1"], f["spans=2+"] = float(n == 0), float(n == 1), float(n >= 2)
    f["declines"] = float(bool(_DECLINES.search(ctx.output)))
    f["log_answer_chars"] = math.log1p(len(ctx.output))
    if second is not None and second.strip():
        same, _ = await asyncio.to_thread(_same_answer, ctx, second)
        f["agrees"] = float(same)
    return f


def probability(model: dict[str, Any], f: dict[str, float]) -> float:
    """P(escalating fixes the answer): the cheap answer is wrong and frontier's would be right."""
    z = model["intercept"]
    for name, mean, scale, coef in zip(
        model["features"], model["mean"], model["scale"], model["coef"], strict=True
    ):
        z += coef * (f.get(name, 0.0) - mean) / scale
    return 1.0 / (1.0 + math.exp(-z))


def frontier_cost(text: str, task_type: str, price: Price, model: dict[str, Any]) -> float | None:
    """Estimated dollars for frontier to answer: input from the text (4 characters per token), output from
    the typical answer length for the task type in the training runs. None when the price is unknown."""
    out = model["output_tokens"].get(task_type, model["output_tokens"]["default"])
    return price.cost(int(estimated_tokens(text)), int(out))


def should_escalate(p: float, cost: float | None, max_usd_per_fix: float) -> bool:
    """Escalate when a fixed answer is expected to cost no more than the limit. Unknown cost: escalate."""
    return cost is None or cost <= max_usd_per_fix * p


@lru_cache(maxsize=4)
def load_model(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


async def escalation_model(ctx: CheckContext) -> CheckResult:
    """The v3 check: fails (escalates) when the classifier expects escalating to be worth its cost. Any
    doubt escalates: no trained model, no second answer and no price all fail towards frontier."""
    name = "escalation_model"
    cfg = ctx.classifier
    model = load_model(cfg.artifact) if cfg else None
    if model is None:
        return CheckResult(name=name, passed=False, detail="no trained classifier: escalating")
    second = None
    if not _open_ended(ctx):
        call = await ctx.ask_model(ctx.task.user, system=ctx.task.system, tier=ctx.tier)
        second = call.output if call.error is None else None
    p = probability(model, await signals(ctx, second))
    entries = ctx.client.entries if ctx.client else {}
    price = entries[Tier.frontier].price if Tier.frontier in entries else Price()
    text = "\n".join(part for part in (ctx.task.system, ctx.task.user) if part)
    cost = frontier_cost(text, ctx.profile.task_type, price, model)
    limit = cfg.max_usd_per_fix if cfg.max_usd_per_fix is not None else model["max_usd_per_fix"]
    escalate = should_escalate(p, cost, limit)
    per_fix = f"${cost / max(p, 1e-6):.3f} per fixed answer" if cost is not None else "frontier price unknown"
    detail = f"P(escalating fixes it)={p:.2f}, {per_fix}, limit ${limit:.3f}"
    return CheckResult(name=name, passed=not escalate, detail=detail)


CHECKS["escalation_model"] = escalation_model
