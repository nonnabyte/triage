"""Choose the first tier for a request, according to the version's `start` setting."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from triage.config import Price, StrategiesConfig, StrategyConfig
from triage.features import estimated_tokens, prompt_features
from triage.schemas import Decision, Profile, Task, Tier

if TYPE_CHECKING:
    from triage.models import ModelClient


# --- v1: rules --------------------------------------------------------------------------------------------

# v1 routes on the kind of task, not on the wording of the question. A request goes to cheap only when its
# task type is one the cheap model handles as well as frontier; everything else, including a request of
# unknown type, goes to frontier.
#
# Finer rules (question words, word overlap, worked examples) were tried and removed: they separated easy
# from hard prompts on the eval set but did no better than chance on the external test set
# (triage.eval.rules_report), which matches published results on prompt-surface features.
#
# S4 is the cost rule. No prompt signal separates the long extractions the cheap model gets right from the
# ones it gets wrong, so the risk is taken where it pays most: on requests whose estimated saving is at
# least `rules.min_saving_usd` (configs/strategies.yaml). That threshold is v1's accuracy/cost dial.
Rule = tuple[str, Callable[[dict[str, Any]], bool]]

_STAY_CHEAP: list[Rule] = [
    (
        "S1 math word problem, cheap model reasons",
        lambda f: f["task_type"] == "misc_math" and f["cheap_reasons"],
    ),
    ("S2 classification into listed labels", lambda f: f["task_type"] == "misc_classification"),
    (
        "S3 named fields from a document",
        lambda f: f["task_type"] == "contract_extraction" and f["wants_json"] and not f["wants_span_list"],
    ),
    (
        "S4 long extraction: estimated saving at or above the threshold",
        lambda f: (
            f["task_type"] == "contract_extraction"
            and f["saving"] is not None
            and f["saving"] >= f["min_saving"]
        ),
    ),
]
# Why a request that no stay-cheap rule covers goes to frontier. These explain the decision; they do not
# make it, because frontier is already the default.
_RISKS: list[Rule] = [
    ("U0 unknown task type", lambda f: f["task_type"] == "unknown"),
    (
        "M0 math word problem, cheap model does not reason",
        lambda f: f["task_type"] == "misc_math" and not f["cheap_reasons"],
    ),
    ("K0 code: the cheap model is markedly weaker", lambda f: f["task_type"] == "coding"),
    ("Q0 document QA: the cheap model is weaker, little to save", lambda f: f["task_type"] == "document_qa"),
    (
        "X1 extraction: estimated saving below the threshold",
        lambda f: (
            f["task_type"] == "contract_extraction"
            and f["saving"] is not None
            and f["saving"] < f["min_saving"]
        ),
    ),
]
DEFAULT_REASON = "default: frontier (no stay-cheap rule matched)"


def rules(
    task: Task,
    profile: Profile,
    cheap_reasons: bool,
    saving_usd: float | None = None,
    min_saving_usd: float = math.inf,
) -> tuple[Tier, list[str]]:
    """v1: cheap when a stay-cheap rule matches, otherwise frontier. Returns the tier and every rule that
    matched: the stay-cheap rules for a cheap decision, the risks for a frontier one.

    `saving_usd` is the estimated dollars saved by answering on cheap; None (prices unknown) switches the
    cost rule off."""
    f = prompt_features(task)
    f |= {"task_type": profile.task_type, "cheap_reasons": cheap_reasons}
    f |= {"saving": saving_usd, "min_saving": min_saving_usd}
    cheap = [name for name, rule in _STAY_CHEAP if rule(f)]
    if cheap:
        return Tier.cheap, cheap
    return Tier.frontier, [name for name, rule in _RISKS if rule(f)] or [DEFAULT_REASON]


def input_saving(task: Task, cheap: Price, frontier: Price) -> float | None:
    """Estimated dollars saved on the request's input by answering on cheap; None when a price is unknown.
    Input only: before the call the answer's length is not known, and on long documents input dominates."""
    if cheap.input is None or frontier.input is None:
        return None
    text = "\n".join(p for p in (task.system, task.user) if p)
    return estimated_tokens(text) * (frontier.input - cheap.input) / 1e6


async def choose_start(
    cfg: StrategyConfig,
    task: Task,
    profile: Profile,
    strategies: StrategiesConfig,
    client: ModelClient,
) -> Decision:
    if cfg.start == "frontier":
        return Decision(start_tier=Tier.frontier, reasons=["always frontier"])
    if cfg.start == "cheap":
        return Decision(start_tier=Tier.cheap, reasons=["always cheap"])
    if cfg.start == "rules":
        cheap, frontier = client.entries.get(Tier.cheap), client.entries.get(Tier.frontier)
        tier, reasons = rules(
            task,
            profile,
            cheap_reasons=bool(cheap and cheap.reasoning_levels),
            saving_usd=input_saving(task, cheap.price, frontier.price) if cheap and frontier else None,
            min_saving_usd=strategies.rules.min_saving_usd,
        )
        return Decision(start_tier=tier, reasons=reasons)
    raise ValueError(f"unknown start policy '{cfg.start}'")
