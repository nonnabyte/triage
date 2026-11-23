"""The router's core loop: profile → decide → call → check → escalate.

Versions without checks (baseline, v1) return the first answer. Versions with checks (v2, v3) move one tier
up when an answer fails its check, times out or errors. Frontier is the last tier and is never checked.
"""

from __future__ import annotations

import itertools
import logging
import time

from triage import escalation, tracing  # noqa: F401  (escalation registers the v3 check)
from triage.checks import CHECKS, CheckContext, run_checks
from triage.config import StrategiesConfig
from triage.models import ModelClient
from triage.policies import choose_start
from triage.profile import build_profile
from triage.schemas import Call, CheckOutcome, CheckResult, Task, Totals, TriageBlock

log = logging.getLogger("triage.cascade")


class TierNotConfigured(ValueError):
    """The version needs a tier that has no model in configs/models.yaml."""


class CascadeFailed(RuntimeError):
    """The final answer call failed (error or timeout). Carries the block so the spend is still reported."""

    def __init__(self, block: TriageBlock, error: str):
        super().__init__(error)
        self.block = block


def validate_strategies(strategies: StrategiesConfig) -> None:
    for name, cfg in strategies.versions.items():
        unknown = sorted(c for c in cfg.all_checks() if c not in CHECKS)
        if unknown:
            raise ValueError(f"configs/strategies.yaml: {name} uses unknown checks {unknown}")


def _totals(calls: list[Call], latency_ms: float) -> Totals:
    priced = [c for c in calls if c.cost_usd is not None]
    return Totals(
        cost_usd=sum(c.cost_usd for c in priced),
        wasted_cost_usd=sum(c.cost_usd for c in priced if c.check is not None and not c.check.passed),
        overhead_cost_usd=sum(c.cost_usd for c in priced if c.purpose == "judge"),
        latency_ms=latency_ms,
        # Calls that errored report no tokens, so only answered calls need a price to be complete.
        cost_complete=all(c.cost_usd is not None for c in calls if c.error is None),
    )


async def run(
    task: Task, strategy: str, client: ModelClient, strategies: StrategiesConfig
) -> tuple[str, TriageBlock]:
    """Return the final answer and the triage block. Raises CascadeFailed if the last answer call failed."""
    start_time = time.perf_counter()
    cfg = strategies.versions[strategy]
    profile = build_profile(task)
    decision = await choose_start(cfg, task, profile, strategies, client)

    ladder = client.ladder()
    if decision.start_tier not in ladder:
        raise TierNotConfigured(f"tier '{decision.start_tier}' has no model; set it in configs/models.yaml")
    check_names = cfg.checks_for(profile.task_type)
    escalate = bool(check_names)

    counter = itertools.count(1)
    calls: list[Call] = []  # answer calls and judge calls, in the order they happened
    answers: list[Call] = []
    tier = decision.start_tier
    while True:
        call = await client.complete(tier, task, seq=next(counter))
        calls.append(call)
        answers.append(call)
        if not escalate or tier == ladder[-1]:
            break
        if call.error is not None:
            call.check = CheckOutcome(
                passed=False, results=[CheckResult(name="call", passed=False, detail=call.error)]
            )
        else:
            ctx = CheckContext(
                task=task,
                profile=profile,
                output=call.output or "",
                tier=tier,
                client=client,
                next_seq=lambda: next(counter),
                classifier=strategies.classifier,
            )
            with tracing.trace_check(tier) as span:
                call.check = await run_checks(check_names, ctx)
                tracing.record_check(span, call.check)
            calls.extend(ctx.extra_calls)
        if call.check.passed:
            break
        tier = ladder[ladder.index(tier) + 1]

    final = answers[-1]
    block = TriageBlock(
        strategy=strategy,
        profile=profile,
        decision=decision,
        final_tier=final.tier,
        escalation_path=[c.tier for c in answers],
        calls=calls,
        # A reused frontier answer took no time here; count the time it took when it was first made.
        totals=_totals(
            calls, (time.perf_counter() - start_time) * 1000 + sum(c.latency_ms for c in calls if c.reused)
        ),
    )
    log.info(
        "%s %s -> %s | %d calls | cost $%.6f%s | %.0f ms%s",
        strategy,
        profile.task_type,
        " -> ".join(block.escalation_path),
        len(calls),
        block.totals.cost_usd,
        "" if block.totals.cost_complete else " (some prices unknown)",
        block.totals.latency_ms,
        f" | FAILED: {final.error}" if final.error else "",
    )
    if final.error is not None:
        raise CascadeFailed(block, final.error)
    return final.output or "", block
