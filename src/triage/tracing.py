"""Langfuse tracing: one trace per routed request, every model call a generation inside it.

On when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and TRIAGE_TRACING isn't false. PydanticAI's
OpenTelemetry instrumentation records each model call (model, tokens, thinking, latency); this module adds
the `route-request` root (user message in, final answer out, routing decision and our cost in metadata)
and a `check-answer` guardrail observation per check.

TRIAGE_TRACE_CONTENT=false keeps prompts and answers out of Langfuse: only names, tiers, tokens, cost and
timings are sent. Environment (LANGFUSE_TRACING_ENVIRONMENT) defaults to `eval` in eval mode, else
`development`, so harness runs don't mix with real traffic.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from triage.config import Settings
from triage.schemas import CheckOutcome, Task, TriageBlock

log = logging.getLogger("triage.tracing")

# PydanticAI also reports provider token breakdowns (e.g. Gemini's thoughts_tokens, already inside
# output_tokens) as gen_ai.usage.details.*; Langfuse adds those to the usage total and double-counts.
USAGE_DETAILS_PREFIX = "gen_ai.usage.details."

_enabled = False
_content = True


def setup_tracing(settings: Settings) -> bool:
    """Turn tracing on if configured. Call after load_env(), before any agent runs."""
    global _enabled, _content
    if not settings.tracing:
        return False
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval" if settings.eval_mode else "development")
    os.environ.setdefault("OTEL_SERVICE_NAME", "triage")
    from langfuse import Langfuse  # after the env is loaded, so the client sees the keys

    Langfuse(mask_otel_spans=_move_usage_details)
    Agent.instrument_all(InstrumentationSettings(include_content=settings.trace_content))
    _enabled, _content = True, settings.trace_content
    log.info("Langfuse tracing on (environment %s)", os.environ["LANGFUSE_TRACING_ENVIRONMENT"])
    return True


def _move_usage_details(*, params: Any) -> Any:
    """Keep provider token breakdowns visible as metadata instead of as extra usage."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for ident, span in params.spans.items():
        details = {k: v for k, v in span.attributes.items() if k.startswith(USAGE_DETAILS_PREFIX)}
        if details:
            patches[ident] = OtelSpanPatch(
                delete_attributes=tuple(details),
                set_attributes={
                    f"triage.usage.{k.removeprefix(USAGE_DETAILS_PREFIX)}": v for k, v in details.items()
                },
            )
    return MaskOtelSpansResult(span_patches=patches)


def flush() -> None:
    """Send buffered traces. Call before a short-lived process (the harness, a script) exits."""
    if _enabled:
        from langfuse import get_client

        get_client().flush()


class RequestTrace:
    """Handle on the `route-request` root; a no-op when tracing is off."""

    def __init__(self, observation: Any = None):
        self._obs = observation

    def finish(self, answer: str | None, block: TriageBlock | None, error: str | None = None) -> None:
        if self._obs is None:
            return
        metadata: dict[str, Any] = {}
        if block is not None:
            metadata = {
                "task_type": block.profile.task_type,
                "task_type_source": block.profile.task_type_source,
                "start_tier": block.decision.start_tier,
                "decision_reasons": block.decision.reasons,
                "escalation_path": block.escalation_path,
                "final_tier": block.final_tier,
                "models": [
                    f"{c.tier}:{c.model}" + (f"@{c.reasoning}" if c.reasoning else "") for c in block.calls
                ],
                # Our price from configs/models.yaml; Langfuse's own cost needs the model in its price table.
                "cost_usd": block.totals.cost_usd,
                "wasted_cost_usd": block.totals.wasted_cost_usd,
                "overhead_cost_usd": block.totals.overhead_cost_usd,
                "cost_complete": block.totals.cost_complete,
            }
        update: dict[str, Any] = {"metadata": metadata}
        if _content:
            update["output"] = answer
        if error is not None:
            update.update(level="ERROR", status_message=error)
        self._obs.update(**update)


@contextmanager
def trace_request(
    task: Task, strategy: str, *, session_id: str | None = None, metadata: dict[str, str] | None = None
) -> Iterator[RequestTrace]:
    """Wrap one routed request. Model calls made inside become generations under it."""
    if not _enabled:
        yield RequestTrace()
        return
    from langfuse import get_client, propagate_attributes

    with (
        get_client().start_as_current_observation(
            as_type="chain",
            name="route-request",
            input=task.user if _content else None,
            metadata={"strategy": strategy, **(metadata or {})},
        ) as obs,
        propagate_attributes(
            trace_name="route-request", session_id=session_id, tags=[f"strategy:{strategy}"]
        ),
    ):
        yield RequestTrace(obs)


@contextmanager
def trace_check(tier: str) -> Iterator[RequestTrace]:
    """Wrap the checks on one answer as a guardrail observation."""
    if not _enabled:
        yield RequestTrace()
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(
        as_type="guardrail", name="check-answer", metadata={"tier": tier}
    ) as obs:
        yield RequestTrace(obs)


def record_check(trace: RequestTrace, outcome: CheckOutcome) -> None:
    if trace._obs is not None:
        level: Literal["DEFAULT", "WARNING"] = "DEFAULT" if outcome.passed else "WARNING"
        trace._obs.update(output=outcome.model_dump(mode="json"), level=level)
