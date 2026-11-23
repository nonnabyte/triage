"""Calls one tier and returns a fully logged Call: tokens, cost, latency, retries, timeout."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse

from triage.config import ModelEntry, ModelsConfig, Settings
from triage.providers import build_model, fake_entry
from triage.schemas import LADDER, Call, Task, Tier

log = logging.getLogger("triage.models")

# Throttling, request timeout and server errors can succeed on a retry; auth, bad request or a
# model that isn't enabled (400/401/403/404) never will.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
RETRY_BASE_DELAY_S = 0.5


def is_transient(error: BaseException) -> bool:
    if isinstance(error, TimeoutError | httpx.TransportError | ConnectionError):
        return True
    if isinstance(error, ModelHTTPError):
        return error.status_code in RETRYABLE_STATUS
    # Other API errors without an HTTP status are network-level failures.
    return isinstance(error, ModelAPIError)


def saved_frontier_calls(run_dir: Path, entry: ModelEntry) -> dict[tuple[str, int], Call]:
    """The answered frontier calls of a baseline run, keyed by (item id, repeat). Only a run made with the
    same model at the same reasoning effort can stand in for a fresh call."""
    saved: dict[tuple[str, int], Call] = {}
    for line in (run_dir / "calls.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        calls = (row.get("triage") or {}).get("calls") or []
        if row.get("status") != 200 or len(calls) != 1:
            continue
        call = Call(**{**calls[0], "check": None})
        if call.model != entry.model or call.reasoning != entry.reasoning or call.error is not None:
            raise ValueError(
                f"{run_dir.name} was answered by {call.model} ({call.reasoning}), not by the frontier tier "
                f"{entry.model} ({entry.reasoning}): its answers cannot be reused"
            )
        saved[(row["id"], int(row["run"]))] = call
    return saved


class ModelClient:
    def __init__(self, settings: Settings, models: ModelsConfig):
        self.backend = settings.backend
        self.eval_mode = settings.eval_mode  # v3 routes eval prompts leave-one-out
        self.entries: dict[Tier, ModelEntry] = {}
        if settings.backend == "fake":
            # Smart stays off in fake mode, matching an unassigned smart slot.
            self.entries = {Tier.cheap: fake_entry(Tier.cheap), Tier.frontier: fake_entry(Tier.frontier)}
        elif settings.model:
            # One model answers everything: it fills the frontier slot, which the baseline always uses.
            if settings.model not in models.catalog:
                raise ValueError(f"TRIAGE_MODEL: '{settings.model}' is not in configs/models.yaml")
            self.entries = {Tier.frontier: models.catalog[settings.model].at_level(settings.reasoning)}
        else:
            for tier in LADDER:
                entry = models.tier_entry(tier)
                if entry is not None:
                    self.entries[tier] = entry.at_level(settings.reasoning)
            # Frontier is the last rung of every version; a version that starts lower needs that tier too,
            # which the cascade reports per request (TierNotConfigured).
            if Tier.frontier not in self.entries:
                raise ValueError("configs/models.yaml: set tiers.frontier before using TRIAGE_BACKEND=real")
        # The second-opinion model for cross_agreement. Fake mode and single-model runs have none, so the
        # check falls back to asking the same tier again.
        self.verifier: ModelEntry | None = None
        if settings.backend != "fake" and not settings.model and models.verifier:
            self.verifier = models.catalog[models.verifier].at_level(settings.reasoning)
        # Eval runs: frontier answers saved by the baseline run, by (item id, repeat).
        self._saved_frontier: dict[tuple[str, int], Call] = {}
        if settings.reuse_frontier and settings.eval_mode and Tier.frontier in self.entries:
            self._saved_frontier = saved_frontier_calls(settings.reuse_frontier, self.entries[Tier.frontier])
        # One agent per tier; models are built lazily so a missing key only fails the tier that needs it.
        self._agents: dict[Tier | str, Agent] = {}

    def ladder(self) -> list[Tier]:
        """Enabled tiers, cheapest first. Frontier is always last."""
        return [t for t in LADDER if t in self.entries]

    def _agent(self, key: Tier | str, entry: ModelEntry) -> Agent:
        if key not in self._agents:
            # retries=0: by default PydanticAI answers an empty reply with a hidden, billed follow-up call
            # ("Please return text."), which would mask exactly the failures the checks need to see.
            self._agents[key] = Agent(build_model(entry), name=f"triage-{key}", retries=0)
        return self._agents[key]

    async def complete(
        self,
        tier: Tier,
        task: Task,
        seq: int,
        purpose: Literal["answer", "judge"] = "answer",
        verifier: bool = False,
    ) -> Call:
        """Call the tier's model, or with `verifier` the second-opinion model (logged under the same tier,
        with its own provider, model and price)."""
        if tier == Tier.frontier and purpose == "answer" and task.eval_key in self._saved_frontier:
            return self._saved_frontier[task.eval_key].model_copy(update={"seq": seq, "reused": True})
        use_verifier = verifier and self.verifier is not None
        entry = self.verifier if use_verifier else self.entries[tier]
        agent = self._agent("verifier" if use_verifier else tier, entry)
        settings = dict(entry.settings)
        if task.max_tokens is not None:
            settings["max_tokens"] = task.max_tokens
        if task.temperature is not None:
            settings["temperature"] = task.temperature

        start = time.perf_counter()
        retries, timed_out, error = 0, False, None
        output: str | None = None
        truncated = False
        input_tokens = output_tokens = 0
        for attempt in range(entry.max_retries + 1):
            retries = attempt
            with capture_run_messages() as messages:
                try:
                    result = await asyncio.wait_for(
                        agent.run(task.user, instructions=task.system, model_settings=settings),
                        timeout=entry.timeout_s,
                    )
                    output = result.output or ""
                    input_tokens = result.usage.input_tokens or 0
                    output_tokens = result.usage.output_tokens or 0
                    truncated = result.response.finish_reason == "length"
                    error, timed_out = None, False
                    break
                except UnexpectedModelBehavior:
                    # The model answered with no text. That's an answer (an empty one), not an outage:
                    # keep its billed tokens, don't retry, and let the checks fail it.
                    responses = [m for m in messages if isinstance(m, ModelResponse)]
                    output = ""
                    input_tokens = sum(r.usage.input_tokens or 0 for r in responses)
                    output_tokens = sum(r.usage.output_tokens or 0 for r in responses)
                    truncated = any(r.finish_reason == "length" for r in responses)
                    error, timed_out = None, False
                    break
                except Exception as e:
                    timed_out = isinstance(e, TimeoutError)
                    error = f"timed out after {entry.timeout_s}s" if timed_out else f"{type(e).__name__}: {e}"
                    error = error[:300]
                    retry = is_transient(e) and attempt < entry.max_retries
                    log.warning(
                        "%s call to %s failed (attempt %d, %s): %s",
                        tier,
                        entry.key,
                        attempt + 1,
                        "retrying" if retry else "giving up",
                        error,
                    )
                    if not retry:
                        break
            await asyncio.sleep(RETRY_BASE_DELAY_S * 2**attempt)
        latency_ms = (time.perf_counter() - start) * 1000

        call = Call(
            seq=seq,
            purpose=purpose,
            tier=tier,
            provider=entry.provider,
            model=entry.model,
            reasoning=entry.reasoning,
            latency_ms=latency_ms,
            retries=retries,
            timed_out=timed_out,
            truncated=truncated,
            error=error,
            output=output,
        )
        if output is not None:
            call.input_tokens, call.output_tokens = input_tokens, output_tokens
            call.cost_usd = entry.price.cost(input_tokens, output_tokens)
        return call
