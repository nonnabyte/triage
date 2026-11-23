"""API contract: OpenAI chat-completions request and response, plus the `triage` block.

Only runtime fields live here. Dataset fields (label, expected answer, check type) never reach the service.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field


class Tier(StrEnum):
    cheap = "cheap"
    smart = "smart"
    frontier = "frontier"


LADDER = [Tier.cheap, Tier.smart, Tier.frontier]

Strategy = Literal["baseline", "v1", "v2", "v3"]
ReasoningLevel = Literal["low", "medium", "high"]
STRATEGIES: tuple[str, ...] = get_args(Strategy)

TASK_TYPES: tuple[str, ...] = (
    "document_qa",
    "contract_extraction",
    "coding",
    "misc_classification",
    "misc_math",
    "misc_sentiment",
    "misc_planning",
)


# --- Request -------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    """OpenAI chat-completions request. Single-shot rules are enforced in api.py."""

    model_config = ConfigDict(extra="allow")

    # Required by OpenAI clients; ignored by the router, which picks the real model itself.
    model: str = "triage"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    # Optional hint: metadata.task_type (one of TASK_TYPES). Inferred when absent.
    metadata: dict[str, str] | None = None
    stream: bool | None = None
    tools: list[Any] | None = None
    n: int | None = None


class Task(BaseModel):
    """What the router works on: the caller's single task."""

    user: str
    system: str | None = None
    task_type_hint: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # Eval runs only: the dataset item and repeat this request is, so a frontier answer saved by the
    # baseline run can be reused instead of paid for again.
    eval_key: tuple[str, int] | None = None


# --- Response: triage block ----------------------------------------------------


class Profile(BaseModel):
    task_type: str
    task_type_source: Literal["provided", "inferred", "unknown"]
    expected_output: Literal["json", "code", "number", "label", "text"]


class Decision(BaseModel):
    start_tier: Tier
    reasons: list[str] = []
    # A score behind the start decision, when the policy has one; None otherwise.
    score: float | None = None


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class CheckOutcome(BaseModel):
    passed: bool
    results: list[CheckResult]


class Call(BaseModel):
    seq: int
    purpose: Literal["answer", "judge"] = "answer"
    tier: Tier
    provider: str
    model: str
    # Fixed reasoning effort for reasoning models; None for models that don't reason.
    reasoning: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # None when the model's price isn't filled in yet.
    cost_usd: float | None = None
    latency_ms: float
    retries: int = 0
    timed_out: bool = False
    # The model hit max_tokens: the answer is cut off (or empty, if thinking used the whole budget).
    truncated: bool = False
    error: str | None = None
    # None when no check ran (final tier, or a version without checks).
    check: CheckOutcome | None = None
    # True when this is the baseline run's saved answer for the same prompt and repeat (eval runs only),
    # reported with its original tokens, cost and latency.
    reused: bool = False
    # The model's answer. Returned only in eval mode, so the harness can score rejected answers.
    output: str | None = None


class Totals(BaseModel):
    cost_usd: float
    wasted_cost_usd: float
    overhead_cost_usd: float
    latency_ms: float
    # False when an answered call's model has no price, so cost_usd is a lower bound. Calls that errored
    # report no tokens and count as $0.
    cost_complete: bool


class TriageBlock(BaseModel):
    strategy: str
    profile: Profile
    decision: Decision
    final_tier: Tier
    escalation_path: list[Tier]
    calls: list[Call]
    totals: Totals


# --- Response: OpenAI fields ---------------------------------------------------


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = "triage"
    choices: list[Choice]
    usage: Usage
    triage: TriageBlock
