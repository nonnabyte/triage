"""Prompt signals for the v1 rules, computed before any model call.

Every signal is a property of the request itself, read through `profile`, so none depends on a particular
prompt template: the task type, the estimated length, and for an extraction whether it asks for named
fields or for a list of verbatim spans.
"""

from __future__ import annotations

import re
from typing import Any

from triage.profile import build_profile, split_prompt
from triage.schemas import Task

_SPAN_LIST = (
    r"JSON (array|list)|\b(all|every|each|list of)\b[^.?!\n]{0,40}\b(spans|passages|clauses|quotes)\b"
)


def wants_span_list(text: str) -> bool:
    """The request asks for a list of verbatim spans (rather than named fields or one answer)."""
    return bool(re.search(_SPAN_LIST, split_prompt(text).frame, re.IGNORECASE))


def estimated_tokens(text: str) -> float:
    """No tokenizer at routing time: 4 characters per token."""
    return len(text) / 4


def prompt_features(task: Task) -> dict[str, Any]:
    text = "\n".join(p for p in (task.system, task.user) if p)
    profile = build_profile(task)
    parts = split_prompt(text)
    return {
        "task_type": profile.task_type,
        "tokens": estimated_tokens(text),
        "has_document": bool(parts.document),
        "wants_json": profile.expected_output == "json",
        "wants_span_list": bool(re.search(_SPAN_LIST, parts.frame, re.IGNORECASE)),
    }
