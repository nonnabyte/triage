import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
# Tests never send traces, even with Langfuse keys in .env.
os.environ["TRIAGE_TRACING"] = "false"

from triage.api import create_app  # noqa: E402
from triage.config import Settings  # noqa: E402


@pytest.fixture
def make_client():
    """Factory: a TestClient for the service on fake models, running the given version."""

    def make(strategy: str = "v2", eval_mode: bool = False) -> TestClient:
        settings = Settings(backend="fake", strategy=strategy, eval_mode=eval_mode)
        return TestClient(create_app(settings))

    return make


@pytest.fixture
def ask():
    """POST one user message to /v1/chat/completions."""

    def post(client: TestClient, content: str, headers: dict | None = None, **extra):
        body = {"model": "triage", "messages": [{"role": "user", "content": content}], **extra}
        return client.post("/v1/chat/completions", json=body, headers=headers)

    return post
