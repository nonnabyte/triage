"""Which failures are worth retrying."""

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from triage.models import is_transient


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retries_throttling_and_server_errors(status):
    assert is_transient(ModelHTTPError(status_code=status, model_name="m"))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_does_not_retry_auth_or_bad_requests(status):
    # e.g. an expired Bedrock key (403) or a model not enabled in the project (404)
    assert not is_transient(ModelHTTPError(status_code=status, model_name="m"))


def test_retries_timeouts_and_network_errors():
    assert is_transient(TimeoutError())
    assert is_transient(httpx.ConnectError("refused"))


def test_does_not_retry_bugs():
    assert not is_transient(ValueError("bad"))
