"""Turn a catalog entry into a PydanticAI model. Adapted from the vertex-model-test scripts.

vertex_gemini  Gemini on Vertex AI. GOOGLE_CLOUD_PROJECT + GOOGLE_API_KEY (or gcloud login).
gemini_api     Gemini on the Gemini API (AI Studio), with GOOGLE_API_KEY alone. The same models as Vertex
               under a separate quota, for when the project's Vertex quota is exhausted.
vertex_maas    Open "-maas" models on Vertex's OpenAI-compatible endpoint.
               GOOGLE_API_KEY + GOOGLE_CLOUD_PROJECT.
bedrock        Amazon Bedrock. AWS_BEARER_TOKEN_BEDROCK, or an `aws login` session. AWS_REGION.
fake           Scripted models for tests and keyless development.
"""

from __future__ import annotations

import os
from functools import lru_cache

import httpx
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from triage.config import ModelEntry, Price
from triage.schemas import Tier


class ProviderConfigError(RuntimeError):
    """A model can't be built because a required environment variable is missing."""


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderConfigError(f"{name} is not set (see .env.example)")
    return value


def build_model(entry: ModelEntry) -> Model:
    if entry.provider == "vertex_gemini":
        return _vertex_gemini(entry)
    if entry.provider == "gemini_api":
        return _gemini_api(entry)
    if entry.provider == "vertex_maas":
        return _vertex_maas(entry)
    if entry.provider == "bedrock":
        return _bedrock(entry)
    if entry.provider == "fake":
        return fake_model(Tier(entry.model))
    raise ProviderConfigError(f"unknown provider '{entry.provider}'")


def _vertex_gemini(entry: ModelEntry) -> Model:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    # Always address our project and a location. A key alone uses Express Mode, which doesn't serve every
    # model (Gemini 4 Argon returns 404 there).
    key = os.environ.get("GOOGLE_API_KEY", "").strip() or None  # else gcloud application-default login
    provider = GoogleCloudProvider(
        api_key=key, project=_env("GOOGLE_CLOUD_PROJECT"), location=entry.region or "global"
    )
    return GoogleModel(entry.model, provider=provider)


def _gemini_api(entry: ModelEntry) -> Model:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleModel(entry.model, provider=GoogleProvider(api_key=_env("GOOGLE_API_KEY")))


@lru_cache
def _google_key_http_client(key: str) -> httpx.AsyncClient:
    """One shared connection pool for every Vertex open model. Per-call timeouts are enforced in models.py."""

    async def use_google_key(request: httpx.Request) -> None:
        # The OpenAI client sends "Authorization: Bearer <key>", which Google reads as an OAuth token.
        request.headers.pop("authorization", None)
        request.headers["x-goog-api-key"] = key

    return httpx.AsyncClient(event_hooks={"request": [use_google_key]}, timeout=600)


def _vertex_maas(entry: ModelEntry) -> Model:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    key = _env("GOOGLE_API_KEY")
    project = _env("GOOGLE_CLOUD_PROJECT")
    region = entry.region or "us-central1"
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    client = AsyncOpenAI(
        base_url=f"https://{host}/v1/projects/{project}/locations/{region}/endpoints/openapi",
        api_key="unused",
        http_client=_google_key_http_client(key),
    )
    return OpenAIChatModel(entry.model, provider=OpenAIProvider(openai_client=client))


def _bedrock(entry: ModelEntry) -> Model:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    region = entry.region or _env("AWS_REGION")
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip() or None  # else boto3 uses `aws login`
    return BedrockConverseModel(entry.model, provider=BedrockProvider(region_name=region, api_key=token))


# --- Fake models ------------------------------------------------------------------

# Put a marker in the prompt to script the fake cheap and smart tiers:
#   [fake:fail]   return an empty answer, so the non_empty check fails and the request escalates
#   [fake:error]  raise an error, as a provider outage would
# and this one for every tier, frontier included:
#   [fake:down]   every tier raises an error
FAKE_FAIL = "[fake:fail]"
FAKE_ERROR = "[fake:error]"
FAKE_DOWN = "[fake:down]"
FAKE_CUT = "[fake:cut]"

FAKE_PRICES = {
    Tier.cheap: Price(input=0.10, output=0.40),
    Tier.smart: Price(input=0.50, output=1.50),
    Tier.frontier: Price(input=1.25, output=10.00),
}


def fake_entry(tier: Tier) -> ModelEntry:
    return ModelEntry(
        key=f"fake-{tier}",
        provider="fake",
        model=str(tier),
        price=FAKE_PRICES[tier],
        timeout_s=5,
        max_retries=0,
    )


def fake_model(tier: Tier) -> Model:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages[-1].parts[-1].content)
        if FAKE_DOWN in prompt:
            raise RuntimeError(f"fake {tier} model is down")
        if tier is not Tier.frontier:
            if FAKE_ERROR in prompt:
                raise RuntimeError(f"fake {tier} model error")
            if FAKE_FAIL in prompt:
                return ModelResponse(parts=[TextPart("")])
        if FAKE_CUT in prompt:  # hit max_tokens mid-answer
            return ModelResponse(parts=[TextPart(f"[{tier}] answer to")], finish_reason="length")
        return ModelResponse(parts=[TextPart(f"[{tier}] answer to: {prompt[:60]}")])

    return FunctionModel(respond, model_name=f"fake-{tier}")
