"""HTTP service: OpenAI chat-completions endpoint, single-shot."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triage import cascade, tracing
from triage.config import Settings, get_settings, load_env, load_models, load_strategies
from triage.models import ModelClient
from triage.schemas import (
    STRATEGIES,
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    Task,
    TriageBlock,
    Usage,
)

STRATEGY_HEADER = "x-triage-strategy"
# Eval mode only: the harness tags each request with its run and dataset item, for grouping traces.
EVAL_RUN_HEADER = "x-triage-eval-run"
EVAL_ITEM_HEADER = "x-triage-eval-item"
EVAL_REPEAT_HEADER = "x-triage-eval-repeat"


def _error(status: int, code: str, message: str, triage: TriageBlock | None = None) -> JSONResponse:
    body: dict = {
        "error": {
            "type": "invalid_request_error" if status < 500 else "api_error",
            "code": code,
            "message": message,
        }
    }
    if triage is not None:
        body["triage"] = triage.model_dump(mode="json")
    return JSONResponse(status_code=status, content=body)


def to_task(req: ChatRequest) -> Task | str:
    """Return the Task, or an error message if the request isn't a single-shot task."""
    if req.stream:
        return "stream is not supported"
    if req.tools:
        return "tools are not supported"
    if req.n not in (None, 1):
        return "n must be 1"
    msgs = req.messages
    system = None
    if msgs[0].role in ("system", "developer"):
        system, msgs = msgs[0].content, msgs[1:]
    if len(msgs) != 1 or msgs[0].role != "user":
        return "send exactly one user message, optionally preceded by one system message"
    hint = (req.metadata or {}).get("task_type")
    return Task(
        user=msgs[0].content,
        system=system,
        task_type_hint=hint,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


def setup_logging(level: str) -> None:
    """Send triage.* logs to stderr next to uvicorn's own logs."""
    logger = logging.getLogger("triage")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


def create_app(settings: Settings | None = None) -> FastAPI:
    load_env()
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    models = load_models(settings.models_config)
    strategies = load_strategies(settings.strategies_config)
    cascade.validate_strategies(strategies)
    client = ModelClient(settings, models)
    tracing.setup_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        tracing.flush()

    app = FastAPI(title="Triage", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _error(400, "invalid_request", f"{where}: {first.get('msg', 'invalid request')}")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "backend": settings.backend,
            "strategy": settings.strategy,
            "tiers": [str(t) for t in client.ladder()],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatRequest, request: Request) -> JSONResponse:
        task = to_task(req)
        if isinstance(task, str):
            return _error(400, "unsupported_request", task)

        strategy = settings.strategy
        override = request.headers.get(STRATEGY_HEADER)
        if settings.eval_mode and override:
            if override not in STRATEGIES:
                return _error(
                    400, "unknown_strategy", f"{STRATEGY_HEADER} must be one of {', '.join(STRATEGIES)}"
                )
            strategy = override

        session_id, trace_meta = None, {}
        if settings.eval_mode:
            session_id = request.headers.get(EVAL_RUN_HEADER)
            for key, header in (("eval_item", EVAL_ITEM_HEADER), ("eval_repeat", EVAL_REPEAT_HEADER)):
                if value := request.headers.get(header):
                    trace_meta[key] = value
            item, repeat = request.headers.get(EVAL_ITEM_HEADER), request.headers.get(EVAL_REPEAT_HEADER)
            if item and repeat and repeat.isdigit():
                task = task.model_copy(update={"eval_key": (item, int(repeat))})

        with tracing.trace_request(task, strategy, session_id=session_id, metadata=trace_meta) as trace:
            try:
                answer, block = await cascade.run(task, strategy, client, strategies)
            except NotImplementedError as e:
                trace.finish(None, None, str(e))
                return _error(501, "not_implemented", str(e))
            except cascade.TierNotConfigured as e:
                trace.finish(None, None, str(e))
                return _error(503, "tier_not_configured", str(e))
            except cascade.CascadeFailed as e:
                trace.finish(None, e.block, str(e))
                return _error(502, "model_failed", str(e), triage=_public(e.block, settings))
            trace.finish(answer, block)

        resp = ChatResponse(
            id=f"trg_{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            choices=[Choice(message=AssistantMessage(content=answer))],
            usage=Usage(
                prompt_tokens=sum(c.input_tokens for c in block.calls),
                completion_tokens=sum(c.output_tokens for c in block.calls),
                total_tokens=sum(c.input_tokens + c.output_tokens for c in block.calls),
            ),
            triage=_public(block, settings),
        )
        return JSONResponse(content=resp.model_dump(mode="json"))

    return app


def _public(block: TriageBlock, settings: Settings) -> TriageBlock:
    """Per-call answers are only returned in eval mode."""
    if settings.eval_mode:
        return block
    return block.model_copy(update={"calls": [c.model_copy(update={"output": None}) for c in block.calls]})
