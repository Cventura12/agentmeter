"""LangChain callback integration for agentmeter."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from threading import Lock
from typing import Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from agentmeter.core import Span
from agentmeter.emitter import get_emitter
from agentmeter.tracer import SpanKind


class _FallbackCallbackHandler:
    """Fallback base class used only when langchain is unavailable."""

    pass


_BaseCallbackHandler: type[object] = _FallbackCallbackHandler
_LANGCHAIN_IMPORT_ERROR: Exception | None = None
try:
    callbacks_module = importlib.import_module("langchain_core.callbacks")
    callback_handler_cls = getattr(callbacks_module, "BaseCallbackHandler", None)
    if isinstance(callback_handler_cls, type):
        _BaseCallbackHandler = callback_handler_cls
    else:
        raise ImportError("langchain_core.callbacks.BaseCallbackHandler not found")
except ImportError as exc:  # pragma: no cover - exercised when optional dep missing
    _LANGCHAIN_IMPORT_ERROR = exc

LLMResult = object
SpanOutcome = Literal["success", "failure", "timeout", "unknown"]


def _instruction() -> str:
    """Return dependency installation instruction."""
    return "LangChain is required. Install with: pip install agentmeter[langchain]"


def _now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def _as_str(value: object) -> str | None:
    """Normalize non-empty string values."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _as_int(value: object) -> int | None:
    """Convert values to integer where possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize_uuid(value: object) -> str:
    """Convert run IDs into valid UUID strings."""
    raw = _as_str(value)
    if raw is None:
        return str(uuid4())
    try:
        return str(UUID(raw))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, raw))


def _capability_from_span_kind(kind: SpanKind) -> str:
    """Map span kind to taxonomy capability."""
    if kind == SpanKind.LLM:
        return "generation.summary"
    if kind == SpanKind.CHAIN:
        return "classification.intent"
    if kind == SpanKind.RETRIEVER:
        return "retrieval.semantic"
    if kind == SpanKind.TOOL:
        return "transformation.format"
    if kind == SpanKind.GUARDRAIL:
        return "verification.schema"
    if kind == SpanKind.EVALUATOR:
        return "verification.factcheck"
    return "planning.multi_step"


def _extract_tokens(response: object) -> tuple[int | None, int | None]:
    """Extract token usage from LangChain llm output structures."""
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage")
        if isinstance(token_usage, dict):
            prompt_tokens = _as_int(token_usage.get("prompt_tokens")) or _as_int(token_usage.get("input_tokens"))
            completion_tokens = _as_int(token_usage.get("completion_tokens")) or _as_int(token_usage.get("output_tokens"))
            return prompt_tokens, completion_tokens

        usage_metadata = llm_output.get("usage_metadata")
        if isinstance(usage_metadata, dict):
            prompt_tokens = _as_int(usage_metadata.get("input_tokens"))
            completion_tokens = _as_int(usage_metadata.get("output_tokens"))
            return prompt_tokens, completion_tokens

    # LangChain also surfaces per-generation usage in generation_info.
    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        for group in generations:
            if not isinstance(group, list):
                continue
            for generation in group:
                info = getattr(generation, "generation_info", None)
                if isinstance(info, dict):
                    usage = info.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = _as_int(usage.get("prompt_tokens")) or _as_int(usage.get("input_tokens"))
                        completion_tokens = _as_int(usage.get("completion_tokens")) or _as_int(usage.get("output_tokens"))
                        if prompt_tokens is not None or completion_tokens is not None:
                            return prompt_tokens, completion_tokens
    return None, None


class _PendingSpan:
    """Mutable span state between start/end callbacks."""

    def __init__(
        self,
        execution_id: str,
        span_kind: SpanKind,
        caller_agent_id: str,
        callee_agent_id: str,
        started_at: datetime,
    ) -> None:
        self.execution_id = execution_id
        self.span_kind = span_kind
        self.caller_agent_id = caller_agent_id
        self.callee_agent_id = callee_agent_id
        self.started_at = started_at
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cost_usd: float | None = None
        self.outcome: SpanOutcome = "unknown"
        self.error_message: str | None = None
        self.metadata: dict[str, str] = {
            "framework": "langchain",
            "span_kind": span_kind.name,
        }

    def set_outcome(self, outcome: str) -> None:
        """Set normalized outcome for pending span."""
        if outcome in {"success", "failure", "timeout", "unknown"}:
            self.outcome = cast(SpanOutcome, outcome)
        else:
            self.outcome = "unknown"

    def set_error(self, error: BaseException | object) -> None:
        """Attach error details and mark failure outcome."""
        self.error_message = str(error)
        self.outcome = "failure"


class AgentMeterCallbackHandler(_BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """LangChain callback handler that emits agentmeter spans."""

    def __init__(self, agent_id: str) -> None:
        if _LANGCHAIN_IMPORT_ERROR is not None:
            raise ImportError(_instruction()) from _LANGCHAIN_IMPORT_ERROR
        super().__init__()
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        self.agent_id = agent_id
        self._pending: dict[str, _PendingSpan] = {}
        self._lock = Lock()

    def _start_span(
        self,
        run_id: object,
        parent_run_id: object,
        span_kind: SpanKind,
        serialized: object,
        metadata: dict[str, str] | None = None,
    ) -> None:
        run_key = _normalize_uuid(run_id)
        execution_id = _normalize_uuid(parent_run_id) if parent_run_id is not None else run_key
        if isinstance(serialized, dict):
            name = _as_str(serialized.get("name")) or _as_str(serialized.get("id"))
        else:
            name = None
        if name is None:
            name = span_kind.name.lower()

        pending = _PendingSpan(
            execution_id=execution_id,
            span_kind=span_kind,
            caller_agent_id=self.agent_id,
            callee_agent_id=name,
            started_at=_now_utc(),
        )
        if metadata:
            pending.metadata.update(metadata)
        with self._lock:
            self._pending[run_key] = pending

    def _finish_span(self, run_id: object, outcome: str = "success") -> None:
        run_key = _normalize_uuid(run_id)
        with self._lock:
            pending = self._pending.pop(run_key, None)
        if pending is None:
            return

        pending.set_outcome(outcome)
        ended_at = _now_utc()
        span_model = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=pending.execution_id,
            caller_agent_id=pending.caller_agent_id,
            callee_agent_id=pending.callee_agent_id,
            capability=_capability_from_span_kind(pending.span_kind),
            started_at=pending.started_at,
            ended_at=ended_at,
            input_tokens=pending.input_tokens,
            output_tokens=pending.output_tokens,
            cost_usd=pending.cost_usd,
            outcome=pending.outcome,
            error_message=pending.error_message,
            metadata=pending.metadata,
        )
        get_emitter().emit_span_safe(span_model)

    def _mark_error(self, run_id: object, error: BaseException | object) -> None:
        run_key = _normalize_uuid(run_id)
        with self._lock:
            pending = self._pending.get(run_key)
        if pending is None:
            return
        pending.set_error(error)

    # Chain hooks
    def on_chain_start(self, serialized: object, inputs: object, **kwargs: object) -> None:
        del inputs
        self._start_span(
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            span_kind=SpanKind.CHAIN,
            serialized=serialized,
        )

    def on_chain_end(self, outputs: object, **kwargs: object) -> None:
        del outputs
        self._finish_span(kwargs.get("run_id"), outcome="success")

    def on_chain_error(self, error: BaseException, **kwargs: object) -> None:
        self._mark_error(kwargs.get("run_id"), error)
        self._finish_span(kwargs.get("run_id"), outcome="failure")

    # LLM hooks
    def on_llm_start(self, serialized: object, prompts: object, **kwargs: object) -> None:
        del prompts
        self._start_span(
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            span_kind=SpanKind.LLM,
            serialized=serialized,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        run_key = _normalize_uuid(kwargs.get("run_id"))
        prompt_tokens, completion_tokens = _extract_tokens(response)
        with self._lock:
            pending = self._pending.get(run_key)
            if pending is not None:
                pending.input_tokens = prompt_tokens
                pending.output_tokens = completion_tokens
        self._finish_span(kwargs.get("run_id"), outcome="success")

    def on_llm_error(self, error: BaseException, **kwargs: object) -> None:
        self._mark_error(kwargs.get("run_id"), error)
        self._finish_span(kwargs.get("run_id"), outcome="failure")

    # Tool hooks
    def on_tool_start(self, serialized: object, input_str: str, **kwargs: object) -> None:
        del input_str
        self._start_span(
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            span_kind=SpanKind.TOOL,
            serialized=serialized,
        )

    def on_tool_end(self, output: object, **kwargs: object) -> None:
        del output
        self._finish_span(kwargs.get("run_id"), outcome="success")

    def on_tool_error(self, error: BaseException, **kwargs: object) -> None:
        self._mark_error(kwargs.get("run_id"), error)
        self._finish_span(kwargs.get("run_id"), outcome="failure")

    # Retriever hooks
    def on_retriever_start(self, serialized: object, query: str, **kwargs: object) -> None:
        del query
        self._start_span(
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            span_kind=SpanKind.RETRIEVER,
            serialized=serialized,
        )

    def on_retriever_end(self, documents: object, **kwargs: object) -> None:
        del documents
        self._finish_span(kwargs.get("run_id"), outcome="success")
