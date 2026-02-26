"""Developer-facing tracing APIs for recording agent execution spans.

This module exposes:
- `Tracer`: thread-safe tracer using `threading.local()` span context stacks.
- `AsyncTracer`: async-aware tracer using `contextvars.ContextVar` stacks.

Both tracers share the same public interface:
- `span(...)` context manager
- `record_tokens(...)`
- `finish(...)`
- `get_current_execution_id()`
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock, local
from types import TracebackType
from typing import Literal, cast
from uuid import uuid4

from .cohort import AnonymizedSpanMetric, CohortExporter
from .core import ExecutionTrace, Span
from .emitter import get_cohort_export_config, get_emitter
from .exceptions import TraceNotActiveError
from .taxonomy import Capability

SpanOutcome = Literal["success", "failure", "timeout", "unknown"]
ExecutionOutcome = Literal["success", "failure", "partial", "unknown"]

_SPAN_OUTCOMES = frozenset({"success", "failure", "timeout", "unknown"})
_EXECUTION_OUTCOMES = frozenset({"success", "failure", "partial", "unknown"})
_COHORT_WRITE_LOCK = Lock()


class SpanKind(str, Enum):
    """OpenInference-style span kinds used in tracing APIs."""

    LLM = "llm"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    TOOL = "tool"
    AGENT = "agent"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    GUARDRAIL = "guardrail"
    EVALUATOR = "evaluator"

    @classmethod
    def from_value(cls, value: str | SpanKind) -> SpanKind | None:
        """Parse a span kind from enum instance, member name, or value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        for member in cls:
            if member.value == lowered or member.name.lower() == lowered:
                return member
        return None


_SPAN_KIND_TO_CAPABILITY: dict[SpanKind, Capability] = {
    SpanKind.LLM: Capability.GENERATION_SUMMARY,
    SpanKind.CHAIN: Capability.PLANNING_MULTI_STEP,
    SpanKind.RETRIEVER: Capability.RETRIEVAL_SEMANTIC,
    SpanKind.TOOL: Capability.TRANSFORMATION_FORMAT,
    SpanKind.AGENT: Capability.PLANNING_ROUTING,
    SpanKind.EMBEDDING: Capability.RETRIEVAL_SEMANTIC,
    SpanKind.RERANKER: Capability.RETRIEVAL_STRUCTURED,
    SpanKind.GUARDRAIL: Capability.VERIFICATION_SCHEMA,
    SpanKind.EVALUATOR: Capability.VERIFICATION_FACTCHECK,
}


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _trace_outcome_from_spans(spans: list[Span]) -> ExecutionOutcome:
    """Infer trace outcome from underlying span outcomes."""
    if not spans:
        return "unknown"
    success_count = sum(1 for span in spans if span.outcome == "success")
    failure_count = sum(1 for span in spans if span.outcome in {"failure", "timeout"})
    if success_count == len(spans):
        return "success"
    if success_count == 0 and failure_count > 0:
        return "failure"
    if success_count > 0 and failure_count > 0:
        return "partial"
    return "unknown"


def _append_cohort_metrics(path: Path, metrics: list[AnonymizedSpanMetric]) -> None:
    """Append cohort metrics as JSONL to a local file."""
    if not metrics:
        return
    rendered = CohortExporter.to_jsonl(metrics)
    if not rendered:
        return
    with _COHORT_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")


def _span_kind_from_capability(capability: Capability) -> SpanKind:
    """Infer a reasonable span kind for a taxonomy capability."""
    category = capability.value.split(".", 1)[0]
    if category == "retrieval":
        return SpanKind.RETRIEVER
    if category == "transformation":
        return SpanKind.TOOL
    if category == "verification":
        return SpanKind.GUARDRAIL
    if category == "planning":
        return SpanKind.AGENT
    if category == "classification":
        return SpanKind.CHAIN
    return SpanKind.LLM


def _normalize_capability_and_kind(value: str | Capability | SpanKind) -> tuple[str, SpanKind]:
    """Normalize tracer input into taxonomy capability and canonical span kind."""
    parsed_kind = SpanKind.from_value(value)
    if parsed_kind is not None:
        mapped = _SPAN_KIND_TO_CAPABILITY[parsed_kind]
        return mapped.value, parsed_kind
    if isinstance(value, Capability):
        return value.value, _span_kind_from_capability(value)
    capability = Capability.from_string(value)
    return capability.value, _span_kind_from_capability(capability)


class SpanContext:
    """Mutable builder and context manager for constructing immutable `Span` records."""

    def __init__(self, tracer: _BaseTracer, callee_id: str, capability: str | Capability | SpanKind) -> None:
        self._tracer = tracer
        self.span_id: str = str(uuid4())
        self.parent_span_id: str | None = None
        self.callee_id: str = callee_id
        normalized_capability, normalized_kind = _normalize_capability_and_kind(capability)
        self.capability: str = normalized_capability
        self.span_kind: SpanKind = normalized_kind
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cost_usd: float | None = None
        self.outcome: SpanOutcome = "unknown"
        self.error_message: str | None = None
        self.metadata: dict[str, str] = {}
        self._closed = False

    def set_cost(self, cost_usd: float) -> None:
        """Set span cost in USD."""
        if cost_usd < 0:
            raise ValueError("cost must be non-negative")
        self.cost_usd = cost_usd

    def set_outcome(self, outcome: SpanOutcome) -> None:
        """Set the final span outcome."""
        if outcome not in _SPAN_OUTCOMES:
            raise ValueError(f"outcome must be one of: {sorted(_SPAN_OUTCOMES)}")
        self.outcome = outcome

    def set_metadata(self, key: str, value: str) -> None:
        """Attach string metadata to the span."""
        if not key:
            raise ValueError("metadata key must be non-empty")
        self.metadata[key] = value

    def _set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set input/output token counts on this span."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __enter__(self) -> SpanContext:
        """Enter span context and register the span on the tracer stack."""
        self._tracer._open_span(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Close span context, persist span, and never suppress exceptions."""
        del tb
        self._tracer._close_span(self, exc)
        return False

    async def __aenter__(self) -> SpanContext:
        """Async context entry equivalent of `__enter__`."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Async context exit equivalent of `__exit__`."""
        return self.__exit__(exc_type, exc, tb)


class _BaseTracer:
    """Shared tracer behavior for thread-local and contextvar-backed implementations."""

    def __init__(
        self,
        agent_id: str,
        capability: str | Capability | SpanKind,
        *,
        session_id: str | None = None,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        self.agent_id = agent_id
        normalized_capability, normalized_kind = _normalize_capability_and_kind(capability)
        self.capability = normalized_capability
        self.span_kind = normalized_kind
        self.session_id = session_id
        self._execution_id: str | None = str(uuid4())
        self._started_at = _utc_now()
        self._spans: list[Span] = []
        self._finished_trace: ExecutionTrace | None = None
        self._lock = Lock()
        self._open_span_count = 0

    def span(self, callee_id: str, capability: str | Capability | SpanKind) -> SpanContext:
        """Create a span context manager for one downstream agent call."""
        self._ensure_open()
        return SpanContext(self, callee_id=callee_id, capability=capability)

    def record_tokens(self, input: int, output: int) -> None:
        """Set token counts on the currently active span for this context."""
        self._ensure_open()
        current = self._peek_span()
        if current is None:
            raise TraceNotActiveError("No active span in the current context.")
        current._set_tokens(input_tokens=input, output_tokens=output)

    def finish(self, outcome: str = "success") -> ExecutionTrace:
        """Close the execution and return an immutable `ExecutionTrace`."""
        if outcome not in _EXECUTION_OUTCOMES:
            raise ValueError(f"outcome must be one of: {sorted(_EXECUTION_OUTCOMES)}")
        normalized_outcome = cast(ExecutionOutcome, outcome)
        with self._lock:
            if self._finished_trace is not None:
                return self._finished_trace
            if self._execution_id is None:
                raise TraceNotActiveError("No active execution.")
            if self._open_span_count > 0:
                raise TraceNotActiveError("Cannot finish while spans are still active.")
            trace = ExecutionTrace(
                execution_id=self._execution_id,
                root_agent_id=self.agent_id,
                started_at=self._started_at,
                ended_at=_utc_now(),
                spans=list(self._spans),
                outcome=normalized_outcome,
            )
            self._finished_trace = trace
            self._execution_id = None
        try:
            get_emitter().emit_trace(trace)
        except Exception as exc:  # noqa: BLE001
            print(f"[agentmeter] emitter error: {exc}", file=sys.stderr)
        cohort_export_enabled, cohort_path = get_cohort_export_config()
        if cohort_export_enabled and cohort_path is not None:
            metrics = self.export_cohort_metrics(orchestrator="custom")
            _append_cohort_metrics(cohort_path, metrics)
        return trace

    def get_current_execution_id(self) -> str | None:
        """Return the active execution UUID, or `None` after `finish()`."""
        return self._execution_id

    def export_cohort_metrics(self, *, orchestrator: str | None = None) -> list[AnonymizedSpanMetric]:
        """Export current execution spans as anonymized cohort metrics."""
        trace = self._finished_trace
        if trace is None:
            if self._execution_id is None and not self._spans:
                return []
            execution_id = self._execution_id if self._execution_id is not None else str(uuid4())
            trace = ExecutionTrace(
                execution_id=execution_id,
                root_agent_id=self.agent_id,
                started_at=self._started_at,
                ended_at=_utc_now(),
                spans=list(self._spans),
                outcome=_trace_outcome_from_spans(self._spans),
            )
        return CohortExporter.from_trace(trace, orchestrator=orchestrator)

    def _ensure_open(self) -> None:
        """Raise when operations are attempted after execution finalization."""
        if self._execution_id is None:
            raise TraceNotActiveError("Execution has already been finished.")

    def _open_span(self, handle: SpanContext) -> None:
        """Push a span onto the current context stack and start timing."""
        self._ensure_open()
        parent = self._peek_span()
        handle.parent_span_id = parent.span_id if parent is not None else None
        handle.started_at = _utc_now()
        self._push_span(handle)
        with self._lock:
            self._open_span_count += 1

    def _close_span(self, handle: SpanContext, exc: BaseException | None) -> None:
        """Pop a span, derive outcome/error metadata, and persist immutable `Span`."""
        self._pop_span(handle)
        if handle._closed:
            return
        handle.ended_at = _utc_now()
        if exc is not None:
            handle.outcome = "failure"
            handle.error_message = str(exc)
        if handle.started_at is None:
            handle.started_at = handle.ended_at

        metadata = dict(handle.metadata)
        if self.session_id is not None:
            metadata.setdefault("session_id", self.session_id)
        metadata.setdefault("caller_capability", self.capability)
        metadata.setdefault("caller_span_kind", self.span_kind.name)
        metadata.setdefault("span_kind", handle.span_kind.name)

        execution_id = self._execution_id
        if execution_id is None:
            raise TraceNotActiveError("Execution has already been finished.")

        span = Span(
            span_id=handle.span_id,
            parent_span_id=handle.parent_span_id,
            execution_id=execution_id,
            caller_agent_id=self.agent_id,
            callee_agent_id=handle.callee_id,
            capability=handle.capability,
            started_at=handle.started_at,
            ended_at=handle.ended_at,
            input_tokens=handle.input_tokens,
            output_tokens=handle.output_tokens,
            cost_usd=handle.cost_usd,
            outcome=handle.outcome,
            error_message=handle.error_message,
            metadata=metadata,
        )
        with self._lock:
            self._spans.append(span)
            self._open_span_count -= 1
        handle._closed = True
        get_emitter().emit_span_safe(span)

    def _peek_span(self) -> SpanContext | None:
        """Return the top span for the current context, if any."""
        raise NotImplementedError

    def _push_span(self, handle: SpanContext) -> None:
        """Push a span to the current context stack."""
        raise NotImplementedError

    def _pop_span(self, expected: SpanContext) -> None:
        """Pop a span from the current context stack."""
        raise NotImplementedError


class Tracer(_BaseTracer):
    """
    Thread-safe execution tracer for multi-agent systems.
    Captures spans, costs, and outcomes automatically.
    """

    def __init__(
        self,
        agent_id: str,
        capability: str | Capability | SpanKind,
        *,
        session_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, capability=capability, session_id=session_id)
        self._thread_local = local()

    def _stack(self) -> list[SpanContext]:
        """Get or initialize the current thread's span stack."""
        stack = getattr(self._thread_local, "span_stack", None)
        if stack is None:
            stack = []
            setattr(self._thread_local, "span_stack", stack)
        return stack

    def _peek_span(self) -> SpanContext | None:
        """Return the top-most span in the current thread."""
        stack = self._stack()
        return stack[-1] if stack else None

    def _push_span(self, handle: SpanContext) -> None:
        """Push a span onto the current thread stack."""
        self._stack().append(handle)

    def _pop_span(self, expected: SpanContext) -> None:
        """Pop a span from the current thread stack."""
        stack = self._stack()
        if not stack:
            raise TraceNotActiveError("No active span in the current context.")
        if stack[-1] is expected:
            stack.pop()
            return
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is expected:
                del stack[index]
                return
        raise TraceNotActiveError("Span not found in the current context.")


class AsyncTracer(_BaseTracer):
    """Async-aware execution tracer using `ContextVar` for span stacks."""

    def __init__(
        self,
        agent_id: str,
        capability: str | Capability | SpanKind,
        *,
        session_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, capability=capability, session_id=session_id)
        self._span_stack: ContextVar[tuple[SpanContext, ...]] = ContextVar(
            f"agentmeter_async_span_stack_{id(self)}",
            default=(),
        )

    def _peek_span(self) -> SpanContext | None:
        """Return the top-most span in the current async context."""
        stack = self._span_stack.get()
        return stack[-1] if stack else None

    def _push_span(self, handle: SpanContext) -> None:
        """Push a span onto the current async context stack."""
        stack = self._span_stack.get()
        self._span_stack.set(stack + (handle,))

    def _pop_span(self, expected: SpanContext) -> None:
        """Pop a span from the current async context stack."""
        stack = self._span_stack.get()
        if not stack:
            raise TraceNotActiveError("No active span in the current context.")
        if stack[-1] is expected:
            self._span_stack.set(stack[:-1])
            return
        mutable = list(stack)
        for index in range(len(mutable) - 1, -1, -1):
            if mutable[index] is expected:
                del mutable[index]
                self._span_stack.set(tuple(mutable))
                return
        raise TraceNotActiveError("Span not found in the current context.")
