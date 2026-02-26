"""OpenAI Agents SDK tracing integration for agentmeter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from typing import Callable
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from agentmeter.core import ExecutionTrace, Span
from agentmeter.emitter import get_emitter
from agentmeter.tracer import SpanKind

_MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}

_REGISTERED = False


def _instruction() -> str:
    """Return dependency installation instruction."""
    return "OpenAI Agents SDK is required. Install with: pip install agentmeter[openai-agents]"


def _as_str(value: object) -> str | None:
    """Normalize non-empty strings."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _as_int(value: object) -> int | None:
    """Normalize integers from common numeric types."""
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


def _parse_dt(value: object) -> datetime | None:
    """Parse datetime values and normalize to UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _ensure_uuid(value: object) -> str:
    """Return canonical UUID string from arbitrary identifiers."""
    raw = _as_str(value)
    if raw is None:
        return str(uuid4())
    try:
        return str(UUID(raw))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, raw))


def _now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def _span_kind_from_type(span_type: str) -> SpanKind:
    """Map OpenAI Agents span types to agentmeter span kinds."""
    normalized = span_type.lower()
    if normalized in {"agent", "reasoning"}:
        return SpanKind.AGENT
    if normalized in {"function", "tool", "tool_call", "computer_action"}:
        return SpanKind.TOOL
    if normalized in {"generation", "response", "llm", "chat_completion"}:
        return SpanKind.LLM
    return SpanKind.AGENT


def _capability_from_kind(kind: SpanKind) -> str:
    """Map span kind to taxonomy capability."""
    if kind == SpanKind.LLM:
        return "generation.summary"
    if kind == SpanKind.TOOL:
        return "transformation.format"
    if kind == SpanKind.RETRIEVER:
        return "retrieval.semantic"
    if kind == SpanKind.CHAIN:
        return "classification.intent"
    if kind == SpanKind.GUARDRAIL:
        return "verification.schema"
    if kind == SpanKind.EVALUATOR:
        return "verification.factcheck"
    return "planning.multi_step"


def _extract_usage(usage: object) -> tuple[int | None, int | None]:
    """Extract input/output token counts from usage-like objects."""
    if usage is None:
        return None, None

    if isinstance(usage, dict):
        prompt = _as_int(usage.get("prompt_tokens")) or _as_int(usage.get("input_tokens"))
        completion = _as_int(usage.get("completion_tokens")) or _as_int(usage.get("output_tokens"))
        return prompt, completion

    prompt = _as_int(getattr(usage, "prompt_tokens", None)) or _as_int(getattr(usage, "input_tokens", None))
    completion = _as_int(getattr(usage, "completion_tokens", None)) or _as_int(getattr(usage, "output_tokens", None))
    return prompt, completion


def _cost_from_model(model: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Compute estimated cost from local model price table."""
    if model is None:
        return None
    prices = _MODEL_PRICES_PER_1K.get(model)
    if prices is None:
        return None
    prompt_rate, completion_rate = prices
    in_cost = ((input_tokens or 0) / 1000.0) * prompt_rate
    out_cost = ((output_tokens or 0) / 1000.0) * completion_rate
    return round(in_cost + out_cost, 6)


def _trace_outcome(spans: list[Span]) -> str:
    """Infer trace outcome from collected span outcomes."""
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


def setup_openai_agents_tracing(agent_id: str) -> None:
    """Enable OpenAI Agents SDK tracing and route spans to the global emitter."""
    global _REGISTERED

    if not agent_id:
        raise ValueError("agent_id must be non-empty")
    if _REGISTERED:
        return

    try:
        import agents  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_instruction()) from exc

    tracing_module = getattr(agents, "tracing", None)
    if tracing_module is None:
        try:
            import agents.tracing as tracing_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_instruction()) from exc

    tracing_processor_cls = getattr(tracing_module, "TracingProcessor", object)

    class AgentMeterOpenAIProcessor(tracing_processor_cls):  # type: ignore[misc,valid-type]
        """Tracing processor that emits agentmeter spans and traces."""

        def __init__(self) -> None:
            self._trace_spans: dict[str, list[Span]] = {}
            self._trace_start: dict[str, datetime] = {}
            self._trace_root_agent: dict[str, str] = {}
            self._span_start: dict[str, datetime] = {}

        def on_trace_start(self, trace: object) -> None:
            trace_id = _ensure_uuid(getattr(trace, "trace_id", None) or getattr(trace, "id", None))
            started_at = _parse_dt(getattr(trace, "started_at", None)) or _now_utc()
            self._trace_spans.setdefault(trace_id, [])
            self._trace_start[trace_id] = started_at
            self._trace_root_agent[trace_id] = agent_id

        def on_trace_end(self, trace: object) -> None:
            trace_id = _ensure_uuid(getattr(trace, "trace_id", None) or getattr(trace, "id", None))
            spans = self._trace_spans.pop(trace_id, [])
            start = self._trace_start.pop(trace_id, _now_utc())
            end = _parse_dt(getattr(trace, "ended_at", None)) or _now_utc()
            if end < start:
                end = start
            outcome = _as_str(getattr(trace, "outcome", None))
            if outcome not in {"success", "failure", "partial", "unknown"}:
                outcome = _trace_outcome(spans)
            trace_model = ExecutionTrace(
                execution_id=trace_id,
                root_agent_id=self._trace_root_agent.pop(trace_id, agent_id),
                started_at=start,
                ended_at=end,
                spans=spans,
                outcome=outcome,
            )
            emitter = get_emitter()
            try:
                emitter.emit_trace(trace_model)
            except Exception as exc:  # noqa: BLE001
                print(f"[agentmeter] emitter error: {exc}", file=sys.stderr)

        def on_span_start(self, span: object) -> None:
            span_id = _ensure_uuid(getattr(span, "span_id", None) or getattr(span, "id", None))
            started_at = _parse_dt(getattr(span, "started_at", None)) or _now_utc()
            self._span_start[span_id] = started_at

        def on_span_end(self, span: object) -> None:
            span_id = _ensure_uuid(getattr(span, "span_id", None) or getattr(span, "id", None))
            trace_id = _ensure_uuid(getattr(span, "trace_id", None) or getattr(span, "traceId", None))
            parent_span_id = _as_str(getattr(span, "parent_id", None) or getattr(span, "parent_span_id", None))
            if parent_span_id is not None:
                parent_span_id = _ensure_uuid(parent_span_id)

            span_data = getattr(span, "span_data", None)
            span_type = _as_str(getattr(span_data, "type", None)) or _as_str(getattr(span, "type", None)) or "agent"
            span_kind = _span_kind_from_type(span_type)
            model = _as_str(getattr(span_data, "model", None)) or _as_str(getattr(span, "model", None))
            usage = getattr(span_data, "usage", None) or getattr(span, "usage", None)
            input_tokens, output_tokens = _extract_usage(usage)
            cost = _cost_from_model(model, input_tokens, output_tokens) if span_kind == SpanKind.LLM else None

            started_at = self._span_start.pop(span_id, _parse_dt(getattr(span, "started_at", None)) or _now_utc())
            ended_at = _parse_dt(getattr(span, "ended_at", None)) or _now_utc()
            if ended_at < started_at:
                ended_at = started_at

            error_obj = getattr(span, "error", None)
            error_message = _as_str(str(error_obj)) if error_obj is not None else None
            outcome = "failure" if error_message else "success"

            callee = (
                _as_str(getattr(span_data, "name", None))
                or _as_str(getattr(span, "name", None))
                or model
                or span_type
            )
            if callee is None:
                callee = "unknown"

            metadata = {
                "framework": "openai_agents",
                "span_kind": span_kind.name,
                "span_type": span_type,
            }
            if model is not None:
                metadata["model"] = model
            raw_trace_id = _as_str(getattr(span, "trace_id", None))
            raw_span_id = _as_str(getattr(span, "span_id", None))
            if raw_trace_id is not None:
                metadata["raw_trace_id"] = raw_trace_id
            if raw_span_id is not None:
                metadata["raw_span_id"] = raw_span_id

            span_model = Span(
                span_id=span_id,
                parent_span_id=parent_span_id,
                execution_id=trace_id,
                caller_agent_id=agent_id,
                callee_agent_id=callee,
                capability=_capability_from_kind(span_kind),
                started_at=started_at,
                ended_at=ended_at,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                outcome=outcome,
                error_message=error_message,
                metadata=metadata,
            )
            self._trace_spans.setdefault(trace_id, []).append(span_model)
            get_emitter().emit_span_safe(span_model)

        def shutdown(self) -> None:
            return None

        def force_flush(self) -> None:
            return None

    processor = AgentMeterOpenAIProcessor()
    register_fn: Callable[[object], object] | None = getattr(agents, "add_trace_processor", None)
    if register_fn is None:
        register_fn = getattr(tracing_module, "add_trace_processor", None)
    if register_fn is None:
        get_provider = getattr(tracing_module, "get_trace_provider", None)
        if callable(get_provider):
            provider = get_provider()
            register_fn = getattr(provider, "register_processor", None)
    if register_fn is None or not callable(register_fn):
        raise RuntimeError("Unable to locate OpenAI Agents tracing hook registration API.")

    register_fn(processor)
    _REGISTERED = True
