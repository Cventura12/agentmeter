"""LiteLLM callback integration for agentmeter."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from agentmeter.core import Span
from agentmeter.emitter import get_emitter
from agentmeter.tracer import SpanKind

_REGISTERED = False


def _instruction() -> str:
    """Return dependency installation instruction."""
    return "LiteLLM is required. Install with: pip install agentmeter[litellm]"


def _now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def _to_dt(value: object) -> datetime | None:
    """Parse datetime-like objects and normalize to UTC."""
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


def _as_str(value: object) -> str | None:
    """Convert non-empty string values."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _as_int(value: object) -> int | None:
    """Convert values to integers when possible."""
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


def _as_float(value: object) -> float | None:
    """Convert numeric values to float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _ensure_uuid(value: object) -> str:
    """Convert arbitrary ids to UUID strings."""
    raw = _as_str(value)
    if raw is None:
        return str(uuid4())
    try:
        return str(UUID(raw))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, raw))


def _extract_usage(response: object) -> tuple[int | None, int | None]:
    """Extract prompt/completion tokens from LiteLLM responses."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if isinstance(usage, dict):
        prompt = _as_int(usage.get("prompt_tokens")) or _as_int(usage.get("input_tokens"))
        completion = _as_int(usage.get("completion_tokens")) or _as_int(usage.get("output_tokens"))
        return prompt, completion

    prompt = _as_int(getattr(usage, "prompt_tokens", None)) or _as_int(getattr(usage, "input_tokens", None))
    completion = _as_int(getattr(usage, "completion_tokens", None)) or _as_int(getattr(usage, "output_tokens", None))
    return prompt, completion


def _extract_model(kwargs_payload: object, response: object) -> str:
    """Extract model string from callback data."""
    if isinstance(kwargs_payload, dict):
        model = _as_str(kwargs_payload.get("model"))
        if model is not None:
            return model
    model = _as_str(getattr(response, "model", None))
    if model is None and isinstance(response, dict):
        model = _as_str(response.get("model"))
    return model or "unknown_model"


def _extract_call_id(kwargs_payload: object, response: object) -> str:
    """Extract completion call identifier for execution grouping."""
    if isinstance(kwargs_payload, dict):
        for key in ("litellm_call_id", "call_id", "id"):
            call_id = _as_str(kwargs_payload.get(key))
            if call_id is not None:
                return call_id
    call_id = _as_str(getattr(response, "id", None))
    if call_id is None and isinstance(response, dict):
        call_id = _as_str(response.get("id"))
    return call_id or str(uuid4())


def _emit_span(
    agent_id: str,
    *,
    kwargs_payload: object,
    response: object | None,
    start_time: object | None,
    end_time: object | None,
    error_message: str | None,
    cost: float | None,
) -> None:
    """Build and emit one span to the configured global emitter."""
    response_obj = response if response is not None else {}
    execution_id = _ensure_uuid(_extract_call_id(kwargs_payload, response_obj))
    started_at = _to_dt(start_time) or _now_utc()
    ended_at = _to_dt(end_time) or _now_utc()
    if ended_at < started_at:
        ended_at = started_at

    prompt_tokens, completion_tokens = _extract_usage(response_obj)
    model = _extract_model(kwargs_payload, response_obj)

    if cost is None:
        if isinstance(kwargs_payload, dict):
            cost = _as_float(kwargs_payload.get("response_cost"))
    outcome = "failure" if error_message else "success"
    if error_message and "timeout" in error_message.lower():
        outcome = "timeout"

    span = Span(
        span_id=str(uuid4()),
        parent_span_id=None,
        execution_id=execution_id,
        caller_agent_id=agent_id,
        callee_agent_id=model,
        capability="generation.summary",
        started_at=started_at,
        ended_at=ended_at,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cost_usd=cost,
        outcome=outcome,  # type: ignore[arg-type]
        error_message=error_message,
        metadata={
            "framework": "litellm",
            "span_kind": SpanKind.LLM.name,
            "model": model,
        },
    )
    get_emitter().emit_span_safe(span)


def setup_litellm_tracing(agent_id: str) -> None:
    """Register LiteLLM success/failure callbacks that emit agentmeter spans."""
    global _REGISTERED

    if not agent_id:
        raise ValueError("agent_id must be non-empty")
    if _REGISTERED:
        return

    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_instruction()) from exc

    def _cost_or_none(response_obj: object) -> float | None:
        completion_cost_fn: Callable[[object], object] | None = getattr(litellm, "_completion_cost", None)
        if completion_cost_fn is None:
            return None
        try:
            value = completion_cost_fn(response_obj)
        except Exception:
            return None
        return _as_float(value)

    def _success_callback(*callback_args: object, **callback_kwargs: object) -> None:
        kwargs_payload = callback_kwargs.get("kwargs")
        response = callback_kwargs.get("completion_response")
        start_time = callback_kwargs.get("start_time")
        end_time = callback_kwargs.get("end_time")

        if kwargs_payload is None and len(callback_args) >= 1:
            kwargs_payload = callback_args[0]
        if response is None and len(callback_args) >= 2:
            response = callback_args[1]
        if start_time is None and len(callback_args) >= 3:
            start_time = callback_args[2]
        if end_time is None and len(callback_args) >= 4:
            end_time = callback_args[3]

        _emit_span(
            agent_id,
            kwargs_payload=kwargs_payload,
            response=response,
            start_time=start_time,
            end_time=end_time,
            error_message=None,
            cost=_cost_or_none(response),
        )

    def _failure_callback(*callback_args: object, **callback_kwargs: object) -> None:
        kwargs_payload = callback_kwargs.get("kwargs")
        response = callback_kwargs.get("completion_response") or callback_kwargs.get("response_obj")
        start_time = callback_kwargs.get("start_time")
        end_time = callback_kwargs.get("end_time")
        error_obj = callback_kwargs.get("exception") or callback_kwargs.get("error")

        if kwargs_payload is None and len(callback_args) >= 1:
            kwargs_payload = callback_args[0]
        if response is None and len(callback_args) >= 2:
            response = callback_args[1]
        if start_time is None and len(callback_args) >= 3:
            start_time = callback_args[2]
        if end_time is None and len(callback_args) >= 4:
            end_time = callback_args[3]
        if error_obj is None and len(callback_args) >= 5:
            error_obj = callback_args[4]

        error_message = _as_str(str(error_obj)) if error_obj is not None else "litellm_failure"
        _emit_span(
            agent_id,
            kwargs_payload=kwargs_payload,
            response=response,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message,
            cost=_cost_or_none(response),
        )

    success_callbacks = getattr(litellm, "success_callback", None)
    if success_callbacks is None:
        success_callbacks = []
    if not isinstance(success_callbacks, list):
        success_callbacks = [success_callbacks]
    success_callbacks.append(_success_callback)
    litellm.success_callback = success_callbacks

    failure_callbacks = getattr(litellm, "failure_callback", None)
    if failure_callbacks is None:
        failure_callbacks = []
    if not isinstance(failure_callbacks, list):
        failure_callbacks = [failure_callbacks]
    failure_callbacks.append(_failure_callback)
    litellm.failure_callback = failure_callbacks

    _REGISTERED = True
    print("[agentmeter] LiteLLM tracing enabled", file=sys.stderr)
