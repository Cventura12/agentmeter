"""Anonymized cohort schema and exporters for privacy-preserving analytics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from .core import ExecutionTrace, Span

if TYPE_CHECKING:
    from .tracer import SpanKind

_RECORDED_AT_HOUR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}$")
_ORCHESTRATOR_VALUES = frozenset({"langchain", "openai-agents", "custom", "unknown"})
_MODEL_FAMILY_VALUES = frozenset({"gpt-4", "claude-3", "gemini", "other"})
_SPAN_KIND_VALUES = frozenset(
    {
        "llm",
        "chain",
        "retriever",
        "tool",
        "agent",
        "embedding",
        "reranker",
        "guardrail",
        "evaluator",
    }
)
_SPAN_KIND_NAME_TO_VALUE = {value.upper(): value for value in _SPAN_KIND_VALUES}


def _now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def _round_to_bucket(value: int | float | None, bucket_size: int) -> int:
    """Round non-negative numeric values to nearest integer bucket size."""
    if value is None:
        return 0
    numeric = float(value)
    if numeric <= 0:
        return 0
    return int(((numeric + (bucket_size / 2.0)) // bucket_size) * bucket_size)


def _round_cost(value: float | None) -> float:
    """Round non-negative costs to 4 decimal places."""
    if value is None or value <= 0:
        return 0.0
    return round(float(value), 4)


def _normalize_outcome(value: str | None) -> str:
    """Map raw span outcomes to anonymized cohort outcome values."""
    if value == "success":
        return "success"
    if value == "timeout":
        return "timeout"
    return "failure"


def _parse_span_kind(value: object) -> str | None:
    """Parse span kind from enum-like objects or name/value strings."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _SPAN_KIND_VALUES:
            return lowered
        return _SPAN_KIND_NAME_TO_VALUE.get(value.strip().upper())
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return _parse_span_kind(enum_value)
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return _SPAN_KIND_NAME_TO_VALUE.get(enum_name.strip().upper())
    return None


def _infer_span_kind(span: Span) -> str:
    """Infer span kind from metadata or capability taxonomy prefix."""
    raw_kind = span.metadata.get("span_kind")
    parsed_kind = _parse_span_kind(raw_kind)
    if parsed_kind is not None:
        return parsed_kind
    category = span.capability.split(".", 1)[0]
    if category == "retrieval":
        return "retriever"
    if category == "transformation":
        return "tool"
    if category == "verification":
        return "guardrail"
    if category == "planning":
        return "agent"
    if category == "classification":
        return "chain"
    if category == "extraction":
        return "tool"
    return "llm"


def _normalize_model_family(value: str | None) -> str | None:
    """Collapse model names to coarse model families only."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    if lowered.startswith("gpt-4"):
        return "gpt-4"
    if lowered.startswith("claude-3"):
        return "claude-3"
    if lowered.startswith("gemini"):
        return "gemini"
    return "other"


def _normalize_orchestrator(value: str | None) -> str | None:
    """Normalize orchestrator values to the coarse allowed set."""
    if value is None:
        return "unknown"
    lowered = value.strip().lower()
    if not lowered:
        return "unknown"
    if lowered in _ORCHESTRATOR_VALUES:
        return lowered
    return "unknown"


def _to_recorded_hour(value: datetime | None) -> str:
    """Convert datetimes to `YYYY-MM-DDTHH` UTC hour format."""
    timestamp = value if value is not None else _now_utc()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y-%m-%dT%H")


def _has_more_than_4_decimals(value: float) -> bool:
    """Return True when float has more than 4 decimal places."""
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation:
        return True
    quantized = decimal_value.quantize(Decimal("0.0001"))
    return decimal_value != quantized


class AnonymizedSpanMetric(BaseModel):
    """Privacy-enforced anonymized span metric for cross-tenant cohort analysis."""

    model_config = ConfigDict(strict=True, frozen=True)

    schema_version: str = "0.1.0"
    span_kind: str
    outcome: str
    latency_ms_bucket: int
    cost_usd_bucket: float
    input_token_bucket: int
    output_token_bucket: int
    model_family: str | None = None
    orchestrator: str | None = None
    recorded_at_hour: str
    parent_span_kind: str | None = None
    depth_in_trace: int = 0

    @field_validator("span_kind")
    @classmethod
    def _validate_span_kind(cls, value: str) -> str:
        parsed = _parse_span_kind(value)
        if parsed is None:
            raise ValueError("span_kind must be a valid SpanKind value")
        return parsed

    @field_validator("outcome")
    @classmethod
    def _validate_outcome(cls, value: str) -> str:
        if value not in {"success", "failure", "timeout"}:
            raise ValueError("outcome must be one of: success, failure, timeout")
        return value

    @field_validator("latency_ms_bucket")
    @classmethod
    def _validate_latency_bucket(cls, value: int) -> int:
        if value % 50 != 0:
            raise ValueError("latency_ms_bucket must be divisible by 50")
        return value

    @field_validator("cost_usd_bucket")
    @classmethod
    def _validate_cost_bucket(cls, value: float) -> float:
        if _has_more_than_4_decimals(value):
            raise ValueError("cost_usd_bucket must have at most 4 decimal places")
        return value

    @field_validator("input_token_bucket")
    @classmethod
    def _validate_input_bucket(cls, value: int) -> int:
        if value % 100 != 0:
            raise ValueError("input_token_bucket must be divisible by 100")
        return value

    @field_validator("output_token_bucket")
    @classmethod
    def _validate_output_bucket(cls, value: int) -> int:
        if value % 100 != 0:
            raise ValueError("output_token_bucket must be divisible by 100")
        return value

    @field_validator("model_family")
    @classmethod
    def _validate_model_family(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _MODEL_FAMILY_VALUES:
            raise ValueError("model_family must be coarse (gpt-4, claude-3, gemini, other)")
        return value

    @field_validator("orchestrator")
    @classmethod
    def _validate_orchestrator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _ORCHESTRATOR_VALUES:
            raise ValueError("orchestrator must be one of: langchain, openai-agents, custom, unknown")
        return value

    @field_validator("recorded_at_hour")
    @classmethod
    def _validate_recorded_at_hour(cls, value: str) -> str:
        if _RECORDED_AT_HOUR_PATTERN.fullmatch(value) is None:
            raise ValueError("recorded_at_hour must match YYYY-MM-DDTHH exactly")
        return value

    @field_validator("parent_span_kind")
    @classmethod
    def _validate_parent_span_kind(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = _parse_span_kind(value)
        if parsed is None:
            raise ValueError("parent_span_kind must be a valid SpanKind value")
        return parsed

    @field_validator("depth_in_trace")
    @classmethod
    def _validate_depth(cls, value: int) -> int:
        if value < 0:
            raise ValueError("depth_in_trace must be >= 0")
        return value


class CohortExporter:
    """Exporter utilities for privacy-safe cohort metrics."""

    @staticmethod
    def from_span(
        span: Span,
        *,
        depth: int = 0,
        parent_span_kind: SpanKind | None = None,
        orchestrator: str | None = None,
    ) -> AnonymizedSpanMetric:
        """Build anonymized metric from span with enforced precision stripping."""
        span_kind = _infer_span_kind(span)
        parent_kind_value = _parse_span_kind(parent_span_kind)
        model_hint = span.metadata.get("model") if span.metadata else None
        if model_hint is None:
            model_hint = span.callee_agent_id
        try:
            return AnonymizedSpanMetric(
                span_kind=span_kind,
                outcome=_normalize_outcome(span.outcome),
                latency_ms_bucket=_round_to_bucket(span.latency_ms, 50),
                cost_usd_bucket=_round_cost(span.cost_usd),
                input_token_bucket=_round_to_bucket(span.input_tokens, 100),
                output_token_bucket=_round_to_bucket(span.output_tokens, 100),
                model_family=_normalize_model_family(model_hint),
                orchestrator=_normalize_orchestrator(orchestrator),
                recorded_at_hour=_to_recorded_hour(span.ended_at or span.started_at),
                parent_span_kind=parent_kind_value,
                depth_in_trace=max(0, depth),
            )
        except Exception:
            return AnonymizedSpanMetric(
                span_kind="agent",
                outcome="failure",
                latency_ms_bucket=0,
                cost_usd_bucket=0.0,
                input_token_bucket=0,
                output_token_bucket=0,
                model_family=None,
                orchestrator="unknown",
                recorded_at_hour=_to_recorded_hour(None),
                parent_span_kind=None,
                depth_in_trace=0,
            )

    @staticmethod
    def from_trace(
        trace: ExecutionTrace,
        *,
        orchestrator: str | None = None,
    ) -> list[AnonymizedSpanMetric]:
        """Export all spans in a trace with inferred depth and parent span kind."""
        span_by_id = {span.span_id: span for span in trace.spans}
        children: dict[str | None, list[str]] = defaultdict(list)
        for span in trace.spans:
            parent_id = span.parent_span_id if span.parent_span_id in span_by_id else None
            children[parent_id].append(span.span_id)

        depth_map: dict[str, int] = {}
        root_ids = children.get(None, [])
        queue: list[tuple[str, int]] = [(root_id, 0) for root_id in root_ids]
        visited: set[str] = set()
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            depth_map[current_id] = depth
            for child_id in children.get(current_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        for span in trace.spans:
            if span.span_id not in depth_map:
                depth_map[span.span_id] = 0

        from .tracer import SpanKind as RuntimeSpanKind

        metrics: list[AnonymizedSpanMetric] = []
        for span in trace.spans:
            parent_kind: str | None = None
            parent_kind_enum: SpanKind | None = None
            if span.parent_span_id is not None and span.parent_span_id in span_by_id:
                parent_kind = _infer_span_kind(span_by_id[span.parent_span_id])
                parent_kind_enum = RuntimeSpanKind.from_value(parent_kind)
            metrics.append(
                CohortExporter.from_span(
                    span,
                    depth=depth_map.get(span.span_id, 0),
                    parent_span_kind=parent_kind_enum,
                    orchestrator=orchestrator,
                )
            )
        return metrics

    @staticmethod
    def to_jsonl(metrics: list[AnonymizedSpanMetric]) -> str:
        """Return deterministic JSONL, sorted by (span_kind, recorded_at_hour)."""
        sorted_metrics = sorted(
            metrics,
            key=lambda metric: (
                metric.span_kind,
                metric.recorded_at_hour,
                metric.outcome,
                metric.depth_in_trace,
                metric.parent_span_kind or "",
                metric.model_family or "",
                metric.orchestrator or "",
                metric.latency_ms_bucket,
                metric.cost_usd_bucket,
                metric.input_token_bucket,
                metric.output_token_bucket,
            ),
        )
        return "\n".join(
            json.dumps(metric.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            for metric in sorted_metrics
        )

    @staticmethod
    def validate_privacy(metric: AnonymizedSpanMetric) -> tuple[bool, list[str]]:
        """Validate privacy constraints and return (is_valid, violations)."""
        violations: list[str] = []

        parsed_span_kind = _parse_span_kind(metric.span_kind)
        if parsed_span_kind is None:
            violations.append("span_kind must be a valid SpanKind value")

        if metric.parent_span_kind is not None and _parse_span_kind(metric.parent_span_kind) is None:
            violations.append("parent_span_kind must be a valid SpanKind value")

        if metric.outcome not in {"success", "failure", "timeout"}:
            violations.append("outcome must be success/failure/timeout")

        if metric.latency_ms_bucket % 50 != 0:
            violations.append("latency_ms_bucket must be divisible by 50")

        if _has_more_than_4_decimals(metric.cost_usd_bucket):
            violations.append("cost_usd_bucket must have at most 4 decimal places")

        if metric.input_token_bucket % 100 != 0:
            violations.append("input_token_bucket must be divisible by 100")

        if metric.output_token_bucket % 100 != 0:
            violations.append("output_token_bucket must be divisible by 100")

        if _RECORDED_AT_HOUR_PATTERN.fullmatch(metric.recorded_at_hour) is None:
            violations.append("recorded_at_hour must match YYYY-MM-DDTHH")

        if metric.orchestrator is not None and metric.orchestrator not in _ORCHESTRATOR_VALUES:
            violations.append("orchestrator must be coarse and from allowed set")

        if metric.model_family is not None and metric.model_family not in _MODEL_FAMILY_VALUES:
            violations.append("model_family must be coarse and from allowed set")

        if metric.depth_in_trace < 0:
            violations.append("depth_in_trace must be >= 0")

        return len(violations) == 0, violations
