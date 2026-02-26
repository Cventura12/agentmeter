"""Core data models for agent identity and execution tracing.

This module defines strict Pydantic v2 models for:
1. Persistent agent identity metadata.
2. Immutable span records for inter-agent calls.
3. Immutable execution trace records containing span collections.

Each model includes a `to_event_dict()` helper that emits a flat dictionary
containing JSON-serializable primitives suitable for logging, transport, or
event-stream ingestion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, computed_field, field_validator, model_validator

from .taxonomy import Capability

EventValue = str | int | float | bool | None
EventDict = dict[str, EventValue]

_UTC = timezone.utc
_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_CAPABILITY_VALUES = frozenset(capability.value for capability in Capability)


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(_UTC)


def _validate_uuid(value: str, field_name: str) -> str:
    """Validate a UUID string and normalize it to canonical string form."""
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid UUID string") from exc
    return str(parsed)


def _validate_utc(value: datetime, field_name: str) -> datetime:
    """Ensure a datetime is timezone-aware and explicitly UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware in UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC (offset +00:00)")
    return value


def _to_utc_iso(value: datetime | None) -> str | None:
    """Convert a datetime to an ISO 8601 UTC string (Z suffix)."""
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


class AgentIdentity(BaseModel):
    """Represents a stable, versioned identity record for an agent.

    Fields:
    - `agent_id`: UUID string identifying the agent record.
    - `capability`: Capability taxonomy value such as `extraction.invoice`.
    - `vertical`: Optional vertical context (for example: `construction`).
    - `version`: Semantic version of the deployed agent implementation.
    - `tags`: Arbitrary categorization labels.
    - `created_at`: UTC creation timestamp.
    """

    model_config = ConfigDict(strict=True)

    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    capability: str
    vertical: str | None = None
    version: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        """Validate that `agent_id` is a UUID string."""
        return _validate_uuid(value, "agent_id")

    @field_validator("capability")
    @classmethod
    def _validate_capability(cls, value: str) -> str:
        """Validate capability values against the shared taxonomy enum."""
        if value not in _CAPABILITY_VALUES:
            raise ValueError(f"capability must be one of: {sorted(_CAPABILITY_VALUES)}")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        """Validate semantic version format (semver)."""
        if _SEMVER_PATTERN.fullmatch(value) is None:
            raise ValueError("version must be a valid semantic version (e.g. 1.2.3)")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        """Validate UTC requirements for `created_at`."""
        return _validate_utc(value, "created_at")

    def to_event_dict(self) -> EventDict:
        """Return a flat JSON-safe event representation of the model."""
        return {
            "agent_id": self.agent_id,
            "capability": self.capability,
            "vertical": self.vertical,
            "version": self.version,
            "tags_csv": ",".join(self.tags),
            "created_at": _to_utc_iso(self.created_at),
        }


class Span(BaseModel):
    """Represents a single immutable execution span in an agent call graph.

    A span models one inter-agent call edge, including actor identities, timing,
    token usage, cost, and outcome.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    execution_id: str
    caller_agent_id: str
    callee_agent_id: str
    capability: str
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    outcome: Literal["success", "failure", "timeout", "unknown"] = "unknown"
    error_message: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("span_id")
    @classmethod
    def _validate_span_id(cls, value: str) -> str:
        """Validate that `span_id` is a UUID string."""
        return _validate_uuid(value, "span_id")

    @field_validator("parent_span_id")
    @classmethod
    def _validate_parent_span_id(cls, value: str | None) -> str | None:
        """Validate that `parent_span_id` is a UUID string when provided."""
        if value is None:
            return None
        return _validate_uuid(value, "parent_span_id")

    @field_validator("execution_id")
    @classmethod
    def _validate_execution_id(cls, value: str) -> str:
        """Validate that `execution_id` is a UUID string."""
        return _validate_uuid(value, "execution_id")

    @field_validator("capability")
    @classmethod
    def _validate_capability(cls, value: str) -> str:
        """Validate capability values against the shared taxonomy enum."""
        if value not in _CAPABILITY_VALUES:
            raise ValueError(f"capability must be one of: {sorted(_CAPABILITY_VALUES)}")
        return value

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        """Validate UTC requirements for `started_at`."""
        return _validate_utc(value, "started_at")

    @field_validator("ended_at")
    @classmethod
    def _validate_ended_at(cls, value: datetime | None) -> datetime | None:
        """Validate UTC requirements for `ended_at` when provided."""
        if value is None:
            return None
        return _validate_utc(value, "ended_at")

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def _validate_tokens(cls, value: int | None, info: ValidationInfo) -> int | None:
        """Ensure token count fields are non-negative when provided."""
        if value is not None and value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @field_validator("cost_usd")
    @classmethod
    def _validate_cost(cls, value: float | None) -> float | None:
        """Ensure cost values are non-negative when provided."""
        if value is not None and value < 0:
            raise ValueError("cost_usd must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> "Span":
        """Ensure `ended_at` never predates `started_at`."""
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be before started_at")
        return self

    @computed_field(return_type=float | None)
    @property
    def latency_ms(self) -> float | None:
        """Compute latency in milliseconds when both timestamps are known."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000.0

    def to_event_dict(self) -> EventDict:
        """Return a flat JSON-safe event representation of the model."""
        event: EventDict = {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "execution_id": self.execution_id,
            "caller_agent_id": self.caller_agent_id,
            "callee_agent_id": self.callee_agent_id,
            "capability": self.capability,
            "started_at": _to_utc_iso(self.started_at),
            "ended_at": _to_utc_iso(self.ended_at),
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "metadata_count": len(self.metadata),
        }
        for key, value in sorted(self.metadata.items()):
            normalized_key = re.sub(r"[^0-9A-Za-z_]", "_", key).strip("_") or "key"
            event[f"metadata_{normalized_key}"] = value
        return event


class AgentCall(BaseModel):
    """Backward-compatible call model for existing tracer code paths.

    This model is retained so older code importing `AgentCall` continues to
    import successfully while span-first tracing APIs are phased in.
    """

    model_config = ConfigDict(strict=True)

    call_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    capability: Capability
    latency_ms: float
    cost_usd: float
    success: bool
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None
    parent_call_id: str | None = None
    error_message: str | None = None

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        """Validate UTC requirements for `started_at`."""
        return _validate_utc(value, "started_at")

    @field_validator("ended_at")
    @classmethod
    def _validate_ended_at(cls, value: datetime | None) -> datetime | None:
        """Validate UTC requirements for `ended_at` when provided."""
        if value is None:
            return None
        return _validate_utc(value, "ended_at")

    @field_validator("latency_ms")
    @classmethod
    def _validate_latency(cls, value: float) -> float:
        """Validate non-negative latency values."""
        if value < 0:
            raise ValueError("latency_ms must be non-negative")
        return value

    @field_validator("cost_usd")
    @classmethod
    def _validate_cost(cls, value: float) -> float:
        """Validate non-negative cost values."""
        if value < 0:
            raise ValueError("cost_usd must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> "AgentCall":
        """Default `ended_at` and enforce temporal ordering."""
        if self.ended_at is None:
            self.ended_at = self.started_at + timedelta(milliseconds=self.latency_ms)
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be before started_at")
        return self


class ExecutionTrace(BaseModel):
    """Represents an immutable execution-level trace composed of spans.

    Fields:
    - `execution_id`: UUID string identifying one execution.
    - `root_agent_id`: Entry-point agent identifier for the execution.
    - `started_at` / `ended_at`: Execution wall-clock envelope in UTC.
    - `spans`: Ordered span list for the execution DAG edges.
    - `outcome`: Execution-level outcome classification.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    root_agent_id: str
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None
    spans: list[Span] = Field(default_factory=list)
    outcome: Literal["success", "failure", "partial", "unknown"] = "unknown"

    @field_validator("execution_id")
    @classmethod
    def _validate_execution_id(cls, value: str) -> str:
        """Validate that `execution_id` is a UUID string."""
        return _validate_uuid(value, "execution_id")

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        """Validate UTC requirements for `started_at`."""
        return _validate_utc(value, "started_at")

    @field_validator("ended_at")
    @classmethod
    def _validate_ended_at(cls, value: datetime | None) -> datetime | None:
        """Validate UTC requirements for `ended_at` when provided."""
        if value is None:
            return None
        return _validate_utc(value, "ended_at")

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> "ExecutionTrace":
        """Ensure `ended_at` never predates `started_at`."""
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be before started_at")
        return self

    @computed_field(return_type=float)
    @property
    def total_cost_usd(self) -> float:
        """Compute total cost as the sum of per-span costs."""
        return round(sum(span.cost_usd or 0.0 for span in self.spans), 6)

    @computed_field(return_type=float | None)
    @property
    def total_latency_ms(self) -> float | None:
        """Compute execution latency from `started_at` to `ended_at`."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000.0

    @computed_field(return_type=dict[str, int])
    @property
    def span_kind_breakdown(self) -> dict[str, int]:
        """Compute per-span-kind counts for this trace."""
        counts: dict[str, int] = {}
        for span in self.spans:
            metadata_kind = span.metadata.get("span_kind")
            kind = metadata_kind.strip().upper() if metadata_kind is not None else ""
            if not kind:
                category = Capability.from_string(span.capability).value.split(".", 1)[0]
                if category == "retrieval":
                    kind = "RETRIEVER"
                elif category == "verification":
                    kind = "GUARDRAIL"
                elif category == "transformation":
                    kind = "TOOL"
                elif category == "planning":
                    kind = "AGENT"
                elif category == "classification":
                    kind = "CHAIN"
                elif category == "generation":
                    kind = "LLM"
                elif category == "extraction":
                    kind = "TOOL"
                else:
                    kind = "AGENT"
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def to_event_dict(self) -> EventDict:
        """Return a flat JSON-safe event representation of the model."""
        span_ids = ",".join(span.span_id for span in self.spans)
        span_outcomes = ",".join(span.outcome for span in self.spans)
        return {
            "execution_id": self.execution_id,
            "root_agent_id": self.root_agent_id,
            "started_at": _to_utc_iso(self.started_at),
            "ended_at": _to_utc_iso(self.ended_at),
            "span_count": len(self.spans),
            "span_ids_csv": span_ids,
            "span_outcomes_csv": span_outcomes,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "outcome": self.outcome,
        }
