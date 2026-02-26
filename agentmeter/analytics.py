"""Local analytics over stored JSONL execution traces."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Literal
from uuid import uuid4

from .core import ExecutionTrace, Span
from .taxonomy import Capability
from .tracer import SpanKind

_SPAN_OUTCOMES = frozenset({"success", "failure", "timeout", "unknown"})
_TRACE_OUTCOMES = frozenset({"success", "failure", "partial", "unknown"})


def _utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    """Parse ISO datetime-like values and normalize to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_str(value: object) -> str | None:
    """Return a trimmed non-empty string value."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _as_float(value: object) -> float | None:
    """Return numeric values as float."""
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


def _as_int(value: object) -> int | None:
    """Return integer-like values as int."""
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


def _normalize_span_outcome(value: object) -> str:
    """Normalize span outcomes to the supported literal set."""
    if isinstance(value, str) and value in _SPAN_OUTCOMES:
        return value
    return "unknown"


def _normalize_trace_outcome(value: object) -> str:
    """Normalize trace outcomes to the supported literal set."""
    if isinstance(value, str) and value in _TRACE_OUTCOMES:
        return value
    return "unknown"


def _span_kind_from_capability(capability: str) -> SpanKind:
    """Infer span kind from capability taxonomy category."""
    parsed = Capability.from_string(capability)
    category = parsed.value.split(".", 1)[0]
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
    if category == "extraction":
        return SpanKind.TOOL
    if category == "generation":
        return SpanKind.LLM
    return SpanKind.AGENT


def _span_kind_name(span: Span) -> str:
    """Return uppercase span kind name for a span."""
    metadata_kind = span.metadata.get("span_kind")
    if metadata_kind is not None:
        parsed = SpanKind.from_value(metadata_kind)
        if parsed is not None:
            return parsed.name
        upper = metadata_kind.strip().upper()
        if upper:
            return upper
    inferred = _span_kind_from_capability(span.capability)
    return inferred.name


def _percentile(values: list[float], percentile: float) -> float:
    """Compute percentile using linear interpolation between closest ranks."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    if percentile <= 0.0:
        return values[0]
    if percentile >= 100.0:
        return values[-1]

    rank = (percentile / 100.0) * (len(values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(values) - 1)
    if upper_index == lower_index:
        return values[lower_index]
    weight = rank - lower_index
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight


def _percentile_key(percentile: float) -> str:
    """Format percentile keys such as p50 or p99_5."""
    if percentile.is_integer():
        return f"p{int(percentile)}"
    normalized = str(percentile).replace(".", "_")
    return f"p{normalized}"


def _failure_type(span: Span) -> str:
    """Derive a normalized failure type label from span outcome/message."""
    metadata_error_type = span.metadata.get("error_type")
    if metadata_error_type:
        sanitized_meta = re.sub(r"[^a-z0-9]+", "_", metadata_error_type.strip().lower()).strip("_")
        if sanitized_meta:
            return sanitized_meta
    if span.outcome == "timeout":
        return "timeout"
    if span.error_message:
        prefix = span.error_message.strip().lower().split(":", 1)[0]
        token = prefix.split()[0] if prefix else ""
        sanitized = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
        if sanitized:
            return sanitized
    if span.outcome == "failure":
        return "failure"
    if span.outcome == "unknown":
        return "unknown"
    return "failure"


def _trace_outcome_from_spans(spans: list[Span]) -> str:
    """Infer trace-level outcome from span outcomes."""
    if not spans:
        return "unknown"
    success_count = sum(1 for span in spans if span.outcome == "success")
    failure_like_count = sum(1 for span in spans if span.outcome in {"failure", "timeout"})
    if success_count == len(spans):
        return "success"
    if success_count == 0 and failure_like_count > 0:
        return "failure"
    if success_count > 0 and failure_like_count > 0:
        return "partial"
    return "unknown"


def _bucket_key(timestamp: datetime, bucket: Literal["hour", "day", "week"]) -> str:
    """Create a canonical bucket key for hour/day/week groupings."""
    ts = timestamp.astimezone(timezone.utc)
    if bucket == "hour":
        return ts.strftime("%Y-%m-%dT%H:00:00Z")
    if bucket == "day":
        return ts.strftime("%Y-%m-%d")
    week_start: date = ts.date() - timedelta(days=ts.weekday())
    return week_start.isoformat()


def _span_from_event(event: dict[str, object]) -> Span | None:
    """Parse a span JSONL event emitted by `StdoutEmitter`/`FileEmitter`."""
    execution_id = _as_str(event.get("execution_id"))
    caller_agent_id = _as_str(event.get("caller_agent_id")) or "unknown_caller"
    callee_agent_id = _as_str(event.get("callee_agent_id")) or "unknown_callee"
    if execution_id is None:
        return None

    capability = Capability.from_string(_as_str(event.get("capability")) or "unknown").value
    ended_at = _parse_datetime(event.get("ts"))
    latency_ms = _as_float(event.get("latency_ms"))
    if ended_at is None:
        ended_at = _utc_now()
    started_at = ended_at
    if latency_ms is not None and latency_ms >= 0.0:
        started_at = ended_at - timedelta(milliseconds=latency_ms)

    input_tokens = _as_int(event.get("input_tokens"))
    output_tokens = _as_int(event.get("output_tokens"))
    cost_usd = _as_float(event.get("cost_usd"))
    outcome = _normalize_span_outcome(event.get("outcome"))
    error_message = _as_str(event.get("error_message"))
    span_kind = SpanKind.from_value(_as_str(event.get("span_kind")) or "")
    error_type = _as_str(event.get("error_type"))
    metadata = {
        "span_kind": span_kind.name if span_kind is not None else _span_kind_from_capability(capability).name,
        "source": "event",
    }
    if error_type is not None:
        metadata["error_type"] = error_type

    try:
        return Span(
            span_id=_as_str(event.get("span_id")) or str(uuid4()),
            parent_span_id=_as_str(event.get("parent_span_id")),
            execution_id=execution_id,
            caller_agent_id=caller_agent_id,
            callee_agent_id=callee_agent_id,
            capability=capability,
            started_at=started_at,
            ended_at=ended_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            outcome=outcome,
            error_message=error_message,
            metadata=metadata,
        )
    except Exception:
        return None


def _span_from_object(payload: dict[str, object]) -> Span | None:
    """Parse a direct span object from JSONL."""
    execution_id = _as_str(payload.get("execution_id"))
    caller_agent_id = _as_str(payload.get("caller_agent_id"))
    callee_agent_id = _as_str(payload.get("callee_agent_id"))
    if execution_id is None or caller_agent_id is None or callee_agent_id is None:
        return None

    capability = Capability.from_string(_as_str(payload.get("capability")) or "unknown").value
    started_at = _parse_datetime(payload.get("started_at")) or _utc_now()
    ended_at = _parse_datetime(payload.get("ended_at"))
    if ended_at is None:
        latency = _as_float(payload.get("latency_ms"))
        if latency is not None and latency >= 0.0:
            ended_at = started_at + timedelta(milliseconds=latency)

    metadata: dict[str, str] = {}
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            if isinstance(key, str):
                metadata[key] = value if isinstance(value, str) else str(value)
    if "span_kind" not in metadata:
        metadata["span_kind"] = _span_kind_from_capability(capability).name

    try:
        return Span(
            span_id=_as_str(payload.get("span_id")) or str(uuid4()),
            parent_span_id=_as_str(payload.get("parent_span_id")),
            execution_id=execution_id,
            caller_agent_id=caller_agent_id,
            callee_agent_id=callee_agent_id,
            capability=capability,
            started_at=started_at,
            ended_at=ended_at,
            input_tokens=_as_int(payload.get("input_tokens")),
            output_tokens=_as_int(payload.get("output_tokens")),
            cost_usd=_as_float(payload.get("cost_usd")),
            outcome=_normalize_span_outcome(payload.get("outcome")),
            error_message=_as_str(payload.get("error_message")),
            metadata=metadata,
        )
    except Exception:
        return None


class TraceStore:
    """Lazy JSONL-backed trace store.

    The first call to `load()` scans all `.jsonl` files in the configured path.
    Parsed traces are cached for subsequent reads.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: list[ExecutionTrace] | None = None

    def _jsonl_files(self) -> list[Path]:
        """Return all JSONL files to parse."""
        if self._path.is_file() and self._path.suffix.lower() == ".jsonl":
            return [self._path]
        if self._path.is_dir():
            return sorted(self._path.glob("*.jsonl"))
        return []

    def load(self) -> list[ExecutionTrace]:
        """Load traces from JSONL sources on first call and return cached results."""
        if self._cache is not None:
            return list(self._cache)

        full_traces: dict[str, ExecutionTrace] = {}
        trace_events: dict[str, dict[str, object]] = {}
        spans_by_execution: dict[str, list[Span]] = {}

        for file_path in self._jsonl_files():
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue

                    payload_obj: dict[str, object] = payload
                    event_type = _as_str(payload_obj.get("event"))
                    if event_type == "span":
                        parsed_span = _span_from_event(payload_obj)
                        if parsed_span is not None:
                            spans_by_execution.setdefault(parsed_span.execution_id, []).append(parsed_span)
                        continue
                    if event_type == "trace":
                        execution_id = _as_str(payload_obj.get("execution_id"))
                        if execution_id is not None:
                            trace_events[execution_id] = payload_obj
                        continue

                    if "spans" in payload_obj and "root_agent_id" in payload_obj and "execution_id" in payload_obj:
                        try:
                            full_trace = ExecutionTrace.model_validate_json(line)
                        except Exception:
                            full_trace = None
                        if full_trace is not None:
                            full_traces[full_trace.execution_id] = full_trace
                        continue

                    if "span_id" in payload_obj and "execution_id" in payload_obj:
                        parsed_span = _span_from_object(payload_obj)
                        if parsed_span is not None:
                            spans_by_execution.setdefault(parsed_span.execution_id, []).append(parsed_span)
                        continue

                    if "execution_id" in payload_obj and "root_agent_id" in payload_obj and "outcome" in payload_obj:
                        execution_id = _as_str(payload_obj.get("execution_id"))
                        if execution_id is not None:
                            trace_events[execution_id] = payload_obj

        execution_ids = set(trace_events.keys()) | set(spans_by_execution.keys())
        for execution_id in execution_ids:
            if execution_id in full_traces:
                continue
            spans = sorted(
                spans_by_execution.get(execution_id, []),
                key=lambda span: span.started_at,
            )
            trace_event = trace_events.get(execution_id, {})
            root_agent_id = _as_str(trace_event.get("root_agent_id"))
            if root_agent_id is None:
                root_agent_id = spans[0].caller_agent_id if spans else "unknown_agent"

            started_at = _parse_datetime(trace_event.get("started_at"))
            ended_at = _parse_datetime(trace_event.get("ended_at")) or _parse_datetime(trace_event.get("ts"))
            if started_at is None:
                if spans:
                    started_at = min(span.started_at for span in spans)
                else:
                    started_at = ended_at or _utc_now()
            if ended_at is None:
                if spans:
                    ended_at = max((span.ended_at or span.started_at) for span in spans)
                else:
                    ended_at = started_at
            if ended_at < started_at:
                ended_at = started_at

            outcome = _normalize_trace_outcome(trace_event.get("outcome"))
            if outcome == "unknown" and spans:
                outcome = _trace_outcome_from_spans(spans)

            try:
                reconstructed = ExecutionTrace(
                    execution_id=execution_id,
                    root_agent_id=root_agent_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    spans=spans,
                    outcome=outcome,
                )
            except Exception:
                continue
            full_traces[execution_id] = reconstructed

        self._cache = sorted(full_traces.values(), key=lambda trace: trace.started_at)
        return list(self._cache)

    def load_since(self, since: datetime) -> list[ExecutionTrace]:
        """Return traces with timestamps at or after `since` (UTC-normalized)."""
        since_utc = _parse_datetime(since)
        if since_utc is None:
            since_utc = _utc_now()
        traces = self.load()
        result: list[ExecutionTrace] = []
        for trace in traces:
            ended_at = trace.ended_at or trace.started_at
            if trace.started_at >= since_utc or ended_at >= since_utc:
                result.append(trace)
        return result


class Analytics:
    """Analytics engine over in-memory traces or a lazy `TraceStore`."""

    def __init__(self, store: TraceStore | list[ExecutionTrace]) -> None:
        if isinstance(store, TraceStore):
            self._store: TraceStore | None = store
            self._traces: list[ExecutionTrace] | None = None
        else:
            self._store = None
            self._traces = list(store)

    def _get_traces(self) -> list[ExecutionTrace]:
        """Return the backing trace list."""
        if self._store is not None:
            return self._store.load()
        return list(self._traces or [])

    def _all_spans(self) -> list[Span]:
        """Flatten traces into a span list."""
        spans: list[Span] = []
        for trace in self._get_traces():
            spans.extend(trace.spans)
        return spans

    def _filtered_spans(
        self,
        span_kind: SpanKind | None = None,
        agent_id: str | None = None,
    ) -> list[Span]:
        """Filter spans by optional span kind and callee agent id."""
        spans = self._all_spans()
        filtered: list[Span] = []
        for span in spans:
            if span_kind is not None and _span_kind_name(span) != span_kind.name:
                continue
            if agent_id is not None and span.callee_agent_id != agent_id:
                continue
            filtered.append(span)
        return filtered

    def cost_by_span_kind(self) -> dict[str, float]:
        """Return total cost grouped by span kind."""
        spans = self._all_spans()
        if not spans:
            return {}
        totals: dict[str, float] = {}
        for span in spans:
            kind = _span_kind_name(span)
            totals[kind] = totals.get(kind, 0.0) + (span.cost_usd or 0.0)
        return {kind: round(total, 6) for kind, total in totals.items()}

    def latency_percentiles(
        self,
        span_kind: SpanKind | None = None,
        percentiles: list[float] | None = None,
    ) -> dict[str, float]:
        """Return latency percentile metrics for filtered spans."""
        requested = percentiles if percentiles is not None else [50.0, 95.0, 99.0]
        spans = self._filtered_spans(span_kind=span_kind)
        latencies = sorted(span.latency_ms for span in spans if span.latency_ms is not None)
        if not latencies:
            return {}
        results: dict[str, float] = {}
        for raw in requested:
            percentile_value = float(raw)
            if percentile_value < 0.0 or percentile_value > 100.0:
                continue
            results[_percentile_key(percentile_value)] = round(_percentile(latencies, percentile_value), 3)
        return results

    def success_rate(
        self,
        span_kind: SpanKind | None = None,
        agent_id: str | None = None,
    ) -> float:
        """Return success rate for filtered spans."""
        spans = self._filtered_spans(span_kind=span_kind, agent_id=agent_id)
        if not spans:
            return 0.0
        success_count = sum(1 for span in spans if span.outcome == "success")
        return round(success_count / len(spans), 6)

    def failure_breakdown(self) -> dict[str, int]:
        """Group non-success spans by normalized failure type."""
        spans = self._all_spans()
        if not spans:
            return {}
        counts: dict[str, int] = {}
        for span in spans:
            if span.outcome == "success":
                continue
            key = _failure_type(span)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def agent_leaderboard(self) -> list[dict[str, str | int | float]]:
        """Return per-agent ranking sorted by success then average cost."""
        spans = self._all_spans()
        if not spans:
            return []
        groups: dict[tuple[str, str], list[Span]] = {}
        for span in spans:
            group_key = (span.callee_agent_id, _span_kind_name(span))
            groups.setdefault(group_key, []).append(span)

        rows: list[dict[str, str | int | float]] = []
        for (agent_id, span_kind_name), group_spans in groups.items():
            call_count = len(group_spans)
            success_count = sum(1 for span in group_spans if span.outcome == "success")
            avg_cost = sum((span.cost_usd or 0.0) for span in group_spans) / call_count
            avg_latency = sum((span.latency_ms or 0.0) for span in group_spans) / call_count
            rows.append(
                {
                    "agent_id": agent_id,
                    "span_kind": span_kind_name,
                    "call_count": call_count,
                    "success_rate": round(success_count / call_count, 6),
                    "avg_cost_usd": round(avg_cost, 6),
                    "avg_latency_ms": round(avg_latency, 3),
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row["success_rate"]),
                float(row["avg_cost_usd"]),
                float(row["avg_latency_ms"]),
                str(row["agent_id"]),
                str(row["span_kind"]),
            )
        )
        return rows

    def cost_trend(
        self,
        bucket: Literal["hour", "day", "week"] = "day",
    ) -> list[dict[str, str | float | int]]:
        """Return cost trend grouped by hour/day/week buckets."""
        spans = self._all_spans()
        if not spans:
            return []
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for span in spans:
            key = _bucket_key(span.started_at, bucket)
            totals[key] = totals.get(key, 0.0) + (span.cost_usd or 0.0)
            counts[key] = counts.get(key, 0) + 1
        return [
            {
                "bucket": key,
                "total_cost_usd": round(totals[key], 6),
                "span_count": counts[key],
            }
            for key in sorted(totals.keys())
        ]

    def span_kind_distribution(self) -> dict[str, int]:
        """Return call counts grouped by span kind."""
        spans = self._all_spans()
        if not spans:
            return {}
        counts: dict[str, int] = {}
        for span in spans:
            kind = _span_kind_name(span)
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def expensive_span_kinds(self, top_n: int = 5) -> list[dict[str, str | float | int]]:
        """Return top-N span kinds ranked by total cost."""
        if top_n <= 0:
            return []
        spans = self._all_spans()
        if not spans:
            return []
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for span in spans:
            kind = _span_kind_name(span)
            totals[kind] = totals.get(kind, 0.0) + (span.cost_usd or 0.0)
            counts[kind] = counts.get(kind, 0) + 1
        rows = [
            {
                "span_kind": kind,
                "total_cost_usd": round(totals[kind], 6),
                "avg_cost_usd": round(totals[kind] / counts[kind], 6),
                "count": counts[kind],
            }
            for kind in totals
        ]
        rows.sort(
            key=lambda row: (
                -float(row["total_cost_usd"]),
                -float(row["avg_cost_usd"]),
                -int(row["count"]),
                str(row["span_kind"]),
            )
        )
        return rows[:top_n]

    def slowest_agents(self, top_n: int = 10) -> list[dict[str, str | float | int]]:
        """Return top-N slowest agents ranked by p95 latency."""
        if top_n <= 0:
            return []
        spans = self._all_spans()
        if not spans:
            return []
        groups: dict[tuple[str, str], list[float]] = {}
        for span in spans:
            if span.latency_ms is None:
                continue
            group_key = (span.callee_agent_id, _span_kind_name(span))
            groups.setdefault(group_key, []).append(span.latency_ms)
        if not groups:
            return []

        rows: list[dict[str, str | float | int]] = []
        for (agent_id, span_kind_name), latencies in groups.items():
            sorted_latencies = sorted(latencies)
            rows.append(
                {
                    "agent_id": agent_id,
                    "span_kind": span_kind_name,
                    "p95_latency_ms": round(_percentile(sorted_latencies, 95.0), 3),
                    "call_count": len(sorted_latencies),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["p95_latency_ms"]),
                -int(row["call_count"]),
                str(row["agent_id"]),
                str(row["span_kind"]),
            )
        )
        return rows[:top_n]
