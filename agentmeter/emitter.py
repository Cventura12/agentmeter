"""Event emitters for spans and execution traces."""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Final

from .core import ExecutionTrace, Span

_GREEN: Final[str] = "\033[32m"
_RED: Final[str] = "\033[31m"
_YELLOW: Final[str] = "\033[33m"
_RESET: Final[str] = "\033[0m"


def _dt_to_ts(value: datetime | None) -> str:
    """Return ISO timestamp with UTC `Z` suffix and fallback default."""
    if value is not None:
        return value.isoformat().replace("+00:00", "Z")
    return "1970-01-01T00:00:00Z"


def _span_kind(capability: str) -> str:
    """Map capability families to a human-friendly span kind."""
    category = capability.split(".", 1)[0]
    if category == "extraction":
        return "OCR"
    if category == "retrieval":
        return "RAG"
    if category == "transformation":
        return "TRANSFORM"
    if category == "verification":
        return "VALIDATION"
    return "LLM"


def _span_event_payload(span: Span) -> dict[str, str | float | int | None]:
    """Build a JSON-serializable span payload."""
    metadata_kind = span.metadata.get("span_kind")
    span_kind = metadata_kind if metadata_kind is not None else _span_kind(span.capability)
    return {
        "event": "span",
        "execution_id": span.execution_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "caller_agent_id": span.caller_agent_id,
        "callee_agent_id": span.callee_agent_id,
        "capability": span.capability,
        "span_kind": span_kind,
        "outcome": span.outcome,
        "cost_usd": span.cost_usd,
        "latency_ms": span.latency_ms,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "error_message": span.error_message,
        "error_type": span.metadata.get("error_type"),
        "ts": _dt_to_ts(span.ended_at),
    }


def _trace_event_payload(trace: ExecutionTrace) -> dict[str, str | float | int | None]:
    """Build a JSON-serializable execution trace payload."""
    return {
        "event": "trace",
        "execution_id": trace.execution_id,
        "root_agent_id": trace.root_agent_id,
        "outcome": trace.outcome,
        "span_count": len(trace.spans),
        "total_cost_usd": trace.total_cost_usd,
        "total_latency_ms": trace.total_latency_ms,
        "ts": _dt_to_ts(trace.ended_at),
    }


class BaseEmitter(ABC):
    """Abstract emitter interface for span and execution trace events."""

    @abstractmethod
    def emit_span(self, span: Span) -> None:
        """Emit a single span event."""

    @abstractmethod
    def emit_trace(self, trace: ExecutionTrace) -> None:
        """Emit an execution trace event."""

    def emit_span_safe(self, span: Span) -> None:
        """Never raises. Logs errors to stderr."""
        try:
            self.emit_span(span)
        except Exception as exc:  # noqa: BLE001
            print(f"[agentmeter] emitter error: {exc}", file=sys.stderr)


class StdoutEmitter(BaseEmitter):
    """Emit span/trace events as JSON lines to stdout."""

    def __init__(self, *, silent: bool = False, colorize: bool = True) -> None:
        self._silent = silent
        self._colorize = colorize and "NO_COLOR" not in os.environ

    def _color_for_outcome(self, outcome: str) -> str:
        """Return ANSI color prefix for an event outcome."""
        if outcome == "success":
            return _GREEN
        if outcome in {"failure", "timeout"}:
            return _RED
        return _YELLOW

    def _emit_payload(self, payload: dict[str, str | float | int | None]) -> None:
        """Serialize and print payload, optionally with ANSI color."""
        if self._silent:
            return
        rendered = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        if self._colorize:
            color = self._color_for_outcome(str(payload.get("outcome", "unknown")))
            print(f"{color}{rendered}{_RESET}")
        else:
            print(rendered)

    def emit_span(self, span: Span) -> None:
        """Emit one span payload to stdout."""
        self._emit_payload(_span_event_payload(span))

    def emit_trace(self, trace: ExecutionTrace) -> None:
        """Emit one trace payload to stdout."""
        self._emit_payload(_trace_event_payload(trace))


class FileEmitter(BaseEmitter):
    """Write newline-delimited JSON (JSONL) events with lock-based thread safety."""

    def __init__(self, path: str | Path, *, rotate_mb: float = 10.0) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._rotate_bytes = int(max(0.0, rotate_mb) * 1024 * 1024)

    def _rotated_path(self, index: int) -> Path:
        """Return `path.index` for rotation slots."""
        return Path(f"{self._path}.{index}")

    def _rotate_if_needed(self) -> None:
        """Rotate base log file when current size exceeds configured threshold."""
        if self._rotate_bytes <= 0:
            return
        if not self._path.exists():
            return
        if self._path.stat().st_size <= self._rotate_bytes:
            return

        oldest = self._rotated_path(5)
        if oldest.exists():
            oldest.unlink()
        for index in range(4, 0, -1):
            src = self._rotated_path(index)
            dst = self._rotated_path(index + 1)
            if src.exists():
                src.replace(dst)
        self._path.replace(self._rotated_path(1))

    def _write_payload(self, payload: dict[str, str | float | int | None]) -> None:
        """Append one JSON event line and flush immediately."""
        rendered = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.write("\n")
                handle.flush()

    def emit_span(self, span: Span) -> None:
        """Append a span event to JSONL output."""
        self._write_payload(_span_event_payload(span))

    def emit_trace(self, trace: ExecutionTrace) -> None:
        """Append a trace event to JSONL output."""
        self._write_payload(_trace_event_payload(trace))


class CompositeEmitter(BaseEmitter):
    """Fan out events to multiple emitters and suppress individual failures."""

    def __init__(self, emitters: list[BaseEmitter]) -> None:
        self._emitters = list(emitters)

    def emit_span(self, span: Span) -> None:
        """Emit a span to all children, logging failures and continuing."""
        for emitter in self._emitters:
            try:
                emitter.emit_span(span)
            except Exception as exc:  # noqa: BLE001
                print(f"[agentmeter] emitter error: {exc}", file=sys.stderr)

    def emit_trace(self, trace: ExecutionTrace) -> None:
        """Emit a trace to all children, logging failures and continuing."""
        for emitter in self._emitters:
            try:
                emitter.emit_trace(trace)
            except Exception as exc:  # noqa: BLE001
                print(f"[agentmeter] emitter error: {exc}", file=sys.stderr)


# Global default emitter - silent by default
_default_emitter: BaseEmitter = StdoutEmitter(silent=True)
_cohort_export_enabled: bool = False
_cohort_path: Path | None = None


def configure(
    emitter: BaseEmitter | None = None,
    *,
    silent: bool | None = None,
    log_path: str | Path | None = None,
    cohort_export: bool = False,
    cohort_path: str | Path | None = None,
) -> None:
    """
    Configure the global default emitter.

    configure()                            # silent stdout (default)
    configure(silent=False)                # verbose stdout
    configure(log_path="./logs/am.jsonl")  # file output
    configure(FileEmitter("./logs/am.jsonl"))  # explicit emitter
    """
    global _cohort_export_enabled
    global _cohort_path
    global _default_emitter

    _cohort_export_enabled = cohort_export
    if cohort_export:
        resolved = Path(cohort_path) if cohort_path is not None else Path("./agentmeter_logs/cohort_metrics.jsonl")
        _cohort_path = resolved
    else:
        _cohort_path = Path(cohort_path) if cohort_path is not None else None

    if emitter is not None:
        _default_emitter = emitter
        return
    if log_path is not None:
        file_emitter = FileEmitter(log_path)
        if silent is False:
            _default_emitter = CompositeEmitter([StdoutEmitter(silent=False), file_emitter])
        else:
            _default_emitter = file_emitter
        return
    _default_emitter = StdoutEmitter(silent=True if silent is None else silent)


def get_emitter() -> BaseEmitter:
    """Return the configured global emitter."""
    return _default_emitter


def get_cohort_export_config() -> tuple[bool, Path | None]:
    """Return whether cohort export is enabled and its local output path."""
    return _cohort_export_enabled, _cohort_path
