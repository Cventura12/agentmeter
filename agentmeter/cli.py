"""Command-line interface for local agentmeter analytics and log utilities."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analytics import Analytics, TraceStore
from .core import ExecutionTrace
from .tracer import SpanKind

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _supports_ansi() -> bool:
    """Return whether ANSI colors should be emitted."""
    if "NO_COLOR" in os.environ:
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return False
    return True


def _colorize(text: str, outcome: str, enabled: bool) -> str:
    """Apply ANSI color by outcome when enabled."""
    if not enabled:
        return text
    normalized = outcome.lower()
    if normalized == "success":
        return f"{_GREEN}{text}{_RESET}"
    if normalized in {"failure", "timeout"}:
        return f"{_RED}{text}{_RESET}"
    return f"{_YELLOW}{text}{_RESET}"


def _parse_since_value(value: str) -> timedelta:
    """Parse CLI since values like 1h, 24h, 7d, 30d."""
    mapping: dict[str, timedelta] = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError("since must be one of: 1h, 24h, 7d, 30d")
    return mapping[value]


def _utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def _to_utc_datetime(value: object) -> datetime | None:
    """Parse datetime-ish values into UTC."""
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


def _span_kind_from_capability(capability: str) -> str:
    """Infer span kind name from capability category."""
    prefix = capability.split(".", 1)[0].lower()
    if prefix == "retrieval":
        return SpanKind.RETRIEVER.name
    if prefix == "transformation":
        return SpanKind.TOOL.name
    if prefix == "verification":
        return SpanKind.GUARDRAIL.name
    if prefix == "planning":
        return SpanKind.AGENT.name
    if prefix == "classification":
        return SpanKind.CHAIN.name
    if prefix == "generation":
        return SpanKind.LLM.name
    if prefix == "extraction":
        return SpanKind.TOOL.name
    return SpanKind.AGENT.name


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve path argument to absolute path."""
    return Path(path_value).expanduser().resolve()


def _resolve_tail_file(path_value: str | Path) -> Path:
    """Resolve tail source path to a concrete JSONL file."""
    path = _resolve_path(path_value)
    if path.is_file():
        return path
    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
        if jsonl_files:
            return jsonl_files[0]
        return path / "run.jsonl"
    if path.suffix.lower() == ".jsonl":
        return path
    return path / "run.jsonl"


def _format_duration_ms(value: float | None) -> str:
    """Format latency in milliseconds."""
    if value is None:
        return "0ms"
    return f"{value:,.0f}ms"


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert numbers to float with fallback."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _stats_output(since_label: str, traces: list[ExecutionTrace]) -> int:
    """Print stats report and return exit code."""
    analytics = Analytics(traces)
    execution_count = len(traces)
    success_count = sum(1 for trace in traces if getattr(trace, "outcome", "") == "success")
    success_rate_pct = (success_count / execution_count * 100.0) if execution_count else 0.0
    total_cost = sum(getattr(trace, "total_cost_usd", 0.0) for trace in traces)
    avg_cost = (total_cost / execution_count) if execution_count else 0.0

    divider = "=" * 37
    print(divider)
    print(f"AGENTMETER  |  {since_label}")
    print(divider)
    print(f"Executions:   {execution_count:<6} Success: {success_rate_pct:>5.1f}%")
    print(f"Total cost:   ${total_cost:>5.2f}  Avg/exec: ${avg_cost:.3f}")
    print(divider)
    print()

    print("COST BY SPAN KIND")
    cost_map = analytics.cost_by_span_kind()
    dist_map = analytics.span_kind_distribution()
    if not cost_map:
        print("none         $0.00   ( 0%)    0 spans")
    else:
        for kind, cost in sorted(cost_map.items(), key=lambda item: item[1], reverse=True):
            count = dist_map.get(kind, 0)
            pct = (cost / total_cost * 100.0) if total_cost > 0 else 0.0
            print(f"{kind:<11} ${cost:>5.2f}   ({pct:>2.0f}%) {count:>4} spans")
    print()

    print("LATENCY (p50 / p95)")
    if not dist_map:
        print("none         0ms / 0ms")
    else:
        for kind in sorted(dist_map.keys()):
            span_kind_enum = SpanKind[kind] if kind in SpanKind.__members__ else None
            if span_kind_enum is None:
                continue
            metrics = analytics.latency_percentiles(span_kind=span_kind_enum, percentiles=[50.0, 95.0])
            p50 = metrics.get("p50", 0.0)
            p95 = metrics.get("p95", 0.0)
            print(f"{kind:<11} {_format_duration_ms(p50):>6} / {_format_duration_ms(p95):>7}")
    print()

    print("FAILURE BREAKDOWN")
    failure_map = analytics.failure_breakdown()
    if not failure_map:
        print("none             0")
    else:
        for error_type, count in sorted(failure_map.items(), key=lambda item: (-item[1], item[0])):
            print(f"{error_type:<14} {count:>4}")
    print()

    print("AGENT LEADERBOARD")
    leaderboard = analytics.agent_leaderboard()[:10]
    if not leaderboard:
        print("none")
    else:
        for index, row in enumerate(leaderboard, start=1):
            agent_id = str(row["agent_id"])
            span_kind_label = str(row["span_kind"])
            success_pct = float(row["success_rate"]) * 100.0
            avg_cost_usd = float(row["avg_cost_usd"])
            print(f"#{index:<2} {agent_id:<18} {span_kind_label:<10} {success_pct:>3.0f}%  ${avg_cost_usd:.4f}")
    print(divider)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Run the stats subcommand."""
    store = TraceStore(_resolve_path(args.path))
    if args.since is None:
        traces = store.load()
        since_label = "all time"
    else:
        delta = _parse_since_value(args.since)
        since_dt = _utc_now() - delta
        traces = store.load_since(since_dt)
        since_label = f"last {args.since}"
    return _stats_output(since_label, traces)


def _extract_tail_fields(payload: dict[str, object]) -> tuple[str, str, str, float, float, str]:
    """Build printable tail tuple from event payload."""
    timestamp = _to_utc_datetime(payload.get("ts")) or _to_utc_datetime(payload.get("ended_at")) or _utc_now()
    time_label = timestamp.strftime("%H:%M:%S")
    capability = str(payload.get("capability", "unknown"))

    span_kind_raw = payload.get("span_kind")
    if isinstance(span_kind_raw, str) and span_kind_raw.strip():
        parsed_kind = SpanKind.from_value(span_kind_raw)
        span_kind = parsed_kind.name if parsed_kind is not None else span_kind_raw.strip().upper()
    else:
        span_kind = _span_kind_from_capability(capability)

    callee = str(payload.get("callee_agent_id", "unknown-agent"))
    latency_ms = _safe_float(payload.get("latency_ms"), 0.0)
    cost_usd = _safe_float(payload.get("cost_usd"), 0.0)
    outcome = str(payload.get("outcome", "unknown")).upper()
    return time_label, span_kind, callee, latency_ms, cost_usd, outcome


def _iter_new_lines(path: Path) -> Iterable[str]:
    """Yield newly appended lines from a file, polling every 500ms."""
    offset = 0
    started = False
    wait_announced = False

    while True:
        if not path.exists():
            if not wait_announced:
                print(f"Waiting for log file: {path}")
                wait_announced = True
            time.sleep(0.5)
            continue

        wait_announced = False
        size = path.stat().st_size
        if not started:
            offset = size
            started = True
        elif size < offset:
            offset = 0

        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                offset = handle.tell()
                yield line
        time.sleep(0.5)


def _cmd_tail(args: argparse.Namespace) -> int:
    """Run the tail subcommand."""
    path = _resolve_tail_file(args.path)
    color_enabled = _supports_ansi()
    print(f"Tailing {path} (Ctrl+C to stop)")

    try:
        for raw_line in _iter_new_lines(path):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event") != "span" and "span_id" not in payload:
                continue

            time_label, span_kind, callee, latency_ms, cost_usd, outcome = _extract_tail_fields(payload)
            rendered = (
                f"[{time_label}] {span_kind:<10} {callee:<18} "
                f"{latency_ms:>5.0f}ms  ${cost_usd:>0.4f}  {outcome}"
            )
            print(_colorize(rendered, outcome, color_enabled))
    except KeyboardInterrupt:
        print()
        return 0

    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Run the export subcommand."""
    traces = TraceStore(_resolve_path(args.path)).load()
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        with out_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(trace.model_dump_json())
                handle.write("\n")
        print(f"Exported {len(traces)} traces to {out_path}")
        return 0

    fieldnames = [
        "execution_id",
        "span_id",
        "parent_span_id",
        "caller_agent_id",
        "callee_agent_id",
        "capability",
        "started_at",
        "ended_at",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "outcome",
        "error_message",
        "metadata",
    ]
    span_count = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            for span in trace.spans:
                writer.writerow(
                    {
                        "execution_id": span.execution_id,
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id or "",
                        "caller_agent_id": span.caller_agent_id,
                        "callee_agent_id": span.callee_agent_id,
                        "capability": span.capability,
                        "started_at": span.started_at.isoformat().replace("+00:00", "Z"),
                        "ended_at": (span.ended_at.isoformat().replace("+00:00", "Z") if span.ended_at else ""),
                        "latency_ms": f"{span.latency_ms:.3f}" if span.latency_ms is not None else "",
                        "input_tokens": span.input_tokens if span.input_tokens is not None else "",
                        "output_tokens": span.output_tokens if span.output_tokens is not None else "",
                        "cost_usd": f"{span.cost_usd:.6f}" if span.cost_usd is not None else "",
                        "outcome": span.outcome,
                        "error_message": span.error_message or "",
                        "metadata": json.dumps(span.metadata, separators=(",", ":"), ensure_ascii=True),
                    }
                )
                span_count += 1
    print(f"Exported {span_count} spans to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="agentmeter",
        description="Local analytics and log utilities for agentmeter traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show aggregated execution metrics.",
        description="Render local trace statistics grouped by span kind and agent performance.",
        epilog="Example: agentmeter stats --path ./agentmeter_logs --since 24h",
    )
    stats_parser.add_argument(
        "--path",
        default="./agentmeter_logs",
        help="Trace source path (JSONL file or directory). Default: ./agentmeter_logs",
    )
    stats_parser.add_argument(
        "--since",
        default="24h",
        choices=["1h", "24h", "7d", "30d"],
        help="Time filter window. Choices: 1h, 24h, 7d, 30d. Default: 24h",
    )
    stats_parser.set_defaults(handler=_cmd_stats)

    tail_parser = subparsers.add_parser(
        "tail",
        help="Follow new span events as they are written.",
        description="Poll a JSONL span log every 500ms and print one line per new span.",
        epilog="Example: agentmeter tail --path ./agentmeter_logs/run.jsonl",
    )
    tail_parser.add_argument(
        "--path",
        default="./agentmeter_logs",
        help="Path to a JSONL file or directory containing JSONL logs.",
    )
    tail_parser.set_defaults(handler=_cmd_tail)

    export_parser = subparsers.add_parser(
        "export",
        help="Export trace data for offline analysis.",
        description="Export loaded traces as CSV (one row per span) or JSONL (one trace per line).",
        epilog="Example: agentmeter export --path ./agentmeter_logs --format csv --out report.csv",
    )
    export_parser.add_argument(
        "--path",
        default="./agentmeter_logs",
        help="Trace source path (JSONL file or directory). Default: ./agentmeter_logs",
    )
    export_parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Export format: csv or jsonl. Default: csv",
    )
    export_parser.add_argument(
        "--out",
        default=None,
        help="Output file path. Default: ./report.csv or ./report.jsonl",
    )
    export_parser.set_defaults(handler=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export" and args.out is None:
        args.out = "./report.jsonl" if args.format == "jsonl" else "./report.csv"
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
