import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from agentmeter.analytics import Analytics, TraceStore
from agentmeter.core import ExecutionTrace, Span
from agentmeter.tracer import SpanKind
from tests.conftest import generate_synthetic_traces


class AnalyticsTests(unittest.TestCase):
    def test_empty_data_returns_graceful_defaults(self) -> None:
        analytics = Analytics([])
        self.assertEqual(analytics.cost_by_span_kind(), {})
        self.assertEqual(analytics.latency_percentiles(), {})
        self.assertEqual(analytics.success_rate(), 0.0)
        self.assertEqual(analytics.failure_breakdown(), {})
        self.assertEqual(analytics.agent_leaderboard(), [])
        self.assertEqual(analytics.cost_trend(), [])
        self.assertEqual(analytics.span_kind_distribution(), {})
        self.assertEqual(analytics.expensive_span_kinds(), [])
        self.assertEqual(analytics.slowest_agents(), [])

    def test_single_span_metrics(self) -> None:
        started_at = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        span = Span(
            span_id=str(uuid4()),
            execution_id=str(uuid4()),
            caller_agent_id="planner_agent",
            callee_agent_id="llm_agent",
            capability="generation.summary",
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=1000),
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.25,
            outcome="success",
            metadata={"span_kind": "LLM"},
        )
        trace = ExecutionTrace(
            execution_id=span.execution_id,
            root_agent_id="planner_agent",
            started_at=started_at,
            ended_at=span.ended_at,
            spans=[span],
            outcome="success",
        )
        analytics = Analytics([trace])

        self.assertEqual(analytics.cost_by_span_kind(), {"LLM": 0.25})
        self.assertEqual(analytics.latency_percentiles(), {"p50": 1000.0, "p95": 1000.0, "p99": 1000.0})
        self.assertEqual(analytics.success_rate(), 1.0)
        self.assertEqual(analytics.failure_breakdown(), {})
        self.assertEqual(analytics.span_kind_distribution(), {"LLM": 1})

    def test_all_span_kinds_present_in_leaderboard(self) -> None:
        traces = generate_synthetic_traces(500)
        analytics = Analytics(traces)
        leaderboard = analytics.agent_leaderboard()

        self.assertGreater(len(leaderboard), 0)
        present = {str(row["span_kind"]) for row in leaderboard}
        expected = {kind.name for kind in SpanKind}
        self.assertTrue(expected.issubset(present))

    def test_cost_trend_supports_hour_day_week_buckets(self) -> None:
        traces = generate_synthetic_traces(500)
        analytics = Analytics(traces)
        hourly = analytics.cost_trend(bucket="hour")
        daily = analytics.cost_trend(bucket="day")
        weekly = analytics.cost_trend(bucket="week")

        self.assertGreater(len(hourly), 0)
        self.assertGreater(len(daily), 0)
        self.assertGreater(len(weekly), 0)
        self.assertGreaterEqual(len(hourly), len(daily))
        self.assertGreaterEqual(len(daily), len(weekly))

        self.assertTrue(all("T" in str(row["bucket"]) for row in hourly))
        self.assertTrue(all(len(str(row["bucket"])) == 10 for row in daily))
        self.assertTrue(all(len(str(row["bucket"])) == 10 for row in weekly))

        total_hour = round(sum(float(row["total_cost_usd"]) for row in hourly), 6)
        total_day = round(sum(float(row["total_cost_usd"]) for row in daily), 6)
        total_week = round(sum(float(row["total_cost_usd"]) for row in weekly), 6)
        self.assertEqual(total_hour, total_day)
        self.assertEqual(total_day, total_week)

    def test_trace_store_load_and_load_since(self) -> None:
        traces = generate_synthetic_traces(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace_dump.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(trace.model_dump_json())
                    handle.write("\n")

            store = TraceStore(tmpdir)
            loaded = store.load()
            self.assertEqual(len(loaded), 20)

            midpoint = loaded[10].started_at
            recent = store.load_since(midpoint)
            self.assertGreater(len(recent), 0)
            self.assertTrue(
                all(
                    trace.started_at >= midpoint or (trace.ended_at or trace.started_at) >= midpoint
                    for trace in recent
                )
            )

    def test_trace_store_parses_event_jsonl(self) -> None:
        execution_id = str(uuid4())
        ended = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
        span_event = {
            "event": "span",
            "execution_id": execution_id,
            "span_id": str(uuid4()),
            "caller_agent_id": "planner_agent",
            "callee_agent_id": "tool_agent",
            "capability": "extraction.invoice",
            "span_kind": "TOOL",
            "outcome": "success",
            "cost_usd": 0.003,
            "latency_ms": 800.0,
            "ts": ended.isoformat().replace("+00:00", "Z"),
        }
        trace_event = {
            "event": "trace",
            "execution_id": execution_id,
            "root_agent_id": "planner_agent",
            "outcome": "success",
            "ts": ended.isoformat().replace("+00:00", "Z"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(span_event))
                handle.write("\n")
                handle.write(json.dumps(trace_event))
                handle.write("\n")

            store = TraceStore(path)
            traces = store.load()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].execution_id, execution_id)
            self.assertEqual(len(traces[0].spans), 1)
            self.assertEqual(traces[0].spans[0].metadata["span_kind"], "TOOL")


if __name__ == "__main__":
    unittest.main()
