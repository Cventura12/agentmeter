from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from agentmeter import analytics as analytics_module
from agentmeter.analytics import Analytics, TraceStore
from agentmeter.core import ExecutionTrace, Span


def _span(
    *,
    execution_id: str,
    capability: str = "generation.summary",
    outcome: str = "success",
    error_message: str | None = None,
    metadata: dict[str, str] | None = None,
    ended_at: datetime | None = None,
    cost_usd: float | None = 0.001,
) -> Span:
    started_at = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    return Span(
        span_id=str(uuid4()),
        parent_span_id=None,
        execution_id=execution_id,
        caller_agent_id="planner",
        callee_agent_id="worker",
        capability=capability,
        started_at=started_at,
        ended_at=ended_at,
        cost_usd=cost_usd,
        outcome=outcome,  # type: ignore[arg-type]
        error_message=error_message,
        metadata=metadata or {},
    )


class AnalyticsInternalTests(unittest.TestCase):
    def test_helpers_and_percentiles(self) -> None:
        self.assertIsNone(analytics_module._parse_datetime("bad-date"))
        naive = datetime(2026, 1, 1, 9, 0)
        parsed = analytics_module._parse_datetime(naive)
        self.assertIsNotNone(parsed)
        if parsed is not None:
            self.assertEqual(parsed.utcoffset(), timedelta(0))

        self.assertEqual(analytics_module._as_str("  a "), "a")
        self.assertIsNone(analytics_module._as_str(7))
        self.assertEqual(analytics_module._as_float("3.5"), 3.5)
        self.assertIsNone(analytics_module._as_float(True))
        self.assertEqual(analytics_module._as_int(3.0), 3)
        self.assertIsNone(analytics_module._as_int(3.2))

        self.assertEqual(analytics_module._normalize_span_outcome("success"), "success")
        self.assertEqual(analytics_module._normalize_span_outcome("bad"), "unknown")
        self.assertEqual(analytics_module._normalize_trace_outcome("partial"), "partial")
        self.assertEqual(analytics_module._normalize_trace_outcome("bad"), "unknown")

        self.assertEqual(analytics_module._percentile([], 50.0), 0.0)
        self.assertEqual(analytics_module._percentile([7.0], 95.0), 7.0)
        self.assertEqual(analytics_module._percentile([1.0, 5.0], 0.0), 1.0)
        self.assertEqual(analytics_module._percentile([1.0, 5.0], 100.0), 5.0)
        self.assertEqual(analytics_module._percentile_key(99.5), "p99_5")

    def test_kind_and_failure_helpers(self) -> None:
        self.assertEqual(analytics_module._span_kind_from_capability("retrieval.semantic").name, "RETRIEVER")
        self.assertEqual(analytics_module._span_kind_from_capability("transformation.format").name, "TOOL")
        self.assertEqual(analytics_module._span_kind_from_capability("verification.schema").name, "GUARDRAIL")
        self.assertEqual(analytics_module._span_kind_from_capability("planning.multi_step").name, "AGENT")
        self.assertEqual(analytics_module._span_kind_from_capability("classification.intent").name, "CHAIN")
        self.assertEqual(analytics_module._span_kind_from_capability("generation.summary").name, "LLM")
        self.assertEqual(analytics_module._span_kind_from_capability("unknown").name, "AGENT")

        execution_id = str(uuid4())
        timeout_span = _span(execution_id=execution_id, outcome="timeout")
        failed_span = _span(execution_id=execution_id, outcome="failure", error_message="tool_failure: network")
        unknown_span = _span(execution_id=execution_id, outcome="unknown", metadata={"span_kind": "custom_kind"})
        failure_type_span = _span(
            execution_id=execution_id,
            outcome="failure",
            metadata={"error_type": "schema_violation"},
        )

        self.assertEqual(analytics_module._failure_type(timeout_span), "timeout")
        self.assertEqual(analytics_module._failure_type(failed_span), "tool_failure")
        self.assertEqual(analytics_module._failure_type(unknown_span), "unknown")
        self.assertEqual(analytics_module._failure_type(failure_type_span), "schema_violation")
        self.assertEqual(analytics_module._span_kind_name(unknown_span), "CUSTOM_KIND")

    def test_span_parsers_and_store_paths(self) -> None:
        execution_id = str(uuid4())
        payload = {
            "span_id": str(uuid4()),
            "execution_id": execution_id,
            "caller_agent_id": "planner",
            "callee_agent_id": "tool",
            "capability": "extraction.invoice",
            "started_at": "2026-02-01T12:00:00Z",
            "latency_ms": 250.0,
            "outcome": "failure",
            "metadata": {"error_type": "tool_failure", "span_kind": "TOOL"},
        }
        parsed_span = analytics_module._span_from_object(payload)
        self.assertIsNotNone(parsed_span)
        if parsed_span is not None:
            self.assertEqual(parsed_span.metadata.get("error_type"), "tool_failure")
            self.assertIsNotNone(parsed_span.ended_at)

        self.assertIsNone(analytics_module._span_from_object({}))
        self.assertIsNone(analytics_module._span_from_event({"event": "span"}))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write("not-json\n")
                handle.write(json.dumps({"event": "span", "execution_id": "not-a-uuid"}))
                handle.write("\n")
                handle.write(json.dumps(payload))
                handle.write("\n")
                handle.write(
                    json.dumps(
                        {
                            "event": "trace",
                            "execution_id": execution_id,
                            "root_agent_id": "planner",
                            "outcome": "partial",
                            "started_at": "2026-02-01T12:00:00Z",
                            "ended_at": "2026-02-01T12:00:01Z",
                        }
                    )
                )
                handle.write("\n")
            store = TraceStore(path)
            traces = store.load()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].execution_id, execution_id)

    def test_expensive_and_slowest_agents_paths(self) -> None:
        execution_id = str(uuid4())
        started = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        span_fast = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="agent_a",
            capability="generation.summary",
            started_at=started,
            ended_at=started + timedelta(milliseconds=120),
            cost_usd=0.002,
            outcome="success",
            metadata={"span_kind": "LLM"},
        )
        span_slow = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="agent_a",
            capability="generation.summary",
            started_at=started + timedelta(milliseconds=200),
            ended_at=started + timedelta(milliseconds=1600),
            cost_usd=0.01,
            outcome="success",
            metadata={"span_kind": "LLM"},
        )
        span_no_latency = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="agent_b",
            capability="transformation.format",
            started_at=started,
            ended_at=None,
            cost_usd=0.0,
            outcome="unknown",
            metadata={"span_kind": "TOOL"},
        )
        trace = ExecutionTrace(
            execution_id=execution_id,
            root_agent_id="planner",
            started_at=started,
            ended_at=started + timedelta(seconds=2),
            spans=[span_fast, span_slow, span_no_latency],
            outcome="partial",
        )
        analytics = Analytics([trace])

        self.assertEqual(analytics.expensive_span_kinds(top_n=0), [])
        self.assertEqual(analytics.slowest_agents(top_n=0), [])

        expensive = analytics.expensive_span_kinds(top_n=2)
        self.assertEqual(expensive[0]["span_kind"], "LLM")

        slowest = analytics.slowest_agents(top_n=2)
        self.assertEqual(slowest[0]["agent_id"], "agent_a")


if __name__ == "__main__":
    unittest.main()
