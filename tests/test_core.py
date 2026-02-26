import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentmeter.core import ExecutionTrace, Span


class CoreTests(unittest.TestCase):
    def test_execution_trace_aggregates_values(self) -> None:
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        execution_id = str(uuid4())
        span_one = Span(
            span_id=str(uuid4()),
            execution_id=execution_id,
            caller_agent_id="root_agent",
            callee_agent_id="worker_a",
            capability="generation.summary",
            started_at=started,
            ended_at=started + timedelta(milliseconds=120),
            cost_usd=0.02,
            outcome="success",
            metadata={"span_kind": "LLM"},
        )
        span_two = Span(
            span_id=str(uuid4()),
            execution_id=execution_id,
            caller_agent_id="root_agent",
            callee_agent_id="worker_b",
            capability="extraction.invoice",
            started_at=started + timedelta(milliseconds=130),
            ended_at=started + timedelta(milliseconds=210),
            cost_usd=0.03,
            outcome="failure",
            error_message="tool_failure: network",
            metadata={"span_kind": "TOOL"},
        )
        trace = ExecutionTrace(
            execution_id=execution_id,
            root_agent_id="root_agent",
            started_at=started,
            ended_at=started + timedelta(milliseconds=210),
            spans=[span_one, span_two],
            outcome="partial",
        )

        self.assertAlmostEqual(trace.total_cost_usd, 0.05)
        self.assertAlmostEqual(trace.total_latency_ms or 0.0, 210.0)

    def test_to_event_dict_contains_core_fields(self) -> None:
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trace = ExecutionTrace(
            execution_id=str(uuid4()),
            root_agent_id="root_agent",
            started_at=started,
            ended_at=started + timedelta(milliseconds=10),
            spans=[],
            outcome="unknown",
        )

        event = trace.to_event_dict()
        self.assertEqual(event["root_agent_id"], "root_agent")
        self.assertEqual(event["span_count"], 0)
        self.assertEqual(event["outcome"], "unknown")


if __name__ == "__main__":
    unittest.main()
