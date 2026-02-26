from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentmeter.core import AgentCall, AgentIdentity, ExecutionTrace, Span
from agentmeter.taxonomy import Capability


class CoreExtraTests(unittest.TestCase):
    def test_agent_identity_validation_paths(self) -> None:
        identity = AgentIdentity(
            capability="extraction.invoice",
            version="1.2.3",
            tags=["finance"],
        )
        event = identity.to_event_dict()
        self.assertEqual(event["capability"], "extraction.invoice")
        self.assertEqual(event["tags_csv"], "finance")

        with self.assertRaises(ValueError):
            AgentIdentity(agent_id="not-uuid", capability="extraction.invoice", version="1.2.3")
        with self.assertRaises(ValueError):
            AgentIdentity(capability="bad.capability", version="1.2.3")
        with self.assertRaises(ValueError):
            AgentIdentity(capability="extraction.invoice", version="1.2")

    def test_span_event_dict_and_validation_paths(self) -> None:
        started_at = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
        span = Span(
            span_id=str(uuid4()),
            execution_id=str(uuid4()),
            caller_agent_id="planner",
            callee_agent_id="ocr",
            capability="extraction.invoice",
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=300),
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.002,
            outcome="success",
            metadata={"error.type": "schema_violation", "x-y": "z"},
        )
        event = span.to_event_dict()
        self.assertEqual(event["metadata_error_type"], "schema_violation")
        self.assertEqual(event["metadata_x_y"], "z")
        self.assertEqual(event["metadata_count"], 2)

        with self.assertRaises(ValueError):
            Span(
                span_id=str(uuid4()),
                execution_id=str(uuid4()),
                caller_agent_id="planner",
                callee_agent_id="ocr",
                capability="extraction.invoice",
                started_at=started_at,
                ended_at=started_at - timedelta(milliseconds=1),
                outcome="success",
            )
        with self.assertRaises(ValueError):
            Span(
                span_id=str(uuid4()),
                execution_id=str(uuid4()),
                caller_agent_id="planner",
                callee_agent_id="ocr",
                capability="extraction.invoice",
                started_at=started_at,
                input_tokens=-1,
                outcome="success",
            )
        with self.assertRaises(ValueError):
            Span(
                span_id=str(uuid4()),
                execution_id=str(uuid4()),
                caller_agent_id="planner",
                callee_agent_id="ocr",
                capability="extraction.invoice",
                started_at=started_at,
                cost_usd=-1.0,
                outcome="success",
            )

    def test_agent_call_and_execution_trace_validation(self) -> None:
        started_at = datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc)
        call = AgentCall(
            call_id=str(uuid4()),
            agent_id="planner",
            capability=Capability.PLANNING_ROUTING,
            latency_ms=150.0,
            cost_usd=0.001,
            success=True,
            started_at=started_at,
        )
        self.assertIsNotNone(call.ended_at)

        with self.assertRaises(ValueError):
            AgentCall(
                call_id=str(uuid4()),
                agent_id="planner",
                capability=Capability.PLANNING_ROUTING,
                latency_ms=-1.0,
                cost_usd=0.001,
                success=True,
                started_at=started_at,
            )
        with self.assertRaises(ValueError):
            AgentCall(
                call_id=str(uuid4()),
                agent_id="planner",
                capability=Capability.PLANNING_ROUTING,
                latency_ms=1.0,
                cost_usd=-0.1,
                success=True,
                started_at=started_at,
            )

        execution_id = str(uuid4())
        span_retriever = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="retriever",
            capability="retrieval.semantic",
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=10),
            outcome="success",
            metadata={},
        )
        span_guardrail = Span(
            span_id=str(uuid4()),
            parent_span_id=None,
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="validator",
            capability="verification.schema",
            started_at=started_at + timedelta(milliseconds=20),
            ended_at=started_at + timedelta(milliseconds=40),
            outcome="failure",
            metadata={},
        )
        trace = ExecutionTrace(
            execution_id=execution_id,
            root_agent_id="planner",
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=40),
            spans=[span_retriever, span_guardrail],
            outcome="partial",
        )
        self.assertEqual(trace.span_kind_breakdown["RETRIEVER"], 1)
        self.assertEqual(trace.span_kind_breakdown["GUARDRAIL"], 1)

        with self.assertRaises(ValueError):
            ExecutionTrace(
                execution_id=str(uuid4()),
                root_agent_id="planner",
                started_at=started_at,
                ended_at=started_at - timedelta(seconds=1),
                spans=[],
                outcome="unknown",
            )


if __name__ == "__main__":
    unittest.main()
