import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from agentmeter import configure
from agentmeter.cohort import AnonymizedSpanMetric, CohortExporter
from agentmeter.core import ExecutionTrace, Span
from agentmeter.tracer import SpanKind, Tracer


_SPAN_KIND_TO_CAPABILITY: dict[SpanKind, str] = {
    SpanKind.LLM: "generation.summary",
    SpanKind.CHAIN: "classification.intent",
    SpanKind.RETRIEVER: "retrieval.semantic",
    SpanKind.TOOL: "extraction.invoice",
    SpanKind.AGENT: "planning.multi_step",
    SpanKind.EMBEDDING: "retrieval.structured",
    SpanKind.RERANKER: "retrieval.structured",
    SpanKind.GUARDRAIL: "verification.schema",
    SpanKind.EVALUATOR: "verification.factcheck",
}


def _base_metric_kwargs() -> dict[str, object]:
    return {
        "span_kind": SpanKind.LLM.value,
        "outcome": "success",
        "latency_ms_bucket": 100,
        "cost_usd_bucket": 0.0123,
        "input_token_bucket": 200,
        "output_token_bucket": 100,
        "model_family": "gpt-4",
        "orchestrator": "custom",
        "recorded_at_hour": "2026-02-26T14",
        "parent_span_kind": None,
        "depth_in_trace": 0,
    }


def _span_for_kind(kind: SpanKind) -> Span:
    started = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    return Span(
        span_id=str(uuid4()),
        execution_id=str(uuid4()),
        caller_agent_id="root_agent",
        callee_agent_id=f"{kind.value}_agent",
        capability=_SPAN_KIND_TO_CAPABILITY[kind],
        started_at=started,
        ended_at=started + timedelta(milliseconds=1234),
        input_tokens=345,
        output_tokens=678,
        cost_usd=0.012345,
        outcome="success",
        metadata={"span_kind": kind.name, "model": "gpt-4o-mini"},
    )


class CohortTests(unittest.TestCase):
    def test_latency_bucket_rejects_non_multiples_of_50(self) -> None:
        kwargs = _base_metric_kwargs()
        kwargs["latency_ms_bucket"] = 75
        with self.assertRaises(ValueError):
            AnonymizedSpanMetric(**kwargs)

    def test_cost_bucket_rejects_more_than_four_decimals(self) -> None:
        kwargs = _base_metric_kwargs()
        kwargs["cost_usd_bucket"] = 0.00011
        with self.assertRaises(ValueError):
            AnonymizedSpanMetric(**kwargs)

    def test_token_buckets_reject_non_multiples_of_100(self) -> None:
        kwargs = _base_metric_kwargs()
        kwargs["input_token_bucket"] = 150
        with self.assertRaises(ValueError):
            AnonymizedSpanMetric(**kwargs)
        kwargs = _base_metric_kwargs()
        kwargs["output_token_bucket"] = 250
        with self.assertRaises(ValueError):
            AnonymizedSpanMetric(**kwargs)

    def test_recorded_at_hour_rejects_wrong_format(self) -> None:
        kwargs = _base_metric_kwargs()
        kwargs["recorded_at_hour"] = "2026-02-26T14:00"
        with self.assertRaises(ValueError):
            AnonymizedSpanMetric(**kwargs)

    def test_from_span_produces_valid_metric_for_every_span_kind(self) -> None:
        for kind in SpanKind:
            span = _span_for_kind(kind)
            metric = CohortExporter.from_span(span, depth=1, parent_span_kind=SpanKind.AGENT, orchestrator="custom")
            self.assertEqual(metric.span_kind, kind.value)
            valid, violations = CohortExporter.validate_privacy(metric)
            self.assertTrue(valid, msg=f"{kind.name} violations: {violations}")

    def test_validate_privacy_passes_on_correct_bucketed_data(self) -> None:
        metric = AnonymizedSpanMetric(**_base_metric_kwargs())
        valid, violations = CohortExporter.validate_privacy(metric)
        self.assertTrue(valid)
        self.assertEqual(violations, [])

    def test_validate_privacy_catches_each_precision_violation(self) -> None:
        metric = AnonymizedSpanMetric(**_base_metric_kwargs())
        payload = metric.model_dump()

        invalid_latency = AnonymizedSpanMetric.model_construct(**{**payload, "latency_ms_bucket": 125})
        valid, violations = CohortExporter.validate_privacy(invalid_latency)
        self.assertFalse(valid)
        self.assertTrue(any("latency_ms_bucket" in violation for violation in violations))

        invalid_cost = AnonymizedSpanMetric.model_construct(**{**payload, "cost_usd_bucket": 0.00011})
        valid, violations = CohortExporter.validate_privacy(invalid_cost)
        self.assertFalse(valid)
        self.assertTrue(any("cost_usd_bucket" in violation for violation in violations))

        invalid_input = AnonymizedSpanMetric.model_construct(**{**payload, "input_token_bucket": 50})
        valid, violations = CohortExporter.validate_privacy(invalid_input)
        self.assertFalse(valid)
        self.assertTrue(any("input_token_bucket" in violation for violation in violations))

        invalid_output = AnonymizedSpanMetric.model_construct(**{**payload, "output_token_bucket": 350})
        valid, violations = CohortExporter.validate_privacy(invalid_output)
        self.assertFalse(valid)
        self.assertTrue(any("output_token_bucket" in violation for violation in violations))

        invalid_hour = AnonymizedSpanMetric.model_construct(**{**payload, "recorded_at_hour": "2026-02-26"})
        valid, violations = CohortExporter.validate_privacy(invalid_hour)
        self.assertFalse(valid)
        self.assertTrue(any("recorded_at_hour" in violation for violation in violations))

    def test_to_jsonl_is_deterministically_sorted(self) -> None:
        one = AnonymizedSpanMetric(
            **{
                **_base_metric_kwargs(),
                "span_kind": SpanKind.TOOL.value,
                "recorded_at_hour": "2026-01-01T10",
            }
        )
        two = AnonymizedSpanMetric(
            **{
                **_base_metric_kwargs(),
                "span_kind": SpanKind.AGENT.value,
                "recorded_at_hour": "2026-01-01T09",
            }
        )
        metrics = [one, two]
        first = CohortExporter.to_jsonl(metrics)
        second = CohortExporter.to_jsonl(metrics)
        self.assertEqual(first, second)

        lines = first.splitlines()
        decoded = [json.loads(line) for line in lines]
        ordered_pairs = [(item["span_kind"], item["recorded_at_hour"]) for item in decoded]
        self.assertEqual(ordered_pairs, sorted(ordered_pairs))

    def test_depth_in_trace_computed_for_nested_trace(self) -> None:
        execution_id = str(uuid4())
        started = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        root = Span(
            span_id=str(uuid4()),
            execution_id=execution_id,
            caller_agent_id="planner",
            callee_agent_id="root_agent",
            capability="planning.multi_step",
            started_at=started,
            ended_at=started + timedelta(milliseconds=400),
            outcome="success",
            metadata={"span_kind": SpanKind.AGENT.name},
        )
        child = Span(
            span_id=str(uuid4()),
            parent_span_id=root.span_id,
            execution_id=execution_id,
            caller_agent_id="root_agent",
            callee_agent_id="tool_agent",
            capability="extraction.invoice",
            started_at=started + timedelta(milliseconds=410),
            ended_at=started + timedelta(milliseconds=700),
            outcome="success",
            metadata={"span_kind": SpanKind.TOOL.name},
        )
        grandchild = Span(
            span_id=str(uuid4()),
            parent_span_id=child.span_id,
            execution_id=execution_id,
            caller_agent_id="tool_agent",
            callee_agent_id="retriever_agent",
            capability="retrieval.semantic",
            started_at=started + timedelta(milliseconds=710),
            ended_at=started + timedelta(milliseconds=900),
            outcome="success",
            metadata={"span_kind": SpanKind.RETRIEVER.name},
        )
        trace = ExecutionTrace(
            execution_id=execution_id,
            root_agent_id="planner",
            started_at=started,
            ended_at=started + timedelta(milliseconds=900),
            spans=[root, child, grandchild],
            outcome="success",
        )

        metrics = CohortExporter.from_trace(trace, orchestrator="custom")
        by_kind = {metric.span_kind: metric for metric in metrics}
        self.assertEqual(by_kind[SpanKind.AGENT.value].depth_in_trace, 0)
        self.assertIsNone(by_kind[SpanKind.AGENT.value].parent_span_kind)
        self.assertEqual(by_kind[SpanKind.TOOL.value].depth_in_trace, 1)
        self.assertEqual(by_kind[SpanKind.TOOL.value].parent_span_kind, SpanKind.AGENT.value)
        self.assertEqual(by_kind[SpanKind.RETRIEVER.value].depth_in_trace, 2)
        self.assertEqual(by_kind[SpanKind.RETRIEVER.value].parent_span_kind, SpanKind.TOOL.value)

    def test_tracer_export_cohort_metrics(self) -> None:
        tracer = Tracer("planner_agent", SpanKind.AGENT)
        with tracer.span("tool_agent", SpanKind.TOOL) as span:
            span.set_cost(0.0032)
            span.set_outcome("success")
        tracer.finish()

        metrics = tracer.export_cohort_metrics(orchestrator="custom")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].orchestrator, "custom")
        valid, _violations = CohortExporter.validate_privacy(metrics[0])
        self.assertTrue(valid)

    def test_configure_cohort_export_writes_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort_path = Path(tmpdir) / "cohort.jsonl"
            configure(cohort_export=True, cohort_path=cohort_path, silent=True)
            tracer = Tracer("planner_agent", SpanKind.AGENT)
            with tracer.span("tool_agent", SpanKind.TOOL):
                pass
            tracer.finish()
            self.assertTrue(cohort_path.exists())
            lines = [line for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            configure(cohort_export=False, silent=True)


if __name__ == "__main__":
    unittest.main()
