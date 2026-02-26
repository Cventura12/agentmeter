'''
End-to-end integration test for agentmeter.
Read this test first. It demonstrates the complete instrumentation workflow.

Scenario: A planning agent orchestrates a 4-step document processing workflow:
  1. Retrieve relevant documents  (RETRIEVER)
  2. Extract structured data      (LLM)
  3. Validate output              (GUARDRAIL)  ← this one fails
  4. Generate summary report      (LLM)

The GUARDRAIL step fails with error_type="schema_violation".
The trace finishes with outcome="partial".
'''

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentmeter import SpanKind, Tracer, configure
from agentmeter.analytics import Analytics, TraceStore
from agentmeter.cohort import CohortExporter


class EndToEndTests(unittest.TestCase):
    def test_canonical_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            configure(log_path=tmp_path / "run.jsonl")
            tracer = Tracer("planner-agent-01", SpanKind.AGENT)

            with tracer.span("doc-retriever-01", SpanKind.RETRIEVER) as span:
                span.set_cost(0.0002)
                span.set_outcome("success")

            with tracer.span("invoice-extractor-01", SpanKind.LLM) as span:
                tracer.record_tokens(input=850, output=220)
                span.set_cost(0.004)
                span.set_outcome("success")

            with tracer.span("schema-guardrail-01", SpanKind.GUARDRAIL) as span:
                span.set_cost(0.0)
                span.set_metadata("error_type", "schema_violation")
                span.error_message = "required field 'invoice_date' missing"
                span.set_outcome("failure")

            with tracer.span("report-generator-01", SpanKind.LLM) as span:
                tracer.record_tokens(input=400, output=800)
                span.set_cost(0.006)
                span.set_outcome("success")

            trace = tracer.finish(outcome="partial")

            assert trace.total_cost_usd == 0.0102  # Validates economic accounting across all spans.
            assert trace.outcome == "partial"  # Confirms mixed success/failure roll up to partial trace outcome.
            assert len(trace.spans) == 4  # Ensures the full 4-step orchestrated workflow was recorded.
            assert trace.span_kind_breakdown == {"RETRIEVER": 1, "LLM": 2, "GUARDRAIL": 1}  # Verifies span type classification is accurate for analytics.

            store = TraceStore(tmp_path / "run.jsonl")
            analytics = Analytics(store)
            assert analytics.cost_by_span_kind()["LLM"] == 0.01  # Confirms LLM cost aggregation powers per-kind FinOps reporting.
            assert analytics.success_rate(span_kind=SpanKind.LLM) == 1.0  # Verifies successful LLM reliability metrics for repeated calls.
            assert analytics.success_rate(span_kind=SpanKind.GUARDRAIL) == 0.0  # Confirms failed guardrails are visible as hard reliability regressions.
            assert analytics.failure_breakdown() == {"schema_violation": 1}  # Ensures failure taxonomy comes from structured metadata, not brittle parsing.

            metrics = CohortExporter.from_trace(trace)
            for metric in metrics:
                valid, violations = CohortExporter.validate_privacy(metric)
                assert valid, f"Privacy violation: {violations}"  # Guarantees anonymized export remains structurally privacy-safe.

            for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip().splitlines():
                json.loads(line)  # Confirms every emitted line is valid JSON for downstream ingestion.


if __name__ == "__main__":
    unittest.main()
