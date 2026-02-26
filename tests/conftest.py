"""Shared synthetic trace generation for analytics tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random
from uuid import uuid4

from agentmeter.core import ExecutionTrace, Span
from agentmeter.tracer import SpanKind

_RNG = Random(42)

_SPAN_KIND_RANGES: dict[SpanKind, tuple[tuple[float, float], tuple[float, float]]] = {
    SpanKind.LLM: ((0.0006, 0.02), (450.0, 4200.0)),
    SpanKind.CHAIN: ((0.0001, 0.0022), (120.0, 900.0)),
    SpanKind.RETRIEVER: ((0.00004, 0.001), (40.0, 380.0)),
    SpanKind.TOOL: ((0.0002, 0.012), (140.0, 2600.0)),
    SpanKind.AGENT: ((0.0003, 0.006), (220.0, 1700.0)),
    SpanKind.EMBEDDING: ((0.00002, 0.0005), (20.0, 190.0)),
    SpanKind.RERANKER: ((0.00005, 0.0015), (50.0, 580.0)),
    SpanKind.GUARDRAIL: ((0.00003, 0.001), (35.0, 420.0)),
    SpanKind.EVALUATOR: ((0.0004, 0.009), (260.0, 3100.0)),
}

_SPAN_KIND_CAPABILITY: dict[SpanKind, str] = {
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

_FAILURE_MESSAGES = [
    "tool_failure: provider_500",
    "hallucination: unsupported_claim",
    "schema_error: invalid_payload",
    "rate_limit: retry_exhausted",
]


def _trace_outcome(spans: list[Span]) -> str:
    """Infer trace outcome from span outcomes."""
    success_count = sum(1 for span in spans if span.outcome == "success")
    failure_like_count = sum(1 for span in spans if span.outcome in {"failure", "timeout"})
    if success_count == len(spans):
        return "success"
    if success_count == 0 and failure_like_count > 0:
        return "failure"
    if success_count > 0 and failure_like_count > 0:
        return "partial"
    return "unknown"


def generate_synthetic_traces(count: int = 500) -> list[ExecutionTrace]:
    """Generate deterministic synthetic traces covering all span kinds."""
    kinds = list(SpanKind)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    traces: list[ExecutionTrace] = []

    for trace_index in range(count):
        execution_id = str(uuid4())
        root_agent_id = f"planner_{trace_index % 16}"
        trace_started_at = base_time + timedelta(minutes=trace_index * 30)
        span_count = 2 + (trace_index % 6)
        spans: list[Span] = []
        cursor = trace_started_at

        for span_index in range(span_count):
            if span_index == 0:
                kind = kinds[trace_index % len(kinds)]
            else:
                kind = kinds[_RNG.randrange(len(kinds))]

            cost_range, latency_range = _SPAN_KIND_RANGES[kind]
            cost_usd = round(_RNG.uniform(cost_range[0], cost_range[1]), 6)
            latency_ms = _RNG.uniform(latency_range[0], latency_range[1])
            started_at = cursor + timedelta(milliseconds=_RNG.uniform(2.0, 30.0))
            ended_at = started_at + timedelta(milliseconds=latency_ms)

            outcome_roll = _RNG.random()
            if outcome_roll < 0.82:
                outcome = "success"
                error_message = None
            elif outcome_roll < 0.93:
                outcome = "failure"
                error_message = _FAILURE_MESSAGES[_RNG.randrange(len(_FAILURE_MESSAGES))]
            elif outcome_roll < 0.97:
                outcome = "timeout"
                error_message = "timeout: upstream_service"
            else:
                outcome = "unknown"
                error_message = "unknown: missing_signal"

            span = Span(
                span_id=str(uuid4()),
                parent_span_id=spans[-1].span_id if spans and _RNG.random() < 0.25 else None,
                execution_id=execution_id,
                caller_agent_id=root_agent_id if span_index == 0 else spans[-1].callee_agent_id,
                callee_agent_id=f"{kind.name.lower()}_agent_{_RNG.randint(1, 14)}",
                capability=_SPAN_KIND_CAPABILITY[kind],
                started_at=started_at,
                ended_at=ended_at,
                input_tokens=_RNG.randint(40, 1800),
                output_tokens=_RNG.randint(20, 900),
                cost_usd=cost_usd,
                outcome=outcome,
                error_message=error_message,
                metadata={
                    "span_kind": kind.name,
                    "region": ["us-east", "us-west", "eu-central"][_RNG.randrange(3)],
                },
            )
            spans.append(span)
            cursor = ended_at + timedelta(milliseconds=_RNG.uniform(1.0, 16.0))

        trace = ExecutionTrace(
            execution_id=execution_id,
            root_agent_id=root_agent_id,
            started_at=trace_started_at,
            ended_at=spans[-1].ended_at,
            spans=spans,
            outcome=_trace_outcome(spans),
        )
        traces.append(trace)

    return traces

