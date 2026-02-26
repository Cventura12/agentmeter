import asyncio
import unittest
from uuid import UUID

from agentmeter.exceptions import TraceNotActiveError
from agentmeter.taxonomy import Capability
from agentmeter.tracer import AsyncTracer, Tracer


class TracerTests(unittest.TestCase):
    def test_normal_execution_records_cost(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.PLANNING_MULTI_STEP)
        with tracer.span("ocr_22", Capability.EXTRACTION_INVOICE) as span:
            span.set_cost(0.002)
            span.set_outcome("success")
        trace = tracer.finish()
        self.assertEqual(len(trace.spans), 1)
        self.assertAlmostEqual(trace.total_cost_usd, 0.002)
        self.assertEqual(trace.spans[0].outcome, "success")

    def test_span_auto_timing_and_latency(self) -> None:
        tracer = Tracer(agent_id="agent_a", capability="planning.multi_step")
        with tracer.span("worker_b", "extraction.form"):
            pass
        trace = tracer.finish()
        span = trace.spans[0]
        self.assertIsNotNone(span.started_at)
        self.assertIsNotNone(span.ended_at)
        self.assertIsNotNone(span.latency_ms)
        self.assertGreaterEqual(span.latency_ms or 0.0, 0.0)

    def test_exception_sets_failure_and_reraises(self) -> None:
        tracer = Tracer(agent_id="agent_a", capability=Capability.PLANNING_ROUTING)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with tracer.span("worker_b", Capability.EXTRACTION_RECEIPT):
                raise RuntimeError("boom")
        trace = tracer.finish(outcome="failure")
        self.assertEqual(len(trace.spans), 1)
        self.assertEqual(trace.spans[0].outcome, "failure")
        self.assertEqual(trace.spans[0].error_message, "boom")

    def test_nested_spans_have_parent_ids(self) -> None:
        tracer = Tracer(agent_id="root", capability=Capability.PLANNING_MULTI_STEP)
        with tracer.span("outer_agent", Capability.PLANNING_DECOMPOSITION) as outer:
            with tracer.span("inner_agent", Capability.EXTRACTION_CONTRACT):
                pass
        trace = tracer.finish()
        self.assertEqual(len(trace.spans), 2)
        by_callee = {span.callee_agent_id: span for span in trace.spans}
        self.assertEqual(by_callee["inner_agent"].parent_span_id, outer.span_id)

    def test_record_tokens_sets_current_span(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.GENERATION_CODE)
        with tracer.span("summarizer", Capability.GENERATION_SUMMARY):
            tracer.record_tokens(input=123, output=45)
        trace = tracer.finish()
        span = trace.spans[0]
        self.assertEqual(span.input_tokens, 123)
        self.assertEqual(span.output_tokens, 45)

    def test_record_tokens_without_active_span_raises(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.GENERATION_REPORT)
        with self.assertRaises(TraceNotActiveError):
            tracer.record_tokens(input=1, output=1)

    def test_get_current_execution_id_lifecycle(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.RETRIEVAL_SEMANTIC)
        execution_id = tracer.get_current_execution_id()
        self.assertIsNotNone(execution_id)
        if execution_id is not None:
            UUID(execution_id)
        tracer.finish()
        self.assertIsNone(tracer.get_current_execution_id())

    def test_finish_with_invalid_outcome_raises(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.RETRIEVAL_STRUCTURED)
        with self.assertRaises(ValueError):
            tracer.finish(outcome="bad")  # type: ignore[arg-type]

    def test_set_outcome_with_invalid_value_raises(self) -> None:
        tracer = Tracer(agent_id="my_agent", capability=Capability.TRANSFORMATION_FORMAT)
        with tracer.span("x", Capability.TRANSFORMATION_TRANSLATE) as span:
            with self.assertRaises(ValueError):
                span.set_outcome("not-valid")  # type: ignore[arg-type]
        trace = tracer.finish()
        self.assertEqual(trace.spans[0].outcome, "unknown")

    def test_span_capability_accepts_string_and_enum(self) -> None:
        tracer = Tracer(agent_id="agent_a", capability="planning.multi_step")
        with tracer.span("ocr", "extraction.invoice"):
            pass
        with tracer.span("verify", Capability.VERIFICATION_SCHEMA):
            pass
        trace = tracer.finish()
        capabilities = {span.capability for span in trace.spans}
        self.assertIn("extraction.invoice", capabilities)
        self.assertIn("verification.schema", capabilities)

    def test_finish_while_span_active_raises(self) -> None:
        tracer = Tracer(agent_id="agent_a", capability=Capability.CLASSIFICATION_DOCUMENT)
        span_cm = tracer.span("worker", Capability.EXTRACTION_FORM)
        span_cm.__enter__()
        try:
            with self.assertRaises(TraceNotActiveError):
                tracer.finish()
        finally:
            span_cm.__exit__(None, None, None)
        trace = tracer.finish()
        self.assertEqual(len(trace.spans), 1)


class AsyncTracerTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_normal_execution(self) -> None:
        tracer = AsyncTracer(agent_id="async_agent", capability=Capability.PLANNING_MULTI_STEP)
        async with tracer.span("ocr", Capability.EXTRACTION_INVOICE) as span:
            tracer.record_tokens(input=10, output=4)
            span.set_cost(0.0012)
        trace = tracer.finish()
        self.assertEqual(len(trace.spans), 1)
        self.assertAlmostEqual(trace.total_cost_usd, 0.0012)
        self.assertEqual(trace.spans[0].input_tokens, 10)
        self.assertEqual(trace.spans[0].output_tokens, 4)

    async def test_async_exception_handling(self) -> None:
        tracer = AsyncTracer(agent_id="async_agent", capability=Capability.PLANNING_ROUTING)
        with self.assertRaisesRegex(ValueError, "fail"):
            async with tracer.span("worker", Capability.VERIFICATION_FACTCHECK):
                raise ValueError("fail")
        trace = tracer.finish(outcome="failure")
        self.assertEqual(trace.spans[0].outcome, "failure")
        self.assertEqual(trace.spans[0].error_message, "fail")

    async def test_async_context_isolation_between_tasks(self) -> None:
        tracer = AsyncTracer(agent_id="async_agent", capability=Capability.PLANNING_MULTI_STEP)

        async def worker(callee: str, in_tokens: int, out_tokens: int) -> None:
            async with tracer.span(callee, Capability.EXTRACTION_RECEIPT):
                tracer.record_tokens(input=in_tokens, output=out_tokens)
                await asyncio.sleep(0.01)

        await asyncio.gather(
            worker("worker_a", 11, 3),
            worker("worker_b", 22, 6),
        )
        trace = tracer.finish()
        self.assertEqual(len(trace.spans), 2)
        token_pairs = {(span.input_tokens, span.output_tokens) for span in trace.spans}
        self.assertEqual(token_pairs, {(11, 3), (22, 6)})


if __name__ == "__main__":
    unittest.main()
