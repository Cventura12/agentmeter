import io
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
from typing import Literal
import unittest
from unittest.mock import patch
from uuid import uuid4

from agentmeter.core import ExecutionTrace, Span
from agentmeter.emitter import BaseEmitter, CompositeEmitter, FileEmitter, StdoutEmitter, configure, get_emitter


SpanOutcome = Literal["success", "failure", "timeout", "unknown"]
TraceOutcome = Literal["success", "failure", "partial", "unknown"]


def _sample_span(cost_usd: float = 0.003, outcome: SpanOutcome = "success") -> Span:
    execution_id = str(uuid4())
    span_id = str(uuid4())
    return Span(
        span_id=span_id,
        execution_id=execution_id,
        caller_agent_id="planner_1",
        callee_agent_id="ocr_agent_22",
        capability="extraction.invoice",
        cost_usd=cost_usd,
        outcome=outcome,
        metadata={"tenant": "acme"},
    )


def _sample_trace(outcome: TraceOutcome = "success") -> ExecutionTrace:
    span = _sample_span(outcome=outcome if outcome != "partial" else "success")
    return ExecutionTrace(
        execution_id=span.execution_id,
        root_agent_id="planner_1",
        spans=[span],
        outcome=outcome,
    )


class _FailingEmitter(BaseEmitter):
    def emit_span(self, span: Span) -> None:
        del span
        raise RuntimeError("intentional span failure")

    def emit_trace(self, trace: ExecutionTrace) -> None:
        del trace
        raise RuntimeError("intentional trace failure")


class _CollectingEmitter(BaseEmitter):
    def __init__(self) -> None:
        self.span_count = 0
        self.trace_count = 0

    def emit_span(self, span: Span) -> None:
        del span
        self.span_count += 1

    def emit_trace(self, trace: ExecutionTrace) -> None:
        del trace
        self.trace_count += 1


class EmitterTests(unittest.TestCase):
    def test_concurrent_file_writes_1000_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            emitter = FileEmitter(path)

            def worker() -> None:
                for _ in range(10):
                    emitter.emit_span(_sample_span())

            threads: list[Thread] = [Thread(target=worker) for _ in range(100)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1000)
            parsed = [json.loads(line) for line in lines]
            self.assertTrue(all(item["event"] == "span" for item in parsed))

    def test_file_rotation_keeps_five_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rotate.jsonl"
            emitter = FileEmitter(path, rotate_mb=0.0002)
            for _ in range(250):
                emitter.emit_span(_sample_span())

            rotated = sorted(Path(tmpdir).glob("rotate.jsonl.*"))
            self.assertTrue(any(file.name == "rotate.jsonl.1" for file in rotated))
            self.assertFalse(any(file.name == "rotate.jsonl.6" for file in rotated))
            self.assertLessEqual(len(rotated), 5)

    def test_composite_emitter_partial_failure_does_not_propagate(self) -> None:
        collector = _CollectingEmitter()
        composite = CompositeEmitter([_FailingEmitter(), collector])
        stderr_buffer = io.StringIO()
        span = _sample_span()
        trace = _sample_trace()

        with patch("sys.stderr", stderr_buffer):
            composite.emit_span(span)
            composite.emit_trace(trace)

        self.assertEqual(collector.span_count, 1)
        self.assertEqual(collector.trace_count, 1)
        stderr_value = stderr_buffer.getvalue()
        self.assertIn("[agentmeter] emitter error:", stderr_value)

    def test_no_color_env_disables_ansi_codes(self) -> None:
        span = _sample_span()
        stdout_buffer = io.StringIO()
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            emitter = StdoutEmitter(silent=False, colorize=True)
            with patch("sys.stdout", stdout_buffer):
                emitter.emit_span(span)
        output = stdout_buffer.getvalue().strip()
        self.assertTrue(output.startswith("{") and output.endswith("}"))
        self.assertNotIn("\x1b[", output)

    def test_emit_span_safe_never_raises(self) -> None:
        stderr_buffer = io.StringIO()
        with patch("sys.stderr", stderr_buffer):
            _FailingEmitter().emit_span_safe(_sample_span())
        self.assertIn("[agentmeter] emitter error:", stderr_buffer.getvalue())

    def test_configure_and_get_emitter(self) -> None:
        configure()
        self.assertIsInstance(get_emitter(), StdoutEmitter)
        configure(silent=False)
        self.assertIsInstance(get_emitter(), StdoutEmitter)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "am.jsonl"
            configure(log_path=path)
            self.assertIsInstance(get_emitter(), FileEmitter)


if __name__ == "__main__":
    unittest.main()
