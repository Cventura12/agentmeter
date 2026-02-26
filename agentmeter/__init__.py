"""Public API for the agentmeter package."""

from .core import AgentIdentity, ExecutionTrace, Span
from .emitter import configure, get_emitter
from .tracer import AsyncTracer, SpanContext, SpanKind, Tracer

__version__ = "0.1.0"

__all__ = [
    "Tracer",
    "AsyncTracer",
    "SpanKind",
    "ExecutionTrace",
    "Span",
    "AgentIdentity",
    "SpanContext",
    "configure",
    "get_emitter",
]
