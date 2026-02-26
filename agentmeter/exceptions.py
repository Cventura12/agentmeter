class AgentMeterError(Exception):
    """Base exception for agentmeter."""


class TraceNotActiveError(AgentMeterError):
    """Raised when trace operations are attempted outside an execution context."""


class InvalidAgentCallError(AgentMeterError):
    """Raised when an agent call contains invalid values."""
