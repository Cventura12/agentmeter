"""Zero-config integrations for popular agent frameworks."""

from .langchain import AgentMeterCallbackHandler
from .litellm import setup_litellm_tracing
from .openai_agents import setup_openai_agents_tracing

__all__ = [
    "setup_openai_agents_tracing",
    "AgentMeterCallbackHandler",
    "setup_litellm_tracing",
]
