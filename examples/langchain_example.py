# Run: python examples/langchain_example.py
# Requires: pip install agentmeter[langchain]

from __future__ import annotations

from agentmeter import configure
from agentmeter.integrations import AgentMeterCallbackHandler


def main() -> None:
    from langchain_core.runnables import RunnableLambda

    configure(log_path="./agentmeter_logs/langchain.jsonl")

    handler = AgentMeterCallbackHandler("my-chain")
    chain = RunnableLambda(lambda payload: {"result": payload["query"].upper()})
    result = chain.invoke({"query": "trace this chain"}, config={"callbacks": [handler]})
    print(result)


if __name__ == "__main__":
    main()
