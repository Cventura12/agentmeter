# Run: python examples/litellm_example.py
# Requires: pip install agentmeter[litellm]

from __future__ import annotations

import os

from agentmeter import configure
from agentmeter.integrations import setup_litellm_tracing


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    import litellm

    configure(log_path="./agentmeter_logs/litellm.jsonl")
    setup_litellm_tracing("my-agent")

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in five words."}],
    )
    content = response["choices"][0]["message"]["content"]
    print(content)


if __name__ == "__main__":
    main()
