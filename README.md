# agentmeter

[![CI](https://github.com/Cventura12/agentmeter-/actions/workflows/ci.yml/badge.svg)](https://github.com/Cventura12/agentmeter-/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/agentmeter.svg)](https://pypi.org/project/agentmeter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![OpenInference compatible](https://img.shields.io/badge/OpenInference-compatible-0A7B83)](https://arize-ai.github.io/openinference/)

agentmeter instruments multi-agent AI systems to track execution cost, latency, and outcome at the span level - giving you the economic signal your traces are missing.

Multi-agent systems fail silently and spend invisibly. LangSmith and Phoenix show you what happened. agentmeter tells you what it cost, what failed, and how it compares.

## Quickstart

```python
from agentmeter import Tracer, SpanKind, configure
def retrieve(q: str) -> list[str]:
    return ["doc-a"]
configure(log_path="./agentmeter_logs/run.jsonl")
tracer = Tracer("my-planner", SpanKind.AGENT)
with tracer.span("my-retriever", SpanKind.RETRIEVER) as s:
    docs = retrieve("change-order status")
    s.set_cost(0.0002)
    s.set_outcome("success")
print(f"Total cost: ${tracer.finish().total_cost_usd:.4f}")
```

## SpanKind Reference

| Name | OpenInference-compatible | Cost driver | Example |
| --- | --- | --- | --- |
| `LLM` | Yes (`llm`) | prompt + completion tokens | draft a summary |
| `CHAIN` | Yes (`chain`) | orchestration overhead + downstream calls | run a multi-step chain |
| `RETRIEVER` | Yes (`retriever`) | vector/DB lookup count and latency | fetch top-k documents |
| `TOOL` | Yes (`tool`) | external API/runtime call cost | OCR a PDF |
| `AGENT` | Yes (`agent`) | planning + delegated subcalls | route task to workers |
| `EMBEDDING` | Yes (`embedding`) | embedding model tokens | embed a document chunk |
| `RERANKER` | Yes (`reranker`) | reranker model inference | rerank candidate passages |
| `GUARDRAIL` | Yes (`guardrail`) | policy/model checks | schema and safety gate |
| `EVALUATOR` | Yes (`evaluator`) | evaluation model/runtime calls | score answer quality |

## Configuration

```python
from agentmeter import configure

configure()  # silent stdout default
configure(silent=False)  # print JSON events to stdout
configure(log_path="./agentmeter_logs/run.jsonl")  # JSONL file output
```

## Data Model

`ExecutionTrace` is the immutable top-level record for one execution and includes `execution_id`, `root_agent_id`, `started_at`, `ended_at`, `outcome`, and `spans`, plus computed `total_cost_usd` and `total_latency_ms`. Each `Span` captures `span_id`, `parent_span_id`, `caller_agent_id`, `callee_agent_id`, capability, timestamps, token counts, `cost_usd`, `outcome`, and optional `error_message`/metadata. `AgentIdentity` stores persistent identity fields (`agent_id`, capability, version, tags, created timestamp) for cross-run attribution.

## OpenInference Compatibility

agentmeter SpanKinds align with the OpenInference specification used by Arize Phoenix and LangSmith. Traces can be compared against industry benchmarks using the same vocabulary.

## Contributing

Contributions are welcome, especially around tracer integrations, emitter backends, and model validation hardening. Changes to `SpanKind` require discussion before merge because it is part of the public API and affects downstream compatibility.

## License

MIT
# agentmeter
