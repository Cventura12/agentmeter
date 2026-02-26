# agentmeter SpanKind Reference

| SpanKind | OpenInference value | Cost driver | Typical latency | Description |
| --- | --- | --- | --- | --- |
| `LLM` | `llm` | Prompt/completion token usage | 400-3000ms | Text or multimodal model inference call. |
| `CHAIN` | `chain` | Orchestration overhead + downstream fanout | 20-500ms | Deterministic multi-step orchestration wrapper. |
| `RETRIEVER` | `retriever` | Vector/DB query volume and index type | 10-300ms | Retrieve context from vector, search, or database stores. |
| `TOOL` | `tool` | External API/runtime execution time | 50-2000ms | Non-LLM side-effect call such as OCR, HTTP API, or DB write. |
| `AGENT` | `agent` | Planning depth + delegated subcalls | 80-1200ms | Decision-making agent step that routes or decomposes work. |
| `EMBEDDING` | `embedding` | Embedding model token volume | 20-250ms | Embedding generation call for indexing or retrieval. |
| `RERANKER` | `reranker` | Pairwise scoring model/runtime cost | 40-700ms | Rerank candidate documents/chunks after retrieval. |
| `GUARDRAIL` | `guardrail` | Validation policy/model checks | 20-800ms | Schema, policy, or safety gate over generated outputs. |
| `EVALUATOR` | `evaluator` | Scoring/eval model usage | 80-1500ms | Quality, faithfulness, or regression evaluation pass. |

agentmeter SpanKinds are forward-compatible with the OpenInference semantic convention spec used by Arize Phoenix and LangSmith.

To propose a new `SpanKind`, open an issue with:
1. The cognitive function it represents.
2. Evidence it appears in existing repos or frameworks.
3. Proposed typical cost/latency benchmarks.

Changes require discussion because `SpanKind` is a public API once adopted.
