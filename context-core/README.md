# agent-context-core

Framework-agnostic core for the three context-engineering practices:

- **relevance** (Practice A) — chunk / rerank / select / verbatim preview of oversized tool results,
  with a pluggable raw-content store.
- **disclosure** (Practice B) — lexical tool index and lean-catalog / closed-exchange-fold logic.
- **graph** (Practice D) — Card model, deterministic scan, scoring, similarity matcher, and the
  per-call projection.

## Install

From a clone of [sample-context-engineering](https://github.com/aws-samples/sample-context-engineering):

```bash
uv pip install -e context-core
```

The distribution is `agent-context-core`; the import name is `context_core`. You normally do not install
it directly — every framework binding depends on it: the three `strands-*` plugins, the three `langgraph-*`
middlewares, the four `hermes-*` engines and the OpenCode plugin.

The default reranker (`amazon.rerank-v1:0`) and embedder (`cohere.embed-multilingual-v3`) call Amazon
Bedrock through `boto3`, so using them needs AWS credentials and a region. Both are pluggable, and no AWS
client is created until the first scoring call.

## The neutral contract

`context_core` holds **all decision logic** and imports **no agent framework** — neither `strands` nor
`langchain`/`langgraph`. It operates on a neutral message shape (`context_core.message`): a dict
`{"role": str, "content": [block, ...]}` where a block is a dict such as `{"text": ...}`,
`{"toolUse": ...}`, `{"toolResult": ...}`, or `{"json": ...}`. Each framework binding adapts its native
message type to this shape at the boundary and calls into the core. The contract is machine-checked by
`tests/test_no_framework_import.py`.

## Provenance

The logic here was first written inside the Strands community plugins
([`strands-community-plugins/strands-*`](https://github.com/aws-samples/sample-context-engineering/tree/main/strands-community-plugins))
and recreated here, framework-free, for the LangGraph bindings. The Strands plugins have since moved onto
this package too: each one now keeps only its Strands hooks (`plugin.py`), its SDK imports (`_compat.py`)
and its message conversion (`_adapter.py`), and every decision about content runs here, the same code for
every framework.
