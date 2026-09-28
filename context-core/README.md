# agent-context-core

Framework-agnostic core for the three context-engineering practices:

- **relevance** (Practice A) — chunk / rerank / select / verbatim preview of oversized tool results,
  with a pluggable raw-content store.
- **disclosure** (Practice B) — lexical tool index and lean-catalog / closed-exchange-fold logic.
- **graph** (Practice D) — Card model, deterministic scan, scoring, similarity matcher, and the
  per-call projection.

## Install

```bash
pip install agent-context-core
```

The distribution is `agent-context-core`; the import name is `context_core`. You normally do not install
it directly — the framework bindings depend on it:
[`langgraph-relevance-filter`](https://pypi.org/project/langgraph-relevance-filter/),
[`langgraph-progressive-tool-disclosure`](https://pypi.org/project/langgraph-progressive-tool-disclosure/)
and [`langgraph-context-graph`](https://pypi.org/project/langgraph-context-graph/).

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

The logic here is **recreated from the Strands community plugins**
([`community-plugins/strands-*`](https://github.com/aws-samples/sample-context-engineering/tree/main/community-plugins))
used as **read-only reference**. This package never modifies those packages; it is an independent
implementation that the LangGraph bindings depend on.
