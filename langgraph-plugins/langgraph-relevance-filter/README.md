# langgraph-relevance-filter

LangGraph binding for **Practice A** (relevance filtering). All decision logic — chunk / rerank /
verbatim select / preview, the store, the reranker — lives in `context-core`
(`context_core.relevance`). This package is the thin interface: a `create_agent` middleware
(`wrap_tool_call` to rewrite an oversized tool result into a marker + disclaimer + verbatim preview +
reference, `after_agent` to drop closed retrieval exchanges), the `BaseMessage` ↔ neutral message
adapter, and the `retrieve_all_context` tool.

Tested `langchain` / `langgraph` versions are in [Compatibility](#compatibility).
`include_retrieval_tool` defaults to `True`.

## Install

From a clone of [sample-context-engineering](https://github.com/aws-samples/sample-context-engineering):

```bash
uv pip install -e context-core -e langgraph-plugins/langgraph-relevance-filter
```

This pulls in `agent-context-core`, `langchain` and
`langgraph`. The default reranker (`amazon.rerank-v1:0`) runs on Amazon Bedrock, so the first oversized
result needs AWS credentials and a region. Pass your own reranker through `config` to use something else.

## Compatibility

The package declares `langchain>=1.2.12,<2` and `langgraph>=1.1.5,<2`. These are the oldest versions where
the three LangGraph plugin suites and the validation harness all pass, so the three plugins share one floor and
install together.

| `langchain` | `langgraph` | `langchain-core` | Result |
| --- | --- | --- | --- |
| 1.5.0 | 1.2.22 | 1.8.1 | passes |
| 1.4.2 | 1.2.12 | 1.6.5 | passes |
| 1.3.0 | 1.2.22 | 1.8.1 | passes |
| 1.2.12 – 1.2.18 | 1.1.5 / 1.1.10 | 1.8.1 | passes |
| 1.2.10 | 1.0.8 | 1.2.10 | disclosure fails |
| 1.0.0 – 1.2.0 | 1.0.x | | disclosure and graph fail |

`langgraph` below 1.1.5 could not be tested with `langchain` 1.2.12: those `langgraph` releases accept a
`langgraph-prebuilt` that imports `ExecutionInfo`, which only exists in newer `langgraph`.

## Usage

```python
from langchain.agents import create_agent
from langgraph_relevance_filter import RelevanceFilterMiddleware

agent = create_agent(
    model="...",
    tools=[query_ledger],
    middleware=[RelevanceFilterMiddleware(max_result_tokens=8_000)],
)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
```

Both `invoke` and `ainvoke` / `astream` are supported.

Source, design notes and benchmarks:
[sample-context-engineering](https://github.com/aws-samples/sample-context-engineering).
