# langgraph-relevance-filter

LangGraph binding for **Practice A** (relevance filtering). All decision logic — chunk / rerank /
verbatim select / preview, the store, the reranker — lives in `context-core`
(`context_core.relevance`). This package is the thin interface: a `create_agent` middleware
(`wrap_tool_call` to rewrite an oversized tool result into a marker + disclaimer + verbatim preview +
reference, `after_agent` to drop closed retrieval exchanges), the `BaseMessage` ↔ neutral message
adapter, and the `retrieve_all_context` tool.

Verified against `langchain` 1.x. `include_retrieval_tool` defaults to `True`.

## Install

```bash
pip install langgraph-relevance-filter
```

This pulls in [`agent-context-core`](https://pypi.org/project/agent-context-core/), `langchain` and
`langgraph`. The default reranker (`amazon.rerank-v1:0`) runs on Amazon Bedrock, so the first oversized
result needs AWS credentials and a region. Pass your own reranker through `config` to use something else.

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
