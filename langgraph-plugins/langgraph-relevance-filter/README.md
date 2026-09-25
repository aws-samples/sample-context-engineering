# langgraph-relevance-filter

LangGraph binding for **Practice A** (relevance filtering). All decision logic — chunk / rerank /
verbatim select / preview, the store, the reranker — lives in `context-core`
(`context_core.relevance`). This package is the thin interface: a `create_agent` middleware
(`wrap_tool_call` to rewrite an oversized tool result into a marker + disclaimer + verbatim preview +
reference, `after_agent` to drop closed retrieval exchanges), the `BaseMessage` ↔ neutral message
adapter, and the `retrieve_all_context` tool.

Verified against `langchain` 1.x. `include_retrieval_tool` defaults to `True`.
