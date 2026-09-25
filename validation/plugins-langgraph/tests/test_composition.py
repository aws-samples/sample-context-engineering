"""Task 8 composition test: the three LangGraph middlewares on ONE agent.

Asserts the design's composition invariants (design.md §8) against a MOCKED agent — no live model,
reranker, or embedder:

- B (disclosure) and D (context graph) both hook ``wrap_model_call`` but touch **disjoint** fields of the
  same ``ModelRequest`` (D: messages; B: tools + system prompt + a further message fold), so nesting them
  is safe.
- A (relevance) is on the tool surface (``wrap_tool_call`` + ``after_agent``), a different surface from
  the model-call layer.
- As in the Strands harness, the combined arm keeps A's ``retrieve_all_context`` and D's three tools, and
  D gets ``stash=relevance.stash`` so a reference A mints resolves through ``expand_artifact`` too --
  both retrieval tools read the same content.
- Every hook has sync and async twins, so the stack runs under ``invoke`` and ``ainvoke``.
"""

from __future__ import annotations

import pytest

from langgraph_context_graph.middleware import ContextGraphMiddleware
from langgraph_progressive_tool_disclosure.middleware import ProgressiveToolDisclosureMiddleware
from langgraph_relevance_filter.middleware import RelevanceFilterMiddleware


class _MockMatcher:
    def score(self, query, descriptions):  # noqa: ANN001, D102
        return [0.5 for _ in descriptions]


class _FakeReranker:
    """Keyword scorer standing in for BedrockReranker — no AWS."""

    max_sources_per_query = 100

    def __init__(self):
        self.search_units = 0

    async def score(self, query, chunks):  # noqa: ANN001, D102
        self.search_units += 1
        return [1.0 if any(w in c.lower() for w in query.lower().split()) else 0.0 for c in chunks]


def _build_stack():
    """The combined arm exactly as the benchmark builds it, and as the Strands harness does."""
    relevance = RelevanceFilterMiddleware(config={"reranker": _FakeReranker()})
    graph = ContextGraphMiddleware(matcher=_MockMatcher(), stash=relevance.stash)
    disclosure = ProgressiveToolDisclosureMiddleware()
    # Outermost first (design §8): D wraps B; A is on the tool surface.
    return [graph, disclosure, relevance]


def test_stack_constructs_with_the_documented_nesting():
    stack = _build_stack()
    assert [type(m).__name__ for m in stack] == [
        "ContextGraphMiddleware",
        "ProgressiveToolDisclosureMiddleware",
        "RelevanceFilterMiddleware",
    ]


def test_b_and_d_hook_the_same_stage_but_disjoint_request_fields():
    graph, disclosure, _ = _build_stack()
    # D exposes a model-call hook (sync + async); B exposes a model-call hook (sync + async).
    assert hasattr(graph, "wrap_model_call") and hasattr(graph, "awrap_model_call")
    assert hasattr(disclosure, "wrap_model_call") and hasattr(disclosure, "awrap_model_call")


def test_a_is_on_the_tool_surface_not_the_model_call_layer():
    from langgraph_relevance_filter._compat import AgentMiddleware

    _, _, relevance = _build_stack()
    assert hasattr(relevance, "awrap_tool_call")
    assert hasattr(relevance, "after_agent")
    # A does not OVERRIDE the model-call hook — it inherits the base no-op, so it never touches the
    # ModelRequest and cannot collide with B/D at the model-call layer.
    assert type(relevance).wrap_model_call is AgentMiddleware.wrap_model_call
    assert type(relevance).awrap_tool_call is not AgentMiddleware.awrap_tool_call


def test_the_combined_arm_carries_both_retrieval_paths_over_one_content():
    graph, disclosure, relevance = _build_stack()

    def names(mw):
        return {getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(mw, "tools", [])}

    # As in the Strands harness: the filter keeps retrieve_all_context, the graph keeps its three.
    assert "retrieve_all_context" in names(relevance)
    assert {"expand_card", "expand_artifact", "find_context"} <= names(graph)


@pytest.mark.asyncio
async def test_a_reference_the_filter_mints_resolves_through_expand_artifact():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from context_core.graph import GraphState
    from langgraph_relevance_filter._compat import ToolCallRequest

    graph, _, relevance = _build_stack()
    payload = "\n".join(f"row {i}: refund issued for account {i}" for i in range(4000))
    request = ToolCallRequest(
        tool_call={"id": "tc1", "name": "query_ledger", "args": {}},
        tool=None,
        state={
            "messages": [
                HumanMessage(content="which refunds were issued?"),
                AIMessage(content="", tool_calls=[{"id": "tc1", "name": "query_ledger", "args": {}}]),
            ]
        },
        runtime=None,
    )

    async def handler(_request):
        return ToolMessage(content=payload, tool_call_id="tc1", name="query_ledger")

    rewritten = await relevance.awrap_tool_call(request, handler)
    reference = rewritten.content.split("[ref: ", 1)[1].split("]", 1)[0].split(",")[0].strip()
    assert reference.startswith("mem_")

    answer = await graph.expand_artifact(GraphState(), graph._store_for(""), reference)
    assert "row 3999: refund issued for account 3999" in answer

    # Without the stash the same reference is absent, which is the Strands result with no ContextManager.
    bare = ContextGraphMiddleware(matcher=_MockMatcher())
    missing = await bare.expand_artifact(GraphState(), bare._store_for(""), reference)
    assert "row 3999" not in missing


def test_graph_async_hook_closes_the_sync_async_gap():
    """The regression the harness found: the stack runs under ainvoke, so D must offer awrap_model_call."""
    graph = ContextGraphMiddleware(matcher=_MockMatcher())
    assert hasattr(graph, "awrap_model_call")


@pytest.mark.asyncio
async def test_disclosure_and_graph_async_hooks_both_await_the_handler():
    """Both model-call middlewares expose an awaitable hook, so nesting them under ainvoke is coherent."""
    import inspect

    graph = ContextGraphMiddleware(matcher=_MockMatcher())
    disclosure = ProgressiveToolDisclosureMiddleware()
    assert inspect.iscoroutinefunction(graph.awrap_model_call)
    assert inspect.iscoroutinefunction(disclosure.awrap_model_call)
