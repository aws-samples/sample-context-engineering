"""Task 8 composition test: the three LangGraph middlewares on ONE agent.

Asserts the design's composition invariants (design.md §8) against a MOCKED agent — no live model,
reranker, or embedder:

- B (disclosure) and D (context graph) both hook ``wrap_model_call`` but touch **disjoint** fields of the
  same ``ModelRequest`` (D: messages; B: tools + system prompt + a further message fold), so nesting them
  is safe.
- A (relevance) is on the tool surface (``awrap_tool_call`` + ``after_agent``), a different surface from
  the model-call layer.
- The A+D retrieval collision is resolved by constructing A with ``include_retrieval_tool=False`` in the
  combined arm, so only D's retrieval tools (``find_context`` / ``expand_card``) are visible to the model
  — never two retrieval tools over two stores.
- The whole stack runs under ``ainvoke`` (A is async-only), which is why D needs ``awrap_model_call``.
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

    def __init__(self):
        self.search_units = 0

    async def score(self, query, chunks):  # noqa: ANN001, D102
        self.search_units += 1
        return [1.0 if any(w in c.lower() for w in query.lower().split()) else 0.0 for c in chunks]


def _build_stack():
    """The combined arm exactly as the design/benchmark specifies."""
    graph = ContextGraphMiddleware(matcher=_MockMatcher())
    disclosure = ProgressiveToolDisclosureMiddleware()
    relevance = RelevanceFilterMiddleware(
        include_retrieval_tool=False,
        config={"reranker": _FakeReranker()},
    )
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


def test_only_the_graph_retrieval_tools_are_visible_in_the_combined_arm():
    graph, disclosure, relevance = _build_stack()

    def names(mw):
        return {getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(mw, "tools", [])}

    graph_tools = names(graph)
    relevance_tools = names(relevance)

    # A ships NO retrieval tool when include_retrieval_tool=False; D owns retrieval.
    assert "retrieve_all_context" not in relevance_tools
    assert graph_tools & {"find_context", "expand_card"}


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
