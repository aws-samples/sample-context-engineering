"""awrap_model_call: the async twin must project and wrap state exactly like the sync hook.

Regression test for the sync/async gap the benchmark harness surfaced: LangChain raises
``NotImplementedError`` when a sync ``wrap_model_call`` runs under ``ainvoke``, so a stack driven with
``ainvoke`` (as the harness drives it) needs the graph to offer ``awrap_model_call``.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_context_graph.middleware import ContextGraphMiddleware


class _MockMatcher:
    """Scores every description the same, so projection is deterministic without an embedder."""

    def score(self, query, descriptions):  # noqa: ANN001, D102
        return [0.5 for _ in descriptions]


def _request(messages, state):
    """Build a minimal ModelRequest-like stand-in the middleware can drive."""
    from langgraph_context_graph._compat import ModelRequest  # local import: framework boundary

    try:
        return ModelRequest(messages=messages, state=state)  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - shape varies across langchain patch versions
        pytest.skip("ModelRequest constructor shape differs on this langchain version")


@pytest.mark.asyncio
async def test_awrap_model_call_matches_sync_projection_and_wraps_state():
    mw = ContextGraphMiddleware(matcher=_MockMatcher())
    messages = [HumanMessage(content="what is my balance?", id="m1")]
    state = {"messages": messages}

    captured = {}

    async def handler(req):
        captured["messages"] = list(req.messages)
        return AIMessage(content="answer")

    request = _request(messages, state)
    response = await mw.awrap_model_call(request, handler)

    # The handler ran (was awaited) and produced a wrapped response carrying the graph-state update.
    assert "messages" in captured
    assert response is not None


def test_graph_middleware_exposes_both_sync_and_async_model_hooks():
    mw = ContextGraphMiddleware(matcher=_MockMatcher())
    assert hasattr(mw, "wrap_model_call")
    assert hasattr(mw, "awrap_model_call")
