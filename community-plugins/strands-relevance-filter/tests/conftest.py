"""Shared fixtures and helpers for ``test_plugin.py``.

Local to this file's plugin tests: a controllable fake reranker, a fake agent whose
``.messages`` and ``.model.count_tokens`` the test drives, and a factory that builds a real
``AfterToolCallEvent`` around a given tool_use/result. Nothing here is meant to be imported by
the other test files in this directory.
"""

from __future__ import annotations

from typing import Any

import pytest
from strands.hooks.events import AfterToolCallEvent
from strands.types.tools import ToolResult, ToolUse


class FakeReranker:
    """A ``Reranker`` whose scores the test controls.

    Scores every chunk with ``score_value`` unless a ``scores`` list is provided, in which case it
    returns that list verbatim (padded/truncated to the chunk count). Records every ``score`` call so
    a test can assert whether scoring happened at all.
    """

    def __init__(
        self,
        *,
        score_value: float = 1.0,
        scores: list[float] | None = None,
        max_sources_per_query: int = 100,
    ) -> None:
        self.max_sources_per_query = max_sources_per_query
        self._score_value = score_value
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        self.calls.append((query, list(chunks)))
        if not chunks:
            return []
        if self._scores is not None:
            padded = list(self._scores)[: len(chunks)]
            padded += [0.0] * (len(chunks) - len(padded))
            return padded
        return [self._score_value] * len(chunks)


class FakeModel:
    """A fake ``agent.model`` exposing the single async ``count_tokens`` the size gate calls.

    The returned count is whatever the test sets on ``token_count``, so a test decides on its own
    whether a result is over or under ``max_result_tokens`` without building real content.
    """

    def __init__(self, token_count: int = 0) -> None:
        self.token_count = token_count
        self.count_tokens_calls: list[Any] = []

    async def count_tokens(self, messages: Any) -> int:
        self.count_tokens_calls.append(messages)
        return self.token_count


class FakeAgent:
    """A fake agent exposing only what the plugin reads: ``.messages`` and ``.model``."""

    def __init__(self, *, messages: list[dict] | None = None, token_count: int = 0) -> None:
        self.messages = messages if messages is not None else []
        self.model = FakeModel(token_count=token_count)


def make_after_tool_call_event(
    *,
    tool_use: ToolUse,
    result: ToolResult,
    agent: FakeAgent | None = None,
    selected_tool: Any = None,
    cancel_message: str | None = None,
) -> AfterToolCallEvent:
    """Build a real ``AfterToolCallEvent`` for the plugin's ``_on_after_tool_call`` hook.

    Only the fields the hook reads are populated; ``invocation_state`` is an empty dict and the
    optional signals default to their event defaults. The event's ``agent`` comes from the generic
    ``HookEvent`` base, so it is supplied positionally as the first field.
    """
    return AfterToolCallEvent(
        agent if agent is not None else FakeAgent(),  # HookEvent[...] base .agent field
        selected_tool=selected_tool,
        tool_use=tool_use,
        invocation_state={},
        result=result,
        cancel_message=cancel_message,
    )


@pytest.fixture
def fake_reranker() -> FakeReranker:
    """A reranker that scores every chunk 1.0 and records its calls."""
    return FakeReranker()


@pytest.fixture
def make_event():
    """Expose :func:`make_after_tool_call_event` as a fixture-provided factory."""
    return make_after_tool_call_event
