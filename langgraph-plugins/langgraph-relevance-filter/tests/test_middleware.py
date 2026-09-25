"""Behaviour tests for :class:`RelevanceFilterMiddleware`, against a mocked agent.

No real model and no real reranker is ever called: scoring is a deterministic ``FakeReranker``, the tool
hook is driven with a hand-built ``ToolCallRequest``, and the one end-to-end test uses a fake chat model
that replays a scripted turn sequence. What is asserted is the middleware's contract — the six guards,
the rewritten content (marker + disclaimer + verbatim preview + reference), the retrieval read-back
modes, the end-of-run cleanup, and the ``include_retrieval_tool=False`` opt-out.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from context_core.relevance import InMemoryStore, RerankerError
from langgraph_relevance_filter import RelevanceFilterMiddleware
from langgraph_relevance_filter._compat import ToolCallRequest

# --------------------------------------------------------------------------- fixtures / doubles

TARGET_LINE = "TARGET-ROW refund amount R$ 1,234.56 account 90210"
"""A line that must survive verbatim: it carries a decimal, a currency marker and digit runs."""


def _payload(rows: int = 60) -> str:
    """Build a multi-line payload whose 12th line is :data:`TARGET_LINE`."""
    lines = [f"filler row {index:03d} noise noise noise noise" for index in range(rows)]
    lines[11] = TARGET_LINE
    return "\n".join(lines)


class FakeReranker:
    """Deterministic scorer: a chunk holding ``needle`` scores 1.0, everything else 0.0."""

    max_sources_per_query = 100

    def __init__(self, needle: str = "TARGET-ROW", *, fail: bool = False) -> None:
        self.needle = needle
        self.fail = fail
        self.queries: list[str] = []

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        if self.fail:
            raise RerankerError("scoring unavailable")
        self.queries.append(query)
        return [1.0 if self.needle in chunk else 0.0 for chunk in chunks]


class ExplodingStore:
    """A ``Store`` whose write always fails, exercising the keep-original path."""

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        raise RuntimeError("disk on fire")

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        raise KeyError(reference)


def _middleware(**kwargs: Any) -> RelevanceFilterMiddleware:
    """Build a middleware with small budgets and a fake reranker, unless overridden."""
    config = kwargs.pop("config", None) or {}
    config.setdefault("reranker", FakeReranker())
    config.setdefault("chunk_tokens", 40)
    config.setdefault("preview_tokens", 40)
    kwargs.setdefault("max_result_tokens", 100)
    return RelevanceFilterMiddleware(config=config, **kwargs)


def _request(
    *,
    name: str = "query_ledger",
    args: dict[str, Any] | None = None,
    call_id: str = "tc1",
    messages: list[BaseMessage] | None = None,
    tool_obj: Any = None,
) -> ToolCallRequest:
    """Build a ``ToolCallRequest`` the way ``ToolNode`` would hand one to the hook."""
    state = {
        "messages": messages
        if messages is not None
        else [
            HumanMessage(content="which refunds were issued?"),
            AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args or {}}]),
        ]
    }
    return ToolCallRequest(
        tool_call={"id": call_id, "name": name, "args": args or {}},
        tool=tool_obj,
        state=state,
        runtime=None,
    )


def _handler(result: Any):
    """Wrap ``result`` as the ``handler`` coroutine ``awrap_tool_call`` awaits."""

    async def handler(request: ToolCallRequest) -> Any:
        handler.calls += 1  # type: ignore[attr-defined]
        return result

    handler.calls = 0  # type: ignore[attr-defined]
    return handler


# --------------------------------------------------------------------------- the rewrite


async def test_oversized_result_is_rewritten_to_marker_disclaimer_preview_and_ref():
    middleware = _middleware()
    text = _payload()
    result = ToolMessage(content=text, tool_call_id="tc1", name="query_ledger")

    rewritten = await middleware.awrap_tool_call(_request(), _handler(result))

    assert isinstance(rewritten, ToolMessage)
    content = rewritten.content
    assert content.startswith("[Relevance: tool result, ~")
    # The disclaimer, both of its structural claims.
    assert "[Filtered: this is an EXCERPT, not the whole result" in content
    assert "cannot be computed from this excerpt" in content
    # The with-reference branch names the tool and the budgets.
    assert "retrieve_all_context" in content
    assert "max_chunks`/`max_tokens" in content
    # The preview, and the reference token that resolves it.
    assert TARGET_LINE in content
    assert content.rstrip().endswith("]")
    assert "[ref: " in content
    # The original is never mutated, and identity/ids are preserved.
    assert result.content == text
    assert rewritten.tool_call_id == "tc1"
    assert rewritten.name == "query_ledger"


async def test_preview_selection_is_verbatim_and_bounded():
    middleware = _middleware()
    text = _payload()

    rewritten = await middleware.awrap_tool_call(
        _request(), _handler(ToolMessage(content=text, tool_call_id="tc1"))
    )

    body = rewritten.content.split("\n\n", 1)[1]
    preview = body.rsplit("\n\n[ref: ", 1)[0]
    # Every non-marker segment of the preview is an exact substring of the source.
    segments = [
        segment
        for segment in preview.split("\n[... ")
        if segment and "lines omitted ...]" not in segment.split("\n")[0]
    ]
    for segment in segments:
        assert segment.strip("\n") in text
    # The rewrite is smaller than the original: that is the whole point.
    assert len(rewritten.content) < len(text) + 2_000


async def test_query_carries_the_question_and_the_call_arguments():
    reranker = FakeReranker()
    middleware = _middleware(config={"reranker": reranker})

    await middleware.awrap_tool_call(
        _request(args={"account": "90210"}),
        _handler(ToolMessage(content=_payload(), tool_call_id="tc1")),
    )

    assert reranker.queries, "the reranker was never called"
    query = reranker.queries[0]
    assert "which refunds were issued?" in query
    assert '"account": "90210"' in query
    # The tool name biases the ranking and is deliberately excluded.
    assert "query_ledger" not in query


# --------------------------------------------------------------------------- the six guards


async def test_guard_1_non_tool_message_result_passes_through():
    middleware = _middleware()
    command = Command(update={"messages": []})

    out = await middleware.awrap_tool_call(_request(), _handler(command))

    assert out is command


async def test_guard_2_own_retrieval_tool_is_never_filtered():
    middleware = _middleware()
    result = ToolMessage(content=_payload(), tool_call_id="tc1", name="retrieve_all_context")

    out = await middleware.awrap_tool_call(
        _request(name="retrieve_all_context"), _handler(result)
    )

    assert out is result


async def test_guard_3_delegation_result_is_never_filtered():
    middleware = _middleware()

    @tool
    def delegate(question: str) -> str:
        """Hand the question to a sub-agent."""
        return "answered"

    delegate.return_direct = True
    result = ToolMessage(content=_payload(), tool_call_id="tc1", name="delegate")

    out = await middleware.awrap_tool_call(_request(name="delegate", tool_obj=delegate), _handler(result))

    assert out is result


async def test_guard_4_result_under_the_size_gate_is_kept():
    middleware = _middleware()
    result = ToolMessage(content="only a handful of rows", tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert out is result


async def test_guard_5_should_filter_veto_keeps_the_result():
    seen: list[tuple[str, int]] = []

    def veto(tool_name: str, token_count: int, **kwargs: Any) -> bool:
        seen.append((tool_name, token_count))
        return False

    middleware = _middleware(should_filter=veto)
    result = ToolMessage(content=_payload(), tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert out is result
    assert seen and seen[0][0] == "query_ledger"
    assert seen[0][1] > 100, "the callback is consulted with the estimated token count"


async def test_guard_5_should_filter_may_be_async_and_fails_open():
    async def approve(tool_name: str, token_count: int, **kwargs: Any) -> bool:
        return True

    def explode(tool_name: str, token_count: int, **kwargs: Any) -> bool:
        raise RuntimeError("callback bug")

    for callback in (approve, explode):
        middleware = _middleware(should_filter=callback)
        out = await middleware.awrap_tool_call(
            _request(), _handler(ToolMessage(content=_payload(), tool_call_id="tc1"))
        )
        assert "[Relevance: tool result" in out.content, f"{callback.__name__} should have filtered"


async def test_guard_6_result_without_scorable_text_is_kept():
    middleware = _middleware()
    # A large non-textual part: nothing to score, so the result is left alone.
    blob = {"type": "image", "mime_type": "image/png", "base64": "A" * 1_000}
    result = ToolMessage(content=[blob], tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert out is result


async def test_non_scorable_parts_survive_verbatim_after_the_marker():
    middleware = _middleware()
    blob = {"type": "image", "mime_type": "image/png", "base64": "A" * 40}
    result = ToolMessage(content=[{"type": "text", "text": _payload()}, blob], tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert isinstance(out.content, list)
    assert out.content[0]["type"] == "text"
    assert "[Relevance: tool result" in out.content[0]["text"]
    assert out.content[1] == blob


# --------------------------------------------------------------------------- failure paths


async def test_scoring_failure_keeps_the_original_result():
    middleware = _middleware(config={"reranker": FakeReranker(fail=True)})
    result = ToolMessage(content=_payload(), tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert out is result


async def test_store_failure_keeps_the_original_result():
    middleware = _middleware(store=ExplodingStore())
    result = ToolMessage(content=_payload(), tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    assert out is result


async def test_max_result_tokens_must_be_positive():
    with pytest.raises(ValueError, match="max_result_tokens must be positive"):
        RelevanceFilterMiddleware(max_result_tokens=0)


# --------------------------------------------------------------------------- retrieve_all_context


async def _filtered(middleware: RelevanceFilterMiddleware, text: str | None = None) -> tuple[str, str]:
    """Filter a payload and return ``(rewritten content, reference)``."""
    payload = text if text is not None else _payload()
    out = await middleware.awrap_tool_call(
        _request(), _handler(ToolMessage(content=payload, tool_call_id="tc1"))
    )
    content = out.content if isinstance(out.content, str) else out.content[0]["text"]
    reference = content.rsplit("[ref: ", 1)[1].rstrip("]\n")
    return content, reference


async def test_retrieval_tool_is_registered_and_named():
    middleware = _middleware()
    assert [t.name for t in middleware.tools] == ["retrieve_all_context"]


async def test_retrieve_by_line_range():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    out = await middleware.tools[0].ainvoke(
        {"reference": reference, "line_range": {"start": 12, "end": 12}}
    )

    assert "[Lines 12-12 of 60]" in out
    assert TARGET_LINE in out


async def test_retrieve_by_pattern_returns_only_matching_rows():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    out = await middleware.tools[0].ainvoke(
        {"reference": reference, "pattern": "TARGET-ROW", "context_lines": 0}
    )

    assert "[1 match for /TARGET-ROW/]" in out
    assert TARGET_LINE in out
    assert "filler row 000" not in out


async def test_retrieve_by_max_chunks_uses_the_memoized_relevance_ranking():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    assert reference in middleware._rankings, "the chunk ranking must be memoized against the reference"
    out = await middleware.tools[0].ainvoke({"reference": reference, "max_chunks": 1})

    assert "chunks (most relevant first)" in out
    assert TARGET_LINE in out


async def test_retrieve_everything_with_a_large_budget():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    out = await middleware.tools[0].ainvoke({"reference": reference, "max_tokens": 50_000})

    assert "[Lines 1-60 of 60]" in out
    assert TARGET_LINE in out
    assert "filler row 059" in out


async def test_retrieve_without_options_returns_the_raw_content():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    out = await middleware.tools[0].ainvoke({"reference": reference})

    assert out == _payload()


async def test_retrieve_rejects_unknown_reference_and_bad_budgets():
    middleware = _middleware()
    _content, reference = await _filtered(middleware)

    with pytest.raises(ValueError, match="reference not found"):
        await middleware._retrieve("mem_99_nope")
    with pytest.raises(ValueError, match="max_chunks must be an integer"):
        await middleware._retrieve(reference, max_chunks=0)
    with pytest.raises(ValueError, match="max_tokens must be an integer"):
        await middleware._retrieve(reference, max_tokens=-1)


async def test_retrieve_refuses_to_search_binary_content():
    middleware = _middleware()
    reference = await middleware._store.store("tc1_0", b"\x00\x01binary", "image/png")

    with pytest.raises(ValueError, match="cannot search binary content"):
        await middleware._retrieve(reference, pattern="anything")
    assert await middleware._retrieve(reference) == {
        "type": "image",
        "mime_type": "image/png",
        "base64": "AAFiaW5hcnk=",
    }


# --------------------------------------------------------------------------- after_agent cleanup


def _retrieval_exchange(call_id: str = "r1", payload: str = "the whole result") -> list[BaseMessage]:
    """An assistant retrieval call and its answer, as they sit in the state."""
    return [
        AIMessage(
            content="checking the whole result",
            tool_calls=[{"id": call_id, "name": "retrieve_all_context", "args": {"reference": "mem_1_tc1_0"}}],
            id=f"ai-{call_id}",
        ),
        ToolMessage(content=payload, tool_call_id=call_id, id=f"tm-{call_id}"),
    ]


def test_after_agent_removes_the_closed_retrieval_exchange():
    middleware = _middleware()
    state = {
        "messages": [
            HumanMessage(content="what is the largest refund?", id="h1"),
            ToolMessage(content="[Relevance: ...] excerpt [ref: mem_1_tc1_0]", tool_call_id="tc1", id="tm-tc1"),
            *_retrieval_exchange(),
            AIMessage(content="the largest refund is R$ 1,234.56", id="ai-final"),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    messages = update["messages"]
    assert isinstance(messages[0], RemoveMessage) and messages[0].id == REMOVE_ALL_MESSAGES
    kept_ids = [m.id for m in messages[1:]]
    assert kept_ids == ["h1", "tm-tc1", "ai-final"]
    # The excerpt and its reference stay: the content is still retrievable later.
    assert "[ref: mem_1_tc1_0]" in messages[2].content


def test_after_agent_leaves_an_unclosed_retrieval_call_alone():
    middleware = _middleware()
    state = {
        "messages": [
            HumanMessage(content="totals?", id="h1"),
            _retrieval_exchange()[0],  # the call, with no answering ToolMessage
        ]
    }

    assert middleware.after_agent(state, None) is None


def test_after_agent_keeps_a_mixed_assistant_message_and_its_other_result():
    middleware = _middleware()
    mixed = AIMessage(
        content="two calls",
        tool_calls=[
            {"id": "r1", "name": "retrieve_all_context", "args": {"reference": "mem_1_tc1_0"}},
            {"id": "o1", "name": "query_ledger", "args": {}},
        ],
        id="ai-mixed",
    )
    state = {
        "messages": [
            HumanMessage(content="totals?", id="h1"),
            mixed,
            ToolMessage(content="the whole result", tool_call_id="r1", id="tm-r1"),
            ToolMessage(content="other rows", tool_call_id="o1", id="tm-o1"),
        ]
    }

    update = middleware.after_agent(state, None)

    kept = update["messages"][1:]
    assert [m.id for m in kept] == ["h1", "ai-mixed", "tm-o1"]
    assert [call["name"] for call in kept[1].tool_calls] == ["query_ledger"]


def test_after_agent_ignores_a_state_with_nothing_to_remove():
    middleware = _middleware()
    state = {"messages": [HumanMessage(content="hi", id="h1"), AIMessage(content="hello", id="a1")]}

    assert middleware.after_agent(state, None) is None


async def test_aafter_agent_matches_after_agent():
    middleware = _middleware()
    state = {"messages": [HumanMessage(content="totals?", id="h1"), *_retrieval_exchange()]}

    sync_update = middleware.after_agent(state, None)
    async_update = await middleware.aafter_agent(state, None)

    assert [m.id for m in sync_update["messages"]] == [m.id for m in async_update["messages"]]


# --------------------------------------------------------------------------- include_retrieval_tool=False


async def test_opt_out_keeps_filtering_but_stores_nothing():
    middleware = _middleware(include_retrieval_tool=False)
    result = ToolMessage(content=_payload(), tool_call_id="tc1")

    out = await middleware.awrap_tool_call(_request(), _handler(result))

    content = out.content
    assert "[Relevance: tool result, ~" in content
    assert TARGET_LINE in content, "filtering still produces a verbatim preview"
    # No store, no reference, no retrieval tool, no cleanup.
    assert middleware._store is None
    assert list(middleware.tools) == []
    assert "[ref: " not in content
    assert "retrieve_all_context" not in content
    assert "Say that the result was filtered instead of computing it from the excerpt." in content
    assert middleware._rankings == {}


def test_opt_out_disables_the_cleanup():
    middleware = _middleware(include_retrieval_tool=False)
    state = {"messages": [HumanMessage(content="totals?", id="h1"), *_retrieval_exchange()]}

    assert middleware.after_agent(state, None) is None


async def test_opt_out_retrieval_is_unresolvable():
    middleware = _middleware(include_retrieval_tool=False)

    with pytest.raises(ValueError, match="reference not found"):
        await middleware._retrieve("mem_1_tc1_0")


async def test_explicit_store_is_used_when_given():
    store = InMemoryStore()
    middleware = _middleware(store=store)

    _content, reference = await _filtered(middleware)

    content_bytes, content_type = await store.retrieve(reference)
    assert content_bytes.decode("utf-8") == _payload()
    assert content_type == "text/plain"


# --------------------------------------------------------------------------- end to end, mocked model


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that replays scripted turns, one per call, and records what it was sent."""

    turns: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        turn = self.turns[min(len(self.seen) - 1, len(self.turns) - 1)]
        return ChatResult(generations=[ChatGeneration(message=turn)])


async def test_end_to_end_filters_the_result_and_cleans_up_the_retrieval():
    payload = _payload()

    @tool
    def query_ledger(account: str) -> str:
        """Return every ledger row for an account."""
        return payload

    middleware = _middleware()
    model = ScriptedChatModel(
        turns=[
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "query_ledger", "args": {"account": "90210"}}],
                id="ai-1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "r1",
                        "name": "retrieve_all_context",
                        "args": {"reference": "mem_1_tc1_0", "pattern": "TARGET-ROW", "context_lines": 0},
                    }
                ],
                id="ai-2",
            ),
            AIMessage(content="the largest refund is R$ 1,234.56", id="ai-3"),
        ],
        seen=[],
    )

    from langchain.agents import create_agent

    agent = create_agent(model=model, tools=[query_ledger], middleware=[middleware])
    final = await agent.ainvoke({"messages": [HumanMessage(content="largest refund?", id="h1")]})

    messages = final["messages"]
    # The tool result that entered the state is the excerpt, not the payload.
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1, "the retrieval exchange must not survive the run"
    assert "[Relevance: tool result, ~" in tool_messages[0].content
    assert TARGET_LINE in tool_messages[0].content
    assert payload not in tool_messages[0].content
    # The retrieval call and its (large) answer are gone; the excerpt and the answer stay.
    assert [m.id for m in messages] == ["h1", "ai-1", tool_messages[0].id, "ai-3"]
    # The model really did see the retrieval result before answering.
    retrieved = [m for turn in model.seen for m in turn if isinstance(m, ToolMessage) and m.tool_call_id == "r1"]
    assert retrieved and TARGET_LINE in retrieved[0].content


# --------------------------------------------------------------------------- Strands parity


def test_gate_counts_like_the_strands_default_heuristic():
    from langgraph_relevance_filter.middleware import _approximate_tokens

    # ceil(chars / 4) for text, ceil(len(json.dumps) / 2) for JSON, binary parts not counted.
    assert _approximate_tokens([{"text": "abcde"}]) == 2
    assert _approximate_tokens([{"json": {"k": "v"}}]) == 5  # '{"k": "v"}' is 10 chars
    assert _approximate_tokens([{"text": "abcd"}, {"json": [1]}, {"type": "image"}]) == 1 + 2


def test_sync_wrap_tool_call_filters_under_invoke():
    middleware = _middleware()
    result = ToolMessage(content=_payload(), tool_call_id="tc1", name="query_ledger")
    calls = []

    def handler(request: ToolCallRequest) -> Any:
        calls.append(request)
        return result

    rewritten = middleware.wrap_tool_call(_request(), handler)

    assert len(calls) == 1
    assert rewritten.content.startswith("[Relevance: tool result, ~")
    assert TARGET_LINE in rewritten.content


async def test_sync_wrap_tool_call_also_works_inside_a_running_loop():
    middleware = _middleware()
    result = ToolMessage(content=_payload(), tool_call_id="tc1", name="query_ledger")
    rewritten = middleware.wrap_tool_call(_request(), lambda request: result)
    assert rewritten.content.startswith("[Relevance: tool result, ~")
