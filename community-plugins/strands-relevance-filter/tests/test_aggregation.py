"""Tests for aggregate questions: the excerpt disclaimer, budgeted retrieval, and end-of-turn cleanup.

An excerpt cannot answer a question that needs every row (a maximum, a total, a count). These tests pin
the three pieces that let the model get there: the disclaimer that says what it is looking at, the
``max_chunks``/``max_tokens`` reads that reach the rest, and the removal of those reads from the history
once the turn has ended.
"""

from __future__ import annotations

import pytest
from strands.hooks.events import AfterInvocationEvent
from strands.types.tools import ToolContext, ToolResult, ToolUse

from strands_relevance_filter import InMemoryStore, RelevanceFilter

from conftest import FakeAgent, FakeReranker, make_after_tool_call_event

_MESSAGES = [{"role": "user", "content": [{"text": "what was the largest refund"}]}]
# Six lines of 40 characters each; with chunk_tokens=10 (40 chars) every line is one chunk.
_ROWS = [f"row {i:02d} " + "x" * 32 for i in range(6)]
_TEXT = "\n".join(_ROWS)


def _plugin(reranker: FakeReranker, *, tool: bool = True, preview_tokens: int = 10) -> RelevanceFilter:
    config = {"reranker": reranker, "chunk_tokens": 10, "preview_tokens": preview_tokens, "relevance_threshold": 0.0}
    return RelevanceFilter(max_result_tokens=10, config=config, include_retrieval_tool=tool)


async def _filtered(plugin: RelevanceFilter) -> tuple[FakeAgent, str]:
    agent = FakeAgent(messages=list(_MESSAGES), token_count=100_000)
    plugin.init_agent(agent)  # type: ignore[arg-type]
    tool_use = ToolUse(toolUseId="t1", name="statement", input={"days": 90})
    event = make_after_tool_call_event(
        tool_use=tool_use, result=ToolResult(toolUseId="t1", status="success", content=[{"text": _TEXT}]), agent=agent
    )
    await plugin._on_after_tool_call(event)
    return agent, event.result["content"][0]["text"]


def _ctx(agent: FakeAgent) -> ToolContext:
    tool_use = ToolUse(toolUseId="r1", name="retrieve_all_context", input={})
    return ToolContext(tool_use=tool_use, agent=agent, invocation_state={})  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# The disclaimer.
# --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disclaimer_reports_metadata_and_names_the_retrieval_call() -> None:
    plugin = _plugin(FakeReranker(scores=[0.1, 0.9, 0.2, 0.3, 0.4, 0.5]))
    _, marker = await _filtered(plugin)

    assert marker.startswith("[Relevance: tool result, ~100,000 tokens]")
    assert "EXCERPT" in marker
    assert "original: 6 lines, 6 chunks" in marker
    assert "shown: 1 chunk(s), lines 2" in marker  # the top-scored chunk is line 2
    assert "maximum" in marker and "count" in marker
    assert "`retrieve_all_context`" in marker and "max_chunks" in marker and "pattern" in marker
    assert "[ref: " in marker


@pytest.mark.asyncio
async def test_disclaimer_without_the_tool_says_not_to_aggregate_the_excerpt() -> None:
    plugin = _plugin(FakeReranker(), tool=False)
    _, marker = await _filtered(plugin)

    assert "EXCERPT" in marker
    assert "Say that the result was filtered" in marker
    assert "retrieve_all_context" not in marker


# --------------------------------------------------------------------------------------------------
# Budgeted retrieval.
# --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_chunks_returns_the_most_relevant_chunks_in_document_order() -> None:
    plugin = _plugin(FakeReranker(scores=[0.1, 0.9, 0.2, 0.3, 0.4, 0.8]))
    agent, _ = await _filtered(plugin)

    out = await plugin.retrieve_all_context(
        reference="mem_1_t1_0", tool_context=_ctx(agent), max_chunks=2, max_tokens=1000
    )

    assert "2 of 6 chunks (most relevant first)" in out
    assert _ROWS[1] in out and _ROWS[5] in out
    assert _ROWS[0] not in out and _ROWS[3] not in out
    assert out.index(_ROWS[1]) < out.index(_ROWS[5])  # document order
    assert "lines omitted" in out


@pytest.mark.asyncio
async def test_max_chunks_and_max_tokens_large_enough_return_everything() -> None:
    plugin = _plugin(FakeReranker())
    agent, _ = await _filtered(plugin)

    out = await plugin.retrieve_all_context(
        reference="mem_1_t1_0", tool_context=_ctx(agent), max_chunks=1000, max_tokens=10_000
    )

    assert all(row in out for row in _ROWS)
    assert "lines omitted" not in out


@pytest.mark.asyncio
async def test_max_tokens_lifts_the_default_cap_on_a_pattern_read() -> None:
    plugin = _plugin(FakeReranker())
    agent, _ = await _filtered(plugin)

    capped = await plugin.retrieve_all_context(
        reference="mem_1_t1_0", tool_context=_ctx(agent), pattern="row", context_lines=0
    )
    full = await plugin.retrieve_all_context(
        reference="mem_1_t1_0", tool_context=_ctx(agent), pattern="row", context_lines=0, max_tokens=10_000
    )

    assert not all(row in capped for row in _ROWS)  # max_result_tokens=10 -> ~40 chars
    assert all(row in full for row in _ROWS)


@pytest.mark.asyncio
async def test_max_tokens_alone_reads_the_head_within_the_budget() -> None:
    plugin = _plugin(FakeReranker())
    agent, _ = await _filtered(plugin)

    out = await plugin.retrieve_all_context(reference="mem_1_t1_0", tool_context=_ctx(agent), max_tokens=20)

    assert _ROWS[0] in out
    assert _ROWS[5] not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["max_chunks", "max_tokens"])
@pytest.mark.parametrize("bad", [0, -1, True])
async def test_budget_below_one_is_rejected(field: str, bad: object) -> None:
    plugin = _plugin(FakeReranker())
    agent, _ = await _filtered(plugin)

    with pytest.raises(ValueError):
        await plugin.retrieve_all_context(reference="mem_1_t1_0", tool_context=_ctx(agent), **{field: bad})


@pytest.mark.asyncio
async def test_max_chunks_without_a_ranking_reads_in_document_order() -> None:
    plugin = RelevanceFilter(store=InMemoryStore(), include_retrieval_tool=True, config={"chunk_tokens": 10})
    agent = FakeAgent()
    plugin.init_agent(agent)  # type: ignore[arg-type]
    reference = await plugin._store.store("x_0", _TEXT.encode(), "text/plain")

    out = await plugin.retrieve_all_context(
        reference=reference, tool_context=_ctx(agent), max_chunks=2, max_tokens=1000
    )

    assert "(document order)" in out
    assert _ROWS[0] in out and _ROWS[1] in out and _ROWS[2] not in out


# --------------------------------------------------------------------------------------------------
# End-of-turn cleanup.
# --------------------------------------------------------------------------------------------------


def _use(tool_id: str, name: str) -> dict:
    return {"toolUse": {"toolUseId": tool_id, "name": name, "input": {}}}


def _res(tool_id: str) -> dict:
    return {"toolResult": {"toolUseId": tool_id, "status": "success", "content": [{"text": "x" * 100}]}}


async def _end_turn(plugin: RelevanceFilter, messages: list[dict]) -> list[dict]:
    agent = FakeAgent(messages=messages)
    plugin.init_agent(agent)  # type: ignore[arg-type]
    await plugin._on_after_invocation(AfterInvocationEvent(agent=agent))  # type: ignore[arg-type]
    return agent.messages


@pytest.mark.asyncio
async def test_a_retrieval_exchange_is_removed_at_the_end_of_the_turn() -> None:
    question = {"role": "user", "content": [{"text": "largest refund?"}]}
    call = {"role": "assistant", "content": [_use("s", "statement")]}
    result = {"role": "user", "content": [_res("s")]}
    retrieve = {"role": "assistant", "content": [{"text": "reading the rest"}, _use("r", "retrieve_all_context")]}
    retrieved = {"role": "user", "content": [_res("r")]}
    answer = {"role": "assistant", "content": [{"text": "R$ 911,35 on 2026-07-01"}]}
    messages = [question, call, result, retrieve, retrieved, answer]

    kept = await _end_turn(_plugin(FakeReranker()), messages)

    assert kept == [question, call, result, answer]
    assert kept is messages  # edited in place: agent.messages is the list other plugins hold


@pytest.mark.asyncio
async def test_a_mixed_call_keeps_the_other_tools_and_drops_reasoning() -> None:
    reasoning = {"reasoningContent": {"reasoningText": {"text": "t"}}}
    mixed = {"role": "assistant", "content": [reasoning, _use("r", "retrieve_all_context"), _use("s", "statement")]}
    results = {"role": "user", "content": [_res("r"), _res("s")]}
    answer = {"role": "assistant", "content": [{"text": "done"}]}
    question = {"role": "user", "content": [{"text": "q"}]}

    kept = await _end_turn(_plugin(FakeReranker()), [question, mixed, results, answer])

    assert kept[1]["content"] == [_use("s", "statement")]
    assert kept[2]["content"] == [_res("s")]
    assert kept[3] is answer


@pytest.mark.asyncio
async def test_an_unanswered_retrieval_call_is_left_alone() -> None:
    question = {"role": "user", "content": [{"text": "q"}]}
    messages = [question, {"role": "assistant", "content": [_use("r", "retrieve_all_context")]}]
    snapshot = [dict(m) for m in messages]

    kept = await _end_turn(_plugin(FakeReranker()), messages)

    assert kept == snapshot


@pytest.mark.asyncio
async def test_cleanup_is_a_no_op_with_the_tool_off() -> None:
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        {"role": "assistant", "content": [_use("r", "retrieve_all_context")]},
        {"role": "user", "content": [_res("r")]},
        {"role": "assistant", "content": [{"text": "a"}]},
    ]

    kept = await _end_turn(_plugin(FakeReranker(), tool=False), list(messages))

    assert kept == messages
