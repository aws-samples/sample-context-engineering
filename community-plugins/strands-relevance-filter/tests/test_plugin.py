"""Tests for :class:`strands_relevance_filter.plugin.RelevanceFilter`.

Covers construction validation and inertness, each of the guards in ``_on_after_tool_call``, the
happy-path rewrite, the ``retrieve_context`` read modes, and the ``include_retrieval_tool`` toggle.
The store is a real ``InMemoryStore`` (or a ``tmp_path`` ``FileStore``); only the reranker and the
agent/model are faked, through ``conftest``.
"""

from __future__ import annotations

import pytest
from strands.types.tools import ToolContext, ToolResult, ToolUse

from strands_relevance_filter import InMemoryStore, RelevanceFilter
from strands_relevance_filter.reranker import RerankerError

from conftest import FakeAgent, FakeReranker, make_after_tool_call_event


# --------------------------------------------------------------------------------------------------
# Helpers local to this file.
# --------------------------------------------------------------------------------------------------


def _tool_use(name: str = "search", tool_use_id: str = "tool-1", **inp) -> ToolUse:
    return ToolUse(toolUseId=tool_use_id, name=name, input=inp or {"q": "hello"})


def _result(content: list[dict], *, status: str = "success", tool_use_id: str = "tool-1") -> ToolResult:
    return ToolResult(toolUseId=tool_use_id, status=status, content=content)


def _config(reranker: FakeReranker, **overrides) -> dict:
    """A preview config small enough that any multi-line text is chunked and filtered."""
    cfg = {"reranker": reranker, "chunk_tokens": 1, "preview_tokens": 1, "relevance_threshold": 0.0}
    cfg.update(overrides)
    return cfg


async def _init(plugin: RelevanceFilter, agent: FakeAgent) -> FakeAgent:
    """Run the plugin's ``init_agent`` (resolves the store, applies the tool toggle)."""
    plugin.init_agent(agent)  # type: ignore[arg-type]  # FakeAgent stands in for Agent here.
    return agent


def _tool_context(agent: FakeAgent, tool_use: ToolUse) -> ToolContext:
    """A minimal ToolContext for calling ``retrieve_context`` directly; the tool reads only self."""
    return ToolContext(tool_use=tool_use, agent=agent, invocation_state={})  # type: ignore[arg-type]


# A user turn so ``_build_query`` finds a non-empty question (BedrockReranker would reject empty).
_MESSAGES = [{"role": "user", "content": [{"text": "what is the error"}]}]


# --------------------------------------------------------------------------------------------------
# Construction.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -8000])
def test_construction_rejects_non_positive_max_result_tokens(bad: int) -> None:
    with pytest.raises(ValueError):
        RelevanceFilter(max_result_tokens=bad)


def test_construction_does_no_io() -> None:
    """No reranker and no preview builder are constructed until a result is actually filtered."""
    reranker = FakeReranker()
    plugin = RelevanceFilter(config={"reranker": reranker})

    # The preview builder is unbuilt (would otherwise hold the reranker), and the reranker is untouched.
    assert plugin._preview is None
    assert reranker.calls == []


def test_construction_defaults_store_to_none_until_init() -> None:
    """The store is resolved in init_agent, not in __init__."""
    plugin = RelevanceFilter()
    assert plugin._store is None


# --------------------------------------------------------------------------------------------------
# Guards in _on_after_tool_call: each leaves event.result UNTOUCHED (by identity).
# --------------------------------------------------------------------------------------------------


async def _plugin_over_threshold(reranker: FakeReranker, **kwargs) -> RelevanceFilter:
    """A plugin whose size gate a large token_count will trip, with a small preview config."""
    return RelevanceFilter(store=InMemoryStore(), max_result_tokens=10, config=_config(reranker), **kwargs)


@pytest.mark.asyncio
async def test_guard_cancelled_call_leaves_result_untouched() -> None:
    reranker = FakeReranker()
    plugin = await _plugin_over_threshold(reranker)
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "x" * 5000}])
    event = make_after_tool_call_event(
        tool_use=_tool_use(), result=result, agent=agent, cancel_message="user cancelled"
    )

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is original
    assert reranker.calls == []  # never scored


@pytest.mark.asyncio
async def test_guard_own_retrieve_context_result_is_not_refiltered() -> None:
    reranker = FakeReranker()
    plugin = await _plugin_over_threshold(reranker)
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    retrieval_name = plugin.retrieve_context.tool_name
    result = _result([{"text": "x" * 5000}])
    event = make_after_tool_call_event(tool_use=_tool_use(name=retrieval_name), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is original
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_guard_result_at_or_below_threshold_is_untouched() -> None:
    reranker = FakeReranker()
    plugin = RelevanceFilter(store=InMemoryStore(), max_result_tokens=10, config=_config(reranker))
    # token_count == max_result_tokens: the gate is <=, so equal is under.
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=10))
    result = _result([{"text": "some long text\n" * 100}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is original
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_guard_should_filter_false_leaves_result_untouched() -> None:
    reranker = FakeReranker()
    plugin = RelevanceFilter(
        store=InMemoryStore(),
        max_result_tokens=10,
        config=_config(reranker),
        should_filter=lambda tool_name, token_count, **kw: False,
    )
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "x" * 5000}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is original
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_guard_no_text_or_json_subblock_leaves_result_untouched() -> None:
    reranker = FakeReranker()
    plugin = await _plugin_over_threshold(reranker)
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    # An image-only result carries no scorable text/json sub-block, and an empty text block adds nothing.
    result = _result([{"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}, {"text": ""}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is original
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_should_filter_that_raises_fails_open_and_filters() -> None:
    """A should_filter callback that raises must not stop the filter: the result is still rewritten."""

    def boom(tool_name: str, token_count: int, **kw) -> bool:
        raise RuntimeError("callback bug")

    reranker = FakeReranker()
    plugin = RelevanceFilter(
        store=InMemoryStore(), max_result_tokens=10, config=_config(reranker), should_filter=boom
    )
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "line one\nline two\nline three\n" * 50}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is not original  # filtered anyway
    assert reranker.calls  # scoring did happen
    assert "[Relevance:" in event.result["content"][0]["text"]


# --------------------------------------------------------------------------------------------------
# Happy path: an oversized text result is stored and rewritten.
# --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_rewrites_result_and_stores_content() -> None:
    reranker = FakeReranker()
    store = InMemoryStore()
    plugin = RelevanceFilter(
        store=store, max_result_tokens=10, config=_config(reranker), include_retrieval_tool=True
    )
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    body = "alpha line\nbeta line\ngamma line\n" * 40
    result = _result([{"text": body}], tool_use_id="tool-9")
    event = make_after_tool_call_event(tool_use=_tool_use(tool_use_id="tool-9"), result=result, agent=agent)

    original = event.result
    await plugin._on_after_tool_call(event)

    assert event.result is not original
    marker_text = event.result["content"][0]["text"]
    assert marker_text.startswith("[Relevance: tool result, ~100,000 tokens]")
    assert "[ref: " in marker_text  # single stored block -> singular ref token
    # status and toolUseId are carried over from the original result.
    assert event.result["status"] == "success"
    assert event.result["toolUseId"] == "tool-9"

    # The stored content round-trips verbatim through the reference embedded in the marker.
    reference = marker_text.split("[ref: ", 1)[1].split("]", 1)[0]
    stored_bytes, content_type = await store.retrieve(reference)
    assert stored_bytes.decode("utf-8") == body
    assert content_type == "text/plain"


@pytest.mark.asyncio
async def test_happy_path_preserves_nontextual_subblocks_verbatim_after_marker() -> None:
    reranker = FakeReranker()
    plugin = RelevanceFilter(store=InMemoryStore(), max_result_tokens=10, config=_config(reranker))
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    image_block = {"image": {"format": "png", "source": {"bytes": b"\x89PNGDATA"}}}
    body = "row a\nrow b\nrow c\n" * 40
    result = _result([{"text": body}, image_block])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    await plugin._on_after_tool_call(event)

    new_content = event.result["content"]
    assert "[Relevance:" in new_content[0]["text"]  # filtered text first
    assert new_content[1] == image_block  # non-textual block survives verbatim, in order
    assert new_content[1] is image_block  # by identity: the block was not copied/mutated


@pytest.mark.asyncio
async def test_happy_path_multiple_text_blocks_use_plural_refs_token() -> None:
    reranker = FakeReranker()
    plugin = RelevanceFilter(
        store=InMemoryStore(), max_result_tokens=10, config=_config(reranker), include_retrieval_tool=True
    )
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "first block\n" * 30}, {"text": "second block\n" * 30}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    await plugin._on_after_tool_call(event)

    assert "[refs: " in event.result["content"][0]["text"]


@pytest.mark.asyncio
async def test_default_mode_emits_no_reference_token_and_stores_nothing() -> None:
    """With the retrieval tool off, nothing could resolve a reference, so none is promised."""
    reranker = FakeReranker()
    store = InMemoryStore()
    plugin = RelevanceFilter(store=store, max_result_tokens=10, config=_config(reranker))
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "alpha line\nbeta line\n" * 40}], tool_use_id="tool-7")
    event = make_after_tool_call_event(tool_use=_tool_use(tool_use_id="tool-7"), result=result, agent=agent)

    await plugin._on_after_tool_call(event)

    marker_text = event.result["content"][0]["text"]
    # The result is still filtered -- only the reference token is gone.
    assert marker_text.startswith("[Relevance: tool result, ~100,000 tokens]")
    assert "[ref: " not in marker_text
    assert "[refs: " not in marker_text
    with pytest.raises(Exception):
        await store.retrieve("tool-7_0")


@pytest.mark.asyncio
async def test_filtering_runs_with_no_store_at_all() -> None:
    """Storage is optional: with no store and the tool off, chunking + rerank + rewrite still run."""
    reranker = FakeReranker()
    plugin = RelevanceFilter(max_result_tokens=10, config=_config(reranker))
    agent = await _init(plugin, FakeAgent(messages=_MESSAGES, token_count=100_000))
    result = _result([{"text": "alpha line\nbeta line\n" * 40}])
    event = make_after_tool_call_event(tool_use=_tool_use(), result=result, agent=agent)

    await plugin._on_after_tool_call(event)

    assert plugin._store is None  # no default store is built when nothing can read it back
    assert reranker.calls  # scored
    assert event.result["content"][0]["text"].startswith("[Relevance: tool result, ~100,000 tokens]")


# --------------------------------------------------------------------------------------------------
# retrieve_context read modes.
# --------------------------------------------------------------------------------------------------


async def _store_text(plugin: RelevanceFilter, text: str, *, key: str = "tool-1_0") -> str:
    return await plugin._store.store(key, text.encode("utf-8"), "text/plain")


@pytest.mark.asyncio
async def test_retrieve_context_full_read_returns_stored_text() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    agent = await _init(plugin, FakeAgent())
    text = "one\ntwo\nthree\nfour\nfive"
    reference = await _store_text(plugin, text)

    out = await plugin.retrieve_context(reference=reference, tool_context=_tool_context(agent, _tool_use()))

    assert out == text


@pytest.mark.asyncio
async def test_retrieve_context_line_range_read() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    agent = await _init(plugin, FakeAgent())
    reference = await _store_text(plugin, "one\ntwo\nthree\nfour\nfive")

    out = await plugin.retrieve_context(
        reference=reference, tool_context=_tool_context(agent, _tool_use()), line_range={"start": 2, "end": 3}
    )

    assert "two" in out and "three" in out
    assert "one" not in out and "five" not in out


@pytest.mark.asyncio
async def test_retrieve_context_pattern_read() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    agent = await _init(plugin, FakeAgent())
    reference = await _store_text(plugin, "alpha\nERROR here\nbeta\ngamma")

    out = await plugin.retrieve_context(
        reference=reference, tool_context=_tool_context(agent, _tool_use()), pattern="ERROR", context_lines=0
    )

    assert "ERROR here" in out
    assert "match" in out.lower()


@pytest.mark.asyncio
async def test_retrieve_context_unknown_reference_raises_value_error() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    agent = await _init(plugin, FakeAgent())

    with pytest.raises(ValueError):
        await plugin.retrieve_context(reference="mem_999_missing", tool_context=_tool_context(agent, _tool_use()))


@pytest.mark.asyncio
async def test_retrieve_context_pattern_on_binary_content_raises_value_error() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    agent = await _init(plugin, FakeAgent())
    reference = await plugin._store.store("bin_0", b"\x00\x01\x02binary", "application/octet-stream")

    with pytest.raises(ValueError):
        await plugin.retrieve_context(
            reference=reference, tool_context=_tool_context(agent, _tool_use()), pattern="anything"
        )


# --------------------------------------------------------------------------------------------------
# include_retrieval_tool toggle.
# --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_retrieval_tool_false_drops_tool_but_keeps_hook() -> None:
    plugin = RelevanceFilter(store=InMemoryStore(), include_retrieval_tool=False)
    await _init(plugin, FakeAgent())

    retrieval_name = plugin.retrieve_context.tool_name
    assert all(t.tool_name != retrieval_name for t in plugin.tools)
    # The hook is unaffected: the plugin still subscribes to AfterToolCallEvent.
    assert len(plugin.hooks) == 1


@pytest.mark.asyncio
async def test_include_retrieval_tool_default_drops_tool() -> None:
    plugin = RelevanceFilter(store=InMemoryStore())
    await _init(plugin, FakeAgent())

    retrieval_name = plugin.retrieve_context.tool_name
    assert all(t.tool_name != retrieval_name for t in plugin.tools)


@pytest.mark.asyncio
async def test_include_retrieval_tool_true_keeps_tool() -> None:
    plugin = RelevanceFilter(store=InMemoryStore(), include_retrieval_tool=True)
    await _init(plugin, FakeAgent())

    retrieval_name = plugin.retrieve_context.tool_name
    assert any(t.tool_name == retrieval_name for t in plugin.tools)
