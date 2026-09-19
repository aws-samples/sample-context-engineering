"""Unit tests for the plugin's own reference store.

The assertions are about the contract and the default store alone: a reference in, a decoded block out, a miss that is
``None`` rather than an exception, and an empty store when nothing offloads. The resolution order and the optional
``ContextManager`` Stash bridge belong to ``tools.expand_artifact`` and are asserted there.
"""

from collections.abc import Callable

import pytest

from strands_context_graph import store as store_module
from strands_context_graph.store import (
    InMemoryReferenceStore,
    ReferenceStore,
    absent_message,
    estimate_tokens,
    non_textual_message,
    read_artifact,
    record_references,
    resolve_artifact,
    unknown_message,
)


class _ExplodingStore:
    """A store whose write always fails, for the no-propagation contract."""

    async def retrieve(self, reference: str) -> object | None:
        return None

    def put(self, reference: str, block: object) -> None:
        raise RuntimeError("write refused")


class _ForeignStore:
    """A conforming store with no notion of a name with nothing behind it."""

    def __init__(self) -> None:
        self.blocks: dict[str, object] = {}

    async def retrieve(self, reference: str) -> object | None:
        return self.blocks.get(reference)

    def put(self, reference: str, block: object) -> None:
        self.blocks[reference] = block


def test_default_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryReferenceStore(), ReferenceStore)


def test_fresh_store_knows_nothing() -> None:
    store = InMemoryReferenceStore()

    assert len(store) == 0
    assert store.references() == frozenset()
    assert "mem_1_tu-3_0" not in store


@pytest.mark.asyncio
async def test_put_then_retrieve_returns_the_decoded_block() -> None:
    store = InMemoryReferenceStore()

    store.put("mem_1_tu-3_0", "R$ 1.200,00 in March")

    assert await store.retrieve("mem_1_tu-3_0") == "R$ 1.200,00 in March"
    assert "mem_1_tu-3_0" in store


@pytest.mark.asyncio
async def test_unknown_reference_is_none_not_an_exception() -> None:
    store = InMemoryReferenceStore()

    assert await store.retrieve("never-stored") is None


@pytest.mark.asyncio
async def test_re_putting_a_reference_replaces_the_block() -> None:
    store = InMemoryReferenceStore()

    store.put("ref-1", "first")
    store.put("ref-1", "second")

    assert await store.retrieve("ref-1") == "second"
    assert len(store) == 1


@pytest.mark.asyncio
async def test_noted_reference_is_known_with_no_block() -> None:
    store = InMemoryReferenceStore()

    store.note("ref-1")

    assert "ref-1" in store
    assert await store.retrieve("ref-1") is None


@pytest.mark.asyncio
async def test_note_does_not_erase_a_recorded_block() -> None:
    store = InMemoryReferenceStore()
    store.put("ref-1", "content")

    store.note("ref-1")

    assert await store.retrieve("ref-1") == "content"


@pytest.mark.asyncio
async def test_a_non_textual_block_is_stored_as_given() -> None:
    store = InMemoryReferenceStore()
    block = {"image": {"format": "png"}}

    store.put("ref-1", block)

    assert await store.retrieve("ref-1") is block


@pytest.mark.asyncio
async def test_record_references_pairs_blocks_positionally() -> None:
    store = InMemoryReferenceStore()

    recorded = record_references(store, ["ref-1", "ref-2"], ["first", "second"])

    assert recorded == ("ref-1", "ref-2")
    assert await store.retrieve("ref-1") == "first"
    assert await store.retrieve("ref-2") == "second"


@pytest.mark.asyncio
async def test_record_references_without_blocks_notes_the_names() -> None:
    store = InMemoryReferenceStore()

    recorded = record_references(store, ["ref-1", "ref-2"])

    assert recorded == ("ref-1", "ref-2")
    assert store.references() == frozenset({"ref-1", "ref-2"})
    assert await store.retrieve("ref-1") is None


@pytest.mark.asyncio
async def test_mismatched_counts_record_names_rather_than_guess_a_pairing() -> None:
    store = InMemoryReferenceStore()

    recorded = record_references(store, ["ref-1", "ref-2"], ["only-one"])

    assert recorded == ("ref-1", "ref-2")
    assert await store.retrieve("ref-1") is None
    assert await store.retrieve("ref-2") is None


def test_record_references_deduplicates_keeping_order() -> None:
    store = InMemoryReferenceStore()

    recorded = record_references(store, ["ref-2", "ref-1", "ref-2", ""])

    assert recorded == ("ref-2", "ref-1")


def test_nothing_offloaded_records_nothing_and_does_not_raise() -> None:
    store = InMemoryReferenceStore()

    assert record_references(store, []) == ()
    assert len(store) == 0


def test_a_failing_store_write_does_not_propagate() -> None:
    recorded = record_references(_ExplodingStore(), ["ref-1"], ["block"])

    assert recorded == ()


def test_a_foreign_store_gets_blocks_but_no_bare_names() -> None:
    store = _ForeignStore()

    assert record_references(store, ["ref-1"], ["block"]) == ("ref-1",)
    assert record_references(store, ["ref-2"]) == ()
    assert store.blocks == {"ref-1": "block"}


# ---- resolution order and the optional Stash bridge --------------------------------------------


class _Stash:
    """A stand-in for a ContextManager's Stash: one async read, nothing else."""

    def __init__(self, blocks: dict[str, object] | None = None) -> None:
        self.blocks = blocks or {}

    async def retrieve(self, reference: str) -> object | None:
        return self.blocks.get(reference)


class _FakeContextManager:
    """A stand-in for the SDK's ContextManager, carrying only the attribute the bridge reads."""

    def __init__(self, stash: object | None) -> None:
        self._stash = stash


class _Registry:
    def __init__(self, plugins: object) -> None:
        self._plugins = plugins


class _Agent:
    def __init__(self, plugins: object = None) -> None:
        if plugins is not None:
            self._plugin_registry = _Registry(plugins)


def _with_manager(monkeypatch: pytest.MonkeyPatch, manager: object) -> _Agent:
    """An agent whose registry carries ``manager``, with the bridge's anchor type pointed at the fake."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._CONTEXT_MANAGER_SYMBOL: _FakeContextManager}),
    )
    return _Agent({"cm": manager})


def _symbol_table(overrides: dict[tuple[str, str], object]) -> Callable[[str, str], object | None]:
    """A ``_optional_symbol`` replacement answering from ``overrides`` and ``None`` for everything else."""

    def _lookup(module: str, name: str) -> object | None:
        return overrides.get((module, name))

    return _lookup


@pytest.mark.asyncio
async def test_own_store_hit_resolves_without_consulting_any_bridge() -> None:
    store = InMemoryReferenceStore()
    store.put("ref-1", "R$ 1.200,00 in March")

    resolved = await resolve_artifact(store, _Agent(), "ref-1")

    assert resolved.outcome == "text"
    assert resolved.source == "own"
    assert resolved.text == "R$ 1.200,00 in March"


@pytest.mark.asyncio
async def test_own_store_text_read_whole_is_verbatim() -> None:
    store = InMemoryReferenceStore()
    text = "line one\nR$ 1.500,00\nline three"
    store.put("ref-1", text)

    resolved = await resolve_artifact(store, _Agent(), "ref-1")

    assert resolved.text is not None
    assert read_artifact(resolved.text) == text


@pytest.mark.asyncio
async def test_own_miss_with_a_registered_manager_resolves_against_the_stash(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _with_manager(monkeypatch, _FakeContextManager(_Stash({"ref-1": "stashed text"})))

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "text"
    assert resolved.source == "stash"
    assert resolved.text == "stashed text"


@pytest.mark.asyncio
async def test_own_miss_and_stash_miss_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _with_manager(monkeypatch, _FakeContextManager(_Stash()))

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "unknown"
    assert "ref-1" in unknown_message("ref-1")


@pytest.mark.asyncio
async def test_a_noted_reference_falls_through_to_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryReferenceStore()
    store.note("ref-1")
    agent = _with_manager(monkeypatch, _FakeContextManager(_Stash({"ref-1": "stashed text"})))

    resolved = await resolve_artifact(store, agent, "ref-1")

    assert resolved.source == "stash"


@pytest.mark.asyncio
async def test_no_manager_at_all_is_absent() -> None:
    resolved = await resolve_artifact(InMemoryReferenceStore(), _Agent(), "ref-1")

    assert resolved.outcome == "absent"
    assert "ref-1" in absent_message("ref-1")


@pytest.mark.asyncio
async def test_a_manager_with_stash_false_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _with_manager(monkeypatch, _FakeContextManager(None))

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "absent"


@pytest.mark.asyncio
async def test_an_unreadable_registry_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._CONTEXT_MANAGER_SYMBOL: _FakeContextManager}),
    )

    resolved = await resolve_artifact(InMemoryReferenceStore(), _Agent(["not-a-dict"]), "ref-1")

    assert resolved.outcome == "absent"


@pytest.mark.asyncio
async def test_a_renamed_context_manager_degrades_to_no_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "_optional_symbol", _symbol_table({}))
    agent = _Agent({"cm": _FakeContextManager(_Stash({"ref-1": "stashed text"}))})

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "absent"


@pytest.mark.asyncio
async def test_a_failing_stash_read_does_not_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingStash:
        async def retrieve(self, reference: str) -> object | None:
            raise RuntimeError("backend unreachable")

    agent = _with_manager(monkeypatch, _FakeContextManager(_ExplodingStash()))

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "unknown"


@pytest.mark.asyncio
async def test_a_failing_own_store_read_falls_through_to_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingRead:
        async def retrieve(self, reference: str) -> object | None:
            raise RuntimeError("read refused")

        def put(self, reference: str, block: object) -> None:
            return None

    agent = _with_manager(monkeypatch, _FakeContextManager(_Stash({"ref-1": "stashed text"})))

    resolved = await resolve_artifact(_ExplodingRead(), agent, "ref-1")

    assert resolved.source == "stash"


@pytest.mark.asyncio
async def test_a_non_textual_own_store_block_is_reported_without_a_media_type() -> None:
    store = InMemoryReferenceStore()
    store.put("ref-1", {"image": {"format": "png", "source": {"bytes": b"..."}}})

    resolved = await resolve_artifact(store, _Agent(), "ref-1")

    assert resolved.outcome == "non_textual"
    message = non_textual_message("ref-1")
    assert "ref-1" in message
    for media_type in ("png", "image/", "mime", "application/"):
        assert media_type not in message


@pytest.mark.asyncio
async def test_a_non_textual_stashed_block_follows_the_same_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _with_manager(monkeypatch, _FakeContextManager(_Stash({"ref-1": {"image": {"format": "png"}}})))

    resolved = await resolve_artifact(InMemoryReferenceStore(), agent, "ref-1")

    assert resolved.outcome == "non_textual"
    assert resolved.source == "stash"


def test_a_targeted_read_is_bounded_by_the_shared_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    assert store_module._MAX_RESULT_TOKENS == 10_000

    calls: list[dict[str, object]] = []

    def _search(text: str, **kwargs: object) -> str:
        calls.append(kwargs)
        return "matched line"

    monkeypatch.setattr(store_module, "_optional_symbol", _symbol_table({store_module._SEARCH_CONTENT_SYMBOL: _search}))

    answer = read_artifact("a\nb\nc", line_range=(1, 2), pattern="b")

    assert answer == "matched line"
    assert calls[0]["max_chars"] == 10_000 * 4
    assert calls[0]["line_range"] == (1, 2)
    assert calls[0]["pattern"] == "b"


def test_a_missing_search_helper_degrades_to_a_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "_optional_symbol", _symbol_table({}))

    with pytest.raises(ValueError, match="targeted reads are unavailable"):
        read_artifact("a\nb", pattern="b")


def test_a_whole_read_needs_no_search_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "_optional_symbol", _symbol_table({}))

    assert read_artifact("a\nb") == "a\nb"


def test_token_estimate_never_reports_nothing() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 100
