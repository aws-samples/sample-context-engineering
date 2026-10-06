"""Routing tests for the standalone-vs-integrated seam of ``resolve_artifact``.

``test_store.py`` already asserts each outcome of the resolution order one at a time. What is asserted here is what no
single-outcome test can state: the whole routing table at once, the **order** the two stores are consulted in, and what
the model is handed on each branch, word for word.

Three guarantees:

- **The table.** Own-store hit, own miss with a Stash hit, and both missing are the three outcomes of the seam
  (Requirements 15.2, 15.3, 15.4), and the prose for a miss is asserted as the exact string, not as a substring, because
  the miss message *is* the answer the model reads.
- **The order.** The own store is consulted first and the bridge is consulted only after it misses — asserted through a
  journal shared by both stores, and again through a bridge that raises when it is touched at all.
- **No model call, on any path.** Resolution is dictionary work. The agent stub trips on every attribute a model call
  would go through, so reaching for one fails the test rather than quietly costing a token.
"""

from collections.abc import Callable

import pytest

from strands_context_graph import store as store_module
from strands_context_graph.store import (
    InMemoryReferenceStore,
    absent_message,
    read_artifact,
    resolve_artifact,
    unknown_message,
)

REFERENCE = "mem_1_tu-3_0"
"""One reference, shaped like a real placeholder, used by every case below."""

CONTENT = "line one\nR$ 1.200,00 in March\nline three"

ABSENT_PROSE = (
    "expand_artifact | no artifact storage holds reference 'mem_1_tu-3_0' on this agent | nothing was ever offloaded "
    "under that reference, which means the full results are already in the conversation"
)
"""The exact answer for "nothing beyond the own store was there to ask" (Requirement 15.4)."""

UNKNOWN_PROSE = (
    "expand_artifact | unknown reference 'mem_1_tu-3_0' | copy a reference exactly as it was shown to you in a turn's "
    "title or preview"
)
"""The exact answer for "storage was asked and did not hold it"."""

_MODEL_ATTRS = frozenset(
    {"model", "invoke_async", "stream_async", "structured_output", "structured_output_async", "converse"}
)
"""Every attribute a model call on an agent would travel through. Reaching for one during resolution is a bug."""


class _Journal:
    """The order the stores were consulted in, shared by the own store and the bridge."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def record(self, source: str, reference: str) -> None:
        self.entries.append(f"{source}:{reference}")


class _RecordingStore:
    """The plugin's own store, conforming to the protocol and writing every read into the journal."""

    def __init__(self, journal: _Journal, blocks: dict[str, object] | None = None) -> None:
        self._journal = journal
        self.blocks: dict[str, object] = dict(blocks or {})

    async def retrieve(self, reference: str) -> object | None:
        self._journal.record("own", reference)
        return self.blocks.get(reference)

    def put(self, reference: str, block: object) -> None:
        self.blocks[reference] = block


class _RecordingStash:
    """A ContextManager's Stash: one async read, written into the same journal."""

    def __init__(self, journal: _Journal, blocks: dict[str, object] | None = None) -> None:
        self._journal = journal
        self.blocks: dict[str, object] = dict(blocks or {})

    async def retrieve(self, reference: str) -> object | None:
        self._journal.record("stash", reference)
        return self.blocks.get(reference)


class _UntouchableStash:
    """A Stash that fails the test if it is read at all."""

    async def retrieve(self, reference: str) -> object | None:
        raise AssertionError(f"the bridge was consulted for {reference!r} after an own-store hit")


class _FakeContextManager:
    """A stand-in for the SDK's ContextManager, carrying only the attribute the bridge reads."""

    def __init__(self, stash: object | None) -> None:
        self._stash = stash


class _Registry:
    def __init__(self, plugins: object) -> None:
        self._plugins = plugins


class _TripwireAgent:
    """An agent carrying a plugin registry and nothing a model call could travel through.

    Any attribute in :data:`_MODEL_ATTRS` raises ``AssertionError`` rather than ``AttributeError``, so it survives the
    ``getattr(agent, name, None)`` idiom and surfaces as a failed test.
    """

    def __init__(self, plugins: object | None = None) -> None:
        if plugins is not None:
            self._plugin_registry = _Registry(plugins)

    def __getattr__(self, name: str) -> object:
        if name in _MODEL_ATTRS:
            raise AssertionError(f"resolution reached the model through {name!r}")
        raise AttributeError(name)


def _symbol_table(overrides: dict[tuple[str, str], object]) -> Callable[[str, str], object | None]:
    """A ``_optional_symbol`` replacement answering from ``overrides`` and ``None`` for everything else."""

    def _lookup(module: str, name: str) -> object | None:
        return overrides.get((module, name))

    return _lookup


@pytest.fixture
def journal() -> _Journal:
    return _Journal()


@pytest.fixture
def bridged(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], _TripwireAgent]:
    """Build an agent whose registry carries a ContextManager holding ``stash``, with the anchor type faked."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._CONTEXT_MANAGER_SYMBOL: _FakeContextManager}),
    )

    def _agent(stash: object) -> _TripwireAgent:
        return _TripwireAgent({"cm": _FakeContextManager(stash)})

    return _agent


@pytest.mark.asyncio
async def test_the_own_store_answers_first_and_the_bridge_is_never_reached(
    journal: _Journal, bridged: Callable[[object], _TripwireAgent]
) -> None:
    store = _RecordingStore(journal, {REFERENCE: CONTENT})

    resolved = await resolve_artifact(store, bridged(_UntouchableStash()), REFERENCE)

    assert resolved.outcome == "text"
    assert resolved.source == "own"
    assert resolved.text is not None
    assert read_artifact(resolved.text) == CONTENT
    assert journal.entries == [f"own:{REFERENCE}"]


@pytest.mark.asyncio
async def test_an_own_miss_reaches_the_bridge_second_and_only_second(
    journal: _Journal, bridged: Callable[[object], _TripwireAgent]
) -> None:
    stash = _RecordingStash(journal, {REFERENCE: "stashed text"})

    resolved = await resolve_artifact(_RecordingStore(journal), bridged(stash), REFERENCE)

    assert resolved.outcome == "text"
    assert resolved.source == "stash"
    assert journal.entries == [f"own:{REFERENCE}", f"stash:{REFERENCE}"]


@pytest.mark.asyncio
async def test_both_stores_missing_hands_the_model_the_unknown_prose_verbatim(
    journal: _Journal, bridged: Callable[[object], _TripwireAgent]
) -> None:
    resolved = await resolve_artifact(_RecordingStore(journal), bridged(_RecordingStash(journal)), REFERENCE)

    assert resolved.outcome == "unknown"
    assert resolved.text is None
    assert resolved.source is None
    assert unknown_message(REFERENCE) == UNKNOWN_PROSE
    assert journal.entries == [f"own:{REFERENCE}", f"stash:{REFERENCE}"]


@pytest.mark.asyncio
async def test_a_standalone_agent_hands_the_model_the_absence_prose_verbatim(journal: _Journal) -> None:
    resolved = await resolve_artifact(_RecordingStore(journal), _TripwireAgent(), REFERENCE)

    assert resolved.outcome == "absent"
    assert resolved.text is None
    assert resolved.source is None
    assert absent_message(REFERENCE) == ABSENT_PROSE
    assert journal.entries == [f"own:{REFERENCE}"]


def test_the_two_miss_messages_name_the_reference_and_differ() -> None:
    """Both misses name the reference back; they are worded apart because only one says storage was never there."""
    assert REFERENCE in ABSENT_PROSE
    assert REFERENCE in UNKNOWN_PROSE
    assert ABSENT_PROSE != UNKNOWN_PROSE


@pytest.mark.asyncio
async def test_the_whole_routing_table_holds_with_the_default_store(
    bridged: Callable[[object], _TripwireAgent],
) -> None:
    """The same three outcomes across the shipped in-memory store, so none of the above rests on the test double."""
    own = InMemoryReferenceStore()
    own.put(REFERENCE, CONTENT)

    hit = await resolve_artifact(own, bridged(_UntouchableStash()), REFERENCE)
    stashed = await resolve_artifact(
        InMemoryReferenceStore(), bridged(_RecordingStash(_Journal(), {REFERENCE: "stashed text"})), REFERENCE
    )
    missed = await resolve_artifact(InMemoryReferenceStore(), bridged(_RecordingStash(_Journal())), REFERENCE)
    standalone = await resolve_artifact(InMemoryReferenceStore(), _TripwireAgent(), REFERENCE)

    assert [answer.outcome for answer in (hit, stashed, missed, standalone)] == [
        "text",
        "text",
        "unknown",
        "absent",
    ]
    assert [answer.source for answer in (hit, stashed, missed, standalone)] == ["own", "stash", None, None]
