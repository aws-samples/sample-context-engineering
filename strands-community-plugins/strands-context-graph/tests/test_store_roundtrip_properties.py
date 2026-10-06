"""Property tests for the round-trip of a decoded text block through the plugin's own store.

Feature: context-graph-plugin, Property 4: Own-store artifact round-trip.

Validates: Requirements 12.5, 15.5.

The claim is the cheapest one in the package to state and the most expensive one to get wrong: whatever text was stored
under a reference, resolving that reference and reading it whole hands back *that* text, character for character. No
truncation, no reformatting, no normalization, and nothing belonging to a neighbouring reference. The round-trip runs
against the shipped :class:`InMemoryReferenceStore` through ``resolve_artifact`` + ``read_artifact``, which is the path
``expand_artifact`` reads through, so the property needs no AWS and no Stash.

Three claims per generated store:

- **Verbatim, and the right one.** Every stored reference resolves to ``"text"`` from source ``"own"`` and reads back
  equal to what went in — asserted across a whole batch of references at once, so a read that returned a neighbour's
  content fails rather than passing by coincidence.
- **No model call.** The agent handed to the resolution is a tripwire: every attribute a model call would travel
  through raises instead of answering, so reaching for one fails the test rather than quietly costing a token
  (Requirement 12.5).
- **No Resolution change.** A graph state carrying Cards and a frozen Turn Choice is snapshotted around the round-trip
  and compared field by field afterwards, and the choice mapping is still not writable. Expanding an artifact is a read
  of a store, not a decision about a Card.
"""

import asyncio
from types import MappingProxyType

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.state import Card, CardChoice, TurnChoice, _GraphState
from strands_context_graph.store import InMemoryReferenceStore, read_artifact, resolve_artifact

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_MODEL_ATTRS = frozenset(
    {"model", "invoke_async", "stream_async", "structured_output", "structured_output_async", "converse"}
)
"""Every attribute a model call on an agent would travel through. Reaching for one during a round-trip is a bug."""


class _TripwireAgent:
    """A standalone agent — no plugin registry — that fails the test if a model call is attempted through it.

    The attributes in :data:`_MODEL_ATTRS` raise ``AssertionError`` rather than ``AttributeError``, so they survive the
    ``getattr(agent, name, None)`` idiom the bridge lookup uses and surface as a failed test instead of a silent miss.
    """

    def __getattr__(self, name: str) -> object:
        if name in _MODEL_ATTRS:
            raise AssertionError(f"the round-trip reached the model through {name!r}")
        raise AttributeError(name)


reference_strategy = st.one_of(
    # The offloader's inline shape and the Stash's standalone shape, plus a plainly arbitrary key: the store is a
    # dictionary and treats a reference as opaque, so the property must not depend on its spelling.
    st.builds("mem_{}_tu-{}_{}".format, st.integers(0, 9), st.integers(0, 9), st.integers(0, 4)),
    st.builds("stash_{}".format, st.integers(0, 99)),
    st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=12),
)

line_strategy = st.one_of(
    st.text(max_size=40),
    # Shapes a real tool return carries and a careless read would normalize: monetary separators, tabular rows,
    # indentation, trailing spaces, accented prose.
    st.builds("R$ {}.{:03d},00".format, st.integers(0, 999_999), st.integers(0, 999)),
    st.builds("US$ {},{:03d}.50".format, st.integers(0, 999_999), st.integers(0, 999)),
    st.builds("| {} | {} |".format, st.integers(0, 9999), st.integers(0, 9999)),
    st.builds("\t{}\t{}".format, st.integers(0, 9999), st.integers(0, 9999)),
    st.builds("   indented {}  ".format, st.integers(0, 9999)),
    st.sampled_from(["", "  ", "saldo de março", "R$\u00a01.200,00", "ça va — 42%", "日本語のテキスト"]),
)

# Joined with newlines and allowed to be empty: a decoded block may be a single line, many lines, or nothing at all.
text_strategy = st.lists(line_strategy, max_size=8).map("\n".join)

# A batch of references stored together, so a read that crossed over to a neighbour is visible.
blocks_strategy = st.dictionaries(reference_strategy, text_strategy, min_size=1, max_size=6)


def _card(title: str, turn: int) -> Card:
    """Build a subject Card standing in for a derived one, carrying no content of its own."""
    return Card(
        title=title,
        kind="subject",
        turn=turn,
        dialogue_ids=(f"m{turn}",),
        evidence_ids=(),
        pairs=(),
        tool_names=frozenset({"get_balance"}),
        references=(),
        numeric_lines=(),
        tags=("balance",),
        description=f"turn {turn}: balance",
    )


def _state_with_a_frozen_choice() -> _GraphState:
    """A graph state whose Cards sit at three different Resolutions, frozen as a Turn Choice for the turn."""
    state = _GraphState()
    resolutions = (("full", "full"), ("description", "description"), ("title", "description"))

    for turn in range(len(resolutions)):
        title = f"card-{turn}"
        state.cards[title] = _card(title, turn)

    state.choice = TurnChoice(
        by_title=MappingProxyType(
            {
                f"card-{turn}": CardChoice(dialogue=dialogue, evidence=evidence)  # type: ignore[arg-type]
                for turn, (dialogue, evidence) in enumerate(resolutions)
            }
        ),
        full_pass=False,
        selected=frozenset({"card-0", "card-1"}),
    )
    state.turn = len(resolutions)

    return state


def _snapshot(state: _GraphState) -> tuple[object, ...]:
    """Everything about ``state`` a round-trip must leave alone, in a comparable shape."""
    return (
        dict(state.cards),
        dict(state.choice.by_title),
        state.choice.full_pass,
        state.choice.selected,
        dict(state.reuse),
        state.turn,
    )


def _round_trip(store: InMemoryReferenceStore, reference: str) -> tuple[str, str | None, str | None]:
    """Resolve ``reference`` against ``store`` and read it whole, as ``expand_artifact`` does.

    Args:
        store: The plugin's own store. Read only.
        reference: The reference to resolve.

    Returns:
        ``(outcome, source, answer)``, where ``answer`` is the whole read for a ``"text"`` outcome and ``None``
        otherwise.
    """

    async def _resolve() -> tuple[str, str | None, str | None]:
        resolved = await resolve_artifact(store, _TripwireAgent(), reference)
        answer = read_artifact(resolved.text) if resolved.text is not None else None
        return resolved.outcome, resolved.source, answer

    return asyncio.run(_resolve())


@given(blocks=blocks_strategy)
@PROPERTY_SETTINGS
def test_every_stored_text_block_reads_back_verbatim(blocks: dict[str, str]) -> None:
    """Feature: context-graph-plugin, Property 4: Own-store artifact round-trip.

    Validates: Requirements 12.5, 15.5.
    """
    store = InMemoryReferenceStore()
    for reference, text in blocks.items():
        store.put(reference, text)

    for reference, text in blocks.items():
        outcome, source, answer = _round_trip(store, reference)

        assert outcome == "text"
        assert source == "own"
        assert answer == text

    # The reads left the store exactly as the writes did: a resolution is not a pop.
    assert store.references() == frozenset(blocks)
    assert len(store) == len(blocks)


@given(blocks=blocks_strategy)
@PROPERTY_SETTINGS
def test_repeated_reads_of_the_same_reference_are_identical(blocks: dict[str, str]) -> None:
    """Feature: context-graph-plugin, Property 4: Own-store artifact round-trip.

    Validates: Requirements 12.5, 15.5.
    """
    store = InMemoryReferenceStore()
    for reference, text in blocks.items():
        store.put(reference, text)

    for reference, text in blocks.items():
        assert _round_trip(store, reference) == _round_trip(store, reference) == ("text", "own", text)


@given(blocks=blocks_strategy)
@PROPERTY_SETTINGS
def test_the_round_trip_changes_no_card_resolution(blocks: dict[str, str]) -> None:
    """Feature: context-graph-plugin, Property 4: Own-store artifact round-trip.

    Validates: Requirements 12.5, 15.5.
    """
    store = InMemoryReferenceStore()
    for reference, text in blocks.items():
        store.put(reference, text)

    state = _state_with_a_frozen_choice()
    before = _snapshot(state)

    for reference in blocks:
        assert _round_trip(store, reference)[0] == "text"

    assert _snapshot(state) == before

    # Still frozen after the reads: the turn's choice is a read-only mapping, not a live dict a tool could edit.
    try:
        state.choice.by_title["card-0"] = CardChoice(dialogue="full", evidence="full")  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover - only reached if the choice stops being immutable
        raise AssertionError("the Turn Choice became writable")
