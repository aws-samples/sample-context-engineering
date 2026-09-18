"""Property tests for the rebuild scan against the incremental construction.

Feature: context-graph-plugin, Property 10: A rebuild-by-scan equals the incremental graph.

Validates: Requirements 14.5, 3.1, 3.11.

The graph is derived state: losing it to a restart has to cost one scan and nothing else, which only holds if the scan
lands on the same graph the turn-by-turn write half built. ``tests/test_cards.py`` pins that on one hand-written
conversation; these properties assert it over generated histories.

The incremental path is simulated the way the write half runs: a turn's Card is derived when the boundary that closes it
arrives, so turn ``i`` is registered against the history as it stood at that moment — the prefix up to and including
that boundary — never against the finished conversation. A scan that agreed only on the complete history would still
lose the graph of an agent interrupted mid-conversation.

Three claims, over the same generated histories:

- the two graphs are equal Card for Card and Link for Link: Titles, turn ordinals, Descriptions, Tags, addressed
  identities, references, tool names and edges, in the same order (Requirement 14.5);
- one Card per closed turn boundary, the turn in progress carrying none (Requirement 3.1);
- Cards group by turn and by nothing else: every identity a Card addresses comes from its own turn's range, so no
  subject classification regrouped messages across turns (Requirement 3.11).

The generators keep the shapes the scan branches on: tool pairs, interleaved pairs, offloaded previews in both
placeholder shapes, numeric lines, messages with no ``tracking_id``, and whole turns where no message carries one.
"""

import copy
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.cards import closed_turn_ranges, derive_and_register, rebuild
from strands_context_graph.state import Card, _GraphState

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Text that stays inside what the scan reads: words, digits, currency and punctuation, never empty.
text_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz 0123456789.,:$-_")),
    min_size=1,
    max_size=40,
)

# Structural steps inside a turn. ``pair`` closes one tool pair and ``interleaved`` straddles two; ``inline_ref``
# carries the offloader placeholder and ``stash_refs`` the Stash one; ``untracked`` carries no durable identity.
step_strategy = st.sampled_from(
    ["assistant", "pair", "interleaved", "inline_ref", "stash_refs", "untracked", "numeric"]
)

# A turn: its ask, its interior steps, and whether any of its messages carries a durable identity at all. A turn drawn
# unaddressed consumes its ordinal without producing a Card (Requirement 3.5), the case that would shift every later
# ordinal if the two paths counted turns differently.
turn_strategy = st.tuples(text_strategy, st.lists(step_strategy, max_size=3), st.booleans())


class _Identities:
    """Hands out sequential durable identities, withholding them for a turn drawn unaddressed."""

    def __init__(self) -> None:
        self._count = 0
        self.addressed = True

    def next(self) -> str | None:
        """Return the next identity, or ``None`` while the current turn is unaddressed."""
        self._count += 1
        return f"m{self._count}" if self.addressed else None


def _tool_use(tool_use_id: str, tracking_id: str | None, name: str = "get_balance") -> dict[str, Any]:
    """Build the assistant half of a tool pair."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
    }
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def _tool_result(tool_use_id: str, tracking_id: str | None, text: str) -> dict[str, Any]:
    """Build the user half of a tool pair."""
    message: dict[str, Any] = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]}}],
    }
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def _text_message(role: str, text: str, tracking_id: str | None) -> dict[str, Any]:
    """Build a plain text message, with or without a durable identity."""
    message: dict[str, Any] = {"role": role, "content": [{"text": text}]}
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def _render(turns: list[tuple[str, list[str], bool]]) -> list[dict[str, Any]]:
    """Render structural turns into a message history, assigning durable identities sequentially.

    Asks carry their turn ordinal so Titles stay distinct: the Title is the Card's identity, and two turns asking the
    same thing are one Card in either path, which would make a count of Cards say nothing about the scan.

    Args:
        turns: One ``(ask, steps, addressed)`` triple per turn, in order.

    Returns:
        The history, opening with a turn boundary and ending with a turn in progress whenever more than one turn was
        drawn.
    """
    messages: list[dict[str, Any]] = []
    identities = _Identities()

    for turn, (drawn_ask, steps, addressed) in enumerate(turns):
        identities.addressed = addressed
        ask = f"t{turn} {drawn_ask}"
        messages.append(_text_message("user", ask, identities.next()))

        for index, step in enumerate(steps):
            first_id = f"tu-{turn}-{index}-a"
            second_id = f"tu-{turn}-{index}-b"

            if step == "assistant":
                messages.append(_text_message("assistant", ask, identities.next()))
            elif step == "untracked":
                # No durable identity even in an addressed turn: no Card addresses it either way.
                messages.append(_text_message("assistant", ask, None))
            elif step == "numeric":
                text = f"balance: R$ 1.200,00\ntotal: 3.400,00 | {ask}"
                messages.append(_text_message("assistant", text, identities.next()))
            elif step == "inline_ref":
                # The offloader inline placeholder, one of the two shapes the scan reads references from.
                messages.append(_tool_use(first_id, identities.next(), name="http_request"))
                preview = f"[image: png, 900 bytes | ref: mem_1_{first_id}_0]"
                messages.append(_tool_result(first_id, identities.next(), preview))
            elif step == "stash_refs":
                messages.append(_tool_use(first_id, identities.next(), name="read_file"))
                preview = f"stashed [refs: {first_id}_0, {second_id}_1] | {ask}"
                messages.append(_tool_result(first_id, identities.next(), preview))
            elif step == "pair":
                messages.append(_tool_use(first_id, identities.next()))
                messages.append(_tool_result(first_id, identities.next(), f"balance: R$ 1.200,00 | {ask}"))
            else:
                # Two pairs opened back to back, each straddling the other.
                messages.append(_tool_use(first_id, identities.next()))
                messages.append(_tool_use(second_id, identities.next(), name="read_file"))
                messages.append(_tool_result(first_id, identities.next(), f"first: 1.200,00 | {ask}"))
                messages.append(_tool_result(second_id, identities.next(), f"second: 3.400,00 | {ask}"))

    return messages


history_strategy = st.lists(turn_strategy, min_size=1, max_size=4).map(_render)

# The four derivation knobs, drawn so the equality is about the scan rather than about one calibration of it.
registration_strategy = st.fixed_dictionaries(
    {
        "description_tokens": st.integers(min_value=10, max_value=200),
        "tags_per_card": st.integers(min_value=1, max_value=6),
        "rarity_weight": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "link_threshold": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    }
)


def _identities_of(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Durable identities of ``messages``, in order and without duplicates."""
    found: dict[str, None] = {}
    for message in messages:
        tracking_id = message.get("tracking_id")
        if tracking_id:
            found.setdefault(tracking_id, None)
    return tuple(found)


def _build_incrementally(messages: list[dict[str, Any]], registration: dict[str, Any]) -> _GraphState:
    """Build the graph turn by turn, each turn seeing only the history that existed when it closed.

    The write half runs on the message that opens the next turn, so turn ``i`` is derived against the prefix ending at
    that boundary. Deriving every turn against the finished conversation instead would hide any dependence on messages
    that had not been written yet.

    Args:
        messages: The whole conversation, as data. Read only.
        registration: The four derivation knobs.

    Returns:
        The state the incremental write half holds after the last closed turn.
    """
    state = _GraphState()

    for turn, (start, stop) in enumerate(closed_turn_ranges(messages)):
        # The boundary at ``stop`` is the message that closed this turn, and the last one the hook had seen.
        prefix = messages[: stop + 1]
        turn_ids = _identities_of(messages[start:stop])
        if not turn_ids:
            # A turn no identity addresses registers no Card, and still consumes its ordinal.
            continue

        derive_and_register(state, prefix, turn_ids, turn, **registration)

    return state


def _fields_of(card: Card) -> tuple[Any, ...]:
    """The Card fields the two paths must agree on, spelled out rather than left to dataclass equality."""
    return (
        card.title,
        card.kind,
        card.turn,
        card.dialogue_ids,
        card.evidence_ids,
        card.pairs,
        card.tool_names,
        card.references,
        card.numeric_lines,
        card.tags,
        card.description,
        card.reference,
        card.content_type,
        card.size_bytes,
    )


@given(messages=history_strategy, registration=registration_strategy)
@PROPERTY_SETTINGS
def test_rebuild_by_scan_equals_the_incremental_graph(
    messages: list[dict[str, Any]], registration: dict[str, Any]
) -> None:
    """Feature: context-graph-plugin, Property 10: A rebuild-by-scan equals the incremental graph.

    Validates: Requirements 14.5, 3.1, 3.11.
    """
    before = copy.deepcopy(messages)
    incremental = _build_incrementally(messages, registration)

    scanned = rebuild(messages, **registration)

    assert list(scanned.cards) == list(incremental.cards)
    for title, card in scanned.cards.items():
        assert _fields_of(card) == _fields_of(incremental.cards[title])
    assert scanned.cards == incremental.cards

    assert list(scanned.links) == list(incremental.links)
    for title, edges in scanned.links.items():
        assert edges == incremental.links[title]

    # Neither path touches the conversation it derives from.
    assert messages == before


@given(messages=history_strategy, registration=registration_strategy)
@PROPERTY_SETTINGS
def test_the_scan_derives_one_card_per_closed_turn_grouped_by_turn_only(
    messages: list[dict[str, Any]], registration: dict[str, Any]
) -> None:
    """Feature: context-graph-plugin, Property 10: A rebuild-by-scan equals the incremental graph.

    Validates: Requirements 14.5, 3.1, 3.11.
    """
    ranges = closed_turn_ranges(messages)
    addressed = {turn: _identities_of(messages[start:stop]) for turn, (start, stop) in enumerate(ranges)}
    eligible = [turn for turn, turn_ids in addressed.items() if turn_ids]

    state = rebuild(messages, **registration)

    # One Card per closed turn boundary that carries an identity, and none for the turn in progress (Requirement 3.1).
    assert state.turn == len(ranges)
    assert [card.turn for card in state.cards.values()] == eligible

    for card in state.cards.values():
        # Grouping is by turn and by nothing else: every identity the Card addresses lies in its own turn's range, so
        # nothing was regrouped by subject across turns (Requirement 3.11).
        assert set(card.dialogue_ids) | set(card.evidence_ids) <= set(addressed[card.turn])
        assert set(card.dialogue_ids) & set(card.evidence_ids) == set()
        assert card.kind == "subject"
