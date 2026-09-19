"""Property tests for the projection integrity of the Removal.

Feature: context-graph-plugin, Property 3: Projection integrity — removal is a subsequence and tool pairs are whole.

Validates: Requirements 3.6, 5.1.

The generators build histories the shape of what the middleware actually receives: interleaved tool pairs (one pair's
result landing after another pair's use), a single assistant message carrying two ``toolUse`` blocks, orphan halves,
messages with no ``tracking_id``, pinned messages, and previews citing ``[refs: x, y]`` markers. Two entry points are
driven over them — ``remove_messages`` with a directly drawn request, and ``apply_removal`` over a graph rebuilt by scan
with a drawn turn choice — and both are held to the same claims:

- the result is a true subsequence: the same message objects by identity, in the same relative order, with no
  duplication, no insertion and no reordering (Requirement 5.1);
- the retained ``toolUse`` and ``toolResult`` blocks carry equal ``toolUseId`` sets, so no pair is ever split;
- the result opens with a ``user`` role whenever the input holds a user message, and is non-empty for a non-empty input;
- a pinned message is always retained;
- the received list, and every dict inside it, come back unmutated;
- the identities of the turn in progress are never removed (Requirement 3.6).
"""

import copy
from types import MappingProxyType
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.cards import closed_turn_ranges, rebuild_into
from strands_context_graph.removal import apply_removal, remove_messages
from strands_context_graph.state import CardChoice, TurnChoice, _GraphState

REGISTRATION = {
    "description_tokens": 100,
    "tags_per_card": 5,
    "rarity_weight": 0.5,
    "link_threshold": 0.5,
}

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

# Structural steps of a turn. ``interleaved`` opens two pairs and closes them out of order; ``multi`` puts two
# ``toolUse`` blocks on one message; ``orphan_use`` and ``orphan_result`` leave a half unpaired; ``untracked`` carries
# no durable identity; ``refs`` cites a Stash marker; ``pinned`` marks a message the way the main agent does.
step_strategy = st.sampled_from(
    ["assistant", "pair", "interleaved", "multi", "orphan_use", "orphan_result", "untracked", "refs", "pinned"]
)

turn_strategy = st.tuples(text_strategy, st.lists(step_strategy, max_size=4))

resolution_strategy = st.sampled_from(["full", "description", "title"])


def _tool_use(tool_use_id: str, tracking_id: str, name: str = "get_balance") -> dict[str, Any]:
    """Build the assistant half of a tool pair."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
        "tracking_id": tracking_id,
    }


def _tool_result(tool_use_id: str, tracking_id: str, text: str) -> dict[str, Any]:
    """Build the user half of a tool pair."""
    return {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]}},
        ],
        "tracking_id": tracking_id,
    }


def _render(turns: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    """Render structural turns into a message history, assigning durable identities sequentially."""
    messages: list[dict[str, Any]] = []
    counter = 0

    def identity() -> str:
        nonlocal counter
        counter += 1
        return f"m{counter}"

    for turn, (ask, steps) in enumerate(turns):
        messages.append({"role": "user", "content": [{"text": ask}], "tracking_id": identity()})

        for index, step in enumerate(steps):
            first_id = f"tu-{turn}-{index}-a"
            second_id = f"tu-{turn}-{index}-b"

            if step == "assistant":
                messages.append({"role": "assistant", "content": [{"text": ask}], "tracking_id": identity()})
            elif step == "untracked":
                # No durable identity: no Card addresses it, so the Removal always sends it (Requirement 3.5).
                messages.append({"role": "assistant", "content": [{"text": ask}]})
            elif step == "refs":
                # The Stash standalone marker, the second placeholder shape the scan reads (Requirement 3.8).
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"text": f"{ask} [refs: mem_1_{first_id}_0, mem_1_{second_id}_0]"}],
                        "tracking_id": identity(),
                    }
                )
            elif step == "pinned":
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"text": ask}],
                        "tracking_id": identity(),
                        "metadata": {"custom": {"pinned": True}},
                    }
                )
            elif step == "pair":
                messages.append(_tool_use(first_id, identity()))
                messages.append(_tool_result(first_id, identity(), f"balance: R$ 1.200,00 | {ask}"))
            elif step == "interleaved":
                # Two pairs opened back to back and closed in the same order, so each pair straddles the other.
                messages.append(_tool_use(first_id, identity()))
                messages.append(_tool_use(second_id, identity()))
                messages.append(_tool_result(first_id, identity(), f"first: 1.200,00 | {ask}"))
                messages.append(_tool_result(second_id, identity(), f"second: 3.400,00 | {ask}"))
            elif step == "multi":
                # One message paired with two different results: the shape the reconciliation must not flip on.
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"toolUse": {"toolUseId": first_id, "name": "get_balance", "input": {}}},
                            {"toolUse": {"toolUseId": second_id, "name": "read_file", "input": {}}},
                        ],
                        "tracking_id": identity(),
                    }
                )
                messages.append(_tool_result(first_id, identity(), f"multi first | {ask}"))
                messages.append(_tool_result(second_id, identity(), f"multi second | {ask}"))
            elif step == "orphan_use":
                messages.append(_tool_use(first_id, identity()))
            else:
                messages.append(_tool_result(first_id, identity(), f"orphan | {ask}"))

    return messages


history_strategy = st.lists(turn_strategy, min_size=1, max_size=4).map(_render)


@st.composite
def request_cases(draw: st.DrawFn) -> tuple[list[dict[str, Any]], frozenset[str]]:
    """A history and a request drawn from the identities it actually carries, plus a few it does not."""
    messages = draw(history_strategy)
    known = _identities_of(messages)
    candidates = [*known, "absent-1", "absent-2"]
    requested = draw(st.lists(st.sampled_from(candidates), max_size=len(candidates)))
    return messages, frozenset(requested)


@st.composite
def choice_cases(draw: st.DrawFn) -> tuple[list[dict[str, Any]], _GraphState, TurnChoice, frozenset[str]]:
    """A history, the graph rebuilt by scan over it, a drawn turn choice, and the turn in progress."""
    messages = draw(history_strategy)
    state = _GraphState()
    rebuild_into(state, messages, **REGISTRATION)

    by_title: dict[str, CardChoice] = {}
    for title in sorted(state.cards):
        if draw(st.booleans()):
            # A Card absent from the choice reads as full content, the fail-safe direction for derivation lag.
            continue
        by_title[title] = CardChoice(dialogue=draw(resolution_strategy), evidence=draw(resolution_strategy))

    choice = TurnChoice(MappingProxyType(by_title), full_pass=False)

    ranges = closed_turn_ranges(messages)
    open_start = ranges[-1][1] if ranges else 0
    return messages, state, choice, frozenset(_identities_of(messages[open_start:]))


def _identities_of(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Durable identities of ``messages``, in order and without duplicates."""
    found: dict[str, None] = {}
    for message in messages:
        tracking_id = message.get("tracking_id")
        if tracking_id:
            found.setdefault(tracking_id, None)
    return tuple(found)


def _is_subsequence(removal: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    """Whether ``removal`` holds the same objects as ``messages``, in the same relative order."""
    remaining = iter(messages)
    return all(any(candidate is element for candidate in remaining) for element in removal)


def _tool_use_ids(messages: list[dict[str, Any]], block: str) -> set[str]:
    """Collect the ``toolUseId`` set carried by ``block`` blocks across ``messages``."""
    return {
        content[block]["toolUseId"]
        for message in messages
        for content in message.get("content", [])
        if isinstance(content, dict) and block in content
    }


def _carriers(messages: list[dict[str, Any]]) -> dict[str, set[int]]:
    """Map each ``toolUseId`` of ``messages`` to the positions carrying it, either half."""
    carriers: dict[str, set[int]] = {}
    for index, message in enumerate(messages):
        for block in ("toolUse", "toolResult"):
            for tool_use_id in _tool_use_ids([message], block):
                carriers.setdefault(tool_use_id, set()).add(index)
    return carriers


def _paired_ids(messages: list[dict[str, Any]]) -> set[str]:
    """The ``toolUseId``s the input itself pairs, i.e. present as both a ``toolUse`` and a ``toolResult``."""
    return _tool_use_ids(messages, "toolUse") & _tool_use_ids(messages, "toolResult")


def _pinned(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages the main agent marked pinned."""
    return [
        message
        for message in messages
        if message.get("metadata", {}).get("custom", {}).get("pinned")  # type: ignore[union-attr]
    ]


def _fingerprint(messages: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Identity of the list, identity of every message dict, and every key set — what a mutation would move."""
    return (id(messages), tuple(id(message) for message in messages), tuple(tuple(message) for message in messages))


def _assert_projection_integrity(
    removal: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    before: list[dict[str, Any]],
    fingerprint: tuple[Any, ...],
) -> None:
    """Assert every claim of Property 3 over one delivery."""
    # A true subsequence: same objects, same relative order, no reordering and no insertion (Requirement 5.1).
    assert _is_subsequence(removal, messages)
    # No duplication: each retained object appears exactly once.
    assert len({id(message) for message in removal}) == len(removal)
    # Tool pairs travel whole: the retained ``toolUse`` and ``toolResult`` ids are equal over every id the input pairs,
    # and an id is either retained by all its carriers or by none of them (Requirements 5.3, 5.4).
    paired = _paired_ids(messages)
    assert _tool_use_ids(removal, "toolUse") & paired == _tool_use_ids(removal, "toolResult") & paired
    retained = {id(message) for message in removal}
    for positions in _carriers(messages).values():
        kept = {index for index in positions if id(messages[index]) in retained}
        assert kept in (set(), positions)
    # Providers reject a conversation that does not open with a user turn (Requirement 5.6), and reject an empty one.
    if messages:
        assert removal
    if any(message.get("role") == "user" for message in messages):
        assert removal[0]["role"] == "user"
    # An explicit pin wins over any request (Requirement 5.7).
    for pinned in _pinned(messages):
        assert any(message is pinned for message in removal)
    # The received list, and every dict in it, come back untouched (Requirement 5.2).
    assert messages == before
    assert _fingerprint(messages) == fingerprint


@given(case=request_cases())
@PROPERTY_SETTINGS
def test_remove_messages_projects_a_whole_subsequence(case: tuple[list[dict[str, Any]], frozenset[str]]) -> None:
    """Feature: context-graph-plugin, Property 3: Projection integrity — removal is a subsequence and tool pairs are whole.

    Validates: Requirements 3.6, 5.1.
    """  # noqa: E501
    messages, requested = case
    before = copy.deepcopy(messages)
    fingerprint = _fingerprint(messages)

    removal = remove_messages(messages, requested)

    _assert_projection_integrity(list(removal), messages, before, fingerprint)
    # A message with no durable identity, or one the request does not name, is always sent — unless it is one half of a
    # pair whose other half was requested, which the reconciliation drops with it (Requirement 5.4).
    for message in messages:
        tracking_id = message.get("tracking_id")
        if _tool_use_ids([message], "toolUse") or _tool_use_ids([message], "toolResult"):
            continue
        if not tracking_id or tracking_id not in requested:
            assert any(candidate is message for candidate in removal)


@given(case=choice_cases())
@PROPERTY_SETTINGS
def test_apply_removal_projects_a_whole_subsequence(
    case: tuple[list[dict[str, Any]], _GraphState, TurnChoice, frozenset[str]],
) -> None:
    """Feature: context-graph-plugin, Property 3: Projection integrity — removal is a subsequence and tool pairs are whole.

    Validates: Requirements 3.6, 5.1.
    """  # noqa: E501
    messages, state, choice, current_turn_ids = case
    before = copy.deepcopy(messages)
    fingerprint = _fingerprint(messages)

    removal, requested = apply_removal(messages, state, choice, current_turn_ids)

    _assert_projection_integrity(list(removal), messages, before, fingerprint)
    # The turn in progress is projected in full content, whatever the choice says (Requirement 3.6).
    assert requested.isdisjoint(current_turn_ids)
    for message in messages:
        if message.get("tracking_id") in current_turn_ids:
            assert any(candidate is message for candidate in removal)
