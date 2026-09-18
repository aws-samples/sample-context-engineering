"""Property tests for the state-isolation slice of the graph.

Feature: context-graph-plugin, Property 1: The live history is never mutated.

Validates: Requirements 1.5, 14.4.

Scope is what is ported today: ``state.py`` and ``cards.py``. The plugin's hooks and middleware are wired by later
tasks, so the lifecycle is exercised where it touches the history and the per-agent state: the derivation and
registration path over generated conversations. Two claims are asserted over every generated history:

- the history is read only — same list object, same message dicts by identity, same keys, same values, and no new key
  such as a metadata field written back onto a message;
- two agents keyed in a weakly keyed map keep independent ``_GraphState``: writing one leaves the other exactly as it
  was, and the two states share no container object.
"""

import copy
import weakref
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.cards import closed_turn_ranges, derive_and_register, rebuild_into
from strands_context_graph.state import _GraphState

DERIVATION = {"description_tokens": 100, "tags_per_card": 5, "rarity_weight": 0.5}
REGISTRATION = {**DERIVATION, "link_threshold": 0.5}

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class _FakeAgent:
    """Stand-in for an Agent: the state map only needs a weak-referenceable object as key."""


# Text that stays inside what the scan reads: words, digits, currency and punctuation, never empty.
text_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz 0123456789.,:$-_")),
    min_size=1,
    max_size=40,
)

# A turn is described structurally first, then rendered into messages with sequential durable identities: the counter
# cannot live inside a strategy, and the identities have to be unique across the whole conversation.
step_strategy = st.sampled_from(["assistant", "pair", "orphan_use", "untracked"])

turn_strategy = st.tuples(text_strategy, st.lists(step_strategy, max_size=3))


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
            tool_use_id = f"tu-{turn}-{index}"
            if step == "assistant":
                messages.append({"role": "assistant", "content": [{"text": ask}], "tracking_id": identity()})
            elif step == "untracked":
                # A message with no durable identity: no Card addresses it (Requirement 3.5).
                messages.append({"role": "assistant", "content": [{"text": ask}]})
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "get_balance", "input": {}}}],
                        "tracking_id": identity(),
                    }
                )
                if step == "pair":
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "status": "success",
                                        "content": [{"text": f"balance: R$ 1.200,00 | {ask}"}],
                                    }
                                }
                            ],
                            "tracking_id": identity(),
                        }
                    )

    return messages


history_strategy = st.lists(turn_strategy, min_size=1, max_size=4).map(_render)


def _identities_of(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Durable identities of ``messages``, in order and without duplicates."""
    found: dict[str, None] = {}
    for message in messages:
        tracking_id = message.get("tracking_id")
        if tracking_id:
            found.setdefault(tracking_id, None)
    return tuple(found)


def _fingerprint(messages: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Identity of the list, identity of every message dict, and every key set — what a mutation would move."""
    return (id(messages), tuple(id(message) for message in messages), tuple(tuple(message) for message in messages))


def _run_derivation(state: _GraphState, messages: list[dict[str, Any]]) -> None:
    """Exercise both write paths over ``messages``: the rebuild scan and one incremental registration."""
    rebuild_into(state, messages, **REGISTRATION)

    ranges = closed_turn_ranges(messages)
    if ranges:
        start, stop = ranges[-1]
        derive_and_register(state, messages, _identities_of(messages[start:stop]), len(ranges) - 1, **REGISTRATION)


@given(messages=history_strategy)
@PROPERTY_SETTINGS
def test_derivation_never_mutates_the_live_history(messages: list[dict[str, Any]]) -> None:
    """Feature: context-graph-plugin, Property 1: The live history is never mutated.

    Validates: Requirements 1.5, 14.4.
    """
    before = copy.deepcopy(messages)
    fingerprint = _fingerprint(messages)

    _run_derivation(_GraphState(), messages)

    assert messages == before
    assert _fingerprint(messages) == fingerprint


@given(first_messages=history_strategy, second_messages=history_strategy)
@PROPERTY_SETTINGS
def test_two_agents_keep_independent_state(
    first_messages: list[dict[str, Any]],
    second_messages: list[dict[str, Any]],
) -> None:
    """Feature: context-graph-plugin, Property 1: The live history is never mutated.

    Validates: Requirements 1.5, 14.4.
    """
    states: weakref.WeakKeyDictionary[_FakeAgent, _GraphState] = weakref.WeakKeyDictionary()
    first, second = _FakeAgent(), _FakeAgent()
    states[first] = _GraphState()
    states[second] = _GraphState()

    _run_derivation(states[first], first_messages)
    fresh = _GraphState()

    # Writing the first agent's state leaves the second one exactly as a fresh state (Requirement 14.4).
    assert states[second].cards == fresh.cards
    assert states[second].links == fresh.links
    assert states[second].turn == fresh.turn
    assert states[second].choice.full_pass is True

    snapshot_cards = dict(states[first].cards)
    snapshot_links = {title: list(edges) for title, edges in states[first].links.items()}

    _run_derivation(states[second], second_messages)

    assert states[first].cards == snapshot_cards
    assert states[first].links == snapshot_links

    # No container is shared between the two states, so a later write on one cannot reach the other.
    assert states[first].cards is not states[second].cards
    assert states[first].links is not states[second].links
    assert states[first].reuse is not states[second].reuse
    assert states[first].vectors is not states[second].vectors
