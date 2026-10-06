"""Unit tests for the write half of a tool call: the ``AfterToolCallEvent`` hook.

The derivation itself is ``cards``' and the recording is ``store``'s, so what is worth asserting here is the
composition: that an offloaded result produces an artifact Card keeping the reference and nothing else, that the same
reference lands in the plugin's own store, that a result no offloader touched writes nothing anywhere, that the store is
always present and never shared between two agents, and that no failure of either half reaches the caller.

The agent is a plain class carrying only ``messages``, so it is weak-referenceable, which both per-agent maps require.
"""

import logging
from collections.abc import Sequence

from strands.hooks.events import AfterToolCallEvent

from strands_context_graph import ContextGraph
from strands_context_graph.store import InMemoryReferenceStore

REFERENCE = "mem_1_tu-3_0"
"""The reference an offloader's placeholder names, in the inline shape."""


class _Agent:
    """The one member the write half reads off an agent."""

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []


class UnusableMatcher:
    """A matcher that fails if it is ever reached: the write half pays for no embedding."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Fail the test rather than answer."""
        raise AssertionError("the write half must not reach the matcher")


def graph() -> ContextGraph:
    """A plugin whose matcher would fail loudly, since nothing here should reach it."""
    return ContextGraph(matcher=UnusableMatcher())


def offloaded(text: str = f"[image: png, 900 bytes | ref: {REFERENCE}]") -> dict:
    """A tool result an offloader replaced by a placeholder naming a reference."""
    return {"toolUseId": "tu-3", "status": "success", "content": [{"text": text}]}


def event(agent: _Agent, result: object, tool_name: str = "http_request") -> AfterToolCallEvent:
    """One ``AfterToolCallEvent`` for ``result``, with the fields the hook reads."""
    return AfterToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"toolUseId": "tu-3", "name": tool_name, "input": {}},
        invocation_state={},
        result=result,  # type: ignore[arg-type]
    )


def test_an_offloaded_result_cards_the_reference_and_records_it() -> None:
    """Requirements 3.7, 15.1: the Card holds the address, the store learns the name, neither holds the content."""
    plugin = graph()
    agent = _Agent()

    plugin._on_after_tool_call(event(agent, offloaded()))

    state = plugin._states[agent]
    card = state.cards[REFERENCE]
    assert card.kind == "artifact"
    assert card.title == card.reference == REFERENCE
    assert card.tool_names == frozenset({"http_request"})
    # Addresses only: an artifact owns no message, so no resolution of its parts can drop one.
    assert card.dialogue_ids == card.evidence_ids == ()
    # The store knows the reference, and has no block behind it: the content was already replaced by the time the hook
    # saw the result, so none is invented.
    store = plugin._store_for(agent)
    assert REFERENCE in store
    assert len(store) == 1


def test_no_raw_content_is_kept_by_either_half() -> None:
    """The content the tool returned is nowhere in the graph or the store: only the reference is."""
    plugin = graph()
    agent = _Agent()
    secret = "0.0.0.0 root password hunter2"

    plugin._on_after_tool_call(event(agent, offloaded(f"{secret}\n[image: png, 900 bytes | ref: {REFERENCE}]")))

    card = plugin._states[agent].cards[REFERENCE]
    assert secret not in card.description
    assert secret not in "".join(card.numeric_lines)
    assert plugin._store_for(agent)._blocks == {REFERENCE: None}


def test_a_result_naming_no_reference_writes_nothing_anywhere() -> None:
    """Requirement 15.9: nothing offloaded is the ordinary path — no Card, an empty store, no exception."""
    plugin = graph()
    agent = _Agent()

    plugin._on_after_tool_call(event(agent, offloaded("plain output, nothing offloaded")))

    assert plugin._states[agent].cards == {}
    assert len(plugin._store_for(agent)) == 0


def test_a_failed_tool_call_is_not_an_artifact() -> None:
    """A failed call carries an exception where a result would be: nothing to address, and nothing raised."""
    plugin = graph()
    agent = _Agent()

    plugin._on_after_tool_call(event(agent, RuntimeError("the tool exploded")))

    assert plugin._states[agent].cards == {}
    assert len(plugin._store_for(agent)) == 0


def test_several_references_are_carded_and_recorded_in_order() -> None:
    """One Card and one recorded reference per reference the placeholder names, without duplicates."""
    plugin = graph()
    agent = _Agent()

    plugin._on_after_tool_call(event(agent, offloaded("stashed [refs: tu-3_0, tu-3_1]")))

    assert set(plugin._states[agent].cards) == {"tu-3_0", "tu-3_1"}
    assert plugin._store_for(agent).references() == frozenset({"tu-3_0", "tu-3_1"})


def test_the_artifact_is_carded_under_the_turn_in_progress() -> None:
    """The ordinal is the graph's own, so the artifact is attributed to the turn that called the tool."""
    plugin = graph()
    agent = _Agent()
    plugin._state_for(agent).turn = 7

    plugin._on_after_tool_call(event(agent, offloaded()))

    assert plugin._states[agent].cards[REFERENCE].turn == 7


def test_the_store_is_always_present_and_is_per_agent() -> None:
    """Requirement 15.1: a store exists before anything offloads, and two agents never share one."""
    plugin = graph()
    first, second = _Agent(), _Agent()

    store = plugin._store_for(first)

    assert isinstance(store, InMemoryReferenceStore)
    assert len(store) == 0
    assert plugin._store_for(first) is store
    assert plugin._store_for(second) is not store

    plugin._on_after_tool_call(event(first, offloaded()))

    assert REFERENCE in plugin._store_for(first)
    assert REFERENCE not in plugin._store_for(second)
    assert plugin._state_for(second).cards == {}


def test_a_failed_derivation_keeps_no_artifact_card_and_does_not_propagate(monkeypatch, caplog) -> None:
    """Requirements 16.4, 16.8: the hook completes, the graph keeps nothing, and the failure stays in the log."""
    from strands_context_graph import plugin as plugin_module

    def explode(*args, **kwargs):
        raise RuntimeError("derivation exploded")

    monkeypatch.setattr(plugin_module, "derive_and_register_artifacts", explode)

    plugin = graph()
    agent = _Agent()

    with caplog.at_level(logging.WARNING, logger="strands_context_graph.plugin"):
        plugin._on_after_tool_call(event(agent, offloaded()))

    assert plugin._states[agent].cards == {}
    assert len(plugin._store_for(agent)) == 0
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None

    # No failure state is kept: the next tool call cards its artifact again.
    monkeypatch.undo()
    plugin._on_after_tool_call(event(agent, offloaded()))
    assert REFERENCE in plugin._states[agent].cards


def test_a_failed_store_write_still_leaves_the_card(monkeypatch, caplog) -> None:
    """The two writes are independent: a store that cannot record still leaves the Card the scan derived."""
    plugin = graph()
    agent = _Agent()

    def explode(self, reference: str) -> None:
        raise RuntimeError("store write failed")

    monkeypatch.setattr(InMemoryReferenceStore, "note", explode)

    with caplog.at_level(logging.WARNING, logger="strands_context_graph.store"):
        plugin._on_after_tool_call(event(agent, offloaded()))

    assert REFERENCE in plugin._states[agent].cards
    assert len(plugin._store_for(agent)) == 0
