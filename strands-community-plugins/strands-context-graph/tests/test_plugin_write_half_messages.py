"""Unit tests for the write half of the turn boundary, run by the projection on every model call (D-1).

Under D-1 there is no ``MessageAddedEvent`` hook: ``project`` closes turns into Cards at the start of every model
call. A model call follows every user message (an ask or a tool result) and never an assistant answer, so the
replay below projects after each user message, which is exactly when a live agent would.

The derivation itself belongs to ``cards`` and is asserted there. What is worth asserting here is the composition: that
a boundary closes the turn before it and nothing else, that a mid-turn message closes nothing, that an absent state in
front of a conversation with a closed boundary is rebuilt by one scan, that the scan equals the incremental graph, that
a vanished message is referenced by nothing, and that no failure of either route reaches the caller or the model.

The agent is a plain class carrying only ``messages``, so it is weak-referenceable, which the per-agent state map
requires.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from context_core.graph.cards import rebuild_into
from context_core.graph.projection import _card_config
from strands_context_graph._compat import InvokeModelContext

from strands_context_graph import ContextGraph


class _Agent:
    """The one member the write half reads off an agent."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = messages if messages is not None else []


class UnusableMatcher:
    """A matcher that fails if it is ever reached: the write half pays for no embedding."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Fail the test rather than answer."""
        raise AssertionError("the write half must not reach the matcher")


def graph() -> ContextGraph:
    """A plugin whose matcher would fail loudly, since nothing here should reach it."""
    return ContextGraph(matcher=UnusableMatcher())


def user(text: str, tracking_id: str | None = None) -> dict[str, Any]:
    """A plain user ask, the shape that opens a turn and closes the one before it."""
    message: dict[str, Any] = {"role": "user", "content": [{"text": text}]}
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def assistant(text: str, tracking_id: str | None = None) -> dict[str, Any]:
    """An assistant answer, which is mid-turn and closes nothing."""
    message: dict[str, Any] = {"role": "assistant", "content": [{"text": text}]}
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def project_call(plugin: ContextGraph, agent: _Agent) -> None:
    """Run the projection of one model call over the agent's current history."""
    context = InvokeModelContext(
        agent=agent,
        messages=list(agent.messages),
        system_prompt=None,
        tool_specs=[],
        tool_choice=None,
        invocation_state={},
        model=object(),
        projected_input_tokens=None,
        dynamic_trailing_blocks=0,
    )
    asyncio.run(plugin._projection.deliver(context))


def add(plugin: ContextGraph, agent: _Agent, message: dict[str, Any]) -> None:
    """Append ``message`` the way the framework does; a user message is followed by a model call, so project."""
    agent.messages.append(message)
    if message["role"] == "user":
        project_call(plugin, agent)


def conversation() -> list[dict[str, Any]]:
    """Three asks and two answers: two closed turns plus a turn in progress."""
    return [
        user("what is the balance of account 42?", "m1"),
        assistant("the balance is 1,204.55 USD", "m2"),
        user("and the overdraft limit?", "m3"),
        assistant("the limit is 500.00 USD", "m4"),
        user("summarize both", "m5"),
    ]


def drive(plugin: ContextGraph, agent: _Agent, messages: list[dict[str, Any]]) -> None:
    """Replay ``messages`` one at a time, projecting where a model call would happen: the incremental build."""
    for message in messages:
        add(plugin, agent, message)


def test_nothing_is_derived_before_the_first_boundary_closes_a_turn() -> None:
    """Requirement 16.5: a turn is closed by what comes after it, so the first turn has no Card yet."""
    plugin = graph()
    agent = _Agent()

    drive(plugin, agent, conversation()[:2])

    state = plugin._states[agent]
    assert state.cards == {}
    assert state.links == {}


def test_a_boundary_closes_the_turn_before_it() -> None:
    """Requirement 3.10: the Card of the turn that just ended, with its Title, Description, Tags and Links."""
    plugin = graph()
    agent = _Agent()

    drive(plugin, agent, conversation()[:3])

    state = plugin._states[agent]
    (card,) = state.cards.values()
    assert card.kind == "subject"
    assert card.turn == 0
    assert card.title
    assert card.description
    assert card.dialogue_ids == ("m1", "m2")
    # A Card exists in the link index even when it links to nothing yet.
    assert card.title in state.links


def test_a_mid_turn_message_closes_nothing() -> None:
    """An assistant answer is inside the turn: no boundary, so no Card is derived and none is disturbed."""
    plugin = graph()
    agent = _Agent()
    drive(plugin, agent, conversation()[:3])
    before = dict(plugin._states[agent].cards)

    add(plugin, agent, assistant("the limit is 500.00 USD", "m4"))

    assert plugin._states[agent].cards == before


def test_each_boundary_closes_exactly_one_more_turn() -> None:
    """The ordinal is the position of the boundary, so the Cards come out in turn order, one per closed turn."""
    plugin = graph()
    agent = _Agent()

    drive(plugin, agent, conversation())

    turns = sorted(card.turn for card in plugin._states[agent].cards.values())
    assert turns == [0, 1]


def test_an_absent_state_is_rebuilt_by_scan_from_the_restored_history() -> None:
    """Requirement 14.5: a process that inherited the conversation derives the whole graph in one scan."""
    plugin = graph()
    restored = conversation()
    agent = _Agent(list(restored))

    # A restore populates ``agent.messages`` directly and fires no event the incremental step could have seen; the next
    # message added is what the fresh process sees first.
    project_call(plugin, agent)

    state = plugin._states[agent]
    assert sorted(card.turn for card in state.cards.values()) == [0, 1]
    # The scan sets the ordinal to the count of closed boundaries (2), and the call that ran it advances it by one.
    assert state.turn == 3


def test_a_rebuild_by_scan_equals_the_incremental_graph() -> None:
    """Requirements 14.3, 14.5: same Cards, same Links, same Tags, whichever route built them."""
    incremental, scanned = graph(), graph()
    walked, restored = _Agent(), _Agent(conversation())

    drive(incremental, walked, conversation())
    project_call(scanned, restored)

    left, right = incremental._states[walked], scanned._states[restored]
    assert left.cards == right.cards
    assert left.links == right.links
    # ``turn`` advances once per model call, and the two routes made different numbers of calls; what they have to
    # agree on is the graph, and the ordinal carried by each Card is part of it.


def test_the_rebuild_is_idempotent() -> None:
    """The scan clears before it writes, so seeing a second boundary-less message leaves the same graph."""
    plugin = graph()
    agent = _Agent(conversation())

    project_call(plugin, agent)
    first = dict(plugin._states[agent].cards)
    rebuild_into(plugin._states[agent], agent.messages, **_card_config(plugin._thresholds()))

    assert plugin._states[agent].cards == first


def test_an_empty_conversation_derives_nothing_and_does_not_raise() -> None:
    """Requirement 14.9: no closed boundary is nothing to derive and nothing to rebuild from."""
    plugin = graph()
    agent = _Agent()

    add(plugin, agent, user("first ask", "m1"))

    state = plugin._states[agent]
    assert state.cards == {}
    # One model call ran, so the ordinal moved once; nothing was closed, so nothing was derived.
    assert state.turn == 1


def test_a_turn_without_durable_identities_yields_no_card() -> None:
    """A turn whose messages carry no ``tracking_id`` has no Card: its messages travel whole."""
    plugin = graph()
    agent = _Agent()

    drive(plugin, agent, [user("untracked ask"), assistant("untracked answer"), user("tracked ask", "m3")])

    assert plugin._states[agent].cards == {}


def test_a_card_references_only_messages_present_in_the_history() -> None:
    """Requirement 14.7: a vanished message is referenced by nothing, so no Resolution decision sees it."""
    plugin = graph()
    restored = conversation()
    # The answer of the first turn never made it into the restored history.
    del restored[1]
    agent = _Agent(list(restored))

    project_call(plugin, agent)

    present = {message["tracking_id"] for message in restored if message.get("tracking_id")}
    for card in plugin._states[agent].cards.values():
        assert set(card.dialogue_ids) <= present
        assert set(card.evidence_ids) <= present


def test_the_scan_leaves_the_vector_cache_alone() -> None:
    """Requirement 14.8: the cache is per process, so a rebuild costs at most an embedding, never a Card."""
    plugin = graph()
    agent = _Agent(conversation())
    state = plugin._state_for(agent)
    state.vectors["carried over"] = ("some description", (0.1, 0.2))

    project_call(plugin, agent)

    after = plugin._states[agent]
    assert after.vectors["carried over"] == ("some description", (0.1, 0.2))
    assert after.cards


def test_a_failed_card_derivation_registers_nothing_and_does_not_propagate(monkeypatch, caplog) -> None:
    """Requirements 16.4, 16.5: the call completes, that turn keeps no Card, the failure stays in the log."""
    from context_core.graph import cards as cards_module

    plugin = graph()
    agent = _Agent()
    drive(plugin, agent, conversation()[:2])

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("description exploded")

    monkeypatch.setattr(cards_module, "derive_card", explode)

    with caplog.at_level(logging.WARNING, logger="context_core.graph.cards"):
        add(plugin, agent, user("and the overdraft limit?", "m3"))

    assert plugin._states[agent].cards == {}
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None

    # No failure state is kept: the next boundary closes its turn again.
    monkeypatch.undo()
    add(plugin, agent, user("summarize both", "m5"))
    assert plugin._states[agent].cards


def test_a_failed_rebuild_keeps_no_card_and_does_not_propagate(monkeypatch, caplog) -> None:
    """Requirement 16.4: a scan that could not run leaves an empty graph, which is a full-content conversation."""
    from context_core.graph import projection as projection_module

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(projection_module, "rebuild_into", explode)

    plugin = graph()
    agent = _Agent(conversation())

    with caplog.at_level(logging.WARNING, logger="context_core.graph.projection"):
        project_call(plugin, agent)

    assert plugin._states[agent].cards == {}
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_two_agents_on_one_instance_keep_independent_graphs() -> None:
    """Requirement 1.5: the state is keyed by the agent, so one agent's boundary is not the other's."""
    plugin = graph()
    first, second = _Agent(), _Agent()

    drive(plugin, first, conversation()[:3])

    assert plugin._states[first].cards
    assert plugin._state_for(second).cards == {}
