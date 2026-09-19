"""A negative test of the ``NullConversationManager`` precondition: what a trimming manager actually costs.

Requirement 1.6 is stated as a precondition and enforced by nothing but a wiring-time ``warnings.warn``
(``test_plugin_init_agent.py`` asserts that notice fires, and this file deliberately does not repeat it). What is
asserted here is the *consequence* the notice warns about, so the precondition is a demonstrated fact rather than a
claim in a docstring:

The graph's fold is meant to be reversible. It collapses a turn to its Description in the per-call message list only,
leaving the turn whole in ``agent.messages``, which is what makes ``expand_card`` able to promise the turn back in full
for the rest of the turn. A trimming conversation manager edits that live list. Once it has run, the messages the graph
folded are *physically gone*: the fold stops being a projection and becomes data loss, ``expand_card`` still answers
"arrives in full" because the Card is still in the graph, and the very next delivery cannot put that content on the wire
at any Resolution. A rebuild-by-scan cannot recover it either, because the scan derives from the trimmed history.

Kept deterministic and offline: no model call (the model raises if it is ever reached), no matcher call (the Turn Choice
is handcrafted, exactly as ``test_projection.py`` does), and the manager is driven directly through
``apply_management``. The wiring notice is swallowed here on purpose — it is asserted elsewhere, and this file is about
what happens after it is ignored.

The test is guarded: it skips rather than fails if the SDK no longer exposes an in-place ``apply_management``, or if the
manager it is driven with stops dropping the turn this file relies on. A guard that skips keeps the negative test honest
— it can only pass by showing the loss, never by accident.
"""

from types import MappingProxyType
from typing import Any

import pytest
from strands import Agent
from strands._middleware.stages import InvokeModelContext
from strands.agent.conversation_manager import NullConversationManager, SlidingWindowConversationManager
from strands.hooks.events import MessageAddedEvent
from strands.models.model import Model

from strands_context_graph import ContextGraph
from strands_context_graph.state import CardChoice, TurnChoice

WINDOW_SIZE = 2
"""Small enough that the trim reaches the folded turn, which is the whole point of the fixture."""

FOLDED_TITLE = "the migration cost breakdown"
"""The user message of the turn the graph folds, and the Title its Card is identified by."""

FOLDED_ANSWER = "we moved every account over to the new cluster overnight"
"""The assistant answer in that same turn. The sentinel whose disappearance this file measures.

Deliberately free of numbers: a numeric line is copied into the Description literally, so a numeric sentinel would still
reach the call inside the folded block and could not tell a collapse apart from a deletion.
"""


class _Model(Model):
    """A model that cannot be called: the loss asserted below is structural, not something a turn revealed."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Fail loudly if a turn is ever run from these tests."""
        raise AssertionError("the model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Fail loudly if a turn is ever run from these tests."""
        raise AssertionError("the model was called")
        yield


def user(text: str, tracking_id: str) -> dict[str, Any]:
    """A plain user ask: a turn boundary, since it carries no tool result."""
    return {"role": "user", "content": [{"text": text}], "tracking_id": tracking_id}


def assistant(text: str, tracking_id: str) -> dict[str, Any]:
    """An assistant answer, part of its turn's dialogue."""
    return {"role": "assistant", "content": [{"text": text}], "tracking_id": tracking_id}


def conversation() -> list[dict[str, Any]]:
    """Three turns: one warm-up, the turn the graph folds, and the open turn carrying the live question.

    The folded turn sits in the middle for the same reason it does in ``test_projection.py``: the removal never drops
    the first user message, so a Card over turn one could not leave the call collapsed at all.
    """
    return [
        user("warm up", "m0"),
        assistant("warming up", "m1"),
        user(FOLDED_TITLE, "m2"),
        assistant(FOLDED_ANSWER, "m3"),
        user("what did we conclude", "m4"),
    ]


def build(manager: Any, graph: ContextGraph) -> Agent:
    """Wire ``graph`` onto an agent under ``manager``, ignoring the wiring notice asserted elsewhere."""
    with pytest.warns() if not isinstance(manager, NullConversationManager) else _quiet():
        return Agent(
            model=_Model(),
            conversation_manager=manager,
            plugins=[graph],
        )


class _quiet:
    """A no-op stand-in for ``pytest.warns`` on the branch where no notice is expected."""

    def __enter__(self) -> None:
        """Nothing to arm."""

    def __exit__(self, *_exc: object) -> None:
        """Nothing to check."""


def derive(graph: ContextGraph, agent: Agent, messages: list[dict[str, Any]]) -> None:
    """Replay ``messages`` through the write half, building the graph the way a live conversation does."""
    for message in messages:
        agent.messages.append(message)
        graph._on_message_added(MessageAddedEvent(agent=agent, message=message))  # type: ignore[arg-type]


def fold_the_middle_turn(graph: ContextGraph, agent: Agent) -> None:
    """Freeze a Turn Choice that puts the middle turn's Card at Description, the collapse this file is about.

    Handcrafted rather than scored, so no matcher and no embedding call is involved and the delivery under test is the
    same one production runs.
    """
    state = graph._states[agent]
    state.choice = TurnChoice(
        by_title=MappingProxyType({FOLDED_TITLE: CardChoice(dialogue="description", evidence="full")}),
        full_pass=False,
    )


def raise_the_middle_turn(graph: ContextGraph, agent: Agent) -> None:
    """Freeze a Turn Choice that puts that same Card back at Full Content, what ``expand_card`` promises."""
    state = graph._states[agent]
    state.choice = TurnChoice(
        by_title=MappingProxyType({FOLDED_TITLE: CardChoice(dialogue="full", evidence="full")}),
        full_pass=False,
    )


def context_over(agent: Agent) -> InvokeModelContext:
    """An ``InvokeModelContext`` over the agent's current live history, as the middleware receives it."""
    return InvokeModelContext(
        agent=agent,
        messages=list(agent.messages),
        system_prompt=None,
        tool_specs=[],
        tool_choice=None,
        invocation_state={},
        model=agent.model,
        projected_input_tokens=0,
        dynamic_trailing_blocks=0,
    )


async def delivered_text(graph: ContextGraph, agent: Agent) -> str:
    """Every piece of text the delivery would send for one call, folded block included, as one string."""
    context = await graph._projection.deliver(context_over(agent))
    return "\n".join(
        block["text"]
        for message in context.messages
        for block in message.get("content", [])
        if isinstance(block, dict) and "text" in block
    )


def identities(agent: Agent) -> set[str]:
    """The Durable Identities still present in the live history."""
    return {message["tracking_id"] for message in agent.messages if message.get("tracking_id")}


@pytest.fixture
def trimmed() -> tuple[ContextGraph, Agent, SlidingWindowConversationManager]:
    """A wired agent with a derived graph and a manager that will trim the folded turn out of the live list."""
    if not callable(getattr(SlidingWindowConversationManager, "apply_management", None)):
        pytest.skip("the SDK no longer exposes an in-place apply_management to drive the trim with")

    graph = ContextGraph()
    manager = SlidingWindowConversationManager(window_size=WINDOW_SIZE)
    agent = build(manager, graph)
    derive(graph, agent, conversation())
    return graph, agent, manager


def trim(agent: Agent, manager: SlidingWindowConversationManager) -> None:
    """Run the manager over the live history, and skip unless it reached the turn this file is about."""
    manager.apply_management(agent)

    if {"m2", "m3"} & identities(agent):
        pytest.skip(f"{type(manager).__name__} no longer trims the folded turn; the negative test has nothing to show")


# ---- the baseline: under the precondition, the fold is reversible ---------------------------------


@pytest.mark.asyncio
async def test_under_the_null_manager_the_folded_turn_survives_in_the_live_history():
    """The precondition met: the fold removes the turn from the call and leaves it whole where it is owned."""
    graph = ContextGraph()
    agent = build(NullConversationManager(), graph)
    derive(graph, agent, conversation())
    fold_the_middle_turn(graph, agent)

    sent = await delivered_text(graph, agent)

    # Collapsed on the wire: the Title is folded in, the assistant's answer is not sent.
    assert FOLDED_ANSWER not in sent
    assert FOLDED_TITLE in sent
    # Still owned by the live history, which is what makes the collapse a projection rather than a deletion.
    assert FOLDED_ANSWER in str(agent.messages)
    assert {"m2", "m3"} <= identities(agent)


@pytest.mark.asyncio
async def test_under_the_null_manager_raising_the_resolution_recovers_the_turn():
    """The reversibility ``expand_card`` promises, shown end to end: back at Full Content the answer is sent again."""
    graph = ContextGraph()
    agent = build(NullConversationManager(), graph)
    derive(graph, agent, conversation())

    fold_the_middle_turn(graph, agent)
    assert FOLDED_ANSWER not in await delivered_text(graph, agent)

    raise_the_middle_turn(graph, agent)
    assert FOLDED_ANSWER in await delivered_text(graph, agent)


# ---- the negative test: a trimming manager makes the same fold a loss -----------------------------


def test_a_trimming_manager_removes_the_folded_turn_from_the_live_history(trimmed):
    """The mechanism of the loss: the manager edits the list the graph relies on owning the content."""
    graph, agent, manager = trimmed
    assert {"m2", "m3"} <= identities(agent)

    trim(agent, manager)

    # Physically gone, not collapsed: no Resolution can address a message that is no longer in the list.
    assert {"m2", "m3"}.isdisjoint(identities(agent))
    assert FOLDED_ANSWER not in str(agent.messages)


def test_the_card_outlives_the_messages_it_addresses(trimmed):
    """The graph is left describing a turn that no longer exists, since it holds addresses and not content."""
    graph, agent, manager = trimmed
    trim(agent, manager)

    card = graph._states[agent].cards[FOLDED_TITLE]

    assert card.dialogue_ids == ("m2", "m3")
    assert set(card.dialogue_ids).isdisjoint(identities(agent))


@pytest.mark.asyncio
async def test_raising_the_resolution_recovers_nothing_after_the_trim(trimmed):
    """The consequence Requirement 1.6 names: Full Content delivers nothing, there being nothing left to deliver."""
    graph, agent, manager = trimmed
    fold_the_middle_turn(graph, agent)
    trim(agent, manager)

    raise_the_middle_turn(graph, agent)
    sent = await delivered_text(graph, agent)

    # Under the null manager this exact call sends the answer back (see the baseline above). Here it cannot.
    assert FOLDED_ANSWER not in sent
    assert FOLDED_TITLE not in sent


def test_expand_card_still_promises_a_turn_it_can_no_longer_produce(trimmed):
    """Why the runtime cannot recover: the tool reads the graph, and the graph still has the Card.

    Nothing about the tool is broken — it answers, it raises the Resolution, it does not raise an exception. It simply
    promises content that the manager deleted, which is exactly why the protection has to be the wiring-time notice.
    """
    from strands_context_graph import tools

    graph, agent, manager = trimmed
    fold_the_middle_turn(graph, agent)
    trim(agent, manager)

    answer = tools.expand_card(graph._states[agent], FOLDED_TITLE, cycle=0, reuse_ttl_cycles=0)

    assert "arrives in full" in answer
    assert graph._states[agent].choice.by_title[FOLDED_TITLE] == CardChoice(dialogue="full", evidence="full")
    assert FOLDED_ANSWER not in str(agent.messages)


def test_a_rebuild_by_scan_cannot_recover_the_trimmed_turn(trimmed):
    """The last door closed: the scan derives from the live history, so the Description goes with the messages."""
    graph, agent, manager = trimmed
    trim(agent, manager)

    state = graph._states[agent]
    graph._rebuild(agent, state)

    assert FOLDED_TITLE not in state.cards
    assert all(FOLDED_ANSWER not in card.description for card in state.cards.values())
