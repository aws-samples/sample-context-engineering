"""The assembled plugin, driven through a real ``Agent``: all four engagement points, together, on one conversation.

Every other test in this package exercises one seam. This one asserts that the seams are actually wired to each other,
which is the only claim no unit test can make: the read half freezes a choice, the middleware turns that choice into the
message list the provider receives, the two write halves build the graph the choice was computed from, and a retrieval
tool reaches back into a turn the choice collapsed — all within one agent, in one conversation, in the order a live run
produces them.

Offline and deterministic. The model is a scripted stub that records what it was handed and never reaches a network, the
similarity matcher is a fixed table so the Turn Choice is a function of the script rather than of an embedding model,
and the only tool the agent owns returns a constant. Nothing here needs credentials, and nothing here can be slow.

Two scenarios, because they need different scripts:

- The collapse scenario (``collapse_run``) tunes the thresholds so exactly one earlier turn lands at Description, and
  scripts the model to call ``expand_card`` for it. That single turn carries the round trip: the first call of the turn
  goes out with that turn folded into a block, the tool raises it, and the *second* call of the same turn — the
  autonomous tool loop's — carries the turn whole again.
- The regression scenario (``expand_threshold=0.0``) needs a script that runs identically with and *without* the plugin,
  so it scripts no retrieval tool: those tools do not exist on the unwired agent. The two runs are then compared call by
  call.

What "field for field" means for the regression switch, precisely. What the provider is handed is the argument list of
``Model.stream``, which the SDK reduces to ``role`` and ``content`` per message, so the comparison is over content and
not over identities the agent assigns at random. The one field that legitimately differs is ``tool_specs``: the plugin
registers its three retrieval tools (Requirement 1.2), which is a registration the switch does not claim to undo, so the
comparison holds those three names aside and requires the rest to match exactly — including the absence of the
``dynamic_trailing_blocks`` argument, the SDK's own signal that a trailing block was folded in.
"""

import asyncio
import copy
import functools
import json
from collections.abc import AsyncGenerator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models.model import Model
from strands.tools.decorator import tool

from strands_context_graph import ContextGraph

# ---- the conversation the scenarios share ---------------------------------------------------------

WARM_UP_ASK = "open the ledger project with me"
"""Turn one. Its Card can never be collapsed — the removal always keeps the first user message — so it is here to make
the graph big enough to be worth scoring, and to be the Card that stays at Full Content."""

MIGRATION_ASK = "the migration cost breakdown"
"""Turn two's user message, and therefore the Title of its Card: a short message is its own Title verbatim."""

MIGRATION_ANSWER = "we moved every account over to the new cluster overnight"
"""Turn two's answer, and the sentinel of this file: present in the payload means the turn travelled whole.

Free of numbers on purpose. A numeric line is copied into the Description literally, so a numeric sentinel would still
reach the provider inside the folded block and could not tell a collapse apart from a full delivery.
"""

LEDGER_ASK = "check the ledger for march"
"""Turn three, the turn that calls the agent's own tool, which is what puts ``AfterToolCallEvent`` on the path."""

RECALL_ASK = "so what did that migration actually cost us"
"""Turn four: the turn whose choice collapses turn two, and whose tool loop asks for it back."""

LEDGER_RESULT = "march ledger: 412 entries, 3 unreconciled"
"""What the agent's tool returns. Constant, so the turn is a function of the script alone."""

PLUGIN_TOOL_NAMES = frozenset({"expand_card", "expand_artifact", "find_context"})
"""The three tools Requirement 1.2 adds to the registry, held aside in the regression comparison."""

COLLAPSED_HEADER = "<collapsed_turns>"
"""The compaction's own marker, which is how the payload is asked whether a folded block reached it."""

HIGH_SIMILARITY = 0.95
"""Table score for a Card the turn's question is about: above ``expand_threshold``, so Full Content."""

LOW_SIMILARITY = 0.30
"""Table score for the Card this file collapses.

Between the two thresholds the scenario configures, and it stays there after propagation: the only Note that could reach
turn two's Card is inherited from a Card that follows it, and at the instant the choice is computed no such Card exists
yet — turn three closes only when turn four's message arrives.
"""

EXPAND_THRESHOLD = 0.90
"""Deliberately far from the default. The scenario wants one Card clearly above and one clearly below, so the assertion
is about the wiring and not about how close a calibrated default sits to a table value."""

COLLAPSE_FLOOR = 0.20
"""The other side of the same interval, wide for the same reason."""

MIN_CARDS = 2
"""Two rather than the default three.

``BeforeInvocationEvent`` fires *before* the turn's message is appended, and a turn is closed by the message that
follows it, so at the instant turn four's choice is computed the graph holds the Cards of turns one and two and not yet
turn three's. Two is what makes a four-turn conversation reach the scoring path at all.
"""


@tool
def ledger_lookup(month: str) -> str:
    """Return a fixed ledger line for ``month``.

    Args:
        month: The month asked for. Echoed nowhere; the answer is constant by design.

    Returns:
        The constant result.
    """
    return LEDGER_RESULT


# ---- the recording model -------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedCall:
    """One provider call, as the model received it.

    Attributes:
        messages: The messages handed to the provider, deep-copied. The SDK reduces each to ``role`` and ``content``
            before this point, so no agent-assigned identity is in here.
        tool_specs: The tool specifications, deep-copied, or ``None``.
        system_prompt: The system prompt.
        tool_choice: The tool choice.
        dynamic_trailing_blocks: The SDK passes this argument only when a trailing block was folded in, so ``None`` here
            means the call carried none.
    """

    messages: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]] | None
    system_prompt: str | None
    tool_choice: Any
    dynamic_trailing_blocks: int | None

    @property
    def text(self) -> str:
        """Every piece of text in the payload, as one string, for asking what reached the provider."""
        return "\n".join(
            block["text"]
            for message in self.messages
            for block in message.get("content", ())
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    @property
    def tool_names(self) -> frozenset[str]:
        """Names of the tools offered on this call."""
        return frozenset(spec["name"] for spec in self.tool_specs or ())

    def comparable(self) -> dict[str, Any]:
        """The fields the regression switch claims to leave untouched, with the plugin's own tools held aside."""
        return {
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "tool_choice": self.tool_choice,
            "dynamic_trailing_blocks": self.dynamic_trailing_blocks,
            "tool_specs": sorted(self.tool_names - PLUGIN_TOOL_NAMES),
        }


class ScriptedModel(Model):
    """A model that replays a script and records every call, so no turn depends on a network.

    Args:
        script: One assistant message per provider call, in order. A message carrying a ``toolUse`` block stops with
            ``tool_use``, which is what drives the agent's tool loop.
    """

    def __init__(self, script: Sequence[dict[str, Any]]) -> None:
        """Keep the script and start the recording."""
        self.script = list(script)
        self.calls: list[RecordedCall] = []

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Not part of any scenario here."""
        raise AssertionError("structured output was requested")
        yield

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Record the call, then replay the next scripted response as a stream of events."""
        self.calls.append(
            RecordedCall(
                messages=copy.deepcopy(list(messages)),
                tool_specs=copy.deepcopy(list(tool_specs)) if tool_specs else None,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                dynamic_trailing_blocks=kwargs.get("dynamic_trailing_blocks"),
            )
        )

        if len(self.calls) > len(self.script):
            raise AssertionError(f"the model was called {len(self.calls)} times for a script of {len(self.script)}")

        for event in _events_for(self.script[len(self.calls) - 1]):
            yield event


def _events_for(message: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Render one scripted assistant message as the stream events the SDK parses."""
    stop_reason = "end_turn"
    yield {"messageStart": {"role": "assistant"}}

    for block in message["content"]:
        if "text" in block:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
            yield {"contentBlockStop": {}}
        if "toolUse" in block:
            stop_reason = "tool_use"
            use = block["toolUse"]
            yield {"contentBlockStart": {"start": {"toolUse": {"name": use["name"], "toolUseId": use["toolUseId"]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(use["input"])}}}}
            yield {"contentBlockStop": {}}

    yield {"messageStop": {"stopReason": stop_reason}}


def text_response(text: str) -> dict[str, Any]:
    """A scripted assistant answer of plain text."""
    return {"role": "assistant", "content": [{"text": text}]}


def tool_response(name: str, tool_use_id: str, **payload: Any) -> dict[str, Any]:
    """A scripted assistant answer that calls a tool."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"name": name, "toolUseId": tool_use_id, "input": payload}}],
    }


# ---- the table matcher ---------------------------------------------------------------------------


@dataclass
class TableMatcher:
    """A similarity matcher of fixed answers: ``LOW_SIMILARITY`` for the Card this file collapses, high for the rest.

    Keyed on the Description's own text rather than on position, so the answer does not move when the graph gains a
    Card. Counting the calls is the other half of its job: the read half is allowed exactly one scoring round per turn.

    Attributes:
        questions: The question of every call, in order.
        documents: The Descriptions of every call, in order.
    """

    questions: list[str] = field(default_factory=list)
    documents: list[tuple[str, ...]] = field(default_factory=list)

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one score per Description, in the order received."""
        self.questions.append(question)
        self.documents.append(tuple(descriptions))
        return [LOW_SIMILARITY if MIGRATION_ASK in text else HIGH_SIMILARITY for text in descriptions]

    @property
    def calls(self) -> int:
        """How many scoring rounds have been asked for."""
        return len(self.questions)


@dataclass
class ForbiddenMatcher:
    """A matcher that records being consulted, which under ``expand_threshold=0.0`` must never happen.

    It records rather than raises: the read half fails open, so a raise would be swallowed into a full pass and the
    payload comparison would still succeed. A flag is the only evidence that survives.

    Attributes:
        consulted: Whether anything asked for a similarity.
    """

    consulted: bool = False

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Record the call and answer with nothing, which the graph reads as "send everything"."""
        self.consulted = True
        return []


# ---- the spy, for "did all four engagement points fire" -------------------------------------------


class SpyGraph(ContextGraph):
    """The real plugin, recording which engagement point ran and in what order.

    Only recording: every handler delegates to the implementation under test, and the delivery is wrapped on the
    ``Projection`` instance *before* registration, so what the middleware registry receives is the wrapper around the
    real delivery rather than a stand-in for it.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Wire the recorder around the real delivery."""
        super().__init__(**kwargs)
        self.fired: list[str] = []
        real_deliver = self._projection.deliver

        @functools.wraps(real_deliver)
        async def recording_deliver(context: Any) -> Any:
            self.fired.append("delivery")
            return await real_deliver(context)

        self._projection.deliver = recording_deliver  # type: ignore[method-assign]

    def _on_before_invocation(self, event: Any) -> None:
        self.fired.append("before_invocation")
        super()._on_before_invocation(event)

    def _on_message_added(self, event: Any) -> None:
        self.fired.append("message_added")
        super()._on_message_added(event)

    def _on_after_tool_call(self, event: Any) -> None:
        self.fired.append("after_tool_call")
        super()._on_after_tool_call(event)


# ---- the scenarios -------------------------------------------------------------------------------


def build_agent(model: ScriptedModel, graph: ContextGraph | None) -> Agent:
    """An agent under the precondition, with the ledger tool, wired to ``graph`` when one is given."""
    return Agent(
        model=model,
        tools=[ledger_lookup],
        conversation_manager=NullConversationManager(),
        plugins=[graph] if graph is not None else [],
    )


COLLAPSE_SCRIPT = (
    text_response("ready when you are"),
    text_response(MIGRATION_ANSWER),
    tool_response("ledger_lookup", "use-ledger", month="march"),
    text_response("three entries are still open"),
    tool_response("expand_card", "use-expand", titles=[MIGRATION_ASK]),
    text_response("the migration cost is in the turn you just gave me back"),
)
"""Six provider calls over four turns: turn three's tool loop takes two, turn four's takes two."""

PLAIN_SCRIPT = (
    text_response("ready when you are"),
    text_response(MIGRATION_ANSWER),
    tool_response("ledger_lookup", "use-ledger", month="march"),
    text_response("three entries are still open"),
    text_response("nothing further"),
)
"""The same conversation with no retrieval tool call, so it replays identically on an agent with no plugin."""

ASKS = (WARM_UP_ASK, MIGRATION_ASK, LEDGER_ASK, RECALL_ASK)
"""The four user messages, in order."""


def run(script: Sequence[dict[str, Any]], graph: ContextGraph | None) -> tuple[Agent, ScriptedModel]:
    """Drive the whole four-turn conversation and hand back the agent and what the provider saw.

    Driven through ``asyncio.run`` so the fixtures below are ordinary synchronous ones: the conversation is over by the
    time a test reads anything, and everything a test then inspects is recorded data with no tie to a loop.
    """

    async def conversation() -> tuple[Agent, ScriptedModel]:
        model = ScriptedModel(script)
        agent = build_agent(model, graph)
        for ask in ASKS:
            await agent.invoke_async(ask)
        return agent, model

    return asyncio.run(conversation())


@pytest.fixture
def collapse_run() -> tuple[SpyGraph, Agent, ScriptedModel, TableMatcher]:
    """The tuned scenario: one turn collapsed, then asked for back by name."""
    matcher = TableMatcher()
    graph = SpyGraph(
        expand_threshold=EXPAND_THRESHOLD,
        collapse_floor=COLLAPSE_FLOOR,
        min_cards=MIN_CARDS,
        matcher=matcher,
    )
    agent, model = run(COLLAPSE_SCRIPT, graph)
    return graph, agent, model, matcher


# ---- all four engagement points, on one conversation ---------------------------------------------


def test_all_four_engagement_points_fire_on_one_conversation(collapse_run):
    """Requirement 1.1, observed rather than inspected: each of the four ran while a real agent ran."""
    graph, _agent, _model, _matcher = collapse_run

    assert {"before_invocation", "message_added", "after_tool_call", "delivery"} == set(graph.fired)


def test_the_read_half_precedes_every_delivery_of_its_turn(collapse_run):
    """The ordering the frozen choice depends on: a delivery never runs before the turn's choice exists."""
    graph, _agent, _model, _matcher = collapse_run
    ordered = [step for step in graph.fired if step in {"before_invocation", "delivery"}]

    assert ordered[0] == "before_invocation"


def test_the_write_halves_built_the_graph_the_choice_was_computed_from(collapse_run):
    """The two write halves left a graph behind, keyed to this agent and holding the collapsed turn's Card."""
    graph, agent, _model, _matcher = collapse_run
    state = graph._states[agent]

    assert MIGRATION_ASK in state.cards
    assert state.cards[MIGRATION_ASK].kind == "subject"
    # One Card per closed turn: three of the four turns are closed by the time the run ends.
    assert len(state.cards) == len(ASKS) - 1


def test_the_live_history_still_holds_every_turn(collapse_run):
    """The whole point of a projection: nothing the plugin did removed anything from ``agent.messages``."""
    _graph, agent, _model, _matcher = collapse_run
    history = str(agent.messages)

    assert MIGRATION_ANSWER in history
    assert all(ask in history for ask in ASKS)


def test_the_matcher_is_consulted_once_for_the_turn_that_reaches_the_scoring_path(collapse_run):
    """Requirement 11.3: one scoring round per turn, and none at all for a turn the graph's size settles."""
    _graph, _agent, _model, matcher = collapse_run

    # Turns one to three are below ``min_cards`` at the instant their choice is computed, so only turn four scores.
    assert matcher.calls == 1
    assert matcher.questions == [RECALL_ASK]
    assert any(MIGRATION_ASK in document for document in matcher.documents[0])


# ---- the collapsed block reaches the provider ----------------------------------------------------


def calls_of_the_last_turn(model: ScriptedModel) -> list[RecordedCall]:
    """The provider calls of turn four: the last two of the collapse script."""
    return model.calls[-2:]


def test_the_collapsed_turn_is_folded_into_the_payload(collapse_run):
    """The middleware's whole job, seen from the provider: the turn is gone from the messages and named in a block."""
    _graph, _agent, model, _matcher = collapse_run
    first_call, _second_call = calls_of_the_last_turn(model)

    assert MIGRATION_ANSWER not in first_call.text
    assert COLLAPSED_HEADER in first_call.text
    assert MIGRATION_ASK in first_call.text
    # The SDK's own count of folded blocks, which is how the provider is told the last message carries one.
    assert first_call.dynamic_trailing_blocks == 1


def test_the_folded_block_rides_on_the_last_user_message(collapse_run):
    """Where the injection primitive puts it, which is what keeps the block out of a message of its own."""
    _graph, _agent, model, _matcher = collapse_run
    first_call, _second_call = calls_of_the_last_turn(model)
    last = first_call.messages[-1]

    assert last["role"] == "user"
    assert COLLAPSED_HEADER in "\n".join(
        block["text"] for block in last["content"] if isinstance(block.get("text"), str)
    )


def test_the_earlier_turns_the_choice_kept_still_reach_the_provider(collapse_run):
    """A collapse is per Card: the Card above the threshold travelled whole on the very same call."""
    _graph, _agent, model, _matcher = collapse_run
    first_call, _second_call = calls_of_the_last_turn(model)

    assert WARM_UP_ASK in first_call.text
    assert RECALL_ASK in first_call.text


# ---- the retrieval tool round trip ---------------------------------------------------------------


def test_a_retrieval_tool_round_trips_within_the_turn(collapse_run):
    """The round trip end to end: collapsed on the first call, asked for by name, whole on the second.

    This is the assertion no unit test reaches. The tool is invoked by the agent's own tool loop, from a tool
    specification the plugin registered, and what proves it worked is the *next* provider call of the same turn.
    """
    _graph, _agent, model, _matcher = collapse_run
    first_call, second_call = calls_of_the_last_turn(model)

    assert MIGRATION_ANSWER not in first_call.text
    assert MIGRATION_ANSWER in second_call.text
    # Nothing left to collapse, so the delivery returns the received context and folds no block at all.
    assert COLLAPSED_HEADER not in second_call.text
    assert second_call.dynamic_trailing_blocks is None


def test_the_tool_answered_rather_than_raised(collapse_run):
    """Requirement 12.2's answer is in the history, which is also how the model learned the turn was coming back."""
    _graph, agent, _model, _matcher = collapse_run
    results = [
        block["toolResult"]
        for message in agent.messages
        for block in message.get("content", ())
        if isinstance(block, dict) and "toolResult" in block
    ]
    expand_results = [result for result in results if result["toolUseId"] == "use-expand"]

    assert len(expand_results) == 1
    assert expand_results[0]["status"] == "success"
    assert "arrives in full" in str(expand_results[0]["content"])


def test_the_three_retrieval_tools_are_offered_to_the_provider(collapse_run):
    """Requirement 1.2 from the provider's side: exactly the three, alongside the agent's own tool."""
    _graph, _agent, model, _matcher = collapse_run

    assert PLUGIN_TOOL_NAMES <= model.calls[0].tool_names
    assert model.calls[0].tool_names == PLUGIN_TOOL_NAMES | {"ledger_lookup"}


# ---- the regression switch: expand_threshold=0.0 --------------------------------------------------


@pytest.fixture
def regression_runs() -> tuple[ScriptedModel, ScriptedModel, ForbiddenMatcher]:
    """The same conversation twice: once with the switch thrown, once with no plugin at all."""
    matcher = ForbiddenMatcher()
    graph = ContextGraph(
        expand_threshold=0.0,
        # The relation ``collapse_floor <= expand_threshold`` holds the floor down with the ceiling.
        collapse_floor=0.0,
        min_cards=MIN_CARDS,
        matcher=matcher,
    )
    _with_agent, with_plugin = run(PLAIN_SCRIPT, graph)
    _without_agent, without_plugin = run(PLAIN_SCRIPT, None)
    return with_plugin, without_plugin, matcher


def test_the_regression_switch_produces_the_same_number_of_calls(regression_runs):
    """The conversation ran the same way on both agents, which is the precondition of comparing the calls."""
    with_plugin, without_plugin, _matcher = regression_runs

    assert len(with_plugin.calls) == len(without_plugin.calls) == len(PLAIN_SCRIPT)


def test_the_regression_switch_produces_field_for_field_identical_calls(regression_runs):
    """Requirement 2.20: with the switch thrown, the provider cannot tell the plugin is installed.

    Compared call by call rather than in bulk, so a failure names the call that diverged.
    """
    with_plugin, without_plugin, _matcher = regression_runs

    for index, (wired, bare) in enumerate(zip(with_plugin.calls, without_plugin.calls, strict=True)):
        assert wired.comparable() == bare.comparable(), f"call {index} diverged"


def test_the_regression_switch_folds_no_trailing_block(regression_runs):
    """The identity is delivered by returning the received context, so no call carries a block."""
    with_plugin, _without_plugin, _matcher = regression_runs

    assert all(call.dynamic_trailing_blocks is None for call in with_plugin.calls)
    assert all(COLLAPSED_HEADER not in call.text for call in with_plugin.calls)


def test_the_regression_switch_pays_for_no_similarity_call(regression_runs):
    """The switch short-circuits before the matcher, so the identity costs nothing remote either.

    ``ForbiddenMatcher`` raising would have been swallowed by the read half's own fail-open, so the assertion is that
    the payloads stayed identical *and* that the matcher recorded nothing — the run reaching it would have degraded to a
    full pass and still compared equal.
    """
    with_plugin, _without_plugin, matcher = regression_runs

    assert not matcher.consulted
    # The three retrieval tools are still registered: the switch silences the projection, not Requirement 1.2.
    assert PLUGIN_TOOL_NAMES <= with_plugin.calls[0].tool_names
