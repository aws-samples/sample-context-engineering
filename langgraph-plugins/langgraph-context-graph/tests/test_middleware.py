"""Behaviour tests for :class:`ContextGraphMiddleware`, against a mocked model call and a mock matcher.

No agent is built, no provider is reached and no embedding is ever computed: the matcher is a deterministic
``MockMatcher`` whose ``calls`` list is itself an assertion (a projection pays for at most one scoring round,
and for none at all where the choice is short-circuited), and the model call is a ``RecordingHandler`` that
captures the request the middleware handed it. What is asserted is the practice -- Cards off the closed
turns, Resolution within the body budget, the projected ``override``, the full-pass identity, the untouched
persisted state, the two retrieval tools, and the one wiring-time notice.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from context_core.graph import CardChoice, GraphState
from langgraph_context_graph import ContextGraphMiddleware
from langgraph_context_graph._compat import (
    Command,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolRuntime,
)

TOPIC_1 = "question 1 about topic 1 costing 100 reais"
TOPIC_2 = "question 2 about topic 2 costing 200 reais"
TOPIC_3 = "question 3 about topic 3 costing 300 reais"
TOPIC_4 = "question 4 about topic 4 costing 400 reais"


# --------------------------------------------------------------------------- doubles / helpers


class MockMatcher:
    """Scores by substring lookup, records every call, and never embeds anything.

    The middleware resolves its default matcher lazily, so passing one of these is what keeps the whole
    suite free of a network call.
    """

    def __init__(self, scores: dict[str, float] | None = None, *, default: float = 0.10) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.vector_calls: list[tuple[str, ...]] = []

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        self.calls.append((question, tuple(descriptions)))
        return [self._score_of(description) for description in descriptions]

    def _score_of(self, description: str) -> float:
        for marker, value in self.scores.items():
            if marker in description:
                return value
        return self.default


class VectorMockMatcher(MockMatcher):
    """A matcher that also publishes description vectors, which is what lets ``similar`` links form."""

    def vectors(self, descriptions: Sequence[str]) -> list[tuple[float, ...]]:
        self.vector_calls.append(tuple(descriptions))
        # Identical vectors, so every pair measures at similarity 1.0 and links.
        return [(1.0, 0.0) for _ in descriptions]


class RecordingHandler:
    """Stands in for the model call: records the request it was given and answers with one message."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    @property
    def request(self) -> ModelRequest:
        assert len(self.requests) == 1, f"expected exactly one model call, got {len(self.requests)}"
        return self.requests[0]

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self.request.messages)


def turn(n: int, *, tool: bool = False) -> list[BaseMessage]:
    """One closed turn: the question, optionally a tool pair, and the answer.

    Every message carries an ``id``, which is what a real agent's ``add_messages`` reducer assigns and what
    the adapter carries across as the Durable Identity a Card addresses.
    """
    messages: list[BaseMessage] = [
        HumanMessage(content=f"question {n} about topic {n} costing {n}00 reais", id=f"u{n}")
    ]
    if tool:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"q": n}, "id": f"tc{n}", "type": "tool_call"}],
                id=f"a{n}",
            )
        )
        messages.append(
            ToolMessage(content=f"lookup says {n}00 reais over 12{n} units", tool_call_id=f"tc{n}", id=f"t{n}")
        )
    messages.append(AIMessage(content=f"answer {n}: it costs {n}00 reais", id=f"z{n}"))
    return messages


def conversation(closed: int = 4, *, question: str = "what did topic 1 cost?") -> list[BaseMessage]:
    """``closed`` closed turns plus the open one, which no Card ever covers."""
    messages: list[BaseMessage] = []
    for n in range(1, closed + 1):
        messages.extend(turn(n, tool=(n == 1)))
    messages.append(HumanMessage(content=question, id="open"))
    return messages


def request_for(messages: list[BaseMessage], state: dict | None = None) -> ModelRequest:
    """A ``ModelRequest`` carrying ``messages``, with the agent state the middleware reads and writes."""
    return ModelRequest(
        model=None,
        messages=list(messages),
        state=state if state is not None else {"messages": list(messages)},
    )


def project_once(middleware: ContextGraphMiddleware, messages=None, state=None):
    """Run one model call through the middleware and return ``(handler, agent state)``."""
    messages = conversation() if messages is None else messages
    state = {"messages": list(messages)} if state is None else state
    handler = RecordingHandler()
    middleware.wrap_model_call(request_for(messages, state), handler)
    return handler, state


def _runtime(state: dict, tool_call_id: str) -> ToolRuntime:
    """A ``ToolRuntime`` as the tool node would build it, minus everything these tools do not read."""
    return ToolRuntime(
        state=state,
        context=None,
        config=None,
        stream_writer=None,
        tool_call_id=tool_call_id,
        store=None,
    )



# ---- the write half: Cards off the closed turns ---------------------------------------------------


def test_cards_are_derived_from_the_closed_turns_only():
    """One Card per closed turn, titled by its question. The open turn is covered by none."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)

    _, state = project_once(middleware)

    graph = state["context_graph"]
    assert list(graph.cards) == [TOPIC_1, TOPIC_2, TOPIC_3, TOPIC_4]
    assert all(card.kind == "subject" for card in graph.cards.values())
    # The open turn's question titles nothing: no Card can drop it below full content.
    assert "what did topic 1 cost?" not in graph.cards
    # The Card addresses the messages by the ids the adapter carried over, never by their content.
    assert graph.cards[TOPIC_1].dialogue_ids == ("u1", "z1")
    assert graph.cards[TOPIC_1].evidence_ids == ("a1", "t1")


def test_a_history_without_message_ids_yields_no_card():
    """No Durable Identity, no Card: an id-less history projects whole rather than raising."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    messages = [message.model_copy(update={"id": None}) for message in conversation()]

    handler, state = project_once(middleware, messages)

    assert state["context_graph"].cards == {}
    assert handler.messages == messages


def test_one_scoring_round_per_call():
    """The graph is scored once against the turn's question, and the matcher sees the Descriptions."""
    matcher = MockMatcher({"topic 1": 0.9})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)

    project_once(middleware)

    assert len(matcher.calls) == 1
    question, descriptions = matcher.calls[0]
    assert question == "what did topic 1 cost?"
    assert len(descriptions) == 4


def test_below_min_cards_nothing_is_scored_at_all():
    """The short circuit is reached before the matcher: a small graph pays for no embedding."""
    matcher = MockMatcher()
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=10)

    handler, state = project_once(middleware)

    assert matcher.calls == []
    assert state["context_graph"].choice.full_pass is True
    assert handler.messages == conversation()


# ---- the read half: Resolution, budget, delivery --------------------------------------------------


def test_resolution_follows_the_note_and_the_body_budget():
    """Above the threshold is Full Content while the budget lasts, then one rung down -- never Title."""
    matcher = MockMatcher({"topic 1": 0.95, "topic 2": 0.90})
    # Two dialogue messages per turn at the core's 250-token fallback estimate, so a budget of 500 admits
    # exactly one Card at Full Content and the next one has to step down.
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1, body_budget=500)

    _, state = project_once(middleware)

    choice = state["context_graph"].choice
    assert choice.full_pass is False
    assert choice.by_title[TOPIC_1].dialogue == "full"
    # It cleared the threshold too, so the budget is the only reason it stepped down, and by one rung.
    assert choice.by_title[TOPIC_2].dialogue == "description"
    # Below the floor: Title only.
    assert choice.by_title[TOPIC_4].dialogue == "title"


def test_override_reflects_the_projection():
    """The call the model sees carries the removal and the folded Descriptions; the input list does not."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    messages = conversation()

    handler, _ = project_once(middleware, messages)

    projected = handler.messages
    assert len(projected) < len(messages)
    # The turn kept at Full Content is still there, whole.
    assert HumanMessage(content=TOPIC_1, id="u1") in projected
    # The collapsed turns are gone from the call.
    assert not any(message.id == "u4" for message in projected)
    # ... and what left is described in a block folded onto the last user message, which stays last.
    folded = projected[-1]
    assert isinstance(folded, HumanMessage)
    rendered = folded.text if callable(getattr(folded, "text", None)) else str(folded.content)
    assert "what did topic 1 cost?" in rendered
    assert TOPIC_4 in rendered
    # The guidance names the tools this middleware actually registers.
    assert "expand_card" in rendered


def test_a_full_pass_is_the_identical_call():
    """``expand_threshold=0.0`` and ``collapse_floor=0.0``: no override, no scoring, the same request."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(
        matcher=matcher,
        min_cards=1,
        expand_threshold=0.0,
        collapse_floor=0.0,
    )
    messages = conversation()
    state = {"messages": list(messages)}
    request = request_for(messages, state)
    handler = RecordingHandler()

    middleware.wrap_model_call(request, handler)

    # The same request object, so the call is identical to the no-middleware one rather than equal to it.
    assert handler.request is request
    assert handler.request.messages is request.messages
    assert handler.messages == messages
    # A full pass takes the decision before the matcher is reached.
    assert matcher.calls == []
    assert state["context_graph"].choice.full_pass is True


def test_persisted_state_is_never_mutated():
    """``state["messages"]`` comes out as it went in, and the graph handed in is not written to."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    messages = conversation()
    state = {"messages": list(messages)}

    project_once(middleware, messages, state)
    first = state["context_graph"]
    turn_at_first_call = first.turn
    cards_at_first_call = dict(first.cards)

    # A second call carrying the graph the first produced.
    followup = [*messages, AIMessage(content="answer 5", id="z5"), HumanMessage(content="and topic 2?", id="open2")]
    state["messages"] = list(followup)
    handler, state = project_once(middleware, followup, state)

    second = state["context_graph"]
    assert second is not first
    assert first.turn == turn_at_first_call
    assert first.cards == cards_at_first_call
    # Every message of the persisted list survives, by object identity, whatever the call carried.
    assert state["messages"] == followup
    assert len(handler.messages) < len(followup)


def test_the_graph_update_travels_on_the_response():
    """The durable write is a ``Command`` touching ``context_graph`` and nothing else."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    messages = conversation()
    state = {"messages": list(messages)}

    response = middleware.wrap_model_call(request_for(messages, state), RecordingHandler())

    assert isinstance(response, ExtendedModelResponse)
    assert list(response.command.update) == ["context_graph"]
    assert response.command.update["context_graph"] is state["context_graph"]
    assert [message.content for message in response.model_response.result] == ["ok"]


def test_a_bare_ai_message_answer_is_normalised():
    """A handler answering with an ``AIMessage`` still gets the graph update attached."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    messages = conversation()

    response = middleware.wrap_model_call(
        request_for(messages),
        lambda request: AIMessage(content="bare"),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert isinstance(response.model_response, ModelResponse)
    assert "context_graph" in response.command.update


def test_an_inner_middlewares_command_keeps_its_own_keys():
    """Merged, not replaced: only ``context_graph`` is written."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    inner = ExtendedModelResponse(
        model_response=ModelResponse(result=[AIMessage(content="ok")]),
        command=Command(update={"something_else": 1}),
    )

    response = middleware.wrap_model_call(request_for(conversation()), lambda request: inner)

    assert response is inner
    assert response.command.update["something_else"] == 1
    assert isinstance(response.command.update["context_graph"], GraphState)


def test_a_state_without_a_graph_starts_from_a_fresh_one():
    """A first call carries no graph, which is a full pass and therefore the untouched call."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    messages = turn(1)

    handler, state = project_once(middleware, messages)

    assert handler.messages == messages
    assert state["context_graph"].turn == 1


# ---- the tools -----------------------------------------------------------------------------------


def test_the_three_retrieval_tools_are_registered_with_model_facing_arguments_only():
    """The injected ``runtime`` is not part of the schema the model is shown.

    Registered in the Strands plugin's own order, ``expand_artifact`` between the two that read the
    conversation's turns, so the set a model is shown is the same set in the same order on both bindings.
    """
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    assert [each.name for each in middleware.tools] == ["expand_card", "expand_artifact", "find_context"]
    assert all(isinstance(each, BaseTool) for each in middleware.tools)
    by_name = {each.name: each for each in middleware.tools}
    assert sorted(by_name["expand_card"].args) == ["titles"]
    assert sorted(by_name["find_context"].args) == ["need", "tag"]
    assert sorted(by_name["expand_artifact"].args) == ["line_range", "pattern", "reference"]
    assert "earlier turns" in by_name["expand_card"].description


def test_expand_card_raises_a_collapsed_card_to_full_content():
    """Both axes, and the fed-back note is what carries the request into the next turn."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)
    graph = state["context_graph"]
    assert graph.choice.by_title[TOPIC_4].dialogue == "title"

    answer = middleware.expand_card(graph, [TOPIC_4])

    assert TOPIC_4 in answer
    assert graph.choice.by_title[TOPIC_4] == CardChoice(dialogue="full", evidence="full")
    # The Card the choice kept is untouched by somebody else's elevation.
    assert graph.choice.by_title[TOPIC_1].dialogue == "full"
    assert graph.choice.full_pass is False
    assert TOPIC_4 in graph.reuse
    assert graph.retrieval_cycles == 1


def test_expand_card_names_a_title_it_cannot_find():
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    _, state = project_once(middleware)

    answer = middleware.expand_card(state["context_graph"], ["no such turn"])

    assert "no earlier turn" in answer
    assert "'no such turn'" in answer
    assert state["context_graph"].reuse == {}


def test_expand_card_reports_a_partial_batch():
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)

    answer = middleware.expand_card(state["context_graph"], [TOPIC_4, "no such turn"])

    assert f"'{TOPIC_4}' arrives in full" in answer
    assert "nothing was raised for it" in answer


def test_expand_card_accepts_a_scalar_title():
    """The schema says array; a model that sends the string anyway is answered rather than corrected."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)

    assert TOPIC_4 in middleware.expand_card(state["context_graph"], TOPIC_4)


def test_expand_card_leaves_a_full_pass_alone():
    """Every Card is already whole, and an entry would cost the delivery its identity short circuit."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1, expand_threshold=0.0, collapse_floor=0.0)
    _, state = project_once(middleware)
    graph = state["context_graph"]

    middleware.expand_card(graph, [TOPIC_4])

    assert graph.choice.full_pass is True
    assert dict(graph.choice.by_title) == {}
    # The request still crosses the turn boundary, through the fed-back note.
    assert TOPIC_4 in graph.reuse


def test_find_context_answers_with_the_candidates_that_clear_the_floor():
    matcher = VectorMockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)
    graph = state["context_graph"]

    answer = middleware.find_context(graph, "topic 1 cost")

    assert "1 earlier turn(s) match 'topic 1 cost'" in answer
    assert f"- title: {TOPIC_1}" in answer
    # Below the floor, so not offered as a candidate.
    assert f"- title: {TOPIC_4}" not in answer
    # The ``similar`` edge, traversed: neighbours of a candidate, which the ranking cannot see.
    assert "related turns:" in answer
    assert TOPIC_1 in graph.reuse


def test_find_context_names_what_it_did_not_find():
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    _, state = project_once(middleware)
    graph = state["context_graph"]

    answer = middleware.find_context(graph, "quarterly bonus policy")

    assert "nothing in this conversation matches 'quarterly bonus policy'" in answer
    assert graph.reuse == {}


def test_find_context_narrows_by_tag():
    matcher = MockMatcher({"topic": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)
    graph = state["context_graph"]

    narrowed = middleware.find_context(graph, "anything", tag="lookup")
    missing = middleware.find_context(graph, "anything", tag="no-such-tag")

    assert f"- title: {TOPIC_1}" in narrowed
    assert "among the turns tagged 'no-such-tag'" in missing


def test_the_retrieval_budget_is_spent_per_turn():
    """A ceiling of ``n`` admits exactly ``n`` calls, then answers with the instruction to stop asking."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1, max_retrieval_cycles=1)
    _, state = project_once(middleware)
    graph = state["context_graph"]

    first = middleware.expand_card(graph, [TOPIC_4])
    second = middleware.expand_card(graph, [TOPIC_2])
    third = middleware.find_context(graph, "topic 2")

    assert "arrives in full" in first
    assert "already spent its 1 retrieval calls" in second
    assert "already spent its 1 retrieval calls" in third
    # The refusal costs no budget of its own.
    assert graph.retrieval_cycles == 1


def test_a_tool_call_returns_the_state_update_and_the_answer():
    """The whole tool path: the graph is read off the runtime state and written back in a ``Command``."""
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)
    expand_card = {each.name: each for each in middleware.tools}["expand_card"]

    command = expand_card.invoke({"titles": [TOPIC_4], "runtime": _runtime(state, "call-7")})

    assert isinstance(command, Command)
    graph = command.update["context_graph"]
    assert graph is state["context_graph"]
    assert graph.choice.by_title[TOPIC_4].dialogue == "full"
    (message,) = command.update["messages"]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-7"
    assert TOPIC_4 in message.content


def test_a_tool_reached_before_any_projection_gets_a_fresh_graph():
    """No graph in state is answered as an empty graph, not as somebody else's."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    find_context = {each.name: each for each in middleware.tools}["find_context"]

    command = find_context.invoke({"need": "anything", "runtime": _runtime({"messages": []}, "call-1")})

    assert isinstance(command.update["context_graph"], GraphState)
    assert command.update["context_graph"].cards == {}
    assert "nothing in this conversation matches" in command.update["messages"][0].content


# ---- the wiring-time notice -----------------------------------------------------------------------


class FakeSummarizationMiddleware:
    """Stands in for a middleware that rewrites the persisted message list."""


class FakeTrimmingMiddleware:
    pass


class FakeMonitoringMiddleware:
    """Stands in for a middleware that removes nothing."""


def test_the_pruning_notice_fires_exactly_once():
    """One notice per construction, naming the first offender, and the instance is registered anyway."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        middleware = ContextGraphMiddleware(
            matcher=MockMatcher(),
            middleware=[FakeMonitoringMiddleware(), FakeSummarizationMiddleware(), FakeTrimmingMiddleware()],
        )

    assert len(caught) == 1
    assert "FakeSummarizationMiddleware" in str(caught[0].message)
    assert "state['messages']" in str(caught[0].message)
    # Degrades and never blocks: the tools are registered exactly as they are without the notice.
    assert [each.name for each in middleware.tools] == ["expand_card", "expand_artifact", "find_context"]


def test_nothing_is_said_about_a_middleware_list_that_prunes_nothing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraphMiddleware(matcher=MockMatcher(), middleware=[FakeMonitoringMiddleware()])
        ContextGraphMiddleware(matcher=MockMatcher())

    assert caught == []


def test_the_middleware_does_not_warn_about_itself():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        middleware = ContextGraphMiddleware(matcher=MockMatcher())
        ContextGraphMiddleware(matcher=MockMatcher(), middleware=[middleware])

    assert caught == []


# ---- configuration --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expand_threshold": 1.5},
        {"expand_threshold": True},
        {"collapse_floor": -0.1},
        {"link_threshold": "0.5"},
        {"collapse_floor": 0.9, "expand_threshold": 0.5},
        {"description_tokens": 0},
        {"tags_per_card": 2.5},
        {"neighbors_per_candidate": -1},
        {"reuse_ttl_cycles": -1},
        {"min_cards": 0},
        {"body_budget": 0},
        {"max_retrieval_cycles": 0},
        {"matcher": object()},
    ],
)
def test_invalid_configuration_is_refused_by_the_constructor(kwargs):
    with pytest.raises(ValueError):
        ContextGraphMiddleware(**kwargs)


def test_the_persisted_graph_survives_the_copy_langgraph_makes_of_it():
    """LangGraph deep-copies every state update, so nothing in the graph may be a ``mappingproxy``.

    The core freezes the turn choice behind a ``MappingProxyType``, which ``deepcopy`` cannot carry at all.
    It is flattened on the way into the agent state -- by the projection and by a tool alike -- and the
    freeze costs nothing there, the next projection recomputing the choice rather than extending it.
    """
    matcher = MockMatcher({"topic 1": 0.95})
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)
    _, state = project_once(middleware)

    assert type(state["context_graph"].choice.by_title) is dict
    assert copy.deepcopy(state["context_graph"]) is not None

    command = {each.name: each for each in middleware.tools}["expand_card"].invoke(
        {"titles": [TOPIC_4], "runtime": _runtime(state, "call-1")}
    )

    assert type(command.update["context_graph"].choice.by_title) is dict
    assert copy.deepcopy(command.update["context_graph"]) is not None


def test_the_default_matcher_is_never_built_when_one_is_supplied():
    """Construction opens no client, and a supplied matcher keeps it that way for the whole run."""
    matcher = MockMatcher()
    middleware = ContextGraphMiddleware(matcher=matcher, min_cards=1)

    project_once(middleware)

    assert middleware._resolved_matcher is None


# ---- end to end, on a real agent with a scripted model --------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that replays scripted turns, one per call, and records what it was sent."""

    turns: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        turn = self.turns[min(len(self.seen) - 1, len(self.turns) - 1)]
        return ChatResult(generations=[ChatGeneration(message=turn)])


def test_end_to_end_the_agent_sees_the_projection_and_keeps_its_history():
    """On a real ``create_agent`` graph: the call is projected, the state is not, and the tool wires up.

    The model is scripted to reach for ``expand_card`` on the first turn, which exercises the whole tool
    path through LangGraph's own tool node -- the ``Command`` it returns is what puts the fed-back note in
    the persisted graph, and that note is the only thing that crosses the turn boundary: the elevation
    itself ends with the turn, since the next projection recomputes the choice from the graph.
    """
    from langchain.agents import create_agent

    middleware = ContextGraphMiddleware(matcher=MockMatcher({"topic 1": 0.95}), min_cards=1)
    model = ScriptedChatModel(
        turns=[
            AIMessage(
                content="",
                tool_calls=[{"id": "r1", "name": "expand_card", "args": {"titles": [TOPIC_4]}}],
                id="ai-1",
            ),
            AIMessage(content="topic 1 cost 100 reais", id="ai-2"),
        ],
        seen=[],
    )
    history = conversation()

    agent = create_agent(model=model, tools=[], middleware=[middleware])
    final = agent.invoke({"messages": history})

    # The provider saw the projection: fewer messages than the history, and the folded block with them.
    assert len(model.seen[0]) < len(history)
    assert any("expand_card" in str(message.content) for message in model.seen[0])
    # The persisted history keeps every message it started with, plus what the run added.
    assert [message.id for message in history] == [
        message.id for message in final["messages"][: len(history)]
    ]
    # The graph travelled in the agent state, and the tool's request crossed the turn boundary as a note.
    graph = final["context_graph"]
    assert isinstance(graph, GraphState)
    assert list(graph.cards) == [TOPIC_1, TOPIC_2, TOPIC_3, TOPIC_4]
    assert TOPIC_4 in graph.reuse
    # Two model calls, so the second projection recomputed the choice rather than inheriting the elevation.
    assert len(model.seen) == 2
    assert final["messages"][-1].content == "topic 1 cost 100 reais"
