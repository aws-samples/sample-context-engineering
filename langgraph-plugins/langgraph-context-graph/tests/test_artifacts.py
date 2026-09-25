"""The artifact path: what a tool return is stored as, and what ``expand_artifact`` answers with.

The half of the Strands plugin the binding did not have. Three things are asserted here and nowhere else:
the ``wrap_tool_call``/``awrap_tool_call`` pair records a tool return in the conversation's reference store
and derives the artifact Card of any reference the return names; ``expand_artifact`` reads that store whole,
by line range and by pattern, with the Strands plugin's own messages for every miss; and
``include_artifact_tool=False`` leaves the tool unregistered, which the guidance block then stops
advertising on its own.

No agent is reached for most of it -- a ``ToolCallRequest`` is assembled by hand, the way the tool node
assembles one -- and the end-to-end test drives a real ``create_agent`` graph under both ``invoke`` and
``ainvoke`` with a scripted model, because LangChain bridges neither direction of the tool hook.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool as make_tool

from context_core.graph import GraphState
from context_core.graph import store as core_store
from langgraph_context_graph import ContextGraphMiddleware
from langgraph_context_graph._compat import Command, ToolCallRequest, ToolRuntime

REFERENCE = "mem_1_tc9_0"
"""A reference in the shape an offloader's placeholder names it: ``mem_<counter>_<call>_<block>``."""

ARTIFACT = "\n".join(f"line {n}: value {n}00 reais" for n in range(1, 21))
"""Twenty numbered lines, so a line range and a pattern each select a knowable part of it."""


# --------------------------------------------------------------------------- doubles / helpers


class MockMatcher:
    """Scores every description the same and never embeds anything."""

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        return [0.5 for _ in descriptions]


def runtime_for(state: dict | None = None, tool_call_id: str = "tc9", thread: str | None = None) -> ToolRuntime:
    """A ``ToolRuntime`` as the tool node would build it, optionally carrying a thread id."""
    config = {"configurable": {"thread_id": thread}} if thread is not None else None
    return ToolRuntime(
        state=state if state is not None else {"messages": []},
        context=None,
        config=config,
        stream_writer=None,
        tool_call_id=tool_call_id,
        store=None,
    )


def request_for(
    name: str = "lookup",
    tool_call_id: str = "tc9",
    state: dict | None = None,
    thread: str | None = None,
) -> ToolCallRequest:
    """A ``ToolCallRequest`` for one call, with the agent state the middleware reads."""
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": tool_call_id, "type": "tool_call"},
        tool=None,
        state=state if state is not None else {"messages": []},
        runtime=runtime_for(state, tool_call_id, thread),
    )


def returned(text: str, tool_call_id: str = "tc9") -> ToolMessage:
    """The ``ToolMessage`` a tool answered with."""
    return ToolMessage(content=text, tool_call_id=tool_call_id, id=f"t-{tool_call_id}")


def middleware_with_stored_artifact(text: str = ARTIFACT) -> tuple[ContextGraphMiddleware, GraphState]:
    """A middleware whose store holds ``text`` under ``tc9_0``, plus a graph to read it against."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    middleware.wrap_tool_call(request_for(), lambda _request: returned(text))
    return middleware, GraphState()


def read(middleware: ContextGraphMiddleware, state: GraphState, reference: str, **kwargs) -> str:
    """Drive the async artifact body from a sync test, the way the sync tool body does."""
    store = middleware._store_for("")
    return asyncio.run(middleware.expand_artifact(state, store, reference, **kwargs))


# ---- the write half: the store and the artifact Card ----------------------------------------------


def test_the_raw_return_is_stored_under_the_call_it_came_from():
    """Sync path: the return's text is in the store under ``<tool_call_id>_<block>``, verbatim."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    result = middleware.wrap_tool_call(request_for(), lambda _request: returned("lookup says 400 reais"))

    store = middleware._store_for("")
    assert "tc9_0" in store
    assert asyncio.run(store.retrieve("tc9_0")) == "lookup says 400 reais"
    # The handler's own answer is what the tool node receives: never a Command, never a copy.
    assert isinstance(result, ToolMessage)
    assert result.content == "lookup says 400 reais"


@pytest.mark.asyncio
async def test_the_raw_return_is_stored_on_the_async_path_too():
    """LangChain bridges neither direction, so the async twin has to do its own recording."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    async def handler(_request):
        return returned("lookup says 400 reais")

    await middleware.awrap_tool_call(request_for(), handler)

    assert await middleware._store_for("").retrieve("tc9_0") == "lookup says 400 reais"


def test_each_conversation_gets_its_own_store():
    """A reference discovered in one thread means nothing in another."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    middleware.wrap_tool_call(request_for(thread="a"), lambda _r: returned("thread a content"))

    assert "tc9_0" in middleware._store_for("a")
    assert "tc9_0" not in middleware._store_for("b")


def test_the_middlewares_own_answers_are_not_stored():
    """The skip rule: a retrieval answer is not a tool return to be addressed.

    A whole read echoes the artifact's entire text, so storing it would keep a second copy of the same
    content under a second reference.
    """
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    for name in ("expand_card", "expand_artifact", "find_context"):
        middleware.wrap_tool_call(request_for(name=name), lambda _r: returned("retrieval prose"))

    assert len(middleware._store_for("")) == 0


def test_a_command_return_is_left_alone():
    """A ``Command`` is a state update, not a return with content: nothing to address."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    result = middleware.wrap_tool_call(request_for(), lambda _r: Command(update={"messages": []}))

    assert isinstance(result, Command)
    assert len(middleware._store_for("")) == 0


def test_an_artifact_card_is_derived_from_the_reference_the_return_names():
    """A return carrying an offloader's placeholder yields one artifact Card, titled by the reference.

    The Card holds the **address**: no copy of the content is kept in the graph, which is why nothing in it
    can rot when the content behind the reference changes.
    """
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    graph = GraphState()
    state = {"messages": [HumanMessage(content="what did it cost?", id="u1")], "context_graph": graph}
    preview = f"[Relevance: tool result, ~9,000 tokens]\n\nline 1: value 100 reais\n\n[ref: {REFERENCE}]"

    middleware.wrap_tool_call(request_for(state=state), lambda _r: returned(preview))

    card = graph.cards[REFERENCE]
    assert card.kind == "artifact"
    assert card.reference == REFERENCE
    # The address is recorded in the store as a name, exactly as the Strands hook records it: the content
    # behind it was replaced before this middleware saw the return, so no block is invented for it.
    assert REFERENCE in middleware._store_for("")
    assert asyncio.run(middleware._store_for("").retrieve(REFERENCE)) is None


def test_a_return_naming_no_reference_registers_no_artifact_card():
    """The nothing-offloaded path: no artifact Card, and not a failure either."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    graph = GraphState()
    state = {"messages": [HumanMessage(content="what did it cost?", id="u1")], "context_graph": graph}

    middleware.wrap_tool_call(request_for(state=state), lambda _r: returned("lookup says 400 reais"))

    assert graph.cards == {}


def test_a_tool_call_seen_before_any_projection_still_stores_its_return():
    """No graph yet is no Card, never a raise: the reference is minted from the call, not from the graph."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())

    middleware.wrap_tool_call(request_for(state={"messages": []}), lambda _r: returned(ARTIFACT))

    assert asyncio.run(middleware._store_for("").retrieve("tc9_0")) == ARTIFACT


# ---- the read half: expand_artifact ---------------------------------------------------------------


def test_a_whole_read_returns_the_content_verbatim_behind_the_cost_notice():
    """The text is untouched, character for character, and the answer states what asking whole cost."""
    middleware, graph = middleware_with_stored_artifact()

    answer = read(middleware, graph, "tc9_0")

    assert answer.endswith(f"\n\n{ARTIFACT}")
    assert "whole artifact 'tc9_0'" in answer
    assert "next time pass line_range or pattern to read only the part you need" in answer
    assert graph.retrieval_cycles == 1


def test_a_line_range_read_returns_only_those_lines():
    """The targeted read is delegated to the host search helper this binding registers."""
    middleware, graph = middleware_with_stored_artifact()

    answer = read(middleware, graph, "tc9_0", line_range={"start": 3, "end": 5})

    assert "line 3: value 300 reais" in answer
    assert "line 5: value 500 reais" in answer
    assert "line 6: value 600 reais" not in answer


def test_a_pattern_read_returns_only_the_matching_lines():
    middleware, graph = middleware_with_stored_artifact()

    answer = read(middleware, graph, "tc9_0", pattern="value 1700")

    assert "line 17: value 1700 reais" in answer
    assert "line 1: value 100 reais" not in answer


def test_a_malformed_line_range_is_named_back_in_the_strands_words():
    """The validation message is the Strands plugin's, character for character."""
    middleware, graph = middleware_with_stored_artifact()
    line_range = {"start": "one", "end": 5}

    answer = read(middleware, graph, "tc9_0", line_range=line_range)

    assert answer == (
        f"expand_artifact | line_range=<{line_range!r}> is not a pair of integers | pass "
        '{"start": <int>, "end": <int>}, 1-indexed and inclusive'
    )


def test_a_line_range_outside_the_content_carries_the_helpers_own_refusal():
    """The core's search helper already words it; the tool only names the reference it was asked for."""
    middleware, graph = middleware_with_stored_artifact()

    answer = read(middleware, graph, "tc9_0", line_range={"start": 900, "end": 950})

    assert answer.startswith("expand_artifact | reference 'tc9_0' | ")
    assert "beyond content length" in answer


def test_a_reference_nothing_holds_is_named_back_as_a_miss():
    """The binding has no second storage layer to ask, so a miss is the 'absent' message.

    The Strands plugin distinguishes ``absent`` from ``unknown`` by whether a ``ContextManager`` Stash
    answered. LangGraph has no equivalent, no ``"context_manager"`` host symbol is registered, and the
    conversation's own store is therefore the only layer: every miss reads as absent.
    """
    middleware, graph = middleware_with_stored_artifact()

    answer = read(middleware, graph, "no-such-reference")

    assert answer == (
        "expand_artifact | no artifact storage holds reference 'no-such-reference' on this agent | nothing "
        "was ever offloaded under that reference, which means the full results are already in the "
        "conversation"
    )
    # A miss records no fed-back note and changes no Resolution.
    assert graph.reuse == {}


def test_non_textual_content_is_reported_without_a_media_type():
    """A decoded block carries no media type to name, so none is invented."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    middleware._store_for("").put("tc9_0", {"image": {"format": "png"}})

    answer = read(middleware, GraphState(), "tc9_0")

    assert answer == (
        "expand_artifact | reference 'tc9_0' holds non-textual content | line_range and pattern do not "
        "apply to it, and it cannot be returned as text"
    )


def test_a_successful_read_records_the_fed_back_note_on_the_artifact_card():
    """The content is in this answer; what crosses into the next turn is the note on its Card."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    graph = GraphState()
    state = {"messages": [HumanMessage(content="what did it cost?", id="u1")], "context_graph": graph}
    preview = f"line 1: value 100 reais\n\n[ref: {REFERENCE}]"
    middleware.wrap_tool_call(request_for(state=state), lambda _r: returned(preview))
    # The store holds the address only, so the content is put there the way the offloader's store would.
    middleware._store_for("").put(REFERENCE, ARTIFACT)

    answer = read(middleware, graph, REFERENCE)

    assert ARTIFACT in answer
    assert REFERENCE in graph.reuse


def test_the_retrieval_budget_refuses_expand_artifact_like_the_other_two():
    """One ceiling across the three tools, checked before the call's own increment."""
    middleware, graph = middleware_with_stored_artifact()
    middleware._max_retrieval_cycles = 1

    first = read(middleware, graph, "tc9_0")
    second = read(middleware, graph, "tc9_0")

    assert "whole artifact" in first
    assert second == (
        "expand_artifact | this turn has already spent its 1 retrieval calls | no further recovery is "
        "available on this turn: answer from what the summary and the messages already give you, and state "
        "plainly which part you could not verify"
    )
    # The refusal costs no budget of its own.
    assert graph.retrieval_cycles == 1


def test_the_tool_answers_through_both_of_its_bodies():
    """``invoke`` and ``ainvoke`` reach the same answer: a coroutine-only tool would refuse the first."""
    middleware, _ = middleware_with_stored_artifact()
    artifact_tool = {each.name: each for each in middleware.tools}["expand_artifact"]
    state = {"messages": [], "context_graph": GraphState()}

    sync_command = artifact_tool.invoke({"reference": "tc9_0", "runtime": runtime_for(state, "r1")})
    async_command = asyncio.run(
        artifact_tool.ainvoke({"reference": "tc9_0", "runtime": runtime_for(state, "r2")})
    )

    for command, call in ((sync_command, "r1"), (async_command, "r2")):
        assert isinstance(command, Command)
        (message,) = command.update["messages"]
        assert message.tool_call_id == call
        assert ARTIFACT in message.content
        assert isinstance(command.update["context_graph"], GraphState)


# ---- configuration and wiring ---------------------------------------------------------------------


def test_the_model_reads_the_strands_description_verbatim():
    """The text that travels in the tool spec is the Strands docstring, ``tool_context`` renamed."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher())
    artifact_tool = {each.name: each for each in middleware.tools}["expand_artifact"]

    description = artifact_tool.description
    assert description.startswith(
        "Read a stored artifact that an earlier turn of THIS conversation referred to by address."
    )
    assert "use the tool that minted it" in description
    assert "runtime: Injected by the framework. Not user-facing." in description
    assert "tool_context" not in description


def test_include_artifact_tool_false_leaves_the_tool_unregistered():
    """Excluded rather than registered and then removed, and the other two are unaffected."""
    middleware = ContextGraphMiddleware(matcher=MockMatcher(), include_artifact_tool=False)

    assert [each.name for each in middleware.tools] == ["expand_card", "find_context"]


@pytest.mark.parametrize("value", [1, "yes", None])
def test_include_artifact_tool_is_refused_unless_it_is_a_bool(value):
    with pytest.raises(ValueError, match="include_artifact_tool"):
        ContextGraphMiddleware(matcher=MockMatcher(), include_artifact_tool=value)


def test_the_guidance_names_the_artifact_tool_when_it_is_registered():
    """The guidance reads the registered set on every render, so the switch needs no second one."""
    graph_on = ContextGraphMiddleware(matcher=MockMatcher())
    graph_off = ContextGraphMiddleware(matcher=MockMatcher(), include_artifact_tool=False)

    assert "expand_artifact" in graph_on._thresholds.retrieval_tools
    assert "expand_artifact" not in graph_off._thresholds.retrieval_tools


def test_the_search_helper_is_registered_as_a_host_symbol():
    """Without this registration every targeted read degrades to 'targeted reads are unavailable'."""
    assert core_store.HOST_SYMBOLS["search_content"] == ("context_core.relevance.search", "_search_content")
    # The two this binding has no neutral equivalent for stay unregistered.
    assert "context_manager" not in core_store.HOST_SYMBOLS
    assert "extract_text" not in core_store.HOST_SYMBOLS


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


@make_tool("lookup")
def lookup(topic: str) -> str:
    """Look a topic up.

    Args:
        topic: What to look up.
    """
    return ARTIFACT


def scripted_agent() -> tuple[object, ScriptedChatModel, ContextGraphMiddleware]:
    """A real ``create_agent`` graph whose model calls ``lookup`` and then reads the artifact back."""
    from langchain.agents import create_agent

    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    model = ScriptedChatModel(
        turns=[
            AIMessage(
                content="",
                tool_calls=[{"id": "data-1", "name": "lookup", "args": {"topic": "costs"}}],
                id="ai-1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read-1",
                        "name": "expand_artifact",
                        "args": {"reference": "data-1_0", "line_range": {"start": 7, "end": 8}},
                    }
                ],
                id="ai-2",
            ),
            AIMessage(content="line 7 says 700 reais", id="ai-3"),
        ],
        seen=[],
    )
    agent = create_agent(model=model, tools=[lookup], middleware=[middleware])
    return agent, model, middleware


def assert_artifact_was_read(final: dict, middleware: ContextGraphMiddleware) -> None:
    """The run reached the artifact through the store the tool hook filled.

    Deliberately free of any ``await``, so the same assertions serve the sync and the async run: the store's
    membership test is synchronous and only its ``retrieve`` is a coroutine.
    """
    answers = [
        message.content
        for message in final["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-1"
    ]
    assert answers, "the agent never produced an expand_artifact answer"
    assert "line 7: value 700 reais" in answers[0]
    assert "line 9: value 900 reais" not in answers[0]
    assert "data-1_0" in middleware._store_for("")


def test_end_to_end_under_invoke_the_agent_reads_the_stored_artifact():
    """The whole path on a real graph: the tool hook stores the return, the model reads part of it back."""
    agent, _model, middleware = scripted_agent()

    final = agent.invoke({"messages": [HumanMessage(content="what did topic 1 cost?", id="u1")]})

    assert_artifact_was_read(final, middleware)
    assert asyncio.run(middleware._store_for("").retrieve("data-1_0")) == ARTIFACT


@pytest.mark.asyncio
async def test_end_to_end_under_ainvoke_the_agent_reads_the_stored_artifact():
    """The same run under ``ainvoke``, which reaches the async twin of both the hook and the tool."""
    agent, _model, middleware = scripted_agent()

    final = await agent.ainvoke({"messages": [HumanMessage(content="what did topic 1 cost?", id="u1")]})

    assert_artifact_was_read(final, middleware)
    assert await middleware._store_for("").retrieve("data-1_0") == ARTIFACT


@make_tool("offload")
def offload(topic: str) -> str:
    """Look a topic up, answering the way a tool behind an offloader answers.

    Args:
        topic: What to look up.
    """
    return f"[Relevance: tool result, ~9,000 tokens]\n\nline 1: value 100 reais\n\n[ref: {REFERENCE}]"


def test_end_to_end_the_artifact_card_reaches_the_next_projection():
    """The Card written by the tool hook is in the graph the next model call and the final state read.

    This is what the hook returning the handler's own answer -- rather than a ``Command`` carrying a state
    update -- costs and does not cost: the Card travels on the graph object already in the state, which the
    next projection copies forward.
    """
    from langchain.agents import create_agent

    middleware = ContextGraphMiddleware(matcher=MockMatcher(), min_cards=1)
    model = ScriptedChatModel(
        turns=[
            AIMessage(
                content="",
                tool_calls=[{"id": "off-1", "name": "offload", "args": {"topic": "costs"}}],
                id="ai-1",
            ),
            AIMessage(content="it cost 100 reais", id="ai-2"),
        ],
        seen=[],
    )
    agent = create_agent(model=model, tools=[offload], middleware=[middleware])

    final = agent.invoke({"messages": [HumanMessage(content="what did topic 1 cost?", id="u1")]})

    graph = final["context_graph"]
    assert REFERENCE in graph.cards
    assert graph.cards[REFERENCE].kind == "artifact"


def test_stash_must_have_a_retrieve_method():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="stash="):
        ContextGraphMiddleware(stash=object())


def test_expand_artifact_falls_back_to_the_stash():
    class _Stash:
        async def retrieve(self, reference):
            if reference == "mem_1_tc1_0":
                return "from the stash\nsecond line"
            raise KeyError(reference)

    middleware = ContextGraphMiddleware(stash=_Stash())
    answer = asyncio.run(middleware.expand_artifact(GraphState(), middleware._store_for(""), "mem_1_tc1_0"))
    assert "from the stash" in answer
