"""Progressive tool disclosure, as one LangChain model call sees it.

Every test runs against a mocked model side: either :class:`RecordingHandler`, which captures the
request the middleware handed on, or :class:`ScriptedModel`, which replays prepared replies. No call
leaves the process.
"""

from __future__ import annotations

import pytest
from conftest import (
    DOMAIN_TOOLS,
    CountingHandler,
    ScriptedModel,
    bound_tools,
    make_request,
    make_tool_call_request,
    make_tool_runtime,
    names_of,
    run,
)
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from context_core.disclosure import CATALOG_PROMPT_HEADER, FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME
from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware
from langgraph_progressive_tool_disclosure.middleware import _merge_loads

DISCLOSURE_TOOLS = {FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME}
HIDDEN = "list_investment_transactions"


@pytest.fixture
def middleware() -> ProgressiveToolDisclosureMiddleware:
    """A default instance: 80-character catalog lines, three-cycle TTL, lexical search."""
    return ProgressiveToolDisclosureMiddleware()


# --------------------------------------------------------------------------------------------------
# What the call carries.
# --------------------------------------------------------------------------------------------------


def test_first_call_carries_only_the_disclosure_tools(middleware):
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="my balance?")])
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS


def test_the_request_is_not_mutated(middleware):
    tools = bound_tools(middleware)
    request = make_request(tools=tools, messages=[HumanMessage(content="hi")])
    run(middleware, request)
    assert names_of(request.tools) == {t.name for t in tools}


def test_always_available_tools_are_carried_from_the_first_call():
    middleware = ProgressiveToolDisclosureMiddleware(always_available=["current_time"])
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="what time is it?")])
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS | {"current_time"}


def test_a_loaded_tool_is_carried_on_the_following_call(middleware):
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="my balance?"), AIMessage(content="loading")],
        state={"messages": [], "loaded_tools": {"get_balance": 0}},
    )
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS | {"get_balance"}


def test_a_load_recorded_on_the_current_cycle_is_not_yet_carried(middleware):
    """The model cannot have seen a schema loaded by a call it made on this very cycle."""
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="my balance?")],
        state={"messages": [], "loaded_tools": {"get_balance": 0}},
    )
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS


def test_an_unbound_loaded_name_cannot_resurrect(middleware):
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="hi"), AIMessage(content="x")],
        state={"messages": [], "loaded_tools": {"unregistered_tool": 0}},
    )
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS


def test_the_request_passes_through_when_the_disclosure_tools_are_not_bound(middleware):
    request = make_request(tools=DOMAIN_TOOLS, messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    assert seen is request


def test_provider_native_tool_dicts_are_understood(middleware):
    declared = [
        {"type": "function", "function": {"name": "get_balance", "description": "Balance of an account."}},
        *middleware.tools,
    ]
    request = make_request(
        tools=declared,
        messages=[HumanMessage(content="hi"), AIMessage(content="x")],
        state={"messages": [], "loaded_tools": {"get_balance": 0}},
    )
    seen, _ = run(middleware, request)
    assert names_of(seen.tools) == DISCLOSURE_TOOLS | {"get_balance"}


# --------------------------------------------------------------------------------------------------
# The catalog.
# --------------------------------------------------------------------------------------------------


def test_the_catalog_is_appended_to_the_system_message(middleware):
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="my balance?")],
        system_message=SystemMessage(content="You are a banking assistant."),
    )
    seen, _ = run(middleware, request)
    prompt = seen.system_message.text

    assert prompt.startswith("You are a banking assistant.")
    assert "# Tools available on request" in prompt
    assert f"Call `{GET_TOOL_DETAILS_NAME}` with the names you need" in prompt
    assert "MUST NOT call them directly" in prompt
    assert "- get_balance:" in prompt
    assert f"- {HIDDEN}:" in prompt


def test_the_catalog_omits_what_the_call_already_carries(middleware):
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="hi"), AIMessage(content="x")],
        system_message=SystemMessage(content="sys"),
        state={"messages": [], "loaded_tools": {"get_balance": 0}},
    )
    seen, _ = run(middleware, request)
    prompt = seen.system_message.text

    assert "- get_balance:" not in prompt
    assert f"- {HIDDEN}:" in prompt
    assert f"- {FIND_TOOLS_NAME}" not in prompt


def test_the_catalog_summary_is_budgeted():
    middleware = ProgressiveToolDisclosureMiddleware(catalog_chars=20)
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    lines = [line for line in seen.system_message.text.splitlines() if line.startswith(f"- {HIDDEN}:")]
    assert len(lines) == 1
    summary = lines[0].split(": ", 1)[1]
    assert len(summary) <= 20


def test_a_catalog_is_created_when_the_caller_set_no_system_message(middleware):
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    assert seen.system_message is not None
    assert CATALOG_PROMPT_HEADER.splitlines()[0] in seen.system_message.text


def test_existing_system_blocks_are_kept_apart(middleware):
    """A caller using the list form is usually placing cache checkpoints; collapsing it would move them."""
    request = make_request(
        tools=bound_tools(middleware),
        messages=[HumanMessage(content="hi")],
        system_message=SystemMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
    )
    seen, _ = run(middleware, request)
    blocks = seen.system_message.content_blocks
    assert [b["text"] for b in blocks[:2]] == ["a", "b"]
    assert len(blocks) == 3


def test_catalog_chars_none_sends_no_catalog_and_only_the_two_tools():
    middleware = ProgressiveToolDisclosureMiddleware(catalog_chars=None)
    system = SystemMessage(content="You are a banking assistant.")
    request = make_request(
        tools=bound_tools(middleware), messages=[HumanMessage(content="my balance?")], system_message=system
    )
    seen, _ = run(middleware, request)

    assert seen.system_message is system
    assert names_of(seen.tools) == DISCLOSURE_TOOLS


# --------------------------------------------------------------------------------------------------
# TTL: forgetting through inactivity, renewal through use.
# --------------------------------------------------------------------------------------------------


def _history(cycles: int) -> list:
    """A closed history of ``cycles`` plain assistant turns, so the cycle counter reads ``cycles``."""
    messages = [HumanMessage(content="turn")]
    for index in range(cycles):
        messages.append(AIMessage(content=f"reply {index}"))
    return messages


@pytest.mark.parametrize(("cycles", "still_loaded"), [(1, True), (2, True), (3, False), (5, False)])
def test_a_loaded_tool_is_forgotten_after_the_ttl(cycles, still_loaded):
    middleware = ProgressiveToolDisclosureMiddleware(ttl_cycles=2)
    request = make_request(
        tools=bound_tools(middleware),
        messages=_history(cycles),
        state={"messages": [], "loaded_tools": {"get_balance": 0}},
    )
    seen, _ = run(middleware, request)
    assert ("get_balance" in names_of(seen.tools)) is still_loaded


def test_a_call_renews_the_tool_past_its_original_ttl():
    middleware = ProgressiveToolDisclosureMiddleware(ttl_cycles=2)
    messages = [
        HumanMessage(content="turn"),
        AIMessage(content="reply 0"),
        AIMessage(content="reply 1"),
        # Cycle 2 calls the tool, which renews it.
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "get_balance", "args": {"account_id": "1"}}]),
        ToolMessage(content="1000", tool_call_id="tc1"),
        AIMessage(content="reply 3"),
    ]
    request = make_request(
        tools=bound_tools(middleware), messages=messages, state={"messages": [], "loaded_tools": {"get_balance": 0}}
    )
    seen, _ = run(middleware, request)
    # Loaded on cycle 0 with a two-cycle TTL, it would have expired by cycle 4; the call on cycle 2 kept it.
    assert "get_balance" in names_of(seen.tools)


def test_merge_loads_keeps_the_later_cycle():
    assert _merge_loads({"a": 1, "b": 7}, {"a": 4}) == {"a": 4, "b": 7}
    assert _merge_loads({"a": 4}, {"a": 1}) == {"a": 4}
    assert _merge_loads(None, {"a": 1}) == {"a": 1}
    assert _merge_loads({"a": 1}, None) == {"a": 1}


def test_merge_loads_does_not_mutate_its_arguments():
    left = {"a": 1}
    _merge_loads(left, {"a": 9, "b": 2})
    assert left == {"a": 1}


# --------------------------------------------------------------------------------------------------
# The fold.
# --------------------------------------------------------------------------------------------------


def _two_turns_with_a_closed_exchange() -> list:
    """A first turn that used a tool, a second turn in flight. Only the first is foldable."""
    return [
        HumanMessage(content="turn one"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": HIDDEN, "args": {"account_id": "1", "since": "Jan"}}]),
        ToolMessage(content="3 transactions", tool_call_id="tc1"),
        AIMessage(content="you made 3 transactions"),
        HumanMessage(content="turn two"),
    ]


def test_a_closed_exchange_of_a_tool_the_call_lacks_is_folded(middleware):
    request = make_request(tools=bound_tools(middleware), messages=_two_turns_with_a_closed_exchange())
    seen, _ = run(middleware, request)
    text = "\n".join(m.text for m in seen.messages)

    assert f"The tool {HIDDEN} was called and the result was: 3 transactions" in text
    # The call shape is gone: nothing is left for the model to copy.
    assert not any(getattr(m, "tool_calls", None) for m in seen.messages)
    assert not any(isinstance(m, ToolMessage) for m in seen.messages)
    # The evidence survived.
    assert "3 transactions" in text
    assert "turn one" in text and "turn two" in text


def test_a_closed_exchange_of_a_carried_tool_is_left_alone(middleware):
    messages = _two_turns_with_a_closed_exchange()
    request = make_request(
        tools=bound_tools(middleware), messages=messages, state={"messages": [], "loaded_tools": {HIDDEN: 0}}
    )
    seen, _ = run(middleware, request)

    assert HIDDEN in names_of(seen.tools)
    assert seen.messages == messages
    assert any(isinstance(m, ToolMessage) for m in seen.messages)


def test_the_turn_in_flight_is_returned_unmodified(middleware):
    messages = [
        *_two_turns_with_a_closed_exchange(),
        AIMessage(content="", tool_calls=[{"id": "tc2", "name": "get_balance", "args": {"account_id": "1"}}]),
        ToolMessage(content="1000", tool_call_id="tc2"),
    ]
    request = make_request(
        tools=bound_tools(middleware), messages=messages, state={"messages": [], "loaded_tools": {"get_balance": 1}}
    )
    seen, _ = run(middleware, request)

    # Identity, not equality: a reasoning model rejects a rebuilt latest assistant message.
    assert seen.messages[-1] is messages[-1]
    assert seen.messages[-2] is messages[-2]
    assert f"The tool {HIDDEN} was called" in "\n".join(m.text for m in seen.messages)


def test_a_disclosure_exchange_is_dropped_with_no_sentence(middleware):
    messages = [
        HumanMessage(content="turn one"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": GET_TOOL_DETAILS_NAME, "args": {"names": [HIDDEN]}}]),
        ToolMessage(content="Loaded.", tool_call_id="tc1"),
        AIMessage(content="ready"),
        HumanMessage(content="turn two"),
    ]
    request = make_request(tools=bound_tools(middleware), messages=messages)
    seen, _ = run(middleware, request)
    text = "\n".join(m.text for m in seen.messages)

    assert GET_TOOL_DETAILS_NAME not in text
    assert "Loaded." not in text
    assert "was called and the result was" not in text


def test_an_error_result_is_folded_as_a_failure(middleware):
    messages = _two_turns_with_a_closed_exchange()
    messages[2] = ToolMessage(content="account closed", tool_call_id="tc1", status="error")
    request = make_request(tools=bound_tools(middleware), messages=messages)
    seen, _ = run(middleware, request)
    assert f"The tool {HIDDEN} was called and failed with: account closed" in "\n".join(
        m.text for m in seen.messages
    )


def test_a_partially_folded_exchange_keeps_the_surviving_pair_adjacent(middleware):
    """One assistant message called both a hidden and a carried tool; only the hidden half may fold."""
    messages = [
        HumanMessage(content="turn one"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": HIDDEN, "args": {"account_id": "1", "since": "Jan"}},
                {"id": "tc2", "name": "get_balance", "args": {"account_id": "1"}},
            ],
        ),
        ToolMessage(content="3 transactions", tool_call_id="tc1"),
        ToolMessage(content="1000", tool_call_id="tc2"),
        AIMessage(content="done"),
        HumanMessage(content="turn two"),
    ]
    request = make_request(
        tools=bound_tools(middleware), messages=messages, state={"messages": [], "loaded_tools": {"get_balance": 0}}
    )
    seen, _ = run(middleware, request)

    calls = [c["name"] for m in seen.messages for c in (getattr(m, "tool_calls", None) or ())]
    results = [m for m in seen.messages if isinstance(m, ToolMessage)]

    # The hidden tool's call shape is gone and its evidence is a sentence; the carried one is untouched.
    assert calls == ["get_balance"]
    assert [m.tool_call_id for m in results] == ["tc2"]
    assert f"The tool {HIDDEN} was called and the result was: 3 transactions" in "\n".join(
        m.text for m in seen.messages
    )
    # The surviving result still follows the very message that asked for it.
    caller = next(index for index, m in enumerate(seen.messages) if getattr(m, "tool_calls", None))
    answer = seen.messages[caller + 1]
    assert isinstance(answer, ToolMessage)
    assert answer.tool_call_id == "tc2"


def test_a_single_turn_history_is_never_folded(middleware):
    """Everything from the opening user message on is the turn in flight, so nothing is foldable yet."""
    messages = _two_turns_with_a_closed_exchange()[:4]
    request = make_request(tools=bound_tools(middleware), messages=messages)
    seen, _ = run(middleware, request)
    assert seen.messages == messages


# --------------------------------------------------------------------------------------------------
# The guessed-call guard.
# --------------------------------------------------------------------------------------------------


def _guard(middleware, name, *, args=None, messages=None, loaded=None):
    """Push one tool call through the guard and report the outcome and whether the tool ran.

    The default history is the realistic one: the model loaded on cycle 0 and is calling on cycle 1, so
    two assistant messages are already in state by the time the guard runs.
    """
    call_args = args if args is not None else {"account_id": "1"}
    history = (
        messages
        if messages is not None
        else [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"id": "l1", "name": GET_TOOL_DETAILS_NAME, "args": {"names": ["get_balance"]}}],
            ),
            ToolMessage(content="Loaded.", tool_call_id="l1"),
            AIMessage(content="", tool_calls=[{"id": "tc1", "name": name, "args": call_args}]),
        ]
    )
    state = {"messages": history, "loaded_tools": dict(loaded or {})}
    request = make_tool_call_request(name=name, args=call_args, tools=bound_tools(middleware), state=state)
    handler = CountingHandler()
    return middleware.wrap_tool_call(request, handler), handler


def test_a_call_to_a_catalog_name_is_cancelled(middleware):
    outcome, handler = _guard(middleware, "get_balance")

    assert isinstance(outcome, ToolMessage)
    assert outcome.status == "error"
    assert "did not run" in outcome.content
    assert GET_TOOL_DETAILS_NAME in outcome.content
    assert "get_balance" in outcome.content
    # Nothing ran, and nothing was loaded on the model's behalf.
    assert handler.calls == []


def test_a_call_to_a_loaded_tool_runs(middleware):
    outcome, handler = _guard(middleware, "get_balance", loaded={"get_balance": 0})
    assert outcome == "ran"
    assert len(handler.calls) == 1


def test_a_load_from_the_same_batch_does_not_sanction_a_guess(middleware):
    """``get_tool_details`` running beside the guessed call recorded the current cycle, not an earlier one."""
    outcome, handler = _guard(middleware, "get_balance", loaded={"get_balance": 1})
    assert isinstance(outcome, ToolMessage)
    assert handler.calls == []


def test_a_tool_with_no_required_parameter_is_exempt(middleware):
    outcome, handler = _guard(middleware, "current_time", args={})
    assert outcome == "ran"
    assert len(handler.calls) == 1


def test_the_disclosure_tools_are_never_cancelled(middleware):
    outcome, handler = _guard(middleware, GET_TOOL_DETAILS_NAME, args={"names": ["get_balance"]})
    assert outcome == "ran"
    assert len(handler.calls) == 1


def test_an_always_available_tool_is_never_cancelled():
    middleware = ProgressiveToolDisclosureMiddleware(always_available=["get_balance"])
    outcome, handler = _guard(middleware, "get_balance")
    assert outcome == "ran"
    assert len(handler.calls) == 1


def test_an_unbound_name_is_left_to_the_agent(middleware):
    outcome, handler = _guard(middleware, "no_such_tool")
    assert outcome == "ran"
    assert len(handler.calls) == 1


# --------------------------------------------------------------------------------------------------
# The two tools.
# --------------------------------------------------------------------------------------------------


def test_the_tools_are_registered_with_the_expected_names(middleware):
    assert [t.name for t in middleware.tools] == [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME]


def test_the_runtime_parameter_is_not_exposed_to_the_model(middleware):
    for registered in middleware.tools:
        assert "runtime" not in registered.args


def test_get_tool_details_records_the_names_into_loaded_tools(middleware):
    runtime = make_tool_runtime(
        tools=bound_tools(middleware),
        state={"messages": [HumanMessage(content="hi"), AIMessage(content="loading")]},
    )
    command = middleware._load(["get_balance", HIDDEN], runtime)

    assert isinstance(command, Command)
    assert command.update["loaded_tools"] == {"get_balance": 0, HIDDEN: 0}
    answer = command.update["messages"][0]
    assert isinstance(answer, ToolMessage)
    assert "Loaded." in answer.content
    assert "- get_balance:" in answer.content
    # The schema itself is not in the answer: it travels in the next call's tool list.
    assert "inputSchema" not in answer.content and "properties" not in answer.content


def test_get_tool_details_names_what_is_not_a_tool(middleware):
    command = middleware._load(["get_balance", "nope"], make_tool_runtime(tools=bound_tools(middleware)))
    assert command.update["loaded_tools"] == {"get_balance": 0}
    assert "Not a tool, ignored: nope" in command.update["messages"][0].content


def test_get_tool_details_tolerates_a_bare_string_and_duplicates(middleware):
    runtime = make_tool_runtime(tools=bound_tools(middleware))
    assert middleware._load("get_balance", runtime).update["loaded_tools"] == {"get_balance": 0}
    assert middleware._load(["get_balance", "get_balance"], runtime).update["loaded_tools"] == {"get_balance": 0}


def test_get_tool_details_guides_an_empty_request(middleware):
    command = middleware._load([], make_tool_runtime(tools=bound_tools(middleware)))
    assert "loaded_tools" not in command.update
    assert "as a list" in command.update["messages"][0].content


def test_get_tool_details_will_not_load_itself(middleware):
    command = middleware._load([GET_TOOL_DETAILS_NAME], make_tool_runtime(tools=bound_tools(middleware)))
    assert "loaded_tools" not in command.update


def test_find_tools_lists_matches_without_loading_them(middleware):
    runtime = make_tool_runtime(tools=bound_tools(middleware))
    answer = middleware._search("list the transactions of an investment account", runtime)

    assert "Nothing is loaded yet" in answer
    assert f"- {HIDDEN}:" in answer
    assert GET_TOOL_DETAILS_NAME in answer


def test_find_tools_never_lists_itself(middleware):
    answer = middleware._search("find tools and get tool details", make_tool_runtime(tools=bound_tools(middleware)))
    assert f"- {FIND_TOOLS_NAME}:" not in answer
    assert f"- {GET_TOOL_DETAILS_NAME}:" not in answer


def test_find_tools_guides_an_empty_need(middleware):
    answer = middleware._search("   ", make_tool_runtime(tools=bound_tools(middleware)))
    assert "Describe what you are trying to do" in answer


def test_find_tools_reports_a_need_nothing_matches(middleware):
    answer = middleware._search("zzzzqqqx", make_tool_runtime(tools=bound_tools(middleware)))
    assert "No tool matches" in answer


def test_find_tools_returns_guidance_when_the_search_raises():
    class Exploding:
        def build(self, specs):
            return None

        def search(self, need, top_k):
            raise RuntimeError("index down")

    middleware = ProgressiveToolDisclosureMiddleware(index=Exploding())
    answer = middleware._search("anything", make_tool_runtime(tools=bound_tools(middleware)))
    assert "unavailable right now" in answer


# --------------------------------------------------------------------------------------------------
# Degradation: nothing here may leave the model without tools.
# --------------------------------------------------------------------------------------------------


def test_a_failing_index_build_does_not_break_the_call():
    class Exploding:
        def build(self, specs):
            raise RuntimeError("build failed")

        def search(self, need, top_k):
            return []

    middleware = ProgressiveToolDisclosureMiddleware(index=Exploding())
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    # Degrades to the request as received: every tool goes out, which is the behaviour with no middleware.
    assert seen is request


def test_a_failing_guard_lets_the_call_through(middleware):
    """Cancelling a legitimate call is the worse outcome, so a guard that cannot decide decides nothing."""

    class Hostile:
        @property
        def tools(self):
            raise RuntimeError("runtime unreadable")

    request = make_tool_call_request(name="get_balance", tools=bound_tools(middleware), state={})
    request = request.override(runtime=Hostile())
    handler = CountingHandler()

    assert middleware.wrap_tool_call(request, handler) == "ran"
    assert len(handler.calls) == 1


# --------------------------------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"catalog_chars": 0},
        {"catalog_chars": True},
        {"catalog_chars": 80.0},
        {"ttl_cycles": 0},
        {"ttl_cycles": True},
        {"top_k": 0},
        {"always_available": "current_time"},
        {"always_available": [""]},
        {"always_available": [1]},
        {"index": object()},
        {"summarizer": "not callable"},
    ],
)
def test_a_bad_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        ProgressiveToolDisclosureMiddleware(**kwargs)


def test_a_summarizer_writes_the_catalog_line():
    middleware = ProgressiveToolDisclosureMiddleware(
        catalog_chars=40, summarizer=lambda spec, limit: f"summary of {spec['name']}"
    )
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    assert f"- {HIDDEN}: summary of {HIDDEN}" in seen.system_message.text


def test_a_failing_summarizer_falls_back_to_truncation():
    def explode(spec, limit):
        raise RuntimeError("no model")

    middleware = ProgressiveToolDisclosureMiddleware(catalog_chars=40, summarizer=explode)
    request = make_request(tools=bound_tools(middleware), messages=[HumanMessage(content="hi")])
    seen, _ = run(middleware, request)
    lines = [line for line in seen.system_message.text.splitlines() if line.startswith(f"- {HIDDEN}:")]
    assert len(lines) == 1
    assert len(lines[0].split(": ", 1)[1]) <= 40


def test_the_catalog_is_byte_stable_across_calls(middleware):
    """The block sits in the system prompt a provider may be caching by prefix."""
    blocks = []
    for cycle in range(3):
        request = make_request(tools=bound_tools(middleware), messages=_history(cycle))
        seen, _ = run(middleware, request)
        blocks.append(seen.system_message.text)
    assert blocks[0] == blocks[1] == blocks[2]


# --------------------------------------------------------------------------------------------------
# End to end, through create_agent, against a scripted model.
# --------------------------------------------------------------------------------------------------


def test_the_agent_discovers_loads_and_calls(middleware):
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"id": "l1", "name": GET_TOOL_DETAILS_NAME, "args": {"names": ["get_balance"]}}],
            ),
            AIMessage(
                content="",
                id="a2",
                tool_calls=[{"id": "b1", "name": "get_balance", "args": {"account_id": "123"}}],
            ),
            AIMessage(content="Your balance is 1000.", id="a3"),
        ],
        calls=[],
    )
    agent = create_agent(model=model, tools=DOMAIN_TOOLS, middleware=[middleware], system_prompt="You are helpful.")

    result = agent.invoke({"messages": [HumanMessage(content="what is my balance?")]})

    # The first call paid for two tools, not five, and read the rest off the catalog.
    assert model.calls[0]["tools"] == sorted(DISCLOSURE_TOOLS)
    assert "# Tools available on request" in model.calls[0]["system"]
    assert "- get_balance:" in model.calls[0]["system"]
    # The load made the schema callable on the next call, and took it out of the catalog.
    assert "get_balance" in model.calls[1]["tools"]
    assert "- get_balance:" not in model.calls[1]["system"]
    assert result["loaded_tools"] == {"get_balance": 0}
    assert result["messages"][-1].content == "Your balance is 1000."


def test_the_agent_cancels_a_guessed_call(middleware):
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"id": "g1", "name": "get_balance", "args": {"account_id": "123"}}],
            ),
            AIMessage(content="Let me load it first.", id="a2"),
        ],
        calls=[],
    )
    agent = create_agent(model=model, tools=DOMAIN_TOOLS, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage(content="what is my balance?")]})

    cancellation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "did not run" in cancellation.content
    assert GET_TOOL_DETAILS_NAME in cancellation.content
    # The guard loaded nothing on the model's behalf.
    assert not result.get("loaded_tools")
    assert middleware.premature_cancellations == 1


@tool
def reconcile_ledger(account_id: str, period: str) -> str:
    """Reconcile the general ledger of one account for one accounting period against the bank statement,
    returning every unmatched entry with its amount, posting date and the reason it failed to match."""
    return "ok"


def _summary_agent(middleware, model):
    return create_agent(model=model, tools=[*DOMAIN_TOOLS, reconcile_ledger], middleware=[middleware])


def test_the_default_summarizer_is_the_agents_own_model(middleware):
    model = ScriptedModel(responses=[AIMessage(content="hi", id="a1")], calls=[], summary_calls=[])
    _summary_agent(middleware, model).invoke({"messages": [HumanMessage(content="hello")]})

    # The long description went to the model with the Strands instruction, once, and its usage is billed.
    asked = [call[-1].text for call in model.summary_calls]
    assert any("Tool: reconcile_ledger" in text for text in asked)
    assert all("reconcile_ledger" not in call["system"] or "summarized line" in call["system"] for call in model.calls)
    assert "- reconcile_ledger: summarized line" in model.calls[0]["system"]
    assert middleware.summary_usage["calls"] == len(model.summary_calls)
    assert middleware.summary_usage["inputTokens"] == 40 * len(model.summary_calls)
    # Cached: a second run over the same tools summarizes nothing new.
    before = len(model.summary_calls)
    _summary_agent(middleware, model).invoke({"messages": [HumanMessage(content="again")]})
    assert len(model.summary_calls) == before


async def test_the_default_summarizer_runs_under_ainvoke(middleware):
    model = ScriptedModel(responses=[AIMessage(content="hi", id="a1")], calls=[], summary_calls=[])
    await _summary_agent(middleware, model).ainvoke({"messages": [HumanMessage(content="hello")]})
    assert "- reconcile_ledger: summarized line" in model.calls[0]["system"]
    assert middleware.summary_usage["calls"] >= 1


def test_a_caller_summarizer_replaces_the_model():
    lines = []
    custom = ProgressiveToolDisclosureMiddleware(summarizer=lambda spec, n: lines.append(spec["name"]) or "custom")
    model = ScriptedModel(responses=[AIMessage(content="hi", id="a1")], calls=[], summary_calls=[])
    _summary_agent(custom, model).invoke({"messages": [HumanMessage(content="hello")]})
    assert model.summary_calls == []
    assert "reconcile_ledger" in lines
    assert custom.summary_usage == {}


def test_a_fold_leaves_no_provider_tool_use_part_behind():
    """Regression from the first live run: Bedrock carries each call twice, in ``tool_calls`` and as a
    ``tool_use`` content part. Folding the call must drop both, or Converse rejects the orphan."""
    from langgraph_progressive_tool_disclosure.middleware import _fold_messages

    load = AIMessage(
        content=[
            {"type": "text", "text": "Loading."},
            {"type": "tool_use", "id": "c2", "name": GET_TOOL_DETAILS_NAME, "input": {"names": ["get_balance"]}},
        ],
        id="a2",
        tool_calls=[{"id": "c2", "name": GET_TOOL_DETAILS_NAME, "args": {"names": ["get_balance"]}}],
    )
    history = [
        HumanMessage(content="first", id="h1"),
        load,
        ToolMessage(content="loaded get_balance", tool_call_id="c2", name=GET_TOOL_DETAILS_NAME, id="t2"),
        AIMessage(content="done", id="a3"),
        HumanMessage(content="second", id="h4"),
    ]
    folded = _fold_messages(history, active={"find_tools"})

    call_ids = set()
    for message in folded:
        if isinstance(message, AIMessage):
            call_ids |= {call["id"] for call in message.tool_calls}
            call_ids |= {p.get("id") for p in message.content if isinstance(p, dict) and p.get("type") == "tool_use"} if isinstance(message.content, list) else set()
    answered = {m.tool_call_id for m in folded if isinstance(m, ToolMessage)}
    assert call_ids <= answered


def test_a_projection_by_an_outer_middleware_does_not_reset_the_cycle():
    """Regression from the live run: the context graph wraps this middleware and projects the call's
    messages down. The cycle must come from the persisted history, or a tool loaded on cycle 9 (read
    from state) is never callable on a projected cycle 4 and the model reloads it forever."""
    middleware = ProgressiveToolDisclosureMiddleware(ttl_cycles=3)
    full = _history(10)
    projected = [full[0], *full[-4:]]  # an outer middleware collapsed the older turns
    request = make_request(
        tools=bound_tools(middleware),
        messages=projected,
        state={"messages": full, "loaded_tools": {"get_balance": 9}},
    )
    seen, _ = run(middleware, request)
    assert "get_balance" in names_of(seen.tools)


def test_a_load_numbered_before_messages_were_removed_stays_callable():
    """A middleware that deletes messages from state (the relevance filter's end-of-run cleanup) shrinks
    the AIMessage count, so a load recorded on the longer history reads as a future cycle. It must still
    be callable, or the model reloads it until the count catches up."""
    middleware = ProgressiveToolDisclosureMiddleware(ttl_cycles=3)
    history = _history(5)  # the next call runs on cycle 5
    request = make_request(
        tools=bound_tools(middleware),
        messages=history,
        state={"messages": history, "loaded_tools": {"get_balance": 7}},  # numbered before a removal
    )
    seen, _ = run(middleware, request)
    assert "get_balance" in names_of(seen.tools)
