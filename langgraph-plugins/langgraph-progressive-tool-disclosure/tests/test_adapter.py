"""Adapter boundary tests: the neutral -> LangChain half this binding added for the fold's round trip.

The forward half is covered by the sibling bindings; what is tested here is what the fold needs back —
a neutral message split into the LangChain messages that carry it, with tool results ahead of text and
the original message identity preserved.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_progressive_tool_disclosure._adapter import to_langchain, to_langchain_list, to_neutral


def test_a_human_message_round_trips():
    original = HumanMessage(content="what is my balance?", id="h1")
    out = to_langchain(to_neutral(original))
    assert len(out) == 1
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == "what is my balance?"
    assert out[0].id == "h1"


def test_a_system_message_round_trips():
    out = to_langchain(to_neutral(SystemMessage(content="you are helpful", id="s1")))
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "you are helpful"
    assert out[0].id == "s1"


def test_an_assistant_tool_call_round_trips():
    original = AIMessage(
        content="calling",
        id="a1",
        tool_calls=[{"id": "tc1", "name": "get_balance", "args": {"account_id": "123"}}],
    )
    out = to_langchain(to_neutral(original))
    assert len(out) == 1
    assert out[0].id == "a1"
    assert out[0].tool_calls == [
        {"name": "get_balance", "args": {"account_id": "123"}, "id": "tc1", "type": "tool_call"}
    ]


def test_a_tool_message_round_trips():
    out = to_langchain(to_neutral(ToolMessage(content="1000", tool_call_id="tc1", id="t1")))
    assert len(out) == 1
    assert isinstance(out[0], ToolMessage)
    assert out[0].content == "1000"
    assert out[0].tool_call_id == "tc1"
    assert out[0].id == "t1"


def test_an_error_status_survives_the_round_trip():
    out = to_langchain(to_neutral(ToolMessage(content="boom", tool_call_id="tc9", status="error")))
    assert out[0].status == "error"


def test_a_result_and_a_fold_sentence_split_with_the_result_first():
    """A provider rejects text ahead of the ``toolResult`` answering the previous assistant message."""
    neutral = {
        "role": "user",
        "tracking_id": "t1",
        "content": [
            {"toolResult": {"toolUseId": "tc1", "status": "success", "content": [{"text": "1000"}]}},
            {"text": "The tool list_investment_transactions was called and the result was: 3 transactions"},
        ],
    }
    out = to_langchain(neutral)

    assert [type(m).__name__ for m in out] == ["ToolMessage", "HumanMessage"]
    assert out[0].tool_call_id == "tc1"
    # The identity belongs to the message that existed before the fold, not to the synthetic carrier.
    assert out[0].id == "t1"
    assert out[1].id is None
    assert "3 transactions" in out[1].content


def test_several_results_on_one_neutral_message_split_in_order():
    neutral = {
        "role": "user",
        "tracking_id": "t1",
        "content": [
            {"toolResult": {"toolUseId": "tc1", "status": "success", "content": [{"text": "a"}]}},
            {"toolResult": {"toolUseId": "tc2", "status": "success", "content": [{"text": "b"}]}},
        ],
    }
    out = to_langchain(neutral)
    assert [m.tool_call_id for m in out] == ["tc1", "tc2"]
    assert [m.id for m in out] == ["t1", None]


def test_an_emptied_message_produces_nothing():
    assert to_langchain({"role": "assistant", "content": []}) == []
    assert to_langchain({"role": "user", "content": []}) == []
    assert to_langchain({"role": "system", "content": []}) == []


def test_several_text_blocks_stay_apart():
    out = to_langchain({"role": "user", "content": [{"text": "a"}, {"text": "b"}]})
    assert out[0].content == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def test_a_list_round_trips_in_order():
    originals = [
        HumanMessage(content="turn one", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"id": "tc1", "name": "get_balance", "args": {}}]),
        ToolMessage(content="1000", tool_call_id="tc1", id="t1"),
        AIMessage(content="your balance is 1000", id="a2"),
    ]
    out = to_langchain_list([to_neutral(m) for m in originals])
    assert [type(m).__name__ for m in out] == [type(m).__name__ for m in originals]
    assert [m.id for m in out] == ["h1", "a1", "t1", "a2"]
