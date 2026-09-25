"""Adapter boundary tests: LangChain ``BaseMessage`` <-> the ``context_core`` neutral shape, both ways.

The reverse direction is what the projection needs to hand a rewritten message list back to LangChain, so
the property that matters is the round trip: a conversation converted out and back is the same conversation,
with the same ids, the same tool calls and the same tool results. The two shapes that do **not** exist as a
single LangChain message -- a neutral ``user`` message carrying a tool result, with the compaction's block
folded onto it -- are asserted separately, since that is the one place the mapping is not one to one.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_context_graph._adapter import (
    to_langchain,
    to_langchain_list,
    to_neutral,
    to_neutral_list,
)


def _shape(message: BaseMessage) -> tuple:
    """Everything about a message this binding is responsible for preserving."""
    return (
        type(message),
        message.content,
        message.id,
        tuple(
            (call["name"], tuple(sorted((call.get("args") or {}).items())), call["id"])
            for call in getattr(message, "tool_calls", ()) or ()
        ),
        getattr(message, "tool_call_id", None),
        getattr(message, "status", None),
    )


# ---- LangChain -> neutral -------------------------------------------------------------------------


def test_the_message_id_travels_as_the_tracking_id():
    """The Durable Identity a Card addresses: without it a turn yields no Card at all."""
    assert to_neutral(HumanMessage(content="hello", id="u1")) == {
        "role": "user",
        "content": [{"text": "hello"}],
        "tracking_id": "u1",
    }


def test_a_message_without_an_id_carries_no_identity():
    assert "tracking_id" not in to_neutral(HumanMessage(content="hello"))


def test_a_tool_message_becomes_a_tool_result_on_a_user_message():
    neutral = to_neutral(ToolMessage(content="42 rows", tool_call_id="tc1", id="t1"))

    assert neutral["role"] == "user"
    assert neutral["tracking_id"] == "t1"
    assert neutral["content"] == [
        {"toolResult": {"toolUseId": "tc1", "status": "success", "content": [{"text": "42 rows"}]}}
    ]


# ---- neutral -> LangChain -------------------------------------------------------------------------


def test_the_conversation_round_trips():
    """Out and back is the same conversation: types, content, ids, tool calls and tool results."""
    messages: list[BaseMessage] = [
        SystemMessage(content="you are helpful", id="s1"),
        HumanMessage(content="what did it cost?", id="u1"),
        AIMessage(
            content="let me look",
            tool_calls=[{"name": "lookup", "args": {"q": 1}, "id": "tc1", "type": "tool_call"}],
            id="a1",
        ),
        ToolMessage(content="100 reais", tool_call_id="tc1", id="t1"),
        AIMessage(content="it cost 100 reais", id="z1"),
        HumanMessage(content="and topic 2?", id="u2"),
    ]

    restored = to_langchain_list(to_neutral_list(messages))

    assert [_shape(message) for message in restored] == [_shape(message) for message in messages]


def test_an_error_tool_result_round_trips_as_an_error():
    original = ToolMessage(content="boom", tool_call_id="tc1", status="error", id="t1")

    (restored,) = to_langchain(to_neutral(original))

    assert isinstance(restored, ToolMessage)
    assert restored.status == "error"
    assert restored.content == "boom"


def test_an_assistant_message_with_no_text_round_trips_to_an_empty_content():
    original = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {}, "id": "tc1", "type": "tool_call"}],
        id="a1",
    )

    (restored,) = to_langchain(to_neutral(original))

    assert isinstance(restored, AIMessage)
    assert restored.content == []
    assert [call["name"] for call in restored.tool_calls] == ["lookup"]


def test_several_blocks_render_as_content_parts():
    """One text block collapses to a string; more than one cannot, so they stay addressable parts."""
    folded = {"role": "user", "content": [{"text": "first"}, {"text": "second"}], "tracking_id": "u1"}

    (restored,) = to_langchain(folded)

    assert isinstance(restored, HumanMessage)
    assert restored.content == [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    assert restored.id == "u1"


def test_a_json_block_survives_as_a_content_part():
    (restored,) = to_langchain({"role": "user", "content": [{"json": {"kind": "image", "ref": "abc"}}]})

    assert restored.content == [{"kind": "image", "ref": "abc"}]


def test_a_folded_tool_result_turn_splits_into_the_result_and_the_folded_text():
    """The one shape LangChain has no single message for: the compaction's block on a tool-result turn.

    The tool result has to stay immediately behind its tool call, so the folded text cannot be prepended to
    it and cannot be merged into it either. It travels as its own message, behind the result, and claims no
    Durable Identity: it is per-call content, not a persisted message.
    """
    folded = {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "tc1", "status": "success", "content": [{"text": "100 reais"}]}},
            {"text": "\n\n<collapsed turns>…</collapsed turns>"},
        ],
        "tracking_id": "t1",
    }

    result, appended = to_langchain(folded)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "tc1"
    assert result.id == "t1"
    assert isinstance(appended, HumanMessage)
    assert "collapsed turns" in appended.content
    assert appended.id is None


def test_a_system_message_round_trips():
    (restored,) = to_langchain(to_neutral(SystemMessage(content="be brief", id="s1")))

    assert isinstance(restored, SystemMessage)
    assert restored.content == "be brief"
    assert restored.id == "s1"


def test_an_empty_neutral_list_converts_to_an_empty_list():
    assert to_langchain_list([]) == []


def test_a_message_with_no_content_at_all_is_dropped():
    """Several providers reject an empty message, and it carries nothing to reject it for."""
    assert to_langchain({"role": "user", "content": []}) == []
    assert to_langchain({"role": "assistant", "content": []}) == []
    assert to_langchain({"role": "system", "content": []}) == []
