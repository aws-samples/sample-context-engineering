"""Adapter boundary tests: LangChain BaseMessage <-> context_core neutral shape."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_relevance_filter._adapter import (
    result_block_to_content,
    to_neutral,
    to_neutral_list,
    tool_message_to_result_block,
)


def test_human_message_to_neutral():
    n = to_neutral(HumanMessage(content="what is my balance?"))
    assert n == {"role": "user", "content": [{"text": "what is my balance?"}]}


def test_system_message_to_neutral():
    n = to_neutral(SystemMessage(content="you are helpful"))
    assert n == {"role": "system", "content": [{"text": "you are helpful"}]}


def test_ai_message_with_tool_call_to_neutral():
    msg = AIMessage(
        content="calling",
        tool_calls=[{"id": "tc1", "name": "get_balance", "args": {"acct": "123"}}],
    )
    n = to_neutral(msg)
    assert n["role"] == "assistant"
    assert {"text": "calling"} in n["content"]
    assert {"toolUse": {"toolUseId": "tc1", "name": "get_balance", "input": {"acct": "123"}}} in n["content"]


def test_tool_message_to_result_block():
    tm = ToolMessage(content="42 rows", tool_call_id="tc1")
    block = tool_message_to_result_block(tm)
    assert block["toolResult"]["toolUseId"] == "tc1"
    assert block["toolResult"]["status"] == "success"
    assert block["toolResult"]["content"] == [{"text": "42 rows"}]


def test_tool_message_error_status():
    tm = ToolMessage(content="boom", tool_call_id="tc9", status="error")
    block = tool_message_to_result_block(tm)
    assert block["toolResult"]["status"] == "error"


def test_result_block_single_text_collapses_to_string():
    block = {"toolResult": {"toolUseId": "t", "status": "success", "content": [{"text": "hello"}]}}
    assert result_block_to_content(block) == "hello"


def test_tool_message_roundtrips_through_result_block():
    tm = ToolMessage(content="verbatim payload", tool_call_id="tc1")
    block = tool_message_to_result_block(tm)
    assert result_block_to_content(block) == "verbatim payload"


def test_to_neutral_list_preserves_order():
    msgs = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
    neutral = to_neutral_list(msgs)
    assert [m["content"][0]["text"] for m in neutral] == ["a", "b", "c"]
