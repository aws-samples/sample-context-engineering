"""Adapter round-trip tests (Task 4.2): Hermes OpenAI message <-> neutral, unknown parts survive."""

from __future__ import annotations

import json

from hermes_context_graph._adapter import (
    hermes_to_neutral,
    hermes_to_neutral_list,
    neutral_to_hermes,
    neutral_to_hermes_list,
)


def test_user_message_roundtrip() -> None:
    msg = {"role": "user", "content": "what is the max refund?"}
    assert neutral_to_hermes(hermes_to_neutral(msg)) == msg


def test_system_message_roundtrip() -> None:
    msg = {"role": "system", "content": "You are helpful."}
    assert neutral_to_hermes(hermes_to_neutral(msg)) == msg


def test_assistant_tool_call_roundtrip() -> None:
    msg = {
        "role": "assistant",
        "content": "let me check",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "query_ledger", "arguments": '{"q": "refunds"}'}}
        ],
    }
    back = neutral_to_hermes(hermes_to_neutral(msg))
    assert back["role"] == "assistant"
    assert back["content"] == "let me check"
    assert back["tool_calls"][0]["id"] == "call_1"
    assert back["tool_calls"][0]["function"]["name"] == "query_ledger"
    assert json.loads(back["tool_calls"][0]["function"]["arguments"]) == {"q": "refunds"}


def test_tool_result_roundtrip() -> None:
    msg = {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"}
    back = neutral_to_hermes(hermes_to_neutral(msg))
    assert back["role"] == "tool"
    assert back["tool_call_id"] == "call_1"
    assert back["content"] == "42 rows"


def test_unknown_parts_preserved_verbatim() -> None:
    msg = {
        "role": "assistant",
        "content": "thinking…",
        "reasoning_content": {"opaque": "provider-blob", "sig": "abc"},
        "cache_control": {"type": "ephemeral"},
    }
    neutral = hermes_to_neutral(msg)
    back = neutral_to_hermes(neutral)
    assert back["reasoning_content"] == {"opaque": "provider-blob", "sig": "abc"}
    assert back["cache_control"] == {"type": "ephemeral"}


def test_tool_message_unknown_parts_preserved() -> None:
    msg = {"role": "tool", "tool_call_id": "c1", "content": "ok", "name": "query_ledger"}
    back = neutral_to_hermes(hermes_to_neutral(msg))
    assert back["name"] == "query_ledger"


def test_malformed_tool_arguments_do_not_raise() -> None:
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}],
    }
    neutral = hermes_to_neutral(msg)
    use = [b["toolUse"] for b in neutral["content"] if "toolUse" in b][0]
    assert use["input"] == {"_raw_arguments": "{not json"}


def test_list_roundtrip_order_preserved() -> None:
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert neutral_to_hermes_list(hermes_to_neutral_list(msgs)) == msgs
