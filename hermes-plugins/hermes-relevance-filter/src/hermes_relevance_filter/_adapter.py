"""Adapter between Hermes OpenAI-format chat messages and the ``context_core`` neutral shape.

The ONLY module in this binding that touches both worlds. ``context_core`` never sees a Hermes message;
the binding never leaks a neutral dict into a Hermes request without converting it back here.

Hermes message shape (OpenAI chat format)::

    {"role": "system"|"user"|"assistant"|"tool", "content": str | None,
     "tool_calls": [{"id", "type": "function", "function": {"name", "arguments": <json str>}}],  # assistant
     "tool_call_id": str}                                                                        # tool

Neutral shape (see ``context_core.message``)::

    {"role": ..., "content": [block, ...]}
    block in {"text": str} | {"json": Any}
           | {"toolUse": {"toolUseId", "name", "input"}}
           | {"toolResult": {"toolUseId", "status", "content": [block, ...]}}

Round-trip contract: a message converted to neutral and back is semantically identical for the known
shapes, and any key the adapter does not recognise is preserved verbatim under
``_HERMES_EXTRA_KEY`` so provider-specific parts (reasoning, cache-control, opaque replay items) survive
the trip. ``neutral_to_hermes`` strips that carrier key back out.
"""

from __future__ import annotations

import json
from typing import Any

from context_core.message import NeutralMessage

__all__ = [
    "hermes_to_neutral",
    "hermes_to_neutral_list",
    "neutral_to_hermes",
    "neutral_to_hermes_list",
]

#: Neutral-block key under which an unrecognised Hermes message's extra keys ride, so the round trip is
#: lossless. Never emitted to the model; stripped by ``neutral_to_hermes``.
_HERMES_EXTRA_KEY = "_hermes_extra"

#: Hermes message keys the adapter maps explicitly; everything else (including ``name`` on a tool or
#: assistant message) is preserved verbatim via ``_HERMES_EXTRA_KEY``.
_KNOWN_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id"})


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise a Hermes ``content`` (str, None, or OpenAI content-part list) into neutral blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if isinstance(part, str):
            if part:
                blocks.append({"text": part})
        elif isinstance(part, dict):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append({"text": part["text"]})
            else:
                blocks.append({"json": part})
    return blocks


def _extra_keys(msg: dict[str, Any]) -> dict[str, Any]:
    """The message's keys the adapter does not map, kept for a lossless round trip."""
    return {key: value for key, value in msg.items() if key not in _KNOWN_KEYS}


def hermes_to_neutral(msg: dict[str, Any]) -> NeutralMessage:
    """Convert one Hermes OpenAI-format message to a neutral message dict."""
    role = msg.get("role", "user")
    extra = _extra_keys(msg)

    if role == "tool":
        block = {
            "toolResult": {
                "toolUseId": msg.get("tool_call_id", ""),
                "status": "success",
                "content": _content_to_blocks(msg.get("content")),
            }
        }
        content: list[dict[str, Any]] = [block]
        if extra:
            content.append({_HERMES_EXTRA_KEY: extra})
        # A tool result is carried on a user-role neutral message (the neutral convention).
        return {"role": "user", "content": content}

    content = _content_to_blocks(msg.get("content"))

    if role == "assistant":
        for call in msg.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            raw_args = function.get("arguments", "{}")
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (TypeError, ValueError):
                parsed = {"_raw_arguments": raw_args}
            content.append(
                {
                    "toolUse": {
                        "toolUseId": call.get("id", "") if isinstance(call, dict) else "",
                        "name": function.get("name", ""),
                        "input": parsed,
                    }
                }
            )

    if extra:
        content.append({_HERMES_EXTRA_KEY: extra})
    return {"role": role, "content": content}


def hermes_to_neutral_list(messages: list[dict[str, Any]]) -> list[NeutralMessage]:
    """Convert a list of Hermes messages to neutral messages, in order."""
    return [hermes_to_neutral(message) for message in messages]


def _blocks_to_content(blocks: list[dict[str, Any]]) -> Any:
    """Render neutral text/json blocks back to a Hermes ``content`` value.

    A single text block collapses to a plain string (what a provider normally emits); anything richer
    becomes an OpenAI content-part list.
    """
    renderable = [b for b in blocks if "text" in b or "json" in b]
    if len(renderable) == 1 and "text" in renderable[0]:
        return renderable[0]["text"]
    parts: list[Any] = []
    for block in renderable:
        if "text" in block:
            parts.append({"type": "text", "text": block["text"]})
        elif "json" in block:
            parts.append(block["json"])
    return parts


def neutral_to_hermes(msg: NeutralMessage) -> dict[str, Any]:
    """Convert one neutral message back to a Hermes OpenAI-format message."""
    role = msg.get("role", "user")
    content_blocks = msg.get("content", []) or []

    extra: dict[str, Any] = {}
    tool_result = None
    tool_uses: list[dict[str, Any]] = []
    plain_blocks: list[dict[str, Any]] = []
    for block in content_blocks:
        if _HERMES_EXTRA_KEY in block:
            extra = block[_HERMES_EXTRA_KEY]
        elif "toolResult" in block:
            tool_result = block["toolResult"]
        elif "toolUse" in block:
            tool_uses.append(block["toolUse"])
        else:
            plain_blocks.append(block)

    if tool_result is not None:
        out: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_result.get("toolUseId", ""),
            "content": _blocks_to_content(tool_result.get("content", [])),
        }
        out.update(extra)
        return out

    out = {"role": role, "content": _blocks_to_content(plain_blocks)}
    if tool_uses:
        out["tool_calls"] = [
            {
                "id": use.get("toolUseId", ""),
                "type": "function",
                "function": {
                    "name": use.get("name", ""),
                    "arguments": json.dumps(use.get("input", {})),
                },
            }
            for use in tool_uses
        ]
    out.update(extra)
    return out


def neutral_to_hermes_list(messages: list[NeutralMessage]) -> list[dict[str, Any]]:
    """Convert a list of neutral messages back to Hermes messages, in order."""
    return [neutral_to_hermes(message) for message in messages]
