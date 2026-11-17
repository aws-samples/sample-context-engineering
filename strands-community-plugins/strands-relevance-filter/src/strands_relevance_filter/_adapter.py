"""Strands message and tool-result shapes <-> the neutral shape of :mod:`context_core.message`.

The neutral shape was modeled on the Strands SDK's own message dicts, so this adapter converts nothing: Strands
``Message`` and ``ToolResult`` dicts already *are* neutral messages and blocks. What it guarantees instead:

- **Unknown blocks pass through untouched.** ``reasoningContent``, ``image``, ``document``, ``cachePoint`` blocks and
  message-level keys such as ``metadata`` stay in place, because the core ignores keys it does not recognize and the
  adapter hands over the same dicts.
- **Nothing is mutated.** The core never writes into what it receives; the lists returned here are new, so a binding
  can assign them back to a live SDK object (``agent.messages[:] = ...``) without aliasing the input.

This is the only module of the package that touches both ``strands`` types and :mod:`context_core.message`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from context_core.message import NeutralBlock, NeutralMessage

if TYPE_CHECKING:
    from ._compat import Message, ToolResult

__all__ = ["messages_from_neutral", "messages_to_neutral", "result_from_neutral", "result_to_neutral"]


def messages_to_neutral(messages: Sequence[Message]) -> list[NeutralMessage]:
    """Return the Strands messages as neutral messages: a new list over the same message dicts."""
    return [cast(NeutralMessage, message) for message in messages]


def messages_from_neutral(messages: Sequence[NeutralMessage]) -> list[Message]:
    """Return neutral messages as Strands messages: a new list over the same message dicts."""
    return [cast("Message", message) for message in messages]


def result_to_neutral(result: ToolResult) -> NeutralBlock:
    """Return a Strands ``ToolResult`` as the body of a neutral ``toolResult`` block."""
    return cast(NeutralBlock, result)


def result_from_neutral(block: dict[str, Any]) -> ToolResult:
    """Return the body of a neutral ``toolResult`` block as a Strands ``ToolResult``."""
    return cast("ToolResult", block)
