"""Strands messages <-> the neutral shape of :mod:`context_core.message`.

The neutral shape was modeled on the Strands SDK's own message dicts, so this adapter converts nothing: a Strands
``Message`` already *is* a neutral message. What it guarantees instead:

- **Unknown blocks pass through untouched.** ``reasoningContent``, ``image``, ``document``, ``cachePoint`` blocks and
  message-level keys such as ``metadata`` stay in place, because the core ignores keys it does not recognize and the
  adapter hands over the same dicts.
- **Nothing is mutated.** The core never writes into what it receives; the lists returned here are new.

This is the only module of the package that touches both ``strands`` types and :mod:`context_core.message`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from context_core.disclosure import ToolSpec as NeutralToolSpec
from context_core.message import NeutralMessage

if TYPE_CHECKING:
    from ._compat import Messages, ToolSpec

__all__ = ["messages_from_neutral", "messages_to_neutral", "specs_to_neutral"]


def messages_to_neutral(messages: Messages) -> list[NeutralMessage]:
    """Return the Strands messages as neutral messages: a new list over the same message dicts."""
    return [cast(NeutralMessage, message) for message in messages]


def messages_from_neutral(messages: Sequence[NeutralMessage]) -> Messages:
    """Return neutral messages as Strands messages: a new list over the same message dicts."""
    return cast("Messages", list(messages))


def specs_to_neutral(specs: Sequence[ToolSpec]) -> list[NeutralToolSpec]:
    """Return Strands tool specifications as neutral ones: a new list over the same spec dicts.

    The Strands ``ToolSpec`` is a ``TypedDict`` over the same keys the neutral ``dict`` carries, so nothing is
    converted; this only states that to the type checker.
    """
    return [cast(NeutralToolSpec, spec) for spec in specs]
