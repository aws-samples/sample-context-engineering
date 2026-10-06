"""Token estimate of a tool result, matching the Strands SDK's default heuristic exactly.

The size gate of Practice A must flip at the same result size in every binding, or a result between roughly 8k and 16k
real tokens is filtered on one framework and sent whole on another. The Strands SDK's base ``Model.count_tokens``
(used unless a provider opts into native counting) counts a tool result's content items as:

- ``text``: ``ceil(len(text) / 4)``;
- ``json``: ``ceil(len(json.dumps(obj)) / 2)``, and ``0`` when the object does not serialize;
- binary items (``image``, ``document``, anything else): not counted.

This module reproduces that rule with no framework import, for the bindings whose hook exposes no model to ask
(LangGraph's ``wrap_tool_call``, Hermes' engine). A binding that *does* have the model at hand (Strands) should keep
asking it, because a provider may override the count -- Bedrock's native ``CountTokens`` -- and the gate should honour
that override.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

__all__ = ["approximate_tokens"]

_CHARS_PER_TEXT_TOKEN = 4
_CHARS_PER_JSON_TOKEN = 2


def approximate_tokens(content: Iterable[Any]) -> int:
    """Estimate the tokens of a tool result's content items with the SDK's default heuristic.

    Args:
        content: The result's content items, in the neutral (Strands) shape: dicts carrying ``text`` and/or ``json``.
            Any other key, and any non-dict item, counts zero.

    Returns:
        The estimate, never negative.
    """
    total = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        # Two independent checks, as in the SDK: an item carrying both keys counts both.
        if isinstance(item.get("text"), str):
            total += math.ceil(len(item["text"]) / _CHARS_PER_TEXT_TOKEN)
        if "json" in item:
            try:
                total += math.ceil(len(json.dumps(item["json"])) / _CHARS_PER_JSON_TOKEN)
            except (TypeError, ValueError):
                pass
    return total
