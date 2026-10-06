"""Test seam for the delivery tests: a ``project_state`` that applies a hand-built state's frozen choice.

Under D-1 the delivery calls ``project()``, which recomputes the Turn Choice on every model call. The delivery tests
are about the binding's own contract around that call -- identity on a full pass and on an empty removal, the
``dynamic_trailing_blocks`` it adds, one warning on failure, the state written back -- so they hand ``_Delivery`` this
callable instead: it runs only the core ``deliver`` over the state the test built, choice included. How the choice
itself is computed per call is ``project()``'s contract, tested in ``context-core``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from context_core.graph.projection import Thresholds
from context_core.graph.projection import deliver as core_deliver
from context_core.graph.state import _GraphState

RETRIEVAL_TOOLS = ("expand_card", "expand_artifact", "find_context")


def apply_frozen_choice(
    description_tokens: int,
) -> Callable[[_GraphState | None, Sequence[Any]], tuple[Sequence[Any], _GraphState]]:
    """Return a ``project_state`` that delivers the given state's own choice and returns that state unchanged."""

    def project_state(prior: _GraphState | None, messages: Sequence[Any]) -> tuple[Sequence[Any], _GraphState]:
        state = prior if prior is not None else _GraphState()
        thresholds = Thresholds(description_tokens=description_tokens, retrieval_tools=RETRIEVAL_TOOLS)
        projected, _trailing = core_deliver(state, messages, thresholds)
        return projected, state

    return project_state
