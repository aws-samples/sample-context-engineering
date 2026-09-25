"""The harness checkpointer must restore the graph's state, not block it.

A blocked type is dropped on restore with only a log warning, so the graph arm would silently start
every turn from an empty graph. This is the regression the first live smoke run surfaced.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_core.graph import GraphState  # noqa: E402
from context_core.graph.state import TurnChoice  # noqa: E402
from src.runner import _checkpointer  # noqa: E402


def test_graph_state_survives_a_checkpoint_round_trip(caplog):
    serde = _checkpointer().serde
    state = GraphState()
    state.turn = 7
    # The middleware persists a flattened choice (a plain dict, see _persistable): mirror that.
    state.choice = TurnChoice({}, True)

    with caplog.at_level(logging.WARNING):
        restored = serde.loads_typed(serde.dumps_typed({"context_graph": state}))

    assert not [r for r in caplog.records if "Blocked deserialization" in r.getMessage()]
    assert isinstance(restored["context_graph"], GraphState)
    assert restored["context_graph"].turn == 7
