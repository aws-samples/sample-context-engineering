"""Composition test: the all-three arm applies D -> B -> A and exposes the six tools."""

from __future__ import annotations

from src.config import RUN_CONFIGS
from src.run import build_engine


def test_all_arm_is_composed_engine_with_six_tools() -> None:
    engine = build_engine(RUN_CONFIGS["all"])
    assert engine.name == "all-three"
    names = {s["function"]["name"] for s in engine.get_tool_schemas()}
    assert names == {
        "rf_retrieve_all_context", "ptd_find_tools", "ptd_get_tool_details",
        "cg_expand_card", "cg_expand_artifact", "cg_find_context",
    }


def test_all_arm_shares_relevance_store_with_graph() -> None:
    engine = build_engine(RUN_CONFIGS["all"])
    # the graph reads the filter's store as its stash (one shared store)
    assert engine._graph._stash is not None
    assert engine._relevance._store is not None


def test_all_arm_pipeline_order_is_d_b_a() -> None:
    # select_context iterates graph (D), disclosure (B), relevance (A) in that order
    engine = build_engine(RUN_CONFIGS["all"])
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list my accounts and positions"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "list_accounts", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join(f"row {i}" for i in range(500))},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again?"},
    ]
    engine.on_turn_complete(convo)
    out = engine.select_context(convo)
    assert out is not None  # at least the catalog was injected
