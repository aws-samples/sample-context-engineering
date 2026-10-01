"""RelevanceFilterEngine conformance + behaviour tests (Tasks 6.1, 6.2)."""

from __future__ import annotations

import json

import pytest

from hermes_relevance_filter import RelevanceFilterEngine
from hermes_relevance_filter._compat import ContextEngine


class _FakeReranker:
    """Deterministic scorer: a chunk scores 1.0 if it shares a word with the query, else 0.0."""

    max_sources_per_query = 100

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        terms = set(query.lower().split())
        return [1.0 if terms & set(chunk.lower().split()) else 0.0 for chunk in chunks]


def _engine(**kw) -> RelevanceFilterEngine:
    cfg = {"reranker": _FakeReranker(), "chunk_tokens": 100, "preview_tokens": 100, "relevance_threshold": 0.5}
    cfg.update(kw.pop("config", {}))
    return RelevanceFilterEngine(max_result_tokens=50, config=cfg, **kw)


# ---- conformance (6.1) ----

def test_is_context_engine_subclass() -> None:
    assert isinstance(_engine(), ContextEngine)
    assert RelevanceFilterEngine().name == "relevance-filter"


def test_abstract_members_present() -> None:
    e = RelevanceFilterEngine()
    assert callable(e.update_from_response)
    assert callable(e.should_compress)
    assert callable(e.compress)


def test_tool_schemas_well_formed() -> None:
    schemas = _engine().get_tool_schemas()
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "retrieve_all_context"
    assert "reference" in fn["parameters"]["required"]


def test_handle_tool_call_returns_valid_json_for_unknown() -> None:
    out = _engine().handle_tool_call("nope", {})
    assert json.loads(out)["error"]


def test_select_context_without_rewrites_is_noop() -> None:
    e = RelevanceFilterEngine(config={"reranker": _FakeReranker()})
    msgs = [{"role": "user", "content": "hi"}]
    assert e.select_context(msgs) is None


def test_ctor_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError):
        RelevanceFilterEngine(max_result_tokens=0)


# ---- behaviour (6.2) ----

def _oversized_turn(e: RelevanceFilterEngine, body: str):
    msgs = [
        {"role": "user", "content": "find the error lines"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": body},
    ]
    e.on_turn_complete(msgs)
    return msgs


def test_oversized_result_rewritten_in_select_context() -> None:
    e = _engine()
    body = "error on line one\n" + "\n".join(f"ok line {i}" for i in range(400))
    msgs = _oversized_turn(e, body)
    selected = e.select_context(msgs)
    assert selected is not None
    tool_msg = [m for m in selected if m.get("role") == "tool"][0]
    assert "[Relevance:" in tool_msg["content"]
    assert "[ref:" in tool_msg["content"]
    # verbatim: the matching line survives into the excerpt
    assert "error on line one" in tool_msg["content"]


def test_small_result_not_filtered() -> None:
    e = _engine()
    msgs = _oversized_turn(e, "tiny")
    assert e.select_context(msgs) is None  # under the gate -> no rewrite recorded


def test_full_text_recoverable_via_retrieve_all_context() -> None:
    e = _engine()
    body = "\n".join(f"row {i} value {i*10}" for i in range(400))
    _oversized_turn(e, body)
    ref = next(iter(e._rankings))  # a reference was stored
    out = json.loads(e.handle_tool_call("retrieve_all_context", {"reference": ref}))
    assert "row 399 value 3990" in out["content"]


def test_retrieve_pattern_filters_rows() -> None:
    e = _engine()
    body = "\n".join([f"KEEP {i}" if i % 50 == 0 else f"skip {i}" for i in range(400)])
    _oversized_turn(e, body)
    ref = next(iter(e._rankings))
    out = json.loads(e.handle_tool_call(
        "retrieve_all_context", {"reference": ref, "pattern": "KEEP", "context_lines": 0}))
    assert "KEEP 0" in out["content"] and "KEEP 200" in out["content"]
    assert "skip 1" not in out["content"]


def test_retrieve_unknown_reference_errors() -> None:
    out = json.loads(_engine().handle_tool_call("retrieve_all_context", {"reference": "nope"}))
    assert "not found" in out["error"]


def test_gate_flips_at_documented_size() -> None:
    # max_result_tokens=50 => ceil(chars/4) > 50 => > 200 chars triggers filtering
    e = _engine()
    small = _oversized_turn(e, "a" * 200)  # 200/4 = 50, not > 50
    assert e.select_context(small) is None
    e2 = _engine()
    big = _oversized_turn(e2, "a" * 204)  # 204/4 = 51 > 50
    assert e2.select_context(big) is not None


def test_closed_retrieval_exchange_dropped() -> None:
    e = _engine()
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "r1", "type": "function", "function": {"name": "retrieve_all_context", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "r1", "content": "big recovered content"},
        {"role": "assistant", "content": "the answer is 42"},
    ]
    selected = e.select_context(msgs)
    assert selected is not None
    assert all(m.get("tool_call_id") != "r1" for m in selected)
    assert any(m.get("content") == "the answer is 42" for m in selected)
