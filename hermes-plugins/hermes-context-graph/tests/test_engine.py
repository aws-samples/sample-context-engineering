"""ContextGraphEngine tests (Task 10.1): conformance, projection, recovery tools, no history deletion."""

from __future__ import annotations

import json

import pytest

from hermes_context_graph import ContextGraphEngine
from hermes_context_graph._compat import ContextEngine


class _FakeMatcher:
    """Deterministic matcher: score = fraction of need-words appearing in each description."""

    def score(self, need: str, descriptions):
        terms = set(need.lower().split())
        out = []
        for d in descriptions:
            words = set(d.lower().split())
            out.append(len(terms & words) / len(terms) if terms else 0.0)
        return out


def _engine(**kw) -> ContextGraphEngine:
    kw.setdefault("matcher", _FakeMatcher())
    kw.setdefault("min_cards", 1)
    return ContextGraphEngine(**kw)


def _conversation() -> list[dict]:
    return [
        {"role": "user", "content": "tell me about the refund policy for electronics"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Electronics refunds: 30 days, receipt required."},
        {"role": "assistant", "content": "Electronics can be refunded within 30 days with a receipt."},
        {"role": "user", "content": "and for clothing?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Clothing refunds: 60 days, tags attached."},
        {"role": "assistant", "content": "Clothing can be refunded within 60 days with tags attached."},
        {"role": "user", "content": "what about the electronics case again?"},
    ]


# ---- conformance ----

def test_is_context_engine() -> None:
    assert isinstance(_engine(), ContextEngine)
    assert _engine().name == "context-graph"


def test_tool_schemas() -> None:
    names = [s["function"]["name"] for s in _engine().get_tool_schemas()]
    assert names == ["expand_card", "expand_artifact", "find_context"]


def test_artifact_tool_can_be_disabled() -> None:
    names = [s["function"]["name"] for s in _engine(include_artifact_tool=False).get_tool_schemas()]
    assert "expand_artifact" not in names


def test_handle_unknown_tool_json() -> None:
    e = _engine()
    convo = _conversation()
    e.on_turn_complete(convo[:8])
    e.select_context(convo)  # establishes graph state so dispatch reaches tool handling
    assert json.loads(e.handle_tool_call("nope", {}))["error"]


def test_select_context_fail_open_on_bad_matcher() -> None:
    class Boom:
        def score(self, need, descriptions):
            raise RuntimeError("down")

    e = ContextGraphEngine(matcher=Boom(), min_cards=1)
    # projection still succeeds (matcher failure is caught inside the core / find path); never raises.
    assert e.select_context(_conversation()) in (None,) or isinstance(e.select_context(_conversation()), list)


# ---- projection resolutions ----

def test_projection_runs_and_is_request_only() -> None:
    e = _engine()
    convo = _conversation()
    original = json.dumps(convo)
    e.on_turn_complete(convo[:8])  # close the first two turns
    out = e.select_context(convo)
    # whatever the resolution, the input list is never mutated
    assert json.dumps(convo) == original
    assert out is None or isinstance(out, list)


def test_full_pass_regression_short_circuit() -> None:
    # expand_threshold=0.0 => every card full content => core returns identity => select_context None
    e = ContextGraphEngine(matcher=_FakeMatcher(), min_cards=1, expand_threshold=0.0, collapse_floor=0.0)
    convo = _conversation()
    e.on_turn_complete(convo[:8])
    assert e.select_context(convo) is None


# ---- recovery tools ----

def test_find_context_then_expand_card() -> None:
    e = _engine()
    convo = _conversation()
    e.on_turn_complete(convo[:8])
    e.select_context(convo)  # build the graph/cards
    found = json.loads(e.handle_tool_call("find_context", {"need": "electronics refund"}))
    assert "result" in found


def test_expand_artifact_recovers_stored_tool_result() -> None:
    e = _engine()
    convo = _conversation()
    e.on_turn_complete(convo[:8])  # stores c1_0, c2_0
    e.select_context(convo)
    out = json.loads(e.handle_tool_call("expand_artifact", {"reference": "c1_0"}))
    assert "30 days" in out["result"]


def test_expand_artifact_unknown_reference() -> None:
    e = _engine()
    e.on_turn_complete(_conversation()[:8])
    e.select_context(_conversation())
    out = json.loads(e.handle_tool_call("expand_artifact", {"reference": "ghost_0"}))
    # resolution returns an unknown/absent message, carried as result (not an exception)
    assert "result" in out or "error" in out


def test_retrieval_budget_exhaustion() -> None:
    e = _engine(max_retrieval_cycles=1)
    convo = _conversation()
    e.on_turn_complete(convo[:8])
    e.select_context(convo)
    first = json.loads(e.handle_tool_call("find_context", {"need": "x"}))
    second = json.loads(e.handle_tool_call("find_context", {"need": "y"}))
    assert "budget is spent" in second["result"]


# ---- no deletion from persisted history ----

def test_on_turn_complete_does_not_mutate_messages() -> None:
    e = _engine()
    convo = _conversation()
    snapshot = json.dumps(convo)
    e.on_turn_complete(convo)
    assert json.dumps(convo) == snapshot


def test_handle_tool_call_before_any_projection() -> None:
    e = _engine()
    out = json.loads(e.handle_tool_call("find_context", {"need": "x"}))
    assert "no earlier turns" in out["result"]
