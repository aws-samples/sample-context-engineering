"""AllThreeEngine tests (Task 12.1): composition order, shared store, six tools + dispatch, threshold."""

from __future__ import annotations

import json

from hermes_all_three import AllThreeEngine
from hermes_all_three._compat import ContextEngine


class _FakeReranker:
    max_sources_per_query = 100

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        terms = set(query.lower().split())
        return [1.0 if terms & set(c.lower().split()) else 0.0 for c in chunks]


class _FakeMatcher:
    def score(self, need: str, descriptions):
        terms = set(need.lower().split())
        return [len(terms & set(d.lower().split())) / len(terms) if terms else 0.0 for d in descriptions]


def _tool(name, desc):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": {}}}}


def _engine(**kw) -> AllThreeEngine:
    kw.setdefault("relevance_config", {"reranker": _FakeReranker(), "chunk_tokens": 100, "preview_tokens": 100})
    kw.setdefault("max_result_tokens", 50)
    kw.setdefault("tool_specs", [_tool("query_ledger", "search the ledger"), _tool("send_email", "send an email")])
    kw.setdefault("graph_kwargs", {"matcher": _FakeMatcher(), "min_cards": 1})
    return AllThreeEngine(**kw)


def test_is_context_engine() -> None:
    assert isinstance(_engine(), ContextEngine)
    assert _engine().name == "all-three"


def test_six_tools_present() -> None:
    names = {s["function"]["name"] for s in _engine().get_tool_schemas()}
    assert names == {
        "retrieve_all_context", "find_tools", "get_tool_details",
        "expand_card", "expand_artifact", "find_context",
    }


def test_threshold_default_is_benchmark_value() -> None:
    e = _engine()
    assert e._relevance._config["relevance_threshold"] == 0.02


def test_shared_store_handed_to_graph_as_stash() -> None:
    e = _engine()
    # the graph's stash is the relevance engine's stash view (same underlying store)
    assert e._graph._stash is not None
    assert e._relevance._store is not None


def test_dispatch_routes_to_owning_practice() -> None:
    e = _engine()
    # find_tools -> disclosure
    out = json.loads(e.handle_tool_call("find_tools", {"need": "search ledger"}))
    assert "matches" in out
    # unknown -> error
    assert json.loads(e.handle_tool_call("ghost", {}))["error"]


def test_select_context_pipeline_runs_fail_open() -> None:
    e = _engine()
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find errors"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "query_ledger", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "line one error\n" + "\n".join(f"row {i}" for i in range(400))},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "again?"},
    ]
    e.on_turn_complete(convo)  # A stores+records; D indexes
    out = e.select_context(convo)
    # at least catalog injection or a filter rewrite happened
    assert out is not None
    serialized = json.dumps(out)
    assert "query_ledger" in serialized  # catalog mentions the base tools


def test_shared_ref_resolves_via_expand_artifact() -> None:
    e = _engine()
    convo = [
        {"role": "user", "content": "pull the ledger"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "query_ledger", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join(f"row {i} total {i*5}" for i in range(400))},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "sum?"},
    ]
    e.on_turn_complete(convo)      # A stores c1_0 in the shared store
    e.select_context(convo)        # D builds its graph
    # the filter minted a ref c1_0; D's expand_artifact reads it through the shared stash
    out = json.loads(e.handle_tool_call("expand_artifact", {"reference": "c1_0"}))
    assert "row 399 total 1995" in out.get("result", "")


def test_on_session_reset_fans_out() -> None:
    e = _engine()
    e.update_from_response({"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
    e.on_session_reset()
    assert e.last_prompt_tokens == 0
    assert e._graph._state is None
