"""ProgressiveToolDisclosureEngine tests (Task 8.1)."""

from __future__ import annotations

import json

from hermes_progressive_tool_disclosure import ProgressiveToolDisclosureEngine
from hermes_progressive_tool_disclosure._compat import ContextEngine


def _tool(name: str, desc: str, params: dict | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params or {"type": "object", "properties": {}}}}


_TOOLS = [
    _tool("query_ledger", "Search the financial ledger for transactions and refunds."),
    _tool("send_email", "Send an email message to a recipient."),
    _tool("read_file", "Read the contents of a file from disk."),
]


def _engine(**kw) -> ProgressiveToolDisclosureEngine:
    return ProgressiveToolDisclosureEngine(tool_specs=_TOOLS, **kw)


# ---- conformance ----

def test_is_context_engine() -> None:
    assert isinstance(_engine(), ContextEngine)
    assert _engine().name == "progressive-tool-disclosure"


def test_tool_schemas() -> None:
    names = [s["function"]["name"] for s in _engine().get_tool_schemas()]
    assert names == ["find_tools", "get_tool_details"]


def test_handle_unknown_tool_json() -> None:
    assert json.loads(_engine().handle_tool_call("nope", {}))["error"]


# ---- catalog injection ----

def test_catalog_injected_into_system_message() -> None:
    e = _engine()
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    selected = e.select_context(msgs)
    assert selected is not None
    sys_msg = selected[0]
    # catalog block lists every tool as "- name: summary"
    rendered = sys_msg["content"] if isinstance(sys_msg["content"], str) else json.dumps(sys_msg["content"])
    assert "query_ledger" in rendered and "send_email" in rendered and "read_file" in rendered


def test_catalog_prepended_when_no_system_message() -> None:
    e = _engine()
    selected = e.select_context([{"role": "user", "content": "hi"}])
    assert selected is not None
    assert selected[0]["role"] == "system"


def test_catalog_suppressed_when_chars_none() -> None:
    e = _engine(catalog_chars=None)
    selected = e.select_context([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    # no catalog, no fold needed -> unchanged
    assert selected is None


def test_no_specs_is_noop() -> None:
    e = ProgressiveToolDisclosureEngine(tool_specs=[])
    assert e.select_context([{"role": "user", "content": "hi"}]) is None


# ---- find_tools / get_tool_details ----

def test_find_tools_resolves_against_seeded_set() -> None:
    out = json.loads(_engine().handle_tool_call("find_tools", {"need": "search ledger for refunds"}))
    names = [m["name"] for m in out["matches"]]
    assert "query_ledger" in names


def test_find_tools_requires_need() -> None:
    assert json.loads(_engine().handle_tool_call("find_tools", {"need": ""}))["error"]


def test_get_tool_details_returns_full_spec() -> None:
    out = json.loads(_engine().handle_tool_call("get_tool_details", {"names": ["send_email"]}))
    assert out["tools"]["send_email"]["description"].startswith("Send an email")
    assert "inputSchema" in out["tools"]["send_email"]


def test_get_tool_details_unknown_name() -> None:
    out = json.loads(_engine().handle_tool_call("get_tool_details", {"names": ["ghost"]}))
    assert out["tools"]["ghost"]["error"] == "unknown tool"


# ---- fold ----

def test_fold_removes_stale_detail_exchange() -> None:
    e = _engine()
    # expand a tool so it is active, then build a closed get_tool_details exchange
    e.handle_tool_call("get_tool_details", {"names": ["read_file"]})
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "d1", "type": "function", "function": {"name": "get_tool_details", "arguments": '{"names":["read_file"]}'}}]},
        {"role": "tool", "tool_call_id": "d1", "content": '{"tools": {"read_file": {}}}'},
        {"role": "assistant", "content": "ok let me continue"},
        {"role": "user", "content": "next"},
    ]
    selected = e.select_context(msgs)
    assert selected is not None
    # the get_tool_details call shape should be folded away (no tool_call with name get_tool_details remains active)
    serialized = json.dumps(selected)
    assert '"d1"' not in serialized
