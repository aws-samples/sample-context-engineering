"""Offline tests for the preview-outcome analysis, with a synthetic store and metrics file."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "preview_outcomes", Path(__file__).resolve().parents[1] / "preview_outcomes.py"
)
assert _SPEC and _SPEC.loader
po = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(po)

MARK = po.PREVIEW_MARKER


def _store(tmp_path: Path, run_id: str, turns: list[list[tuple[str, int, bool]]]) -> Path:
    """One arm-A session. ``turns`` is, per user turn, the assistant tool results as (tool, bytes, is_preview)."""
    store = tmp_path / "store"
    store.mkdir(parents=True)
    con = sqlite3.connect(store / "opencode.db")
    con.execute("create table session (id text, title text, time_created integer)")
    con.execute("create table message (id text, session_id text, time_created integer, data text)")
    con.execute("create table part (id text, message_id text, session_id text, time_created integer, data text)")
    sid, clock = "ses_A", 0
    con.execute("insert into session values (?,?,?)", (sid, f"ce60-A-{run_id}-A", clock))
    for index, results in enumerate(turns):
        clock += 1
        con.execute("insert into message values (?,?,?,?)", (f"u{index}", sid, clock, json.dumps({"role": "user"})))
        clock += 1
        con.execute("insert into message values (?,?,?,?)", (f"a{index}", sid, clock, json.dumps({"role": "assistant"})))
        for j, (tool, size, is_preview) in enumerate(results):
            body = (MARK + "x" * (size - len(MARK))) if is_preview else "y" * size
            part = {"type": "tool", "tool": tool, "state": {"status": "completed", "output": body}}
            con.execute("insert into part values (?,?,?,?,?)", (f"p{index}{j}", f"a{index}", sid, clock, json.dumps(part)))
    con.commit()
    con.close()
    return store


def _metrics(run_dir: Path, previews: list[tuple[str, int, int]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"event": "transform", "operation": "skipped", "tool_class": "x", "input_bytes": 10, "output_bytes": 10})]
    for tool, original, shown in previews:
        lines.append(
            json.dumps(
                {"event": "transform", "operation": "previewed", "tool_class": tool, "input_bytes": original, "output_bytes": shown}
            )
        )
    (run_dir / "run-A-A-metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_preview_the_model_re_expands_is_not_a_saving(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        "r",
        [[("ce60-fixture_list_accounts", 500, False)], [("ce60-fixture_list_tx", 4138, True), ("retrieve_context", 33677, False)]],
    )
    _metrics(tmp_path / "out", [("list_tx", 33677, 3948)])
    report = po.analyze("r", tmp_path / "out", store=store)
    item = report["previews"][0]
    assert item["re_expanded"] is True and item["read"] == "full" and item["retrieved_bytes"] == 33677
    assert item["preview_bytes"] == 4138 and item["turn_ordinal"] == 2
    # Without the cleanup (the v3/v4 plugin) the retrieval stays in the history: nothing is saved.
    assert item["in_turn_net_bytes"] == 4138 and item["steady_state_net_bytes"] == 4138
    assert report["total"]["full_reads"] == 1 and report["total"]["steady_state_net_bytes"] == 4138
    assert "no net saving" in report["summary"] and "cleanup=off" in report["summary"]


def test_a_preview_the_model_never_expands_is_a_saving(tmp_path: Path) -> None:
    store = _store(tmp_path, "r", [[("ce60-fixture_read_docs", 4134, True)]])
    _metrics(tmp_path / "out", [("read_docs", 39478, 3944)])
    report = po.analyze("r", tmp_path / "out", store=store)
    assert report["previews"][0]["re_expanded"] is False and report["previews"][0]["read"] == "none"
    assert report["total"]["steady_state_net_bytes"] == 4134 - 39478
    assert "net saving" in report["summary"] and "no net saving" not in report["summary"]


def test_two_previews_are_matched_to_their_own_retrievals_by_size(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        "r",
        [
            [("ce60-fixture_a", 4138, True), ("retrieve_context", 33677, False)],
            [("ce60-fixture_b", 4134, True)],
            [("ce60-fixture_c", 4133, True), ("retrieve_context", 39477, False)],
        ],
    )
    _metrics(tmp_path / "out", [("a", 33677, 3948), ("b", 20000, 3944), ("c", 39477, 3943)])
    report = po.analyze("r", tmp_path / "out", store=store)
    assert [p["re_expanded"] for p in report["previews"]] == [True, False, True]
    assert report["unmatched_retrievals"] == 0
    assert report["total"]["re_expanded"] == 2 and report["total"]["unread"] == 1


def test_missing_inputs_fail_closed_with_a_message_not_a_traceback(tmp_path: Path, capsys) -> None:
    assert po.main(["preview_outcomes.py", "r", str(tmp_path / "nope")]) == 1
    assert "unavailable" in capsys.readouterr().err
    assert po.main(["preview_outcomes.py"]) == 2


def test_cli_prints_a_one_line_summary(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path, "r", [[("ce60-fixture_a", 4138, True)]])
    _metrics(tmp_path / "out", [("a", 10000, 3948)])
    monkeypatch.setattr(po.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home" / ".local" / "share").mkdir(parents=True)
    (tmp_path / "home" / ".local" / "share" / "opencode").symlink_to(store)
    assert po.main(["preview_outcomes.py", "r", str(tmp_path / "out"), "--summary"]) == 0
    assert capsys.readouterr().out.startswith("previews=1 full_reads=0")


def _evidence(run_dir: Path, plugin_options: str) -> None:
    (run_dir / "A-evidence.json").write_text(
        json.dumps({"host_evidence": {"plugin_options": plugin_options}}), encoding="utf-8"
    )


def test_with_the_cleanup_on_a_full_read_costs_only_the_turn_that_made_it(tmp_path: Path) -> None:
    store = _store(
        tmp_path, "r", [[("ce60-fixture_a", 4138, True), ("retrieve_context", 33677, False)]]
    )
    _metrics(tmp_path / "out", [("a", 33677, 3948)])
    _evidence(tmp_path / "out", "preview_style:scoped,retrieval_cleanup:on")
    report = po.analyze("r", tmp_path / "out", store=store)
    item = report["previews"][0]
    assert item["read"] == "full"
    assert item["in_turn_net_bytes"] == 4138
    assert item["steady_state_net_bytes"] == 4138 - 33677
    assert "cleanup=on" in report["summary"] and "no net saving" not in report["summary"]


def test_a_narrow_read_is_partial_and_costs_far_less_than_the_original(tmp_path: Path) -> None:
    store = _store(
        tmp_path, "r", [[("ce60-fixture_a", 4138, True), ("retrieve_context", 2100, False)]]
    )
    _metrics(tmp_path / "out", [("a", 33677, 3948)])
    report = po.analyze("r", tmp_path / "out", store=store)
    item = report["previews"][0]
    assert item["read"] == "partial" and item["re_expanded"] is False
    assert item["in_turn_net_bytes"] == 4138 + 2100 - 33677
    assert report["total"]["partial_reads"] == 1


def test_a_retrieval_in_a_later_turn_still_belongs_to_the_preview_before_it(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        "r",
        [[("ce60-fixture_a", 4138, True)], [("ce60-fixture_x", 100, False)], [("retrieve_context", 33677, False)]],
    )
    _metrics(tmp_path / "out", [("a", 33677, 3948)])
    assert po.analyze("r", tmp_path / "out", store=store)["previews"][0]["read"] == "full"
