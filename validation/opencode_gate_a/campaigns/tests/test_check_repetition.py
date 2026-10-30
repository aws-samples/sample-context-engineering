"""Tests for the campaign acceptance checker, using synthetic payload-free artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_repetition", Path(__file__).resolve().parents[1] / "check_repetition.py"
)
assert _SPEC and _SPEC.loader
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)


@pytest.fixture(autouse=True)
def _default_decision_file(monkeypatch) -> None:
    """The checker reads the decision file named by CE60_DECISION_FILE; a campaign exports it, a test must not inherit it."""
    monkeypatch.delenv("CE60_DECISION_FILE", raising=False)



def _write(out: Path, *, outcome="no_go", a_completed=60, a_timeouts=0, a_codes=None, a_retry=None,
           b_completed=60, residual=0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate-a-decision.json").write_text(json.dumps({"outcome": outcome, "reasons": []}))
    for arm, completed, codes, retry, timeouts in (
        ("baseline", b_completed, [], [], 0),
        ("A", a_completed, a_codes or [], a_retry or [], a_timeouts),
    ):
        (out / f"{arm}-evidence.json").write_text(json.dumps({
            "turns_completed": completed, "correctness_numerator": 29, "correctness_denominator": 30,
            "model_input_tokens_median": 1000.0,
            "host_evidence": {"transform_count": 5, "previewed_count": 1, "input_bytes": 100, "output_bytes": 50,
                              "observability": {"timeout_count": timeouts, "error_codes": codes,
                                                "error_retryable_flags": retry}},
        }))
        (out / f"{arm}-cleanup.json").write_text(json.dumps({"residual_process_count": residual}))


def test_clean_repetition_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path)
    assert cr.check(tmp_path)[0] == cr.CLEAN


def test_not_evaluable_gate_is_a_hard_failure(tmp_path: Path) -> None:
    _write(tmp_path, outcome="not_evaluable")
    assert cr.check(tmp_path)[0] == cr.HARD


def test_a_timeout_is_a_hard_failure(tmp_path: Path) -> None:
    _write(tmp_path, a_timeouts=1)
    assert cr.check(tmp_path)[0] == cr.HARD


def test_residual_processes_are_a_hard_failure(tmp_path: Path) -> None:
    _write(tmp_path, residual=1)
    assert cr.check(tmp_path)[0] == cr.HARD


def test_only_a_retryable_provider_error_may_be_repeated(tmp_path: Path) -> None:
    _write(tmp_path, a_completed=59, a_codes=["APIError"], a_retry=["retryable"])
    assert cr.check(tmp_path)[0] == cr.RETRYABLE_ONLY


def test_a_non_retryable_or_unknown_error_is_never_repeated(tmp_path: Path) -> None:
    _write(tmp_path, a_completed=59, a_codes=["APIError"], a_retry=["not_retryable"])
    assert cr.check(tmp_path)[0] == cr.HARD
    _write(tmp_path, a_completed=59, a_codes=["APIError"], a_retry=["unknown"])
    assert cr.check(tmp_path)[0] == cr.HARD
    _write(tmp_path, a_completed=59, a_codes=["PROMPT_FAILED"], a_retry=["retryable"])
    assert cr.check(tmp_path)[0] == cr.HARD


def test_a_missing_error_record_is_never_assumed_retryable(tmp_path: Path) -> None:
    # An incomplete run with no recorded error class cannot be proven transient: fail closed.
    _write(tmp_path, a_completed=59)
    assert cr.check(tmp_path)[0] == cr.HARD


def test_unreadable_artifacts_fail_closed_without_a_traceback(tmp_path: Path, capsys) -> None:
    assert cr.main(["check_repetition.py", str(tmp_path / "missing")]) == cr.HARD
    assert "hard failure" in capsys.readouterr().out


def _write_v4(out: Path, *, oracle=("v4", "v4"), bridge_attempts=2, turn_attempts=(1, 1), unavailable=0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for arm, version in zip(("baseline", "A"), oracle):
        (out / f"{arm}-evidence.json").write_text(json.dumps({
            "host_evidence": {"oracle_version": version, "retrieval_attempts": bridge_attempts if arm == "A" else None}
        }))
    turns = [{"turn_id": f"t{i}", "retrieval_attempts": n} for i, n in enumerate(turn_attempts)]
    turns += [{"turn_id": f"u{i}", "retrieval_attempts": None} for i in range(unavailable)]
    (out / "A-sanitized.json").write_text(json.dumps({"turn_results": turns}))


def test_v4_accepts_the_v4_oracle_and_reports_consistent_retrieval(tmp_path: Path) -> None:
    _write_v4(tmp_path)
    ok, text = cr.v4_conditions(tmp_path)
    assert ok and "(consistent)" in text and "bridge=2 evaluated=2" in text


def test_v4_rejects_a_run_scored_by_another_oracle(tmp_path: Path) -> None:
    _write_v4(tmp_path, oracle=("v3", "v4"))
    ok, text = cr.v4_conditions(tmp_path)
    assert not ok and "baseline scored by oracle 'v3'" in text


def test_v4_reports_a_count_mismatch_but_fails_a_missing_count(tmp_path: Path) -> None:
    _write_v4(tmp_path, bridge_attempts=3, turn_attempts=(1, 1))
    ok, text = cr.v4_conditions(tmp_path)
    assert ok and "MISMATCH" in text
    # A missing per-turn count is different: the session read failed, so the repetition is not valid.
    _write_v4(tmp_path, turn_attempts=(1, 1), unavailable=1)
    ok, text = cr.v4_conditions(tmp_path)
    assert not ok and "unavailable_turns=1" in text and "have no retrieval count" in text


def test_v4_subcommand_exit_codes_and_unreadable_artifacts(tmp_path: Path, capsys) -> None:
    _write_v4(tmp_path)
    assert cr.main(["check_repetition.py", "v4", str(tmp_path)]) == cr.CLEAN
    _write_v4(tmp_path, oracle=("v3", "v3"))
    assert cr.main(["check_repetition.py", "v4", str(tmp_path)]) == cr.HARD
    assert cr.main(["check_repetition.py", "v4", str(tmp_path / "missing")]) == cr.HARD
    assert "unreadable" in capsys.readouterr().out


import sqlite3


def _store(tmp_path: Path, run_id: str, answers: dict[str, list[str]]) -> Path:
    """A minimal OpenCode store: one session per arm, one user message + one assistant answer per turn."""
    store = tmp_path / "store"
    store.mkdir(parents=True)
    con = sqlite3.connect(store / "opencode.db")
    con.execute("create table session (id text, title text, time_created integer)")
    con.execute("create table message (id text, session_id text, time_created integer, data text)")
    con.execute("create table part (id text, message_id text, session_id text, time_created integer, data text)")
    clock = 0
    for arm, texts in answers.items():
        sid = f"ses_{arm}"
        con.execute("insert into session values (?,?,?)", (sid, f"ce60-{arm}-{run_id}-{arm}", clock))
        for index, text in enumerate(texts):
            clock += 1
            con.execute("insert into message values (?,?,?,?)", (f"u{arm}{index}", sid, clock, json.dumps({"role": "user"})))
            clock += 1
            con.execute("insert into message values (?,?,?,?)", (f"a{arm}{index}", sid, clock, json.dumps({"role": "assistant"})))
            con.execute(
                "insert into part values (?,?,?,?,?)",
                (f"p{arm}{index}", f"a{arm}{index}", sid, clock, json.dumps({"type": "text", "text": text})),
            )
    con.commit()
    con.close()
    return store


PT = "Sim, o conector está com problema e não foi possível sincronizar, mas você pode tentar de novo."
EN = "Yes, the connector is degraded and the last sync failed; you can try again."


def test_language_drift_counts_portuguese_answers_per_arm(tmp_path: Path) -> None:
    store = _store(tmp_path, "r", {"baseline": [EN, EN, PT], "A": [PT, PT, PT, EN]})
    assert cr.language_drift("r", store=store) == {"baseline": 1, "A": 3}
    assert cr.language_drift("missing", store=store) == {"baseline": None, "A": None}


def test_v4_fails_a_repetition_whose_arm_drifted_into_portuguese(tmp_path: Path) -> None:
    _write_v4(tmp_path / "out")
    drifted = _store(tmp_path / "d", "r", {"baseline": [EN] * 6, "A": [PT] * 6})
    ok, text = cr.v4_conditions(tmp_path / "out", "r", store=drifted)
    assert not ok and "A answered in Portuguese in 6 turns (pin failed)" in text

    clean = _store(tmp_path / "c", "r", {"baseline": [EN, PT, EN, EN], "A": [EN] * 4})
    ok, text = cr.v4_conditions(tmp_path / "out", "r", store=clean)
    assert ok and "language_pt_answers={'baseline': 1, 'A': 0}" in text


def test_v4_fails_when_a_session_is_missing_for_the_language_check(tmp_path: Path) -> None:
    _write_v4(tmp_path / "out")
    store = _store(tmp_path / "s", "r", {"baseline": [EN]})
    ok, text = cr.v4_conditions(tmp_path / "out", "r", store=store)
    assert not ok and "A session not found for the language check" in text


def test_v4_can_require_the_plugin_options_arm_a_ran_with(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_v4(out)
    evidence = json.loads((out / "A-evidence.json").read_text())
    evidence["host_evidence"]["plugin_options"] = "preview_style:scoped,retrieval_cleanup:on"
    (out / "A-evidence.json").write_text(json.dumps(evidence))
    ok, text = cr.v4_conditions(out, expect_plugin_options="preview_style:scoped,retrieval_cleanup:on")
    assert ok and "plugin_options=preview_style:scoped,retrieval_cleanup:on" in text
    ok, text = cr.v4_conditions(out, expect_plugin_options="preview_style:scoped,retrieval_cleanup:off")
    assert not ok and "expected 'preview_style:scoped,retrieval_cleanup:off'" in text


def test_v4_requires_plugin_options_to_be_recorded_when_they_are_expected(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_v4(out)  # no plugin_options in the evidence (a v3/v4 plugin run)
    ok, text = cr.v4_conditions(out, expect_plugin_options="preview_style:scoped,retrieval_cleanup:on")
    assert not ok and "ran with plugin options None" in text


# --- practice D (A+D repetitions) ----------------------------------------------------------------


def _graph(out: Path, **overrides) -> None:
    out.mkdir(parents=True, exist_ok=True)
    body = {
        "requests": 100, "checked": 100, "stored_prefix_equal": 100, "mismatched_requests": [],
        "stored_graph_synthetic_messages": 0, "operations": {"projected": 70, "unchanged": 30},
        "removed_messages_total": 900, "block_chars_total": 50000, "rejected_requests": 0,
        "graph_tool_calls": {"expand_card": 3},
    }
    body.update(overrides)
    (out / "A-graph-durability.json").write_text(json.dumps(body))


def test_the_decision_file_name_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path)
    (tmp_path / "gate-a-decision.json").rename(tmp_path / "d-slice-comparison.json")
    assert cr.main(["check_repetition.py", str(tmp_path)]) == cr.HARD  # the default name no longer exists
    monkeypatch.setenv("CE60_DECISION_FILE", "d-slice-comparison.json")
    assert cr.check(tmp_path)[0] == cr.CLEAN


def test_a_clean_graph_run_is_valid_d_evidence(tmp_path: Path) -> None:
    _graph(tmp_path)
    ok, text = cr.graph_conditions(tmp_path)
    assert ok, text
    assert "projected=70" in text and "stored_prefix_equal=100/100" in text


def test_a_graph_that_never_engaged_measured_nothing(tmp_path: Path) -> None:
    _graph(tmp_path, operations={"unchanged": 100})
    ok, text = cr.graph_conditions(tmp_path)
    assert not ok and "never engaged" in text


def test_a_stored_session_that_differs_from_what_requests_were_handed_fails(tmp_path: Path) -> None:
    _graph(tmp_path, stored_prefix_equal=99, mismatched_requests=[42])
    ok, text = cr.graph_conditions(tmp_path)
    assert not ok and "[42]" in text


def test_requests_that_could_not_be_checked_fail_closed(tmp_path: Path) -> None:
    _graph(tmp_path, checked=0, stored_prefix_equal=0)
    ok, text = cr.graph_conditions(tmp_path)
    assert not ok and "only 0 of 100" in text


def test_a_graph_message_in_the_stored_session_fails(tmp_path: Path) -> None:
    _graph(tmp_path, stored_graph_synthetic_messages=2)
    assert not cr.graph_conditions(tmp_path)[0]


def test_a_rejected_bridge_answer_fails(tmp_path: Path) -> None:
    _graph(tmp_path, rejected_requests=1)
    assert not cr.graph_conditions(tmp_path)[0]


def test_too_many_unavailable_requests_fail_but_a_few_are_tolerated(tmp_path: Path) -> None:
    _graph(tmp_path, operations={"projected": 70, "unchanged": 25, "bridge_unavailable": 5})
    assert cr.graph_conditions(tmp_path)[0]
    _graph(tmp_path, operations={"projected": 70, "unchanged": 20, "bridge_unavailable": 6, "error": 4})
    ok, text = cr.graph_conditions(tmp_path)
    assert not ok and "10 of 100" in text


def test_graph_subcommand_exit_codes_and_unreadable_artifacts(tmp_path: Path, capsys) -> None:
    _graph(tmp_path)
    assert cr.main(["check_repetition.py", "graph", str(tmp_path)]) == cr.CLEAN
    assert cr.main(["check_repetition.py", "graph", str(tmp_path / "missing")]) == cr.HARD
    assert "unreadable artifacts" in capsys.readouterr().out
