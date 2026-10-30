"""Offline test for the repetition summary table, with synthetic payload-free artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "summarize_campaigns", Path(__file__).resolve().parents[1] / "summarize_campaigns.py"
)
assert _SPEC and _SPEC.loader
sc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sc)


def _rep(root: Path, name: str, *, base_mi=100000.0, a_mi=120000.0, a_ok=29, reasons=("GATE_MODEL_INPUT_INSUFFICIENT",),
         outcomes=None, plugin_options=None) -> None:
    out = root / name
    out.mkdir(parents=True)
    (out / "gate-a-decision.json").write_text(json.dumps({"outcome": "no_go" if reasons else "go", "reasons": list(reasons)}))
    for arm, ok, mi in (("baseline", 30, base_mi), ("A", a_ok, a_mi)):
        host = {"previewed_count": 2, "retrieval_attempts": 2}
        if arm == "A" and plugin_options:
            host["plugin_options"] = plugin_options
        (out / f"{arm}-evidence.json").write_text(json.dumps({
            "correctness_numerator": ok, "correctness_denominator": 30, "model_input_tokens_median": mi, "host_evidence": host,
        }))
    if outcomes:
        (out / "preview-outcomes.json").write_text(json.dumps({"total": outcomes}))


def test_one_row_per_complete_repetition_with_the_change_and_the_failing_rules(tmp_path: Path) -> None:
    _rep(tmp_path, "ce60-v4-sonnet55-r1", outcomes={"full_reads": 2, "partial_reads": 0, "unread": 0})
    _rep(tmp_path, "ce60-v5-sonnet55-r1", a_mi=80000.0, a_ok=30, reasons=(), plugin_options="preview_style:scoped,retrieval_cleanup:on",
         outcomes={"full_reads": 0, "partial_reads": 2, "unread": 0})
    (tmp_path / "ce60-v4-opus55-r1-ABORTED").mkdir()
    (tmp_path / "ce60-v4-opus55-r1").mkdir()  # incomplete: no decision, so no row
    table = sc.render(sc.collect(tmp_path))
    assert table.count("\n") == 3  # header, separator, two rows
    v4 = next(line for line in table.splitlines() if "| v4 |" in line)
    assert "100,000 -> 120,000" in v4 and "+20.0%" in v4 and "no_go" in v4 and "input" in v4
    assert "2 / 0 / 0" in v4 and "legacy" in v4
    v5 = next(line for line in table.splitlines() if "| v5 |" in line)
    assert "-20.0%" in v5 and "| go |" in v5 and "0 / 2 / 0" in v5 and "preview_style:scoped" in v5


def test_a_repetition_without_a_preview_outcomes_file_still_gets_a_row(tmp_path: Path) -> None:
    _rep(tmp_path, "ce60-v3-opus55-r2", reasons=("GATE_CORRECTNESS_REGRESSION", "GATE_MODEL_INPUT_INSUFFICIENT"))
    line = sc.render(sc.collect(tmp_path)).splitlines()[-1]
    assert "correctness, input" in line and "| - |" in line


def test_main_reports_when_there_is_nothing_to_summarize(tmp_path: Path, capsys) -> None:
    assert sc.main(["summarize_campaigns.py", str(tmp_path)]) == 1
    assert "no complete repetitions" in capsys.readouterr().err


def test_v2_repetitions_are_marked_void_because_their_plugin_was_inert(tmp_path: Path) -> None:
    _rep(tmp_path, "ce60-v2-sonnet55-r1")
    (tmp_path / "ce60-v2-sonnet55-r1" / "INVALID-ARM-A-PREFIX-WRITEBACK.txt").write_text("void")
    assert "inert (write-back bug, void)" in sc.render(sc.collect(tmp_path))
