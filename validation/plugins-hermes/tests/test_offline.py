"""Offline verification (Task 19.1): all five arms build and run with no AWS credentials; re-render works."""

from __future__ import annotations

import json

from src import metrics, report, runner, scenario
from src.config import DEFAULT_CONFIGURATIONS, RUN_CONFIGS
from src.run import build_engine


def _turns():
    return scenario.turns(limit=8)


def test_build_every_arm_offline() -> None:
    for name in DEFAULT_CONFIGURATIONS:
        engine = build_engine(RUN_CONFIGS[name])
        if name == "baseline":
            assert engine is None
        else:
            assert engine is not None
            assert engine.get_tool_schemas() is not None


def test_all_five_arms_run_offline() -> None:
    metrics.set_run_tag("test-offline")
    turns = _turns()
    collectors = runner.run_all(list(DEFAULT_CONFIGURATIONS), turns, build_engine, RUN_CONFIGS, repeats=1)
    assert set(collectors) == set(DEFAULT_CONFIGURATIONS)
    for name, cs in collectors.items():
        collector = cs[0]
        assert len(collector.turns) == len(turns), name
        assert len(collector.calls) >= len(turns), name  # at least one model call per turn
        # no arm crashed every turn
        answered = [t for t in collector.turns if t.response_text]
        assert answered, f"{name} produced no answers"


def test_run_json_shape_and_rerender() -> None:
    metrics.set_run_tag("test-shape")
    turns = _turns()
    collectors = runner.run_all(list(DEFAULT_CONFIGURATIONS), turns, build_engine, RUN_CONFIGS, repeats=1)
    results = {name: metrics.aggregate(cs) for name, cs in collectors.items()}
    results["meta"] = {"framework": "hermes", "mode": "offline"}
    # the report renders from the payload with no model/credentials
    table = report.build_table(results)
    assert "Configuration" in table
    assert "All three combined" in table
    full = report.build_report(results)
    assert "Hermes A/B/D validation" in full
    # round-trips through JSON (what write_outputs persists)
    payload = json.loads(json.dumps(results, default=str))
    assert report.build_table(payload)


def test_engine_tool_counts_increase_with_composition() -> None:
    assert build_engine(RUN_CONFIGS["baseline"]) is None
    assert len(build_engine(RUN_CONFIGS["relevance"]).get_tool_schemas()) == 1
    assert len(build_engine(RUN_CONFIGS["disclosure"]).get_tool_schemas()) == 2
    assert len(build_engine(RUN_CONFIGS["graph"]).get_tool_schemas()) == 3
    assert len(build_engine(RUN_CONFIGS["all"]).get_tool_schemas()) == 6
