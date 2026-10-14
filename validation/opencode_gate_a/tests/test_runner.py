"""Tests for the offline CE-60 runner: assignments, manifests, file ops, Gate A and the safe CLI.

Every test here is offline: no network, no AWS, no provider, no live host. The runner is exercised
through its Python API and its command-line entry point, and the live path is asserted to fail closed
rather than start an incomplete benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _ce60_factories import (
    ELIGIBLE_IDS,
    TURN_IDS,
    build_catalog,
    build_stat_plan,
    raw_run,
)
from opencode_gate_a import runner
from opencode_gate_a.contracts import TURN_COUNT, ContractViolation

_CANARIES = (
    "SECRET-PROMPT-CANARY",
    "SECRET-RESPONSE-CANARY",
    "SECRET-PAYLOAD-CANARY",
    "AKIA-SECRET-CANARY",
    "SECRET-LOG-CANARY",
)

_CAP_DIGEST = "sha256:" + ("1" * 64)
_FIX_DIGEST = "sha256:" + ("2" * 64)


class TestBuildAssignments:
    """The 60-turn assignment builder: exactly 60, deterministic, never truncating."""

    def test_explicit_ids_yield_sixty_assignments(self) -> None:
        """Explicit CE-60 ids produce exactly 60 assignments in order."""
        assignments = runner.build_assignments(TURN_IDS, order_seed="order-1")
        assert len(assignments) == TURN_COUNT
        assert tuple(a.turn_id for a in assignments) == TURN_IDS

    def test_assignments_are_deterministic_in_the_order_seed(self) -> None:
        """The same ids and order seed produce identical seeds; a different seed changes them."""
        a1 = runner.build_assignments(TURN_IDS, order_seed="order-1")
        a2 = runner.build_assignments(TURN_IDS, order_seed="order-1")
        a3 = runner.build_assignments(TURN_IDS, order_seed="order-2")
        assert [a.seed for a in a1] == [a.seed for a in a2]
        assert [a.seed for a in a1] != [a.seed for a in a3]

    def test_wrong_count_fails_closed_never_truncates(self) -> None:
        """Fewer than 60 ids is rejected, not padded or truncated to fit."""
        with pytest.raises(ContractViolation) as exc:
            runner.build_assignments(TURN_IDS[:59], order_seed="order-1")
        assert exc.value.code == "RUN_MANIFEST_INVALID"
        assert "60" in str(exc.value)

    def test_duplicate_ids_are_rejected(self) -> None:
        """A duplicated turn id is rejected."""
        dupes = (TURN_IDS[0],) + TURN_IDS[:59]
        with pytest.raises(ContractViolation) as exc:
            runner.build_assignments(dupes, order_seed="order-1")
        assert exc.value.code == "RUN_MANIFEST_INVALID"

    def test_empty_order_seed_is_rejected(self) -> None:
        """An empty order seed is rejected."""
        with pytest.raises(ContractViolation):
            runner.build_assignments(TURN_IDS, order_seed="  ")

    def test_missing_ids_without_scenario_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no explicit ids and no importable scenario, the builder refuses to guess."""
        monkeypatch.setattr(runner, "load_ce60_turn_ids", lambda: None)
        with pytest.raises(ContractViolation) as exc:
            runner.build_assignments(None, order_seed="order-1")
        assert exc.value.code == "RUN_MANIFEST_INVALID"


class TestBuildManifest:
    """Pilot/full manifest assembly and the local-only / live preconditions."""

    def test_pilot_and_full_offline_are_local_only(self) -> None:
        """Both plans build a valid offline manifest with a local backend and 60 assignments."""
        catalog = build_catalog()
        for scope in ("pilot", "full"):
            manifest = runner.build_manifest(
                run_id=f"run-{scope}",
                arm="baseline",
                host="opencode",
                host_version="1.4.0",
                model_id="model-x",
                capability_manifest_digest=_CAP_DIGEST,
                fixture_catalog_digest=_FIX_DIGEST,
                terminal_catalog_digest=catalog.digest,
                order_seed="order-1",
                scope=scope,
                turn_ids=TURN_IDS,
            )
            assert manifest.execution_mode == "offline"
            assert manifest.backend_id == "local-fake"
            assert len(manifest.turn_assignments) == TURN_COUNT

    def test_unknown_scope_is_rejected(self) -> None:
        """A scope outside pilot/full fails closed."""
        catalog = build_catalog()
        with pytest.raises(ContractViolation) as exc:
            runner.build_manifest(
                run_id="run-x",
                arm="baseline",
                host="opencode",
                host_version="1.4.0",
                model_id="model-x",
                capability_manifest_digest=_CAP_DIGEST,
                fixture_catalog_digest=_FIX_DIGEST,
                terminal_catalog_digest=catalog.digest,
                order_seed="order-1",
                scope="smoke",
                turn_ids=TURN_IDS,
            )
        assert exc.value.code == "RUN_MANIFEST_INVALID"

    def test_offline_with_remote_backend_is_denied(self) -> None:
        """An offline manifest may not carry a remote backend (local-only)."""
        catalog = build_catalog()
        with pytest.raises(ContractViolation) as exc:
            runner.build_manifest(
                run_id="run-x",
                arm="A",
                host="opencode",
                host_version="1.4.0",
                model_id="model-x",
                capability_manifest_digest=_CAP_DIGEST,
                fixture_catalog_digest=_FIX_DIGEST,
                terminal_catalog_digest=catalog.digest,
                order_seed="order-1",
                backend_id="remote-explicit",
                turn_ids=TURN_IDS,
            )
        assert exc.value.code == "ADMISSION_POLICY_DENIED"

    def test_live_requires_approved_stat_plan(self) -> None:
        """A live manifest without an approved stat plan fails closed."""
        catalog = build_catalog()
        with pytest.raises(ContractViolation) as exc:
            runner.build_manifest(
                run_id="run-live",
                arm="A",
                host="opencode",
                host_version="1.4.0",
                model_id="model-x",
                capability_manifest_digest=_CAP_DIGEST,
                fixture_catalog_digest=_FIX_DIGEST,
                terminal_catalog_digest=catalog.digest,
                order_seed="order-1",
                execution_mode="live",
                backend_id="remote-explicit",
                budget_status="configured",
                budget_ceiling_units=10.0,
                budget_currency_or_unit="USD",
                price_provenance="price-sheet-v1",
                execution_admission_id="adm-1",
                admission_policy_digest="sha256:" + ("3" * 64),
                sample_plan_status="approved",
                stat_plan=None,
            )
        assert exc.value.code == "SAMPLE_PLAN_INVALID"

    def test_fully_specified_live_manifest_can_be_authored(self) -> None:
        """A live manifest with admission, budget, price and an approved plan is valid data (no run)."""
        catalog = build_catalog()
        manifest = runner.build_manifest(
            run_id="run-live",
            arm="A",
            host="opencode",
            host_version="1.4.0",
            model_id="model-x",
            capability_manifest_digest=_CAP_DIGEST,
            fixture_catalog_digest=_FIX_DIGEST,
            terminal_catalog_digest=catalog.digest,
            order_seed="order-1",
            execution_mode="live",
            backend_id="remote-explicit",
            budget_status="configured",
            budget_ceiling_units=10.0,
            budget_currency_or_unit="USD",
            price_provenance="price-sheet-v1",
            execution_admission_id="adm-1",
            admission_policy_digest="sha256:" + ("3" * 64),
            sample_plan_status="approved",
            stat_plan=build_stat_plan(),
        )
        assert manifest.execution_mode == "live"
        assert manifest.stat_plan is not None


class TestSerialisationRoundTrip:
    """Digest-verified (de)serialisation of manifests and catalogs."""

    def test_manifest_round_trip_preserves_digest(self) -> None:
        """A manifest serialised and reloaded keeps its digest."""
        catalog = build_catalog()
        manifest = runner.build_manifest(
            run_id="run-1",
            arm="baseline",
            host="opencode",
            host_version="1.4.0",
            model_id="model-x",
            capability_manifest_digest=_CAP_DIGEST,
            fixture_catalog_digest=_FIX_DIGEST,
            terminal_catalog_digest=catalog.digest,
            order_seed="order-1",
            turn_ids=TURN_IDS,
        )
        reloaded = runner.manifest_from_dict(runner.manifest_to_dict(manifest, scope="pilot"))
        assert reloaded.manifest_digest == manifest.manifest_digest

    def test_tampered_manifest_is_rejected_on_reload(self) -> None:
        """Editing a field but keeping the old digest is caught on reload."""
        catalog = build_catalog()
        manifest = runner.build_manifest(
            run_id="run-1",
            arm="baseline",
            host="opencode",
            host_version="1.4.0",
            model_id="model-x",
            capability_manifest_digest=_CAP_DIGEST,
            fixture_catalog_digest=_FIX_DIGEST,
            terminal_catalog_digest=catalog.digest,
            order_seed="order-1",
            turn_ids=TURN_IDS,
        )
        data = runner.manifest_to_dict(manifest)
        data["model_id"] = "model-tampered"
        with pytest.raises(ContractViolation) as exc:
            runner.manifest_from_dict(data)
        assert exc.value.code == "RUN_MANIFEST_INVALID"


def _write_offline_manifest(tmp_path, *, run_id: str, arm: str, catalog) -> str:
    """Write an offline manifest file for ``arm`` and return its path."""
    manifest = runner.build_manifest(
        run_id=run_id,
        arm=arm,
        host="opencode",
        host_version="1.4.0",
        model_id="model-x",
        capability_manifest_digest=_CAP_DIGEST,
        fixture_catalog_digest=_FIX_DIGEST,
        terminal_catalog_digest=catalog.digest,
        order_seed="order-1",
        turn_ids=TURN_IDS,
    )
    path = str(tmp_path / f"{run_id}-manifest.json")
    runner._write_json(path, runner.manifest_to_dict(manifest, scope="pilot"))
    return path


def _write_raw_run(tmp_path, *, run_id: str, model_input: int, name: str, forbidden: bool = False) -> str:
    """Write a raw run file (whose turn ids match the CE-60 ids) and return its path."""
    run = raw_run(run_id=run_id, model_input=model_input, include_forbidden=forbidden)
    # Align raw turn ids with the manifest/catalog ids (the factory uses the same TURN_IDS).
    path = str(tmp_path / name)
    runner._write_json(path, run)
    return path


class TestFileOps:
    """sanitize_file / evaluate_file, including the no-payload persistence guarantee."""

    def test_sanitize_and_evaluate_round_trip(self, tmp_path) -> None:
        """A raw run sanitises and evaluates to a full completion matrix via files."""
        catalog = build_catalog()
        catalog_path = str(tmp_path / "catalog.json")
        runner.write_catalog(catalog, catalog_path)
        manifest_path = _write_offline_manifest(tmp_path, run_id="run-1", arm="baseline", catalog=catalog)
        raw_path = _write_raw_run(tmp_path, run_id="run-1", model_input=100, name="raw.json")

        sanitized_path = str(tmp_path / "sanitized.json")
        eval_path = str(tmp_path / "eval.json")
        runner.sanitize_file(raw_path, manifest_path, sanitized_path, catalog_path=catalog_path)
        ev = runner.evaluate_file(sanitized_path, manifest_path, catalog_path, eval_path)

        assert ev.turns_attempted == TURN_COUNT
        assert ev.turns_completed == TURN_COUNT
        assert ev.correctness_denominator == len(ELIGIBLE_IDS)

    def test_sanitized_file_carries_no_payload(self, tmp_path) -> None:
        """No prompt/response/tool payload/credential/log reaches the persisted sanitized file."""
        catalog = build_catalog()
        catalog_path = str(tmp_path / "catalog.json")
        runner.write_catalog(catalog, catalog_path)
        manifest_path = _write_offline_manifest(tmp_path, run_id="run-1", arm="baseline", catalog=catalog)
        raw_path = _write_raw_run(
            tmp_path, run_id="run-1", model_input=100, name="raw.json", forbidden=True
        )

        sanitized_path = str(tmp_path / "sanitized.json")
        runner.sanitize_file(raw_path, manifest_path, sanitized_path, catalog_path=catalog_path)

        text = Path(sanitized_path).read_text(encoding="utf-8")
        for canary in _CANARIES:
            assert canary not in text
        # The forbidden keys themselves must not appear as keys in the persisted projection.
        blob = json.loads(text)
        for turn in blob["turn_results"]:
            assert "prompt" not in turn
            assert "response" not in turn
            assert "tool_payload" not in turn


class TestGateAFiles:
    """decide_gate_a_files over an isolated same-host baseline/treatment pair."""

    def _prepare_arm(self, tmp_path, *, run_id, arm, model_input):
        """Produce (manifest_path, eval_path) for one arm, offline."""
        catalog = build_catalog()
        catalog_path = str(tmp_path / f"{run_id}-catalog.json")
        runner.write_catalog(catalog, catalog_path)
        manifest_path = _write_offline_manifest(tmp_path, run_id=run_id, arm=arm, catalog=catalog)
        raw_path = _write_raw_run(tmp_path, run_id=run_id, model_input=model_input, name=f"{run_id}-raw.json")
        sanitized_path = str(tmp_path / f"{run_id}-sanitized.json")
        eval_path = str(tmp_path / f"{run_id}-eval.json")
        runner.sanitize_file(raw_path, manifest_path, sanitized_path, catalog_path=catalog_path)
        runner.evaluate_file(sanitized_path, manifest_path, catalog_path, eval_path)
        return manifest_path, eval_path

    def test_go_decision_for_isolated_pair(self, tmp_path) -> None:
        """baseline vs A, same host/model, >=20% input drop -> go."""
        base_m, base_e = self._prepare_arm(tmp_path, run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat_e = self._prepare_arm(tmp_path, run_id="run-a", arm="A", model_input=70)
        out = str(tmp_path / "decision.json")
        decision = runner.decide_gate_a_files(
            baseline_manifest_path=base_m,
            baseline_eval_path=base_e,
            treatment_manifest_path=treat_m,
            treatment_eval_path=treat_e,
            output_path=out,
        )
        assert decision.outcome == "go"
        written = runner._read_json(out)
        assert written["outcome"] == "go"
        assert written["gate_name"] == "Gate_A"


class TestCli:
    """The three safe subcommands and the fail-closed live guard."""

    def _write_catalog_and_turn_ids(self, tmp_path):
        """Write a catalog file and a 60-line turn-ids file; return their paths."""
        catalog = build_catalog()
        catalog_path = str(tmp_path / "catalog.json")
        runner.write_catalog(catalog, catalog_path)
        ids_path = str(tmp_path / "turn_ids.txt")
        with open(ids_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(TURN_IDS) + "\n")
        return catalog_path, ids_path

    def _manifest_argv(self, tmp_path, *, run_id, arm, out, catalog_path, ids_path):
        return [
            "manifest",
            "--run-id", run_id,
            "--arm", arm,
            "--host", "opencode",
            "--host-version", "1.4.0",
            "--model-id", "model-x",
            "--capability-digest", _CAP_DIGEST,
            "--fixture-digest", _FIX_DIGEST,
            "--catalog", catalog_path,
            "--order-seed", "order-1",
            "--turn-ids-file", ids_path,
            "--out", out,
        ]

    def test_manifest_subcommand_writes_file(self, tmp_path) -> None:
        """The manifest subcommand builds and persists a manifest."""
        catalog_path, ids_path = self._write_catalog_and_turn_ids(tmp_path)
        out = str(tmp_path / "manifest.json")
        rc = runner.main(
            self._manifest_argv(
                tmp_path, run_id="run-1", arm="baseline", out=out, catalog_path=catalog_path, ids_path=ids_path
            )
            + ["--full"]
        )
        assert rc == 0
        data = runner._read_json(out)
        assert data["scope"] == "full"
        assert len(data["turn_assignments"]) == TURN_COUNT

    def test_full_pipeline_through_cli(self, tmp_path) -> None:
        """manifest -> evaluate -> gate-a run end to end through the CLI and reach a go."""
        catalog_path, ids_path = self._write_catalog_and_turn_ids(tmp_path)

        base_manifest = str(tmp_path / "base-manifest.json")
        treat_manifest = str(tmp_path / "treat-manifest.json")
        assert runner.main(
            self._manifest_argv(
                tmp_path, run_id="run-base", arm="baseline", out=base_manifest,
                catalog_path=catalog_path, ids_path=ids_path,
            )
        ) == 0
        assert runner.main(
            self._manifest_argv(
                tmp_path, run_id="run-a", arm="A", out=treat_manifest,
                catalog_path=catalog_path, ids_path=ids_path,
            )
        ) == 0

        base_raw = _write_raw_run(tmp_path, run_id="run-base", model_input=100, name="base-raw.json")
        treat_raw = _write_raw_run(tmp_path, run_id="run-a", model_input=70, name="treat-raw.json")

        base_eval = str(tmp_path / "base-eval.json")
        treat_eval = str(tmp_path / "treat-eval.json")
        assert runner.main([
            "evaluate", "--raw", base_raw, "--manifest", base_manifest, "--catalog", catalog_path,
            "--out-sanitized", str(tmp_path / "base-san.json"), "--out-evaluation", base_eval,
        ]) == 0
        assert runner.main([
            "evaluate", "--raw", treat_raw, "--manifest", treat_manifest, "--catalog", catalog_path,
            "--out-sanitized", str(tmp_path / "treat-san.json"), "--out-evaluation", treat_eval,
        ]) == 0

        decision_path = str(tmp_path / "decision.json")
        assert runner.main([
            "gate-a",
            "--baseline-manifest", base_manifest, "--baseline-eval", base_eval,
            "--treatment-manifest", treat_manifest, "--treatment-eval", treat_eval,
            "--out", decision_path,
        ]) == 0
        assert runner._read_json(decision_path)["outcome"] == "go"

    def test_execute_live_fails_closed_with_stable_code(self, tmp_path, capsys) -> None:
        """A fully specified --execute-live request fails closed with the stable code (no run)."""
        catalog_path, ids_path = self._write_catalog_and_turn_ids(tmp_path)
        admission = str(tmp_path / "adm.json")
        stat_plan = str(tmp_path / "plan.json")
        runner._write_json(admission, {"execution_admission_id": "adm-1"})
        runner._write_json(stat_plan, runner._stat_plan_to_dict(build_stat_plan()))
        out = str(tmp_path / "manifest.json")
        rc = runner.main(
            self._manifest_argv(
                tmp_path, run_id="run-live", arm="A", out=out, catalog_path=catalog_path, ids_path=ids_path
            )
            + [
                "--execution-mode", "offline",
                "--execute-live",
                "--i-understand-live-costs",
                "--admission-path", admission,
                "--stat-plan-path", stat_plan,
            ]
        )
        assert rc == 3
        assert runner.LIVE_EXECUTION_REQUIRES_HOST_DRIVER in capsys.readouterr().err

    def test_execute_live_without_confirmation_is_a_usage_error(self, tmp_path) -> None:
        """--execute-live without the explicit confirmation flag is refused before any run."""
        catalog_path, ids_path = self._write_catalog_and_turn_ids(tmp_path)
        out = str(tmp_path / "manifest.json")
        rc = runner.main(
            self._manifest_argv(
                tmp_path, run_id="run-live", arm="A", out=out, catalog_path=catalog_path, ids_path=ids_path
            )
            + ["--execute-live"]
        )
        assert rc == 2
