"""Tests for canonicalisation, digests and the fail-closed contract entities."""

from __future__ import annotations

import dataclasses

import pytest
from _ce60_factories import (
    build_assignments,
    build_catalog,
    build_manifest,
    build_stat_plan,
)
from opencode_gate_a import canonical
from opencode_gate_a.contracts import (
    TURN_COUNT,
    ContractViolation,
    MetricValidity,
    RunManifest,
    StatPlan,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
    TurnAssignment,
)


class TestCanonical:
    """The ``canonical-json-v1`` serialisation and SHA-256 digests."""

    def test_key_order_is_irrelevant_to_the_digest(self) -> None:
        """Two dicts that differ only in key order produce the same canonical bytes and digest."""
        a = {"b": 1, "a": 2, "c": [3, 2, 1]}
        b = {"c": [3, 2, 1], "a": 2, "b": 1}
        assert canonical.canonical_json(a) == canonical.canonical_json(b)
        assert canonical.digest(a) == canonical.digest(b)

    def test_array_order_is_significant(self) -> None:
        """Arrays preserve order, so a reordered list yields a different digest."""
        assert canonical.digest([1, 2, 3]) != canonical.digest([3, 2, 1])

    def test_digest_is_framed_sha256(self) -> None:
        """A digest is ``sha256:`` followed by 64 lowercase hex characters."""
        value = canonical.digest({"x": 1})
        assert value.startswith("sha256:")
        hexpart = value.split(":", 1)[1]
        assert len(hexpart) == 64
        assert hexpart == hexpart.lower()

    def test_non_finite_floats_are_rejected(self) -> None:
        """``NaN``/``Infinity`` are refused so no non-reproducible value can be digested."""
        with pytest.raises(ValueError):
            canonical.canonical_json({"x": float("nan")})
        with pytest.raises(ValueError):
            canonical.digest({"x": float("inf")})

    def test_unicode_is_emitted_literally(self) -> None:
        """Non-ASCII characters are not escaped, so the bytes match across machines."""
        assert canonical.canonical_json({"k": "ação"}) == '{"k":"ação"}'


class TestTurnAssignment:
    """The deterministic per-turn assignment."""

    def test_missing_field_is_rejected(self) -> None:
        """An empty field fails closed with ``RUN_MANIFEST_INVALID``."""
        with pytest.raises(ContractViolation) as exc:
            TurnAssignment(turn_id="t00", seed="", fault_profile_id="f", fault_profile_digest="d")
        assert exc.value.code == "RUN_MANIFEST_INVALID"

    def test_is_frozen(self) -> None:
        """An assignment is immutable."""
        a = build_assignments()[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.seed = "mutated"  # type: ignore[misc]


class TestStatPlan:
    """The closed statistical precondition."""

    def test_build_is_valid_and_digest_matches(self) -> None:
        """A built plan carries a digest that matches its content."""
        plan = build_stat_plan()
        assert plan.digest == plan.content_digest()

    def test_tampered_digest_is_rejected(self) -> None:
        """A plan whose digest does not match its content fails closed."""
        plan = build_stat_plan()
        with pytest.raises(ContractViolation) as exc:
            dataclasses.replace(plan, digest="sha256:" + ("0" * 64))
        assert exc.value.code == "STAT_PLAN_INVALID"

    def test_confidence_out_of_range_is_rejected(self) -> None:
        """A confidence level outside (0, 1) fails closed."""
        with pytest.raises(ContractViolation) as exc:
            StatPlan.build(
                confidence_level=1.5,
                paired_unit="per_turn",
                method="m",
                decision_rule="r",
                minimum_paired_repetitions=5,
                approval_id="a",
                approval_actor="x",
            )
        assert exc.value.code == "STAT_PLAN_INVALID"

    def test_non_positive_repetitions_is_rejected(self) -> None:
        """A non-positive minimum repetition count fails closed."""
        with pytest.raises(ContractViolation):
            StatPlan.build(
                confidence_level=0.95,
                paired_unit="per_turn",
                method="m",
                decision_rule="r",
                minimum_paired_repetitions=0,
                approval_id="a",
                approval_actor="x",
            )


class TestTerminalCatalog:
    """The versioned, immutable terminal classification catalog."""

    def test_build_is_valid(self) -> None:
        """A built catalog has 60 unique turns and matches its digest."""
        catalog = build_catalog()
        assert len(catalog.turn_definitions) == TURN_COUNT
        assert catalog.digest == catalog.content_digest()

    def test_wrong_turn_count_is_rejected(self) -> None:
        """A catalog with a count other than 60 fails closed."""
        definitions = [
            TerminalTurnDefinition(
                turn_id=f"t{i:02d}",
                terminal_matcher="deterministic_predicate",
                accepted_states=("completed_expected",),
                fallback_policy="not_allowed",
                correctness_eligible=i < 30,
            )
            for i in range(59)
        ]
        with pytest.raises(ContractViolation) as exc:
            TerminalOutcomeCatalog.build(
                terminal_catalog_id="c", version="1.0", turn_definitions=definitions
            )
        assert exc.value.code == "TERMINAL_CATALOG_INVALID"

    def test_wrong_eligible_count_is_oracle_mismatch(self) -> None:
        """A catalog without exactly 30 eligible turns fails with ``ORACLE_MISMATCH``."""
        definitions = [
            TerminalTurnDefinition(
                turn_id=f"t{i:02d}",
                terminal_matcher="deterministic_predicate",
                accepted_states=("completed_expected",),
                fallback_policy="not_allowed",
                correctness_eligible=i < 29,
            )
            for i in range(60)
        ]
        with pytest.raises(ContractViolation) as exc:
            TerminalOutcomeCatalog.build(
                terminal_catalog_id="c", version="1.0", turn_definitions=definitions
            )
        assert exc.value.code == "ORACLE_MISMATCH"

    def test_unknown_matcher_is_rejected(self) -> None:
        """A turn definition with an unknown matcher fails closed."""
        with pytest.raises(ContractViolation):
            TerminalTurnDefinition(
                turn_id="t00",
                terminal_matcher="guesswork",  # type: ignore[arg-type]
                accepted_states=("completed_expected",),
                fallback_policy="not_allowed",
                correctness_eligible=True,
            )


class TestRunManifest:
    """The immutable run identity, admission and provenance."""

    def test_build_is_valid_and_digest_matches(self) -> None:
        """A built offline manifest matches its binding digest."""
        manifest = build_manifest()
        assert manifest.manifest_digest == manifest.compute_digest()

    def test_requires_exactly_60_assignments(self) -> None:
        """A manifest with fewer than 60 assignments fails closed."""
        catalog = build_catalog()
        with pytest.raises(ContractViolation) as exc:
            RunManifest.build(
                run_id="r",
                host="opencode",
                host_version="1.0",
                model_id="m",
                backend_id="local-fake",
                arm="baseline",
                capability_manifest_digest="sha256:" + ("1" * 64),
                fixture_catalog_digest="sha256:" + ("2" * 64),
                terminal_catalog_digest=catalog.digest,
                execution_mode="offline",
                order_seed="o",
                turn_assignments=build_assignments()[:59],
                budget_status="not_applicable",
                sample_plan_status="not_applicable",
            )
        assert exc.value.code == "RUN_MANIFEST_INVALID"

    def test_is_immutable(self) -> None:
        """A manifest cannot be mutated after construction (BR7.1)."""
        manifest = build_manifest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.model_id = "other"  # type: ignore[misc]

    def test_tampered_field_breaks_digest(self) -> None:
        """Replacing a bound field without recomputing the digest fails closed."""
        manifest = build_manifest()
        with pytest.raises(ContractViolation) as exc:
            dataclasses.replace(manifest, model_id="tampered")
        assert exc.value.code == "RUN_MANIFEST_INVALID"

    def test_offline_rejects_live_only_fields(self) -> None:
        """An offline manifest carrying an admission id is denied by policy."""
        manifest = build_manifest()
        with pytest.raises(ContractViolation) as exc:
            dataclasses.replace(
                manifest,
                execution_admission_id="adm-1",
                manifest_digest=manifest.manifest_digest,
            )
        assert exc.value.code in {"ADMISSION_POLICY_DENIED", "RUN_MANIFEST_INVALID"}

    def test_live_requires_admission_and_plan(self) -> None:
        """A live manifest without admission, budget, price and plan fails closed."""
        catalog = build_catalog()
        with pytest.raises(ContractViolation):
            RunManifest.build(
                run_id="r",
                host="opencode",
                host_version="1.0",
                model_id="m",
                backend_id="remote-explicit",
                arm="A",
                capability_manifest_digest="sha256:" + ("1" * 64),
                fixture_catalog_digest="sha256:" + ("2" * 64),
                terminal_catalog_digest=catalog.digest,
                execution_mode="live",
                order_seed="o",
                turn_assignments=build_assignments(),
                budget_status="absent",
                sample_plan_status="absent",
            )

    def test_live_is_valid_with_full_admission(self) -> None:
        """A fully-admitted live manifest with an approved plan is valid (shape only, no execution)."""
        catalog = build_catalog()
        manifest = RunManifest.build(
            run_id="r-live",
            host="opencode",
            host_version="1.0",
            model_id="m",
            backend_id="remote-explicit",
            arm="A",
            capability_manifest_digest="sha256:" + ("1" * 64),
            fixture_catalog_digest="sha256:" + ("2" * 64),
            terminal_catalog_digest=catalog.digest,
            execution_mode="live",
            order_seed="o",
            turn_assignments=build_assignments(),
            budget_status="configured",
            sample_plan_status="approved",
            execution_admission_id="adm-1",
            admission_policy_digest="sha256:" + ("3" * 64),
            budget_ceiling_units=100.0,
            budget_currency_or_unit="usd",
            price_provenance="price-sheet-v1",
            stat_plan=build_stat_plan(),
        )
        assert manifest.execution_mode == "live"
        assert manifest.manifest_digest == manifest.compute_digest()


class TestMetricValidity:
    """The explicit per-metric, per-scope validity record."""

    def test_turn_scope_requires_turn_id(self) -> None:
        """A turn-scoped record without a turn id fails closed."""
        with pytest.raises(ContractViolation):
            MetricValidity(scope="turn", run_id="r", metric_name="completion", status="valid")

    def test_run_scope_rejects_turn_id(self) -> None:
        """A run-scoped record carrying a turn id fails closed."""
        with pytest.raises(ContractViolation):
            MetricValidity(
                scope="run",
                run_id="r",
                metric_name="completion",
                status="valid",
                turn_id="t00",
            )

    def test_unknown_status_is_rejected(self) -> None:
        """An out-of-set status fails closed."""
        with pytest.raises(ContractViolation):
            MetricValidity(
                scope="run",
                run_id="r",
                metric_name="completion",
                status="maybe",  # type: ignore[arg-type]
            )
