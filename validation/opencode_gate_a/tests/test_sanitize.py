"""Tests for the sanitisation trust boundary: payload-free results, catalog classification, validity."""

from __future__ import annotations

import json

import pytest
from _ce60_factories import build_catalog, raw_run, raw_turn
from opencode_gate_a.canonical import canonical_json
from opencode_gate_a.contracts import TURN_COUNT, ContractViolation
from opencode_gate_a.sanitize import FORBIDDEN_RAW_KEYS, sanitize_run, sanitize_turn

_CANARIES = (
    "SECRET-PROMPT-CANARY",
    "SECRET-RESPONSE-CANARY",
    "SECRET-PAYLOAD-CANARY",
    "AKIA-SECRET-CANARY",
    "SECRET-LOG-CANARY",
)


class TestSanitizeTurn:
    """Per-turn sanitisation."""

    def test_forbidden_payload_never_reaches_the_result(self) -> None:
        """Prompt, response, tool payload, credentials and logs are dropped (NFR4.1)."""
        catalog = build_catalog()
        turn = sanitize_turn(
            raw_turn("t00", include_forbidden=True), run_id="r", catalog=catalog
        )
        serialized = canonical_json(turn.as_dict())
        for canary in _CANARIES:
            assert canary not in serialized

    def test_result_dict_has_no_forbidden_keys(self) -> None:
        """The sanitized dict contains none of the forbidden raw keys."""
        catalog = build_catalog()
        turn = sanitize_turn(
            raw_turn("t00", include_forbidden=True), run_id="r", catalog=catalog
        )
        assert FORBIDDEN_RAW_KEYS.isdisjoint(turn.as_dict().keys())

    def test_completion_is_classified_by_catalog_not_host(self) -> None:
        """A host ``success`` that fails the matcher becomes invalid_terminal/not_completed (BR7.3)."""
        catalog = build_catalog()
        turn = sanitize_turn(
            raw_turn("t00", host_signal="success", matcher_passed=False),
            run_id="r",
            catalog=catalog,
        )
        assert turn.terminal_state == "invalid_terminal"
        assert turn.completion_status == "not_completed"

    def test_matched_success_is_completed(self) -> None:
        """A host success that passes the matcher is completed_expected."""
        catalog = build_catalog()
        turn = sanitize_turn(raw_turn("t00"), run_id="r", catalog=catalog)
        assert turn.terminal_state == "completed_expected"
        assert turn.completion_status == "completed"

    def test_unknown_host_signal_is_rejected(self) -> None:
        """A raw turn with an unknown host signal fails closed."""
        catalog = build_catalog()
        bad = raw_turn("t00")
        bad["host_signal"] = "weird"
        with pytest.raises(ContractViolation) as exc:
            sanitize_turn(bad, run_id="r", catalog=catalog)
        assert exc.value.code == "RESULT_NOT_SANITIZED"

    def test_missing_metric_becomes_collection_failed_not_zero(self) -> None:
        """A model_input turn with no value is marked collection_failed, never silently zeroed."""
        catalog = build_catalog()
        turn = sanitize_turn(
            raw_turn("t00", model_input=None), run_id="r", catalog=catalog
        )
        mv = turn.validity_for("model_input")
        assert mv is not None
        assert mv.status == "collection_failed"
        assert turn.model_input_tokens is None

    def test_host_not_exposed_invalidates_the_metric(self) -> None:
        """An explicit host_not_exposed status invalidates the metric and drops the value."""
        catalog = build_catalog()
        turn = sanitize_turn(
            raw_turn("t00", retrieval=None, retrieval_status="host_not_exposed"),
            run_id="r",
            catalog=catalog,
        )
        mv = turn.validity_for("retrieval")
        assert mv is not None
        assert mv.status == "host_not_exposed"
        assert turn.retrieval_attempts is None

    def test_non_eligible_turn_marks_correctness_not_applicable(self) -> None:
        """A non-eligible turn records correctness as not_applicable with no score."""
        catalog = build_catalog()
        # t59 is outside the first 30 eligible ids.
        turn = sanitize_turn(
            raw_turn("t59", correctness=None), run_id="r", catalog=catalog
        )
        mv = turn.validity_for("correctness")
        assert mv is not None
        assert mv.status == "not_applicable"
        assert turn.correctness_score is None


class TestSanitizeRun:
    """Whole-run sanitisation."""

    def test_requires_exactly_60_turns(self) -> None:
        """A run with the wrong turn count fails closed."""
        catalog = build_catalog()
        run = raw_run()
        run["turns"] = run["turns"][:59]
        with pytest.raises(ContractViolation) as exc:
            sanitize_run(run, manifest_digest="sha256:" + ("a" * 64), catalog=catalog)
        assert exc.value.code == "RESULT_NOT_SANITIZED"
        assert "60" in str(exc.value)

    def test_produces_60_turns_and_passing_status(self) -> None:
        """A well-formed run yields 60 turns and sanitization_status=passed."""
        catalog = build_catalog()
        result = sanitize_run(
            raw_run(), manifest_digest="sha256:" + ("a" * 64), catalog=catalog
        )
        assert len(result.turn_results) == TURN_COUNT
        assert result.sanitization_status == "passed"
        assert result.result_digest == result.compute_digest()

    def test_host_evidence_is_payload_free(self) -> None:
        """Forbidden keys injected into host evidence do not survive sanitisation."""
        catalog = build_catalog()
        result = sanitize_run(
            raw_run(include_forbidden=True),
            manifest_digest="sha256:" + ("a" * 64),
            catalog=catalog,
        )
        blob = json.dumps(dict(result.host_evidence))
        for canary in _CANARIES:
            assert canary not in blob

    def test_observability_summary_keeps_only_bounded_counts_and_names(self) -> None:
        """The nested diagnostic summary keeps scalars/names and drops payload-shaped canaries."""
        catalog = build_catalog()
        run = raw_run()
        run["host_evidence"]["observability"] = {
            "message_count_max": 8,
            "message_count_median": 4.5,
            "tool_call_count_total": 4,
            "tool_call_count_max": 2,
            "mcp_tool_call_count_total": 1,
            "timeout_count": 1,
            "timeout_turn_ids": ["F002"],
            "tool_names": ["read", "mcp_ce60-fixture_search"],
            "prompt": "SECRET-PROMPT-CANARY",
            "response": "SECRET-RESPONSE-CANARY",
            "arguments": "SECRET-ARGS-CANARY",
            "payload": {"value": "SECRET-PAYLOAD-CANARY"},
        }
        result = sanitize_run(
            run,
            manifest_digest="sha256:" + ("a" * 64),
            catalog=catalog,
        )
        observability = result.host_evidence["observability"]
        assert observability["message_count_max"] == 8
        assert observability["message_count_median"] == 4.5
        assert observability["timeout_turn_ids"] == ["F002"]
        assert observability["tool_names"] == ["read", "mcp_ce60-fixture_search"]
        serialized = json.dumps(dict(result.host_evidence))
        assert "SECRET-ARGS-CANARY" not in serialized
        for canary in _CANARIES:
            assert canary not in serialized

    def test_observability_summary_drops_unbounded_lists(self) -> None:
        """A list beyond the diagnostic bound is omitted rather than persisted."""
        catalog = build_catalog()
        run = raw_run()
        run["host_evidence"]["observability"] = {
            "tool_names": ["read"] * 513,
            "timeout_count": 0,
        }
        result = sanitize_run(
            run,
            manifest_digest="sha256:" + ("a" * 64),
            catalog=catalog,
        )
        assert result.host_evidence["observability"] == {"timeout_count": 0}

        """Sanitising the same raw run twice yields the same result digest."""
        catalog = build_catalog()
        digest_a = sanitize_run(
            raw_run(), manifest_digest="sha256:" + ("a" * 64), catalog=catalog
        ).result_digest
        digest_b = sanitize_run(
            raw_run(), manifest_digest="sha256:" + ("a" * 64), catalog=catalog
        ).result_digest
        assert digest_a == digest_b
