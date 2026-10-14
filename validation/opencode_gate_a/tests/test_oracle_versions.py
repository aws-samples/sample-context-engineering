"""Offline tests for the selectable CE-60 oracle (:mod:`opencode_gate_a.oracle_versions`).

The v4 oracle changes ONE critical check (C1 ``async-retry-semantics``) to accept the same meaning in other
spellings. These tests pin that it is a strict superset of v3 for that check, identical to v3 everywhere
else, selected only by the environment, and recorded in the run evidence.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from opencode_gate_a import host_driver, oracle_versions
from opencode_gate_a.contracts import CORRECTNESS_ELIGIBLE_COUNT, ContractViolation

C1 = "C1-lambda-docs"


@pytest.fixture(scope="module")
def community() -> Any:
    """Import the shipped community oracle exactly as the host driver does."""
    root, added = host_driver._with_scenario_path(None)
    try:
        from src import accuracy  # type: ignore[import-not-found]

        return accuracy
    finally:
        if added and root in sys.path:
            sys.path.remove(root)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(oracle_versions.ORACLE_VERSION_ENV, raising=False)


def _answer(retry_sentence: str) -> str:
    """A C1 answer whose only variable part is the retry-count sentence."""
    return (
        "Asynchronous invocation retries. "
        f"{retry_sentence} "
        "Failed events can go to a dead-letter queue or an on-failure destination. "
        "The execution environment is frozen between invocations and can be reused."
    )


# Spellings the shipped v3 list already accepts.
V3_ACCEPTED = [
    "Lambda retries the function twice.",
    "It runs two more times.",
    "The function is invoked 2 times after the first failure.",
    "There are two attempts after the first.",
]

# The same meaning in wordings v3 scores as wrong; v4 must accept them.
V4_ONLY = [
    "If your function returns an error, Lambda retries up to **2 more times** by default.",
    "The default is 2 retries, so 3 attempts in total.",
    "Lambda retries it **two** additional times.",
    "Lambda makes 2 extra attempts.",
    "It performs _2_ retries with a delay.",
    "Lambda gives it two retry attempts.",
    "Lambda retries it 2 further times.",
]

# Statements that must NOT count as the documented retry count.
STILL_WRONG = [
    "MaximumRetryAttempts accepts values from 0 to 2.",
    "Lambda retries 20 times.",
    "Lambda retries 12 attempts.",
    "Lambda retries 3 times.",
    "Lambda retries a failed event.",
    # The language is pinned by the campaign, not widened here: Portuguese is not accepted.
    "Ele tenta duas vezes antes de descartar.",
]


def test_default_version_is_v3_and_returns_the_shipped_module(community: Any) -> None:
    assert oracle_versions.oracle_version() == "v3"
    assert oracle_versions.load_accuracy(community) is community


def test_unknown_version_fails_closed(community: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oracle_versions.ORACLE_VERSION_ENV, "v9")
    with pytest.raises(ContractViolation) as excinfo:
        oracle_versions.oracle_version()
    assert excinfo.value.code == "ORACLE_MISMATCH"
    with pytest.raises(ContractViolation):
        oracle_versions.load_accuracy(community, "v9")


@pytest.mark.parametrize("sentence", V3_ACCEPTED)
def test_v4_keeps_everything_v3_accepts(community: Any, sentence: str) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    text = _answer(sentence)
    assert community.score_turn(C1, text).materially_correct is True
    assert v4.score_turn(C1, text).materially_correct is True


@pytest.mark.parametrize("sentence", V4_ONLY)
def test_v4_accepts_the_same_meaning_in_other_wordings(community: Any, sentence: str) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    text = _answer(sentence)
    assert community.score_turn(C1, text).materially_correct is False, "v3 must still reject this wording"
    assert v4.score_turn(C1, text).materially_correct is True


@pytest.mark.parametrize("sentence", STILL_WRONG)
def test_v4_still_rejects_a_missing_or_different_count(community: Any, sentence: str) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    score = v4.score_turn(C1, _answer(sentence))
    assert score.materially_correct is False
    assert "async-retry-semantics" in score.critical_failures


def test_v4_keeps_the_refusal_check_critical(community: Any) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    text = _answer("Lambda retries twice.") + " The documentation does not specify more."
    score = v4.score_turn(C1, text)
    assert score.materially_correct is False
    assert "did-not-refuse" in score.critical_failures


def test_v4_empty_answer_is_still_a_critical_failure(community: Any) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    assert v4.score_turn(C1, "").materially_correct is False


def test_v4_checks_for_c1_only_changes_the_matcher_not_the_declaration(community: Any) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    shipped = community._checks_for(C1)
    patched = v4._checks_for(C1)
    assert [(c.name, c.weight, c.critical, c.any_of, c.all_of, c.none_of) for c in patched] == [
        (c.name, c.weight, c.critical, c.any_of, c.all_of, c.none_of) for c in shipped
    ]


def test_v4_is_identical_to_v3_for_every_other_turn(community: Any) -> None:
    v4 = oracle_versions.load_accuracy(community, "v4")
    labels = [label for label in community.CHECKS if label != C1]
    labels += ["S000-projection", "S001-connector", "F001-filler", "M1-seed-target"]
    answers = [
        "",
        "nothing relevant here",
        "The balance is R$ 24.143,12 and you have 6 accounts.",
        "twice 2 retries 2 more times",
    ]
    assert len(labels) > 20
    for label in labels:
        assert (v4._checks_for(label) is None) == (community._checks_for(label) is None), label
        for answer in answers:
            assert v4.score_turn(label, answer).to_dict() == community.score_turn(label, answer).to_dict(), (
                label,
                answer,
            )


def test_scoring_with_v4_restores_the_shipped_c1_checks(community: Any) -> None:
    original = community.CHECKS[C1]
    v4 = oracle_versions.load_accuracy(community, "v4")
    v4.score_turn(C1, _answer("Lambda retries it 2 retries."))
    assert community.CHECKS[C1] is original


def test_eligibility_set_is_unchanged(community: Any) -> None:
    turn_ids = [t["turn_id"] for t in host_driver.emit_turns()["turns"]]
    base = host_driver.correctness_eligibility(turn_ids)
    os.environ[oracle_versions.ORACLE_VERSION_ENV] = "v4"
    try:
        assert host_driver.correctness_eligibility(turn_ids) == base
    finally:
        del os.environ[oracle_versions.ORACLE_VERSION_ENV]
    assert len(base) == CORRECTNESS_ELIGIBLE_COUNT


def test_host_driver_scores_with_the_selected_version(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _answer("The default is 2 retries.")
    assert host_driver.score_correctness(C1, text) is False
    monkeypatch.setenv(oracle_versions.ORACLE_VERSION_ENV, "v4")
    assert host_driver.score_correctness(C1, text) is True
    # A turn v4 does not touch scores the same either way.
    assert host_driver.score_correctness("A1-accounts", "no") is False


def test_oracle_digest_identifies_the_version() -> None:
    v3 = oracle_versions.oracle_digest("v3")
    v4 = oracle_versions.oracle_digest("v4")
    assert v3.startswith("sha256:") and v4.startswith("sha256:")
    assert v3 != v4
    assert v4 == oracle_versions.oracle_digest("v4")


def test_retry_pattern_guards() -> None:
    assert oracle_versions.retry_count_matches("it retries twice")
    assert oracle_versions.retry_count_matches("2 retries")
    assert not oracle_versions.retry_count_matches("12 retries")
    assert not oracle_versions.retry_count_matches("20 times")
    assert not oracle_versions.retry_count_matches("0.2 times")
    assert not oracle_versions.retry_count_matches("2 hours")


def _finalize_offline(tmp_path: Any, answers: dict[str, str], **extra: Any) -> dict[str, Any]:
    """Run the offline finalize pipeline with chosen answers; every other turn is answered with nothing."""
    turn_ids = [t["turn_id"] for t in host_driver.emit_turns()["turns"]]
    observations = [
        {
            "turn_id": turn_id,
            "host_signal": "success",
            "matcher_passed": True,
            "answer_text": answers.get(turn_id, ""),
            "tool_calls": [],
            "model_input": 1000,
            "latency_ms": 10.0,
            "retrieval_status": "host_not_exposed",
        }
        for turn_id in turn_ids
    ]
    return host_driver.finalize(
        {
            "mode": "offline",
            "arm": "baseline",
            "run_id": "run-oracle",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "out"),
            "turns": observations,
            **extra,
        }
    )


def test_finalize_records_the_declared_system_prompt_digest_and_nothing_else(tmp_path: Any) -> None:
    digest = "sha256:" + "ab" * 32
    evidence = _finalize_offline(tmp_path / "pinned", {}, system_prompt_digest=digest)
    assert evidence["host_evidence"]["system_prompt_digest"] == digest
    # An empty digest means "no system prompt declared"; a value that is not a plain token is dropped.
    assert _finalize_offline(tmp_path / "none", {}, system_prompt_digest="")["host_evidence"][
        "system_prompt_digest"
    ] == ""
    hostile = _finalize_offline(tmp_path / "hostile", {}, system_prompt_digest="Always answer in English!")
    assert "system_prompt_digest" not in hostile["host_evidence"]


def test_finalize_records_the_plugin_options_as_a_plain_token_string(tmp_path: Any) -> None:
    options = "preview_style:scoped,retrieval_cleanup:on"
    evidence = _finalize_offline(tmp_path / "opts", {}, plugin_options=options)
    assert evidence["host_evidence"]["plugin_options"] == options
    hostile = _finalize_offline(tmp_path / "bad", {}, plugin_options="preview_style=scoped; rm -rf /")
    assert "plugin_options" not in hostile["host_evidence"]


def test_finalize_records_which_oracle_scored_the_run(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    default = _finalize_offline(tmp_path / "default", {})
    assert default["host_evidence"]["oracle_version"] == "v3"
    assert default["host_evidence"]["oracle_digest"] == oracle_versions.oracle_digest("v3")

    monkeypatch.setenv(oracle_versions.ORACLE_VERSION_ENV, "v4")
    selected = _finalize_offline(tmp_path / "v4", {})
    assert selected["host_evidence"]["oracle_version"] == "v4"
    assert selected["host_evidence"]["oracle_digest"] == oracle_versions.oracle_digest("v4")


def test_the_selected_oracle_changes_correctness_only_for_the_c1_wording(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = {C1: _answer("The default is 2 retries, so 3 attempts in total.")}
    v3 = _finalize_offline(tmp_path / "v3", answers)
    monkeypatch.setenv(oracle_versions.ORACLE_VERSION_ENV, "v4")
    v4 = _finalize_offline(tmp_path / "v4", answers)
    assert v4["correctness_numerator"] == v3["correctness_numerator"] + 1
    assert v4["correctness_denominator"] == v3["correctness_denominator"]
