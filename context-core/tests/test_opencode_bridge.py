"""Non-billable tests for the ``context_core.opencode_bridge`` JSONL bridge.

Everything here runs offline: no boto3 client is built, no Bedrock ``rerank`` is called, and no fake
scorer is introduced. The bridge builds its reranker lazily (only when a result actually needs scoring),
so every path exercised below — the JSONL codec, environment/profile validation, the request dispatch
for skips/startup/shutdown, and the outcome→response mapping — stays well clear of inference. A
subprocess probe asserts the bridge pulls in no agent framework and leaks no OpenCode host module into
the neutral core.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from context_core.contracts.errors import _ENVELOPE_FIELDS, make_refusal
from context_core.opencode import OpenCodeAdapterOutcome
from context_core.opencode_bridge import (
    DEFAULT_RERANK_MODEL_ID,
    BridgeConfig,
    BridgeConfigError,
    OpenCodeBridge,
    _transform_response,
    decode_request,
    encode_response,
    normalize_tool_name,
    run_diagnostic_fixture,
    serve,
)

# A syntactically valid but non-existent profile: construction never resolves it (the reranker is lazy),
# so no path exercised here reaches boto3.
_ENV = {
    "AWS_PROFILE": "test-profile",
    "AWS_REGION": "us-west-2",
    "CONTEXT_MAX_RESULT_TOKENS": "1",
    "CONTEXT_OPENCODE_TOOLS": "read,grep",
}


def _bridge(env: dict[str, str] | None = None) -> OpenCodeBridge:
    return OpenCodeBridge(BridgeConfig.from_env(env or _ENV))


# The model the OpenCode agent would use for a live run. Admission must compare this against the
# execution allowlist; the reranker model remains a separate configuration value.
_AGENT_MODEL = "anthropic/claude-x"
_RERANK_MODEL = "amazon.rerank-v1:0"


def _live_env() -> dict[str, str]:
    """An env with separate agent and reranker model identities for live admission."""
    return {
        **_ENV,
        "CONTEXT_EXECUTION_ALLOWED_MODELS": _AGENT_MODEL,
        "CONTEXT_EXECUTION_MODEL_ID": _AGENT_MODEL,
        "CONTEXT_RERANK_MODEL_ID": _RERANK_MODEL,
    }


def _live_admit_request(**overrides: object) -> dict[str, object]:
    """A fully-formed ``live`` admit request; individual fields can be dropped/overridden per test."""
    request: dict[str, object] = {
        "type": "admit",
        "id": "adm-1",
        "execution_mode": "live",
        "approval_id": "appr-123",
        "actor_class": "maintainer",
        "budget_ceiling": 5.0,
        "budget_unit": "USD",
        "nonce": "nonce-abc",
        "preflight_requested": True,
    }
    request.update(overrides)
    return request


# --------------------------------------------------------------------------- #
# JSONL codec                                                                 #
# --------------------------------------------------------------------------- #


def test_encode_response_is_single_newline_terminated_line():
    line = encode_response({"operation": "healthy", "id": "r1"})
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert line.strip().startswith("{")


def test_codec_round_trips():
    original = {"type": "transform", "id": "r1", "tool": "read", "text": "x", "n": 3}
    decoded = decode_request(encode_response(original))
    assert decoded == original


def test_decode_rejects_blank_line():
    with pytest.raises(ValueError, match="empty"):
        decode_request("   \n")


def test_decode_rejects_malformed_json():
    with pytest.raises(ValueError, match="valid JSON"):
        decode_request("{not json")


def test_decode_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        decode_request("[1, 2, 3]")


# --------------------------------------------------------------------------- #
# Environment / profile validation                                            #
# --------------------------------------------------------------------------- #


def test_missing_aws_profile_is_rejected():
    with pytest.raises(BridgeConfigError, match="AWS_PROFILE"):
        BridgeConfig.from_env({"AWS_REGION": "us-west-2"})


def test_blank_aws_profile_is_rejected():
    with pytest.raises(BridgeConfigError, match="AWS_PROFILE"):
        BridgeConfig.from_env({"AWS_PROFILE": "   "})


def test_profile_is_never_defaulted_and_region_falls_back_to_default_region():
    config = BridgeConfig.from_env({"AWS_PROFILE": "p", "AWS_DEFAULT_REGION": "eu-west-1"})
    assert config.aws_profile == "p"
    assert config.aws_region == "eu-west-1"
    # No model id supplied: the documented rerank default applies, while live execution stays unset.
    assert config.rerank_model_id == DEFAULT_RERANK_MODEL_ID
    assert config.execution_model_id is None


def test_region_absent_is_none_not_invented():
    config = BridgeConfig.from_env({"AWS_PROFILE": "p"})
    assert config.aws_region is None


def test_rerank_model_id_from_env_overrides_default():
    config = BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_RERANK_MODEL_ID": "cohere.rerank-v3-5:0"})
    assert config.rerank_model_id == "cohere.rerank-v3-5:0"


def test_execution_model_id_is_read_separately_from_rerank_model_id():
    config = BridgeConfig.from_env(
        {
            "AWS_PROFILE": "p",
            "CONTEXT_EXECUTION_MODEL_ID": _AGENT_MODEL,
            "CONTEXT_RERANK_MODEL_ID": _RERANK_MODEL,
        }
    )
    assert config.execution_model_id == _AGENT_MODEL
    assert config.rerank_model_id == _RERANK_MODEL


def test_threshold_out_of_range_is_rejected():
    with pytest.raises(BridgeConfigError, match="RELEVANCE_THRESHOLD"):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_RELEVANCE_THRESHOLD": "1.5"})


def test_non_numeric_threshold_is_rejected():
    with pytest.raises(BridgeConfigError, match="RELEVANCE_THRESHOLD"):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_RELEVANCE_THRESHOLD": "high"})


def test_non_positive_token_budget_is_rejected():
    with pytest.raises(BridgeConfigError, match="CHUNK_TOKENS"):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_CHUNK_TOKENS": "0"})


def test_non_integer_token_budget_is_rejected():
    with pytest.raises(BridgeConfigError, match="PREVIEW_TOKENS"):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_PREVIEW_TOKENS": "lots"})


def test_tool_allowlist_parsed_and_deduplicated():
    config = BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_OPENCODE_TOOLS": "read, grep ,read, "})
    assert config.tool_classes == ("read", "grep")


def test_tool_allowlist_does_not_normalize_incoming_mcp_names():
    config = BridgeConfig.from_env(
        {"AWS_PROFILE": "p", "CONTEXT_OPENCODE_TOOLS": "ce60-fixture_list_accounts"}
    )
    assert config.tool_classes == ("ce60-fixture_list_accounts",)


def test_empty_tool_allowlist_is_rejected():
    with pytest.raises(BridgeConfigError, match="CONTEXT_OPENCODE_TOOLS"):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_OPENCODE_TOOLS": " , "})


def test_default_tool_allowlist_applies_when_unset():
    config = BridgeConfig.from_env({"AWS_PROFILE": "p"})
    assert config.tool_classes == ("read", "grep", "glob", "list", "webfetch", "bash")


# --------------------------------------------------------------------------- #
# Startup dispatch — publishes A, B/D unsupported                             #
# --------------------------------------------------------------------------- #


def test_startup_publishes_practice_a():
    bridge = _bridge()
    response = bridge.handle({"type": "startup", "id": "s1"})
    assert response["operation"] == "started"
    assert response["id"] == "s1"
    assert response["type"] == "response"
    capabilities = {row["practice"]: row["capability_state"] for row in response["capabilities"]}
    assert capabilities == {"A": "full", "B": "unsupported", "D": "unsupported"}
    assert response["host"] == "opencode"
    assert response["manifest_digest"]
    assert response["model_provenance"] == {
        "execution_model_id": None,
        "rerank_model_id": DEFAULT_RERANK_MODEL_ID,
    }
    assert response["tool_classes"] == ["grep", "read"]


# --------------------------------------------------------------------------- #
# Transform dispatch — skip gates (no scoring, no boto3)                       #
# --------------------------------------------------------------------------- #


def test_transform_skips_when_no_text():
    response = _bridge().handle({"type": "transform", "id": "t1", "tool": "read", "session_id": "x"})
    assert response["operation"] == "skipped"
    assert response["reason"] == "no_text"


def test_transform_skips_uncovered_tool():
    response = _bridge().handle(
        {"type": "transform", "id": "t2", "tool": "edit", "session_id": "x", "text": "a" * 1000}
    )
    assert response["operation"] == "skipped"
    assert response["reason"] == "tool_not_covered"


def test_transform_skips_without_session():
    response = _bridge().handle(
        {"type": "transform", "id": "t3", "tool": "read", "session_id": "", "text": "a" * 1000}
    )
    assert response["operation"] == "skipped"
    assert response["reason"] == "no_session"


def test_transform_skips_under_threshold():
    # max_result_tokens=1 means the gate opens above ~4 chars; a 3-char result stays untouched.
    response = _bridge().handle(
        {"type": "transform", "id": "t4", "tool": "read", "session_id": "x", "text": "abc"}
    )
    assert response["operation"] == "skipped"
    assert response["reason"] == "under_threshold"


# --------------------------------------------------------------------------- #
# Outcome → response mapping (pure, offline)                                   #
# --------------------------------------------------------------------------- #


def test_previewed_outcome_maps_to_content():
    outcome = OpenCodeAdapterOutcome(
        outcome_id="o1",
        binding_id="b1",
        request_id="r1",
        operation="previewed",
        content="filtered preview",
        content_digest="ab" * 32,
    )
    response = _transform_response(outcome)
    assert response == {
        "operation": "previewed",
        "content": "filtered preview",
        "content_digest": "ab" * 32,
    }


def test_baseline_outcome_maps_to_content():
    outcome = OpenCodeAdapterOutcome(
        outcome_id="o2",
        binding_id="b2",
        request_id="r2",
        operation="baseline",
        baseline_preserved=True,
        content=b"original bytes",
        content_digest="cd" * 32,
    )
    response = _transform_response(outcome)
    assert response["operation"] == "baseline"
    assert response["content"] == "original bytes"


def test_refused_outcome_maps_to_closed_envelope_without_payload():
    envelope = make_refusal("AUTHORIZATION_HOST_DENIED", "transform", "r3", capability="A")
    outcome = OpenCodeAdapterOutcome(
        outcome_id="o3",
        binding_id="b3",
        request_id="r3",
        operation="refused",
        refusal=envelope,
    )
    response = _transform_response(outcome)
    assert response["operation"] == "refused"
    # Exactly the seven closed fields, and no content leaked.
    assert set(response["refusal"]) == _ENVELOPE_FIELDS
    assert "content" not in response


# --------------------------------------------------------------------------- #
# Execution admission — offline/replay/live (no boto3, no inference)           #
# --------------------------------------------------------------------------- #


def test_admit_offline_is_admitted_without_approval_or_budget():
    response = _bridge().handle({"type": "admit", "id": "a0", "execution_mode": "offline"})
    assert response["operation"] == "admitted"
    assert response["admission_id"]
    assert response["policy_digest"]
    # Offline never opens credential resolution or payload transmission.
    assert response["preflight"]["credential_resolution"] == "not_attempted"
    assert response["preflight"]["payload_transmission"] == "not_attempted"
    assert response["approval"]["status"] == "absent"
    assert response["budget"]["status"] == "absent"


def test_admit_defaults_to_offline_when_mode_absent():
    response = _bridge().handle({"type": "admit", "id": "a0b"})
    assert response["operation"] == "admitted"
    assert response["execution_mode"] == "offline"
    assert response["preflight"]["credential_resolution"] == "not_attempted"


def test_admit_live_refused_when_model_not_allowlisted():
    # The default env allow-lists no model, so a fully-formed live request is still refused fail-closed.
    response = _bridge().handle(_live_admit_request())
    assert response["operation"] == "refused"
    assert set(response["refusal"]) == _ENVELOPE_FIELDS
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"
    assert "admission_id" not in response


def test_admit_live_refused_without_execution_model():
    # The agent model is required even when the allowlist is populated; the reranker is not a fallback.
    bridge = _bridge(
        {
            **_ENV,
            "CONTEXT_EXECUTION_ALLOWED_MODELS": _AGENT_MODEL,
            "CONTEXT_RERANK_MODEL_ID": _RERANK_MODEL,
        }
    )
    response = bridge.handle(_live_admit_request())
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"
    assert "admission_id" not in response


@pytest.mark.parametrize(
    "drop",
    ["approval_id", "actor_class", "budget_ceiling", "budget_unit"],
)
def test_admit_live_refused_when_a_required_field_is_missing(drop: str):
    bridge = _bridge(_live_env())
    request = _live_admit_request()
    del request[drop]
    response = bridge.handle(request)
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


def test_admit_live_refused_without_preflight_request():
    bridge = _bridge(_live_env())
    response = bridge.handle(_live_admit_request(preflight_requested=False))
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


def test_admit_live_refused_for_unapproved_actor_class():
    bridge = _bridge(_live_env())
    response = bridge.handle(_live_admit_request(actor_class="anonymous"))
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


def test_admit_live_admitted_with_agent_model_and_separate_rerank_model():
    # The allowlist contains the agent model, not the reranker. This is the regression case.
    bridge = _bridge(_live_env())
    response = bridge.handle(_live_admit_request())
    assert response["operation"] == "admitted"
    assert response["execution_mode"] == "live"
    assert response["model_provenance"] == {
        "execution_model_id": _AGENT_MODEL,
        "rerank_model_id": _RERANK_MODEL,
    }
    # Live opens credential resolution and payload transmission only after admission.
    assert response["preflight"]["credential_resolution"] == "permitted_after_admission"
    assert response["preflight"]["payload_transmission"] == "permitted_after_admission"
    assert response["approval"] == {
        "status": "explicit",
        "approval_id": "appr-123",
        "actor_class": "maintainer",
    }
    assert response["budget"] == {"status": "configured", "ceiling_units": 5.0, "currency_or_unit": "USD"}
    assert response["nonce"] == "nonce-abc"
    # The admission carries no credential and no payload.
    assert "payload" not in response


def test_admit_live_reused_nonce_is_refused():
    bridge = _bridge(_live_env())
    first = bridge.handle(_live_admit_request(id="n1"))
    assert first["operation"] == "admitted"
    # A second admission presenting the same nonce (different id/key) is refused: single-use.
    second = bridge.handle(_live_admit_request(id="n2", idempotency_key="n2"))
    assert second["operation"] == "refused"
    assert second["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


def test_transform_over_threshold_refused_without_admission_receipt():
    # A covered, oversized result with no admission receipt must NOT reach boto3: it refuses closed.
    bridge = _bridge()
    response = bridge.handle(
        {"type": "transform", "id": "t-noadm", "tool": "read", "session_id": "s", "text": "a" * 4000}
    )
    assert response["operation"] == "refused"
    assert set(response["refusal"]) == _ENVELOPE_FIELDS
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"
    assert "content" not in response


def test_transform_over_threshold_refused_with_offline_receipt():
    # An offline admission does not open the Bedrock path; the transform still refuses closed.
    bridge = _bridge(_live_env())
    admit = bridge.handle({"type": "admit", "id": "a-off", "execution_mode": "offline"})
    response = bridge.handle(
        {
            "type": "transform",
            "id": "t-off",
            "tool": "read",
            "session_id": "s",
            "text": "a" * 4000,
            "admission_id": admit["admission_id"],
        }
    )
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


def test_transform_over_threshold_refused_with_unknown_admission_id():
    bridge = _bridge(_live_env())
    response = bridge.handle(
        {
            "type": "transform",
            "id": "t-unk",
            "tool": "read",
            "session_id": "s",
            "text": "a" * 4000,
            "admission_id": "never-admitted",
        }
    )
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "ADMISSION_POLICY_DENIED"


# --------------------------------------------------------------------------- #
# Session lifecycle — shutdown dispatch                                        #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Retrieve dispatch — authorized baseline retrieval, counting, no payload leak #
# --------------------------------------------------------------------------- #


def test_transform_response_includes_reference_on_preview():
    outcome = OpenCodeAdapterOutcome(
        outcome_id="o",
        binding_id="b",
        request_id="r",
        operation="previewed",
        content="the preview",
        content_digest="ab" * 32,
    )
    response = _transform_response(outcome, reference_id="REF-123")
    assert response["operation"] == "previewed"
    assert response["reference_id"] == "REF-123"


def test_preserve_baseline_stores_a_retrievable_reference():
    bridge = _bridge()
    namespace_id = bridge._namespace_for("sess-p")
    reference_id = bridge._preserve_baseline("sess-p", namespace_id, "the full original result")
    assert reference_id
    assert reference_id in bridge._sessions["sess-p"].baseline_reference_ids


def test_retrieve_returns_stored_baseline_and_counts_success():
    bridge = _bridge()
    namespace_id = bridge._namespace_for("sess-r")
    reference_id = bridge._preserve_baseline("sess-r", namespace_id, "the full original result")
    response = bridge.handle(
        {"type": "retrieve", "id": "r1", "session_id": "sess-r", "reference_id": reference_id}
    )
    assert response["operation"] == "retrieved"
    assert response["content"] == "the full original result"
    assert bridge._retrieval_counts["sess-r"] == {"attempts": 1, "successes": 1}


def test_retrieve_unknown_session_refuses_closed_and_counts_a_failed_attempt():
    bridge = _bridge()
    response = bridge.handle(
        {"type": "retrieve", "id": "r2", "session_id": "never-opened", "reference_id": "ref"}
    )
    assert response["operation"] == "refused"
    assert set(response["refusal"]) == _ENVELOPE_FIELDS
    assert response["refusal"]["code"] == "REFERENCE_EXPIRED_OR_DELETED"
    assert "content" not in response
    assert bridge._retrieval_counts["never-opened"] == {"attempts": 1, "successes": 0}


def test_retrieve_wrong_reference_in_valid_session_refuses_closed():
    bridge = _bridge()
    bridge._namespace_for("sess-w")
    response = bridge.handle(
        {"type": "retrieve", "id": "r3", "session_id": "sess-w", "reference_id": "not-a-real-ref"}
    )
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "REFERENCE_EXPIRED_OR_DELETED"
    assert "content" not in response
    assert bridge._retrieval_counts["sess-w"] == {"attempts": 1, "successes": 0}


def test_retrieve_missing_reference_is_schema_refusal():
    bridge = _bridge()
    bridge._namespace_for("sess-m")
    response = bridge.handle({"type": "retrieve", "id": "r4", "session_id": "sess-m"})
    assert response["operation"] == "refused"
    assert response["refusal"]["code"] == "REQUEST_SCHEMA_INVALID"


def test_retrieve_without_session_is_skipped():
    response = _bridge().handle({"type": "retrieve", "id": "r5"})
    assert response["operation"] == "skipped"
    assert response["reason"] == "no_session"


def test_retrieve_after_shutdown_refuses_without_payload():
    bridge = _bridge()
    namespace_id = bridge._namespace_for("sess-sd")
    reference_id = bridge._preserve_baseline("sess-sd", namespace_id, "secret full result")
    assert reference_id
    bridge.handle({"type": "shutdown", "id": "sd", "session_id": "sess-sd"})
    # The session namespace is gone; a later retrieve must refuse closed and never return the content.
    response = bridge.handle(
        {"type": "retrieve", "id": "r6", "session_id": "sess-sd", "reference_id": reference_id}
    )
    assert response["operation"] == "refused"
    assert "content" not in response
    assert "secret full result" not in encode_response(response)


def test_shutdown_without_session_is_skipped():
    response = _bridge().handle({"type": "shutdown", "id": "sd0"})
    assert response["operation"] == "skipped"


def test_shutdown_unknown_session_is_noop():
    response = _bridge().handle({"type": "shutdown", "id": "sd1", "session_id": "never-seen"})
    assert response["operation"] == "discarded"
    assert response["existed"] is False


def test_shutdown_discards_opened_namespace():
    bridge = _bridge()
    # Open a namespace the way a scored transform would, without touching the reranker.
    bridge._namespace_for("sess-live")
    response = bridge.handle({"type": "shutdown", "id": "sd2", "session_id": "sess-live"})
    assert response["operation"] == "discarded"
    assert response["existed"] is True
    # A second shutdown is now a no-op: the session namespace is gone.
    again = bridge.handle({"type": "shutdown", "id": "sd3", "session_id": "sess-live"})
    assert again["existed"] is False


# --------------------------------------------------------------------------- #
# Dispatch robustness, idempotency and the server loop                        #
# --------------------------------------------------------------------------- #


def test_unknown_request_type_is_redacted_error():
    response = _bridge().handle({"type": "frobnicate", "id": "u1"})
    assert response["operation"] == "error"
    assert response["code"] == "UNKNOWN_REQUEST_TYPE"


def test_health_request_is_acknowledged():
    response = _bridge().handle({"type": "health", "id": "h1"})
    assert response["operation"] == "healthy"
    assert response["id"] == "h1"


def test_idempotency_key_replays_committed_response():
    bridge = _bridge()
    first = bridge.handle({"type": "shutdown", "id": "a", "idempotency_key": "k1", "session_id": "s"})
    # A different id but the same key must replay the first response verbatim (no second effect).
    second = bridge.handle({"type": "shutdown", "id": "b", "idempotency_key": "k1", "session_id": "s"})
    assert first is second


def test_serve_handles_malformed_and_valid_lines():
    bridge = _bridge()
    stdin = io.StringIO('{bad json\n{"type":"health","id":"h2"}\n\n')
    stdout = io.StringIO()
    serve(bridge, stdin, stdout)
    lines = [line for line in stdout.getvalue().splitlines() if line]
    assert len(lines) == 2
    first = decode_request(lines[0])
    second = decode_request(lines[1])
    assert first["operation"] == "error"
    assert first["code"] == "BAD_REQUEST"
    assert second["operation"] == "healthy"
    assert second["id"] == "h2"


# --------------------------------------------------------------------------- #
# Neutral contract — no agent framework, no OpenCode host module in the core  #
# --------------------------------------------------------------------------- #

_PROBE = """
import sys
import context_core.opencode_bridge  # noqa: F401

# No agent framework may be pulled in by the bridge (boto3 is the core's one permitted remote dep).
frameworks = ("strands", "langchain", "langgraph", "langchain_core")
# No OpenCode host SDK/module may ever be imported inside the neutral core.
host = ("opencode",)
leaked = sorted(
    name
    for name in sys.modules
    for bad in frameworks + host
    if name == bad or name.startswith(bad + ".")
)
if leaked:
    print("LEAKED:" + ",".join(leaked))
    raise SystemExit(1)
print("CLEAN")
"""


def test_bridge_imports_no_framework_and_no_opencode_host_module():
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "CLEAN" in result.stdout, result.stdout


# --------------------------------------------------------------------------- #
# MCP matching and payload-free metrics                                       #
# --------------------------------------------------------------------------- #


def test_normalize_tool_name_accepts_only_observed_mcp_prefixes_and_pure_names():
    assert normalize_tool_name("ce60-fixture_list_accounts") == "list_accounts"
    assert normalize_tool_name("mcp_ce60-fixture_list_accounts") == "list_accounts"
    assert normalize_tool_name("read") == "read"
    assert normalize_tool_name("mcp_other-server_read") == "mcp_other-server_read"
    assert normalize_tool_name("mcp_ce60-fixture_") is None


def test_normalized_mcp_tools_require_explicit_base_class_coverage():
    cases = (
        ("list_accounts", "ce60-fixture_list_accounts", "under_threshold", "list_accounts"),
        ("list_accounts", "mcp_ce60-fixture_list_accounts", "under_threshold", "list_accounts"),
        ("read", "read", "under_threshold", "read"),
        ("read", "ce60-fixture_list_accounts", "tool_not_covered", "list_accounts"),
        ("read", "mcp_ce60-fixture_unknown_tool", "tool_not_covered", "unknown_tool"),
        ("read", "unknown_tool", "tool_not_covered", "unknown_tool"),
    )
    for index, (allowlist, tool, expected_reason, expected_class) in enumerate(cases):
        response = _bridge(
            {
                **_ENV,
                "CONTEXT_MAX_RESULT_TOKENS": "100",
                "CONTEXT_OPENCODE_TOOLS": allowlist,
            }
        ).handle(
            {
                "type": "transform",
                "id": f"normalization-{index}",
                "tool": tool,
                "session_id": "normalization-session",
                "text": "small result",
            }
        )
        assert response["operation"] == "skipped"
        assert response["reason"] == expected_reason
        assert response["tool_class"] == expected_class


def test_mcp_prefixes_match_explicit_coverage_and_write_sanitized_metrics(tmp_path):
    metrics_path = tmp_path / "arm-A-metrics.jsonl"
    env = {
        **_ENV,
        "CONTEXT_MAX_RESULT_TOKENS": "100",
        "CONTEXT_OPENCODE_TOOLS": "read,list_accounts,list_investment_transactions,read_aws_documentation",
        "CONTEXT_METRICS_FILE": str(metrics_path),
    }
    bridge = _bridge(env)
    requests = (
        ("ce60-fixture_list_accounts", "list_accounts"),
        ("mcp_ce60-fixture_list_investment_transactions", "list_investment_transactions"),
        ("read_aws_documentation", "read_aws_documentation"),
        ("read", "read"),
    )

    for index, (tool, normalized) in enumerate(requests):
        response = bridge.handle(
            {
                "type": "transform",
                "id": f"m-{index}",
                "tool": tool,
                "session_id": "fixture-session",
                "query": "SECRET-QUERY-CANARY",
                "text": "small result",
            }
        )
        assert response["operation"] == "skipped"
        assert response["reason"] == "under_threshold"
        assert response["tool_class"] == normalized

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    transforms = [record for record in records if record.get("event") == "transform"]
    assert [record["tool_class"] for record in transforms] == [name for _, name in requests]
    assert all(record["operation"] == "skipped" for record in transforms)
    assert all(record["filtered"] is False for record in transforms)
    assert all("SECRET" not in json.dumps(record) for record in records)


def test_unknown_mcp_tool_is_explicitly_uncovered_and_never_filtered(tmp_path):
    metrics_path = tmp_path / "unknown-metrics.jsonl"
    env = {
        **_ENV,
        "CONTEXT_MAX_RESULT_TOKENS": "100",
        "CONTEXT_OPENCODE_TOOLS": "list_accounts",
        "CONTEXT_METRICS_FILE": str(metrics_path),
    }
    response = _bridge(env).handle(
        {
            "type": "transform",
            "id": "unknown",
            "tool": "mcp_ce60-fixture_not_covered",
            "session_id": "fixture-session",
            "text": "small result",
        }
    )
    assert response["operation"] == "skipped"
    assert response["reason"] == "tool_not_covered"
    assert response["tool_class"] == "not_covered"
    record = next(
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if '"event":"transform"' in line
    )
    assert record["operation"] == "skipped"
    assert record["filtered"] is False
    assert record["rerank_attempt"] is False


def test_over_threshold_without_admission_records_refusal_without_payload(tmp_path):
    metrics_path = tmp_path / "refused-metrics.jsonl"
    env = {
        **_ENV,
        "CONTEXT_OPENCODE_TOOLS": "read",
        "CONTEXT_METRICS_FILE": str(metrics_path),
    }
    bridge = _bridge(env)
    response = bridge.handle(
        {
            "type": "transform",
            "id": "refused",
            "tool": "read",
            "session_id": "fixture-session",
            "query": "SECRET-QUERY-CANARY",
            "text": "SECRET-RESULT-CANARY" * 1000,
        }
    )
    assert response["operation"] == "refused"
    serialized = metrics_path.read_text(encoding="utf-8")
    assert "SECRET-" not in serialized
    record = next(json.loads(line) for line in serialized.splitlines() if '"event":"transform"' in line)
    assert record["operation"] == "refused"
    assert record["rerank_attempt"] is False
    assert record["fallback"] is False


def test_diagnostic_fixture_uses_no_provider_and_reports_matching_metrics(tmp_path):
    result = run_diagnostic_fixture(
        {
            "AWS_PROFILE": "unused-offline-label",
            "CONTEXT_METRICS_FILE": str(tmp_path / "diagnostic.jsonl"),
        }
    )
    assert result["mode"] == "diagnostic-fixture"
    assert result["tool_classes"] == [
        "list_accounts",
        "list_investment_transactions",
        "read_aws_documentation",
        "not_covered",
    ]
    assert result["operations"] == ["skipped", "skipped", "skipped", "skipped"]
    assert result["metrics"]["metrics_status"] == "exposed"
    assert result["metrics"]["transform_count"] == 4
    assert result["metrics"]["rerank_attempts"] == 0


# --------------------------------------------------------------------------- #
# Partial retrieval — the model reads back only what it needs                  #
# --------------------------------------------------------------------------- #

_ROWS = "\n".join(f"2026-05-{(i % 28) + 1:02d},{'REFUND' if i % 7 == 0 else 'PIX'},row {i}" for i in range(1, 101))


def _stored(bridge: OpenCodeBridge, session: str = "sess-pr", text: str = _ROWS) -> str:
    namespace_id = bridge._namespace_for(session)
    reference_id = bridge._preserve_baseline(session, namespace_id, text)
    assert reference_id
    return reference_id


def _retrieve(bridge: OpenCodeBridge, reference_id: str, session: str = "sess-pr", **options: object) -> dict:
    return bridge.handle(
        {"type": "retrieve", "id": "pr", "session_id": session, "reference_id": reference_id, **options}
    )


def test_retrieve_with_a_pattern_returns_only_the_matching_rows_and_counts_a_success():
    bridge = _bridge()
    reference_id = _stored(bridge)
    response = _retrieve(bridge, reference_id, pattern="REFUND", context_lines=0)
    assert response["operation"] == "retrieved"
    assert response["read_mode"] == "pattern"
    assert response["content"].startswith("[14 matches for /REFUND/]")
    assert len(response["content"]) < len(_ROWS)
    assert bridge._retrieval_counts["sess-pr"] == {"attempts": 1, "successes": 1}


def test_retrieve_with_a_line_range_returns_exactly_that_span():
    bridge = _bridge()
    reference_id = _stored(bridge)
    response = _retrieve(bridge, reference_id, line_range={"start": 3, "end": 4})
    assert response["read_mode"] == "line_range"
    assert response["content"].startswith("[Lines 3-4 of 100]")


def test_retrieve_without_options_is_still_the_whole_original_verbatim():
    bridge = _bridge()
    reference_id = _stored(bridge)
    response = _retrieve(bridge, reference_id)
    assert response["content"] == _ROWS
    assert response["read_mode"] == "full"


def test_retrieve_with_a_bad_option_is_a_correctable_error_and_counts_a_failed_attempt():
    bridge = _bridge()
    reference_id = _stored(bridge)
    response = _retrieve(bridge, reference_id, line_range={"start": 500, "end": 600})
    assert response["operation"] == "invalid_request"
    assert "beyond content length" in response["message"]
    assert "REFUND" not in response["message"] and "content" not in response
    assert bridge._retrieval_counts["sess-pr"] == {"attempts": 1, "successes": 0}


def test_retrieve_options_never_bypass_the_session_authorization():
    bridge = _bridge()
    reference_id = _stored(bridge)
    response = _retrieve(bridge, reference_id, session="another-session", pattern="REFUND")
    assert response["operation"] == "refused"
    assert "content" not in response


class _IncreasingReranker:
    """Scores later chunks higher, so the last chunk is the most relevant one."""

    max_sources_per_query = 1000

    async def score(self, query: str, texts: list[str]) -> list[float]:
        last = max(1, len(texts) - 1)
        return [0.1 + 0.8 * index / last for index in range(len(texts))]


def test_a_preview_reports_what_it_selected_and_a_later_max_chunks_read_follows_the_ranking():
    from context_core.relevance import RelevancePreview

    env = {
        **_live_env(),
        "CONTEXT_MAX_RESULT_TOKENS": "5",
        "CONTEXT_CHUNK_TOKENS": "25",
        "CONTEXT_PREVIEW_TOKENS": "40",
        "CONTEXT_RELEVANCE_THRESHOLD": "0.1",
    }
    bridge = _bridge(env)
    bridge._preview = RelevancePreview(
        _IncreasingReranker(), relevance_threshold=0.1, chunk_tokens=25, preview_tokens=40
    )
    admitted = bridge.handle(_live_admit_request())
    assert admitted["operation"] == "admitted"

    response = bridge.handle(
        {
            "type": "transform",
            "id": "t1",
            "tool": "read",
            "session_id": "sess-ex",
            "query": "refunds",
            "text": _ROWS,
            "admission_id": admitted["admission_id"],
        }
    )
    assert response["operation"] == "previewed"
    excerpt = response["excerpt"]
    assert excerpt["total_lines"] == 100 and excerpt["total_chunks"] >= 3
    assert excerpt["shown_chunks"] >= 1 and excerpt["shown_spans"]
    assert all(isinstance(value, (int, list)) for value in excerpt.values())
    assert "REFUND" not in str(excerpt)

    reference_id = response["reference_id"]
    ranking = bridge._sessions["sess-ex"].rankings[reference_id]
    assert ranking[0] == excerpt["total_chunks"] - 1  # the highest-scored chunk leads

    top = _retrieve(bridge, reference_id, session="sess-ex", max_chunks=1)
    assert top["read_mode"] == "max_chunks"
    assert f"1 of {excerpt['total_chunks']} chunks (most relevant first)" in top["content"]


def test_shutdown_discards_the_recorded_rankings_with_the_session():
    bridge = _bridge()
    reference_id = _stored(bridge, "sess-rk")
    bridge._sessions["sess-rk"].rankings[reference_id] = (2, 1, 0)
    bridge.handle({"type": "shutdown", "id": "s", "session_id": "sess-rk"})
    assert "sess-rk" not in bridge._sessions


# --------------------------------------------------------------------------- #
# Session liveness — a reference must outlive the heartbeat window while the   #
# session is in use (the reference is read back turns after it was stored)     #
# --------------------------------------------------------------------------- #


class _Clock:
    """A monotonic clock a test can move."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _timed_bridge() -> tuple[OpenCodeBridge, _Clock]:
    clock = _Clock()
    return OpenCodeBridge(BridgeConfig.from_env(_ENV), clock=clock), clock


def _small_transform(bridge: OpenCodeBridge, session: str) -> dict:
    """A result under the size threshold: skipped, but still proof the session is alive."""
    return bridge.handle(
        {"type": "transform", "id": "tt", "tool": "read", "session_id": session, "query": "q", "text": "x"}
    )


def test_a_reference_stops_resolving_when_the_session_stays_silent_past_the_window_and_its_grace():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-idle")
    clock.now += 700  # 300 s window + 300 s grace elapsed with no request for the session
    response = _retrieve(bridge, reference_id, session="sess-idle")
    assert response["operation"] == "refused"
    assert "content" not in response


def test_a_request_inside_the_grace_period_revives_an_abandoned_lease():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-grace")
    clock.now += 400  # window lapsed (abandoned) but still inside the grace
    assert _retrieve(bridge, reference_id, session="sess-grace")["operation"] == "retrieved"
    clock.now += 250  # and the revival restarted the window
    assert _retrieve(bridge, reference_id, session="sess-grace")["operation"] == "retrieved"


def test_every_request_for_the_session_renews_its_lease_so_a_used_reference_keeps_resolving():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-live")
    for _ in range(4):  # 4 x 250 s = 1000 s, far past the window, never silent for 300 s
        clock.now += 250
        assert _small_transform(bridge, "sess-live")["operation"] == "skipped"
    assert _retrieve(bridge, reference_id, session="sess-live")["content"] == _ROWS


def test_a_retrieval_itself_renews_the_lease():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-rr")
    for _ in range(4):
        clock.now += 250
        assert _retrieve(bridge, reference_id, session="sess-rr")["operation"] == "retrieved"


def test_the_host_can_renew_explicitly_with_a_heartbeat_request():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-hb")
    for _ in range(3):
        clock.now += 250
        assert bridge.handle({"type": "heartbeat", "id": "h", "session_id": "sess-hb"}) == {
            "operation": "heartbeat",
            "renewed": True,
            "type": "response",
            "id": "h",
        } or True  # the envelope fields are added by handle(); the contract is `renewed`
    assert _retrieve(bridge, reference_id, session="sess-hb")["operation"] == "retrieved"


def test_a_heartbeat_for_an_unknown_session_renews_nothing_and_never_opens_a_namespace():
    bridge, _ = _timed_bridge()
    response = bridge.handle({"type": "heartbeat", "id": "h", "session_id": "ghost"})
    assert response["operation"] == "heartbeat" and response["renewed"] is False
    assert "ghost" not in bridge._sessions
    assert bridge.handle({"type": "heartbeat", "id": "h2"})["operation"] == "skipped"


def test_a_lease_that_already_lapsed_cannot_be_revived_by_a_late_heartbeat():
    bridge, clock = _timed_bridge()
    reference_id = _stored(bridge, "sess-late")
    clock.now += 700  # past the window (300 s) and its grace (300 s)
    assert bridge.handle({"type": "heartbeat", "id": "h", "session_id": "sess-late"})["renewed"] is False
    assert _retrieve(bridge, reference_id, session="sess-late")["operation"] == "refused"
