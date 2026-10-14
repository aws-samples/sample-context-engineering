#!/usr/bin/env bash
# CE-60 OpenCode campaign v3: corrected MCP write-back. Real Bedrock calls: billable.
#
# Why v3: in v2 the plugin's preview never reached the model for MCP tools (it was written to
# `output.output`, which OpenCode ignores for MCP results). v2 arm-A numbers are void for practice A.
#
# Stages (fail closed: stop at the first stage that is not clean):
#   0. Preflight: the offline write-back tests and the real-OpenCode write-back probe must pass.
#   1. Sonnet 5.5 R1-R3, full 60-turn paired runs, declared policy: question + webfetch disabled.
#   2. Opus 5.5 R1-R3, same policy, only after all three Sonnet repetitions are clean.
#
# A repetition is accepted only if ALL hold:
#   - both arms completed 60/60 with zero timeouts and the Gate A decision is go/no_go (never not_evaluable);
#   - arm A's persisted preview is PROVEN to have replaced the original (plugin effect verified), and at
#     least one oversized result was previewed;
#   - no provider error is left unrecovered. A repetition whose only failure is a retryable provider
#     error (HTTP 5xx / isRetryable) is repeated ONCE; anything else stops the campaign.
#
# Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v3.sh [sonnet|opus|all]
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v3-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
APPROVAL="user-approved-ce60-v3-5-5-2026"
PLAN="validation/opencode_gate_a/stat-plan-v3-5-5-3-reps.json"
POLICY="question,webfetch"
WHICH="${1:-all}"

export AWS_PROFILE=claude-code
export AWS_REGION=us-west-2
export PYTHON=python3.13
export CONTEXT_BRIDGE_COMMAND=python3.13
export PYTHONPATH="$ROOT/context-core/src:$ROOT"
export CONTEXT_RERANK_MODEL_ID=amazon.rerank-v1:0
export CONTEXT_OPENCODE_TOOLS=list_accounts,list_investment_positions,list_investment_transactions,get_portfolio_allocation,project_yield,get_connector_status,read_connector_logs,force_connector_sync,rotate_connector_credentials,open_support_case,read_aws_documentation,search_aws_documentation,describe_lambda_function,list_s3_objects,query_dynamodb_table,get_cloudwatch_metrics,describe_iam_role,estimate_aws_cost

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

run_ce60() {
  local run_id="$1" out="$2" model="$3"
  mkdir -p "$out"
  bun .opencode/runner/ce60.ts --execute-live --arm both --model "$model" --approval-id "$APPROVAL" \
    --budget-ceiling 100 --budget-unit USD --stat-plan "$PLAN" --run-id "$run_id" --output-dir "$out" \
    --prompt-timeout-ms 600000 --disable-tools "$POLICY" >"$out/run.log" 2>&1
}

CHECK="validation/opencode_gate_a/campaigns/check_repetition.py"

# Prints a one-line summary and exits 0 (clean), 1 (hard failure) or 3 (only a retryable provider error).
check_full() {
  python3.13 "$CHECK" "$1"
}

# The plugin effect is verified from the persisted OpenCode session, never from bridge metrics alone.
verify_plugin_effect() {
  python3.13 "$CHECK" effect "$1"
}

# stage <label> <run_id> <model>
stage() {
  local label="$1" run_id="$2" model="$3"
  local attempt out summary rc effect
  for attempt in 1 2; do
    out="$RESULTS/$run_id"
    [ "$attempt" -eq 2 ] && { out="$RESULTS/$run_id-retry"; run_id="$run_id-retry"; }
    log "START $label (attempt $attempt)"
    run_ce60 "$run_id" "$out" "$model"
    rc=$?
    summary="$(check_full "$out" 2>&1)"
    local ok=$?
    if [ "$rc" -eq 0 ] && [ "$ok" -eq 0 ]; then
      effect="$(verify_plugin_effect "$run_id" 2>&1)"
      if [ $? -ne 0 ]; then
        log "STOP $label plugin effect NOT verified: $effect | $summary"
        exit 1
      fi
      log "DONE $label $summary | $effect"
      return 0
    fi
    if [ "$ok" -eq 3 ] && [ "$attempt" -eq 1 ]; then
      log "RETRY $label only a retryable provider error: $summary"
      continue
    fi
    log "STOP $label rc=$rc check=$ok $summary"
    exit 1
  done
}

log "CAMPAIGN v3 start (scope=$WHICH policy=$POLICY plan=$PLAN)"

log "PREFLIGHT offline write-back tests"
if ! (cd .opencode && bun test plugins/context-engineering.test.ts >/dev/null 2>&1); then
  log "STOP preflight: offline write-back tests failed"; exit 1
fi
log "PREFLIGHT real-OpenCode write-back probe (Haiku 5.5)"
if ! bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5 >"$RESULTS/probe-writeback-v3-preflight.json" 2>/dev/null; then
  log "STOP preflight: write-back probe failed"; exit 1
fi
log "PREFLIGHT ok"

if [ "$WHICH" = "sonnet" ] || [ "$WHICH" = "all" ]; then
  stage "sonnet55 R1" ce60-v3-sonnet55-r1 "$SONNET"
  stage "sonnet55 R2" ce60-v3-sonnet55-r2 "$SONNET"
  stage "sonnet55 R3" ce60-v3-sonnet55-r3 "$SONNET"
fi
if [ "$WHICH" = "opus" ] || [ "$WHICH" = "all" ]; then
  stage "opus55 R1" ce60-v3-opus55-r1 "$OPUS"
  stage "opus55 R2" ce60-v3-opus55-r2 "$OPUS"
  stage "opus55 R3" ce60-v3-opus55-r3 "$OPUS"
fi

log "CAMPAIGN v3 complete (scope=$WHICH)"
