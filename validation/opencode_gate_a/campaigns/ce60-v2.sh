#!/usr/bin/env bash
# CE-60 OpenCode campaign v2 (F002 fix). Real Bedrock calls: billable.
#
#   1. Variant stage: question tool disabled (the only change), baseline + A, truncated at F002-filler.
#   2. Sonnet 5.5 R1-R3, full 60-turn paired runs, declared policy: question + webfetch disabled.
#   3. Opus 5.5 R1-R3, same policy, only after all three Sonnet repetitions are clean.
#
# Fail closed: the campaign stops at the first stage that is not clean (timeout, incomplete run,
# Gate A not_evaluable, runner error). Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v2.sh
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v2-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
V2_APPROVAL="user-approved-ce60-v2-5-5-2026"
V2_PLAN="validation/opencode_gate_a/stat-plan-v2-5-5-3-reps.json"
V2_POLICY="question,webfetch"

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

# run_ce60 <run_id> <out_dir> <model> <approval_id> <stat_plan> [extra runner flags...]
run_ce60() {
  local run_id="$1" out="$2" model="$3" approval="$4" plan="$5"
  shift 5
  mkdir -p "$out"
  bun .opencode/runner/ce60.ts --execute-live --arm both --model "$model" --approval-id "$approval" \
    --budget-ceiling 100 --budget-unit USD --stat-plan "$plan" --run-id "$run_id" --output-dir "$out" \
    --prompt-timeout-ms 600000 "$@" >"$out/run.log" 2>&1
}

# A truncated diagnostic is clean when both arms reached F002 (19 turns) with zero timeouts.
check_diagnostic() {
  python3.13 -c '
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ok, parts = True, []
for arm in ("baseline", "A"):
    data = json.loads((out / f"{arm}-diagnostic.json").read_text())
    obs = data["observability"]
    sent, timeouts, classes = data["turns_sent"], obs["timeout_count"], obs["timeout_classifications"]
    parts.append(f"{arm}: sent={sent} timeouts={timeouts} classes={classes}")
    ok = ok and sent == 19 and timeouts == 0
print("; ".join(parts))
sys.exit(0 if ok else 1)
' "$1"
}

# A full repetition is clean when Gate A is evaluable and both arms completed 60/60 with no timeout.
check_full() {
  python3.13 -c '
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
gate = json.loads((out / "gate-a-decision.json").read_text())
outcome, reasons = gate["outcome"], ",".join(gate["reasons"]) or "-"
ok = outcome in ("go", "no_go")
parts = [f"gate={outcome} reasons={reasons}"]
for arm in ("baseline", "A"):
    ev = json.loads((out / f"{arm}-evidence.json").read_text())
    obs = ev["host_evidence"].get("observability", {})
    done, num, den = ev["turns_completed"], ev["correctness_numerator"], ev["correctness_denominator"]
    mi, timeouts = ev["model_input_tokens_median"], obs.get("timeout_count")
    parts.append(f"{arm}: completed={done} correct={num}/{den} mi_median={mi} timeouts={timeouts}")
    ok = ok and done == 60 and timeouts == 0
print("; ".join(parts))
sys.exit(0 if ok else 1)
' "$1"
}

# stage_full <label> <run_id> <model>
stage_full() {
  local label="$1" run_id="$2" model="$3"
  local out="$RESULTS/$run_id"
  log "START $label"
  run_ce60 "$run_id" "$out" "$model" "$V2_APPROVAL" "$V2_PLAN" --disable-tools "$V2_POLICY"
  local rc=$?
  local summary
  summary="$(check_full "$out" 2>&1)"
  local ok=$?
  if [ "$rc" -ne 0 ] || [ "$ok" -ne 0 ]; then
    log "STOP $label rc=$rc $summary"
    exit 1
  fi
  log "DONE $label $summary"
}

log "CAMPAIGN v2 start (policy=$V2_POLICY, stat plan=$V2_PLAN)"

variant_out="$RESULTS/ce60-f002-variant-question-off"
log "START variant question-off truncated at F002-filler"
run_ce60 ce60-f002-variant-question-off "$variant_out" "$SONNET" user-approved-ce60-f002-diagnostic-2026 \
  validation/opencode_gate_a/stat-plan-mcp-5-5-3-reps.json --diagnostic-stop-after F002-filler --disable-tools question
rc=$?
summary="$(check_diagnostic "$variant_out" 2>&1)"
ok=$?
if [ "$rc" -ne 0 ] || [ "$ok" -ne 0 ]; then
  log "STOP variant rc=$rc $summary"
  exit 1
fi
log "DONE variant $summary"

stage_full "sonnet55 R1" ce60-v2-sonnet55-r1 "$SONNET"
stage_full "sonnet55 R2" ce60-v2-sonnet55-r2 "$SONNET"
stage_full "sonnet55 R3" ce60-v2-sonnet55-r3 "$SONNET"
stage_full "opus55 R1" ce60-v2-opus55-r1 "$OPUS"
stage_full "opus55 R2" ce60-v2-opus55-r2 "$OPUS"
stage_full "opus55 R3" ce60-v2-opus55-r3 "$OPUS"

log "CAMPAIGN v2 complete"
