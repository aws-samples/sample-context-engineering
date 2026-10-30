#!/usr/bin/env bash
# CE-60 OpenCode campaign v6: plugin revision 2 (fidelity changes plus the two fixes v5 exposed). Billable.
#
# Why v6. Campaign v4 made the verdict trustworthy and was still `no_go` on model input in every repetition: the
# model fetched every preview back whole and the port kept that retrieval in the session. Reading the reference
# Strands filter showed what the port lacked: a retrieval cleanup, partial reads, and an excerpt notice that says
# what an excerpt cannot answer. Campaign v5 ran the first implementation of that (plugin revision 1) and its
# R1 was a valid run: median model input -19.1 % (143,230 -> 115,865), correctness 30/30 vs 30/30, narrow reads
# (6.9 KB instead of 33.7 KB). It also exposed two defects of revision 1, both fixed here (revision 2):
#   1. The bridge never renewed a session's retention lease, so references stopped resolving 300 s after
#      the first preview (last good retrieval +299 s, first failure +300 s) and Gate A reported
#      GATE_RETRIEVAL_INVALID. Now every request for a session renews its lease and the plugin sends a
#      periodic heartbeat.
#   2. Deleting the retrieval exchange from the history left earlier answers quoting figures with no trace of
#      where they came from; the model said it had "claimed to have read content I never retrieved" and re-ran the
#      same retrieval every turn (23 retrievals for 2 previews, also in unrelated turns). Now the result is CLEARED
#      with OpenCode's own marker, "[Old tool result content cleared]", and the call stays visible.
# A 21-turn probe of revision 2 (which spans the 300 s mark) preceded this campaign; its result is in
# results/ce60-v6-probe-a1/.
#
# Declared change against v5 R1 (everything else is identical to v4/v5: harness, oracle v4, pinned docs, English
# pin, tool policy, models, 60 turns, Gate A limits): plugin revision 2. Every repetition records
# `plugin_rev:2` with the switches in its evidence and is rejected if arm A did not run with exactly
# `preview_style:scoped,retrieval_cleanup:on,plugin_rev:2`. The baseline arm loads no plugin.
#
# Pre-registered reading (written before any v6 repetition ran): the plugin is judged by the unchanged Gate A.
# A repetition is a `go` only if the median model input falls by at least 20 % with correctness within 3.0 pp
# and retrieval exact. The hypothesis (revision 2 moves the median down enough, with exact retrieval) is counted
# as supported only if at least 2 of 3 repetitions are `go`, and is reported as it comes out otherwise. v5 R1
# (-19.1 %) is not counted: it ran a different revision.
#
# Stages (fail closed: stop at the first stage that is not clean):
#   0. Preflight: docs cache, offline suites (Bun, Gate A, MCP, context-core), real-OpenCode write-back probe.
#   1. Sonnet 5.5 R1-R3, full 60-turn paired runs. (Opus only on request: `opus` or `all`.)
#
# A repetition is accepted only if ALL hold: both arms 60/60, zero timeouts, Gate A go/no_go (never
# not_evaluable); arm A's persisted preview proven; oracle v4 on both arms; English answers; every arm-A turn has a
# retrieval count; the documented plugin switches; docs cache unchanged; only a retryable provider error is retried
# (once).
#
# Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v6.sh [sonnet|opus|all]   (default: sonnet)
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v6-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
APPROVAL="user-approved-ce60-v6-5-5-2026"
PLAN="validation/opencode_gate_a/stat-plan-v6-5-5-3-reps.json"
POLICY="question,webfetch"
LANGUAGE_PIN="Always answer in English, even when tool results or data are in another language."
PLUGIN_OPTIONS="preview_style:scoped,retrieval_cleanup:on,plugin_rev:2"
WHICH="${1:-sonnet}"
DOCS_STAGING="validation/opencode_gate_a/fixtures/docs-v4"
DOCS_CACHE="validation/community-plugin-A-B-D/.cache"
CAMPAIGNS="validation/opencode_gate_a/campaigns"
CHECK="$CAMPAIGNS/check_repetition.py"
DOCS="$CAMPAIGNS/docs_cache.py"

export AWS_PROFILE=claude-code
export AWS_REGION=us-west-2
export PYTHON=python3.13
export CONTEXT_BRIDGE_COMMAND=python3.13
export PYTHONPATH="$ROOT/context-core/src:$ROOT"
export CONTEXT_RERANK_MODEL_ID=amazon.rerank-v1:0
export CONTEXT_OPENCODE_TOOLS=list_accounts,list_investment_positions,list_investment_transactions,get_portfolio_allocation,project_yield,get_connector_status,read_connector_logs,force_connector_sync,rotate_connector_credentials,open_support_case,read_aws_documentation,search_aws_documentation,describe_lambda_function,list_s3_objects,query_dynamodb_table,get_cloudwatch_metrics,describe_iam_role,estimate_aws_cost
export CE60_ORACLE_VERSION=v4
export CONTEXT_PREVIEW_STYLE=scoped
export CONTEXT_RETRIEVAL_CLEANUP=on

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

run_ce60() {
  local run_id="$1" out="$2" model="$3"
  mkdir -p "$out"
  bun .opencode/runner/ce60.ts --execute-live --arm both --model "$model" --approval-id "$APPROVAL" \
    --budget-ceiling 100 --budget-unit USD --stat-plan "$PLAN" --run-id "$run_id" --output-dir "$out" \
    --prompt-timeout-ms 600000 --disable-tools "$POLICY" --system-prompt "$LANGUAGE_PIN" >"$out/run.log" 2>&1
}

check_full() { python3.13 "$CHECK" "$1"; }
verify_plugin_effect() { python3.13 "$CHECK" effect "$1"; }
verify_docs() { python3.13 "$DOCS" verify --dir "$DOCS_CACHE" --manifest "$DOCS_STAGING/MANIFEST.json" 2>&1; }

# stage <label> <run_id> <model>
stage() {
  local label="$1" run_id="$2" model="$3"
  local attempt out summary rc effect v4 docs outcomes
  for attempt in 1 2; do
    out="$RESULTS/$run_id"
    [ "$attempt" -eq 2 ] && { out="$RESULTS/$run_id-retry"; run_id="$run_id-retry"; }
    log "START $label (attempt $attempt)"
    if ! docs="$(verify_docs)"; then
      log "STOP $label documentation cache no longer matches its pinned digests: $docs"; exit 1
    fi
    run_ce60 "$run_id" "$out" "$model"
    rc=$?
    summary="$(check_full "$out" 2>&1)"
    local ok=$?
    if [ "$rc" -eq 0 ] && [ "$ok" -eq 0 ]; then
      effect="$(verify_plugin_effect "$run_id" 2>&1)"
      if [ $? -ne 0 ]; then
        log "STOP $label plugin effect NOT verified: $effect | $summary"; exit 1
      fi
      v4="$(python3.13 "$CHECK" v4 "$out" "$run_id" "$PLUGIN_OPTIONS" 2>&1)"
      if [ $? -ne 0 ]; then
        log "STOP $label v6 conditions not met: $v4 | $summary"; exit 1
      fi
      if ! docs="$(verify_docs)"; then
        log "STOP $label documentation cache changed during the repetition: $docs"; exit 1
      fi
      cp "$DOCS_STAGING/MANIFEST.json" "$out/docs-manifest.json"
      python3.13 "$CAMPAIGNS/analyze_matched_calls.py" "$run_id" >"$out/matched-calls.json" 2>/dev/null || true
      python3.13 "$CAMPAIGNS/preview_outcomes.py" "$run_id" "$out" >"$out/preview-outcomes.json" 2>/dev/null || true
      outcomes="$(python3.13 "$CAMPAIGNS/preview_outcomes.py" "$run_id" "$out" --summary 2>&1 || true)"
      log "DONE $label $summary | $effect | $v4 | preview-outcomes: $outcomes"
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

log "CAMPAIGN v6 start (scope=$WHICH policy=$POLICY plan=$PLAN oracle=$CE60_ORACLE_VERSION language=en-pinned plugin=$PLUGIN_OPTIONS)"

log "PREFLIGHT documentation cache: install pinned pages, verify, serve through the MCP server"
if ! out="$(python3.13 "$DOCS" install --from "$DOCS_STAGING" --to "$DOCS_CACHE" 2>&1)"; then
  log "STOP preflight: could not install the pinned documentation pages: $out"; exit 1
fi
if ! out="$(verify_docs)"; then log "STOP preflight: documentation cache invalid: $out"; exit 1; fi
if ! out="$(python3.13 "$DOCS" probe 2>&1)"; then log "STOP preflight: MCP server does not serve real docs: $out"; exit 1; fi

log "PREFLIGHT offline test suites"
if ! (cd .opencode && bun test >/dev/null 2>&1); then log "STOP preflight: bun test failed"; exit 1; fi
if ! (cd validation && python3.13 -m pytest opencode_gate_a test_opencode_mcp_server.py -q >/dev/null 2>&1); then
  log "STOP preflight: Gate A / MCP tests failed"; exit 1
fi
if ! context-core/.venv/bin/python -m pytest context-core/tests -q >/dev/null 2>&1; then
  log "STOP preflight: context-core tests failed"; exit 1
fi
log "PREFLIGHT real-OpenCode write-back probe (Haiku 5.5)"
if ! bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5 >"$RESULTS/probe-writeback-v6-preflight.json" 2>/dev/null; then
  log "STOP preflight: write-back probe failed"; exit 1
fi
log "PREFLIGHT ok"

if [ "$WHICH" = "sonnet" ] || [ "$WHICH" = "all" ]; then
  stage "sonnet55 R1" ce60-v6-sonnet55-r1 "$SONNET"
  stage "sonnet55 R2" ce60-v6-sonnet55-r2 "$SONNET"
  stage "sonnet55 R3" ce60-v6-sonnet55-r3 "$SONNET"
fi
if [ "$WHICH" = "opus" ] || [ "$WHICH" = "all" ]; then
  stage "opus55 R1" ce60-v6-opus55-r1 "$OPUS"
  stage "opus55 R2" ce60-v6-opus55-r2 "$OPUS"
  stage "opus55 R3" ce60-v6-opus55-r3 "$OPUS"
fi

log "CAMPAIGN v6 complete (scope=$WHICH)"
