#!/usr/bin/env bash
# CE-60 OpenCode campaign v7: practice D (context graph) on top of A, plugin revision 2. Billable.
#
# This is NOT a Gate A campaign. Gate A judges practice A alone and it was decided in v3-v6 (1 of 3 repetitions `go` in v6:
# the hypothesis was not supported). The owner asked for practice D to be implemented and measured on OpenCode anyway and
# waived the requirement that D wait for a Gate A `go` (FR5.1 / BR5.1). That waiver is recorded here and in the results; it
# makes D `experimental`, never `supported`, and it does not change Gate A, which stays `no_go`.
#
# What is measured. Arm A runs with CONTEXT_GRAPH=on (the context graph on top of the relevance filter), against the
# baseline arm (no plugin) of the SAME repetition, on the same 60-turn scenario, model, oracle (v4), tool policy and
# English pin as v6. The only declared change against v6 is CONTEXT_GRAPH=on, with the benchmark's GRAPH_TUNING (expand 0.62,
# collapse 0.45, link 0.50, 100-token descriptions, 40,000-token body budget, 8 retrievals per turn, reuse 5, embedding
# model cohere.embed-multilingual-v3). The graph is request-only: it leaves messages out of one model request and appends one
# summary block; it never writes to the session. Every repetition records `graph:on` in `plugin_options` and a per-request
# evidence file; the run is rejected as D evidence unless the stored session reproduces what every request was handed.
#
# No Gate A decision is written for these runs: the paired comparison goes to `d-slice-comparison.json`.
#
# Why a measurement and not a promise. A replay of the recorded v6 R1 arm-A session through the binding (before this
# campaign: `python3.13 -m opencode_gate_d.replay`, chars/4, input side only, no output tokens) said D would remove about
# 72 % of the transcript the model reads and, because the projection changes early messages and so invalidates the provider's
# prompt cache, raise the input-side cost by about 74 % (cache writes 8.7x the baseline's). The Strands benchmark shows the
# same shape with caching on (-39..-46 % tokens, +35..+66 % cost). This campaign checks that prediction against the real bill.
#
# Pre-registered reading (written before any v7 repetition ran). Three measures, each per repetition and as the mean of the
# paired differences (A+D against the baseline of the same repetition), reported next to the v6 figures for A alone (the mean
# of R1-R3 of v6: total model input -13.9 %, dollars -11.9 %, correctness parity):
#   T  total model-input tokens, input + cache read + cache write, summed over the 60 turns;
#   C  dollars, from OpenCode's own message `cost` counter (the provider price list applied to input, cache read, cache
#      write and output tokens);
#   K  correct turns on the 30 eligible, oracle v4: A+D minus baseline.
# Labels, fixed now and applied mechanically to the three means:
#   D_USEFUL        mean C is at least 5 points better than A alone in v6 (mean C <= -16.9 %) AND the minimum K over the
#                   repetitions is >= -1;
#   D_TOKENS_ONLY   not D_USEFUL, and mean T is at least 5 points better than A alone in v6 (mean T <= -18.9 %): the tokens
#                   fall and the bill does not follow;
#   D_NOT_WORTH_IT  neither.
# Independently, D_COSTS_CORRECTNESS is added when the minimum K is < -1. The label never changes what D is: `experimental`
# under the owner waiver. A repetition that fails the acceptance below is not counted and is reported as it came out.
#
# Stages (fail closed: stop at the first stage that is not clean):
#   0. Preflight: docs cache, offline suites (Bun, Gate A, Gate D, MCP, context-core), real-OpenCode write-back probe.
#   1. Sonnet 5.5 R1-R3, full 60-turn paired runs. (Opus only on request: `opus` or `all`.)
#
# A repetition is accepted only if ALL hold: both arms 60/60, zero timeouts, a paired comparison that is go/no_go (never
# not_evaluable); arm A's persisted preview proven; oracle v4 on both arms; English answers; every arm-A turn has a retrieval
# count; the documented plugin switches including `graph:on`; the graph engaged and was request-only (`check_repetition.py
# graph`); docs cache unchanged; only a retryable provider error is retried (once).
#
# Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v7.sh [sonnet|opus|all]   (default: sonnet)
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v7-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
APPROVAL="user-approved-ce60-v7-5-5-2026"
PLAN="validation/opencode_gate_a/stat-plan-v7-5-5-3-reps.json"
POLICY="question,webfetch"
LANGUAGE_PIN="Always answer in English, even when tool results or data are in another language."
PLUGIN_OPTIONS="preview_style:scoped,retrieval_cleanup:on,plugin_rev:2,graph:on"
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
# Practice D, on top of A. The comparison file is named so it cannot be read as a Gate A decision.
export CONTEXT_GRAPH=on
export CE60_DECISION_FILE=d-slice-comparison.json

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
  local attempt out summary rc effect v4 graph docs outcomes
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
        log "STOP $label v7 conditions not met: $v4 | $summary"; exit 1
      fi
      graph="$(python3.13 "$CHECK" graph "$out" 2>&1)"
      if [ $? -ne 0 ]; then
        log "STOP $label practice D was not valid evidence: $graph | $summary"; exit 1
      fi
      if ! docs="$(verify_docs)"; then
        log "STOP $label documentation cache changed during the repetition: $docs"; exit 1
      fi
      cp "$DOCS_STAGING/MANIFEST.json" "$out/docs-manifest.json"
      python3.13 "$CAMPAIGNS/analyze_matched_calls.py" "$run_id" >"$out/matched-calls.json" 2>/dev/null || true
      python3.13 "$CAMPAIGNS/preview_outcomes.py" "$run_id" "$out" >"$out/preview-outcomes.json" 2>/dev/null || true
      outcomes="$(python3.13 "$CAMPAIGNS/preview_outcomes.py" "$run_id" "$out" --summary 2>&1 || true)"
      log "DONE $label $summary | $effect | $v4 | $graph | preview-outcomes: $outcomes"
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

log "CAMPAIGN v7 start (scope=$WHICH policy=$POLICY plan=$PLAN oracle=$CE60_ORACLE_VERSION language=en-pinned plugin=$PLUGIN_OPTIONS graph_waiver=owner)"

log "PREFLIGHT documentation cache: install pinned pages, verify, serve through the MCP server"
if ! out="$(python3.13 "$DOCS" install --from "$DOCS_STAGING" --to "$DOCS_CACHE" 2>&1)"; then
  log "STOP preflight: could not install the pinned documentation pages: $out"; exit 1
fi
if ! out="$(verify_docs)"; then log "STOP preflight: documentation cache invalid: $out"; exit 1; fi
if ! out="$(python3.13 "$DOCS" probe 2>&1)"; then log "STOP preflight: MCP server does not serve real docs: $out"; exit 1; fi

# The suites run in a clean environment: the campaign's own switches (CONTEXT_GRAPH, CE60_DECISION_FILE) must not leak into
# tests that assert the defaults.
log "PREFLIGHT offline test suites"
if ! (cd .opencode && env -u CONTEXT_GRAPH -u CE60_DECISION_FILE bun test >/dev/null 2>&1); then log "STOP preflight: bun test failed"; exit 1; fi
if ! (cd validation && env -u CONTEXT_GRAPH -u CE60_DECISION_FILE python3.13 -m pytest opencode_gate_a opencode_gate_d test_opencode_mcp_server.py -q >/dev/null 2>&1); then
  log "STOP preflight: Gate A / Gate D / MCP tests failed"; exit 1
fi
if ! env -u CONTEXT_GRAPH -u CE60_DECISION_FILE context-core/.venv/bin/python -m pytest context-core/tests -q >/dev/null 2>&1; then
  log "STOP preflight: context-core tests failed"; exit 1
fi
log "PREFLIGHT real-OpenCode write-back probe (Haiku 5.5)"
if ! bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5 >"$RESULTS/probe-writeback-v7-preflight.json" 2>/dev/null; then
  log "STOP preflight: write-back probe failed"; exit 1
fi
log "PREFLIGHT ok"

if [ "$WHICH" = "sonnet" ] || [ "$WHICH" = "all" ]; then
  stage "sonnet55 R1" ce60-v7-sonnet55-r1 "$SONNET"
  stage "sonnet55 R2" ce60-v7-sonnet55-r2 "$SONNET"
  stage "sonnet55 R3" ce60-v7-sonnet55-r3 "$SONNET"
fi
if [ "$WHICH" = "opus" ] || [ "$WHICH" = "all" ]; then
  stage "opus55 R1" ce60-v7-opus55-r1 "$OPUS"
  stage "opus55 R2" ce60-v7-opus55-r2 "$OPUS"
  stage "opus55 R3" ce60-v7-opus55-r3 "$OPUS"
fi

log "CAMPAIGN v7 complete (scope=$WHICH)"
