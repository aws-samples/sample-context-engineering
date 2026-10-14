#!/usr/bin/env bash
# CE-60 OpenCode campaign v9: practice D, in-place policy, CACHE-ALIGNED configuration, on top of A, plugin revision 2. Billable.
#
# This is NOT a Gate A campaign. Gate A judges practice A alone and it was decided in v3-v6 (1 of 3 repetitions `go` in v6:
# the hypothesis was not supported). The owner asked for practice D to be implemented and measured on OpenCode anyway and
# waived the requirement that D wait for a Gate A `go` (FR5.1 / BR5.1). That waiver is recorded here and in the results; it
# makes D `experimental`, never `supported`, and it does not change Gate A, which stays `no_go`. The in-place policy is a
# variant of D (closer to the "context curator" idea the design folded into D), reported as such and never as the reference
# algorithm.
#
# Approval. The owner approved the in-place plan ("implement the mode, probe it, run 3 paired repetitions") and campaign v8
# was that run. This campaign is the same mode and the same plan with a corrected configuration (one environment variable),
# run under the same approval id and stat-plan content; the owner was not asked about this extra campaign.
#
# What v7 and v8 measured, Sonnet 5.5, plugin rev 2, A+D against the baseline of the same repetition, 3 repetitions each, K =
# correct turns on the 30 eligible (oracle v4), A+D minus baseline:
#   v7 reference policy      tokens -48.9 %   dollars +215.7 % (+206.5 / +221.5 / +219.0)   K 0 / 0 / 0
#   v8 in-place, dialogue 1  tokens -62.4 %   dollars  +16.6 % (+34.1 / +8.7 / +7.1)       K 0 / 0 / 0   (cache writes 3.0-4.0x)
# v8 was labelled D_TOKENS_ONLY by the rule below. A alone (v6, recomputed): tokens -13.9 %, dollars -11.9 %.
#
# Why v8 did not pay, and what it taught. OpenCode puts a cache checkpoint on the system prompt and on the last two messages
# of every request, and the provider reads a request from cache only up to a checkpoint some earlier request wrote whose
# prefix is byte-identical to this one. In v8 the first call of EVERY turn after the third (57 of 57, in each of the three
# repetitions) read only the system prefix, about 28.7k tokens, and wrote everything after it: 409k-416k of the 596k-637k tokens
# written per repetition were those first calls. The cause is the dialogue tier. A turn's tool-calling messages were replaced
# when it closed but its question and text answers stayed one more turn and were removed the turn after, so a turn changed
# state twice, the second time at a position no earlier request had a checkpoint at, and the whole compact history after the
# system prompt was written again, every turn. (The first cost model, in opencode_gate_d.replay, assumed a hit at any common
# prefix with the previous request; it was wrong for exactly this. The checkpoint-only model reproduces v8's real cache writes
# and dollars within 10 % and 4-8 points; the v8 label would not have changed.)
#
# What changes. Nothing in the code: CONTEXT_GRAPH_KEEP_DIALOGUE=0, so a turn is FINAL the moment it closes. When the next user
# turn begins, the whole assistant side of the turn that just closed (its tool calls and its text answer) is replaced by ONE
# short assistant message directly after the user's question: the turn's Card (title, tools used, the numeric lines copied
# literally, within 100 tokens). The question stays as written. That is the one place where the previous request left a
# checkpoint (the question itself was the last message of the first call of that turn), so the cache is read right up to the
# question and only the summary and the new question are written. Nothing is kept whole beyond the turn in progress, nothing
# is appended at the end, and the decision is a pure function of the transcript (no score, no embedding call, no state).
# `expand_card` returns an old turn's content as the tool result and `find_context` is kept. What it gives up against v8 is
# the model's own previous answers: they are in the Card only as the numeric lines they contain. Everything else is the v8
# setup (scenario, model, oracle v4, tool policy, English pin, GRAPH_TUNING, recall cap).
#
# Pre-registered reading (written before any v9 repetition ran), identical to v7 and v8 so the three campaigns read alike.
# Three measures, each per repetition and as the mean of the paired differences (A+D against the baseline of the same
# repetition; both arms run in every repetition):
#   T  total model-input tokens, input + cache read + cache write, summed over the 60 turns;
#   C  dollars, from OpenCode's own message `cost` counter (the provider price list applied to input, cache read, cache
#      write and output tokens);
#   K  correct turns on the 30 eligible, oracle v4: A+D minus baseline.
# Labels, fixed now and applied mechanically to the three means:
#   D_USEFUL        mean C is at least 5 points better than A alone in v6 (mean C <= -16.9 %) AND the minimum K over the
#                   repetitions is >= -1 (at most one correct turn lost);
#   D_TOKENS_ONLY   not D_USEFUL, and mean T is at least 5 points better than A alone in v6 (mean T <= -18.9 %): the tokens
#                   fall and the bill does not follow;
#   D_NOT_WORTH_IT  neither.
# Independently, D_COSTS_CORRECTNESS is added when the minimum K is < -1. The label never changes what D is: `experimental`
# under the owner waiver. A repetition that fails the acceptance below is not counted and is reported as it came out. The
# configuration is the one set here (keep 0 whole, 0 turns of dialogue) and is not tuned between repetitions or after seeing
# them; this is the last configuration of the in-place policy that will be measured on this scenario.
#
# Prediction, written before any v9 repetition ran. The checkpoint-only cache model, run on the recorded transcripts of the
# three v8 arm-A sessions with this configuration, gives dollars -7 / -24 / -25 % (mean -18 %) against each repetition's
# baseline and cache writes about 1.0-1.5x the baseline's. The same model over-predicted v8's real dollars by 4 / 7 / 8 points,
# so the expectation is about -25 %, with a wide error bar that the model cannot narrow: the recalls (`expand_card`,
# `find_context`: 20 / 12 / 20 per v8 repetition) add model calls and written tokens it does not price, and with no answer text left in
# the history the model may need more of them. Tokens will fall by about as much as in v8. Correctness is what the model cannot
# say.
#
# Stages (fail closed: stop at the first stage that is not clean):
#   0. Preflight: docs cache, offline suites (Bun, Gate A, Gate D, MCP, context-core), real-OpenCode write-back probe.
#   1. Sonnet 5.5 R1-R3, full 60-turn paired runs. (Opus only on request: `opus` or `all`.)
#
# A repetition is accepted only if ALL hold: both arms 60/60, zero timeouts, a paired comparison that is go/no_go (never
# not_evaluable); arm A's persisted preview proven; oracle v4 on both arms; English answers; every arm-A turn has a retrieval
# count; the documented plugin switches including `graph:on,graph_mode:inplace,graph_keep:0,graph_dialogue:0`; the graph
# engaged and was request-only (`check_repetition.py graph`); docs cache unchanged; only a retryable provider error is retried
# (once).
#
# Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v9.sh [sonnet|opus|all]   (default: sonnet)
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v9-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
APPROVAL="user-approved-ce60-v8-5-5-2026"
PLAN="validation/opencode_gate_a/stat-plan-v9-5-5-3-reps.json"
POLICY="question,webfetch"
LANGUAGE_PIN="Always answer in English, even when tool results or data are in another language."
PLUGIN_OPTIONS="preview_style:scoped,retrieval_cleanup:on,plugin_rev:2,graph:on,graph_mode:inplace,graph_keep:0,graph_dialogue:0"
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
# Practice D, on top of A, with the in-place policy in its cache-aligned configuration. The comparison file is named so it cannot be read as a Gate A decision.
export CONTEXT_GRAPH=on
export CONTEXT_GRAPH_MODE=inplace
export CONTEXT_GRAPH_KEEP_RECENT=0
export CONTEXT_GRAPH_KEEP_DIALOGUE=0
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
        log "STOP $label v9 conditions not met: $v4 | $summary"; exit 1
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

log "CAMPAIGN v9 start (scope=$WHICH policy=$POLICY plan=$PLAN oracle=$CE60_ORACLE_VERSION language=en-pinned plugin=$PLUGIN_OPTIONS graph_waiver=owner)"

log "PREFLIGHT documentation cache: install pinned pages, verify, serve through the MCP server"
if ! out="$(python3.13 "$DOCS" install --from "$DOCS_STAGING" --to "$DOCS_CACHE" 2>&1)"; then
  log "STOP preflight: could not install the pinned documentation pages: $out"; exit 1
fi
if ! out="$(verify_docs)"; then log "STOP preflight: documentation cache invalid: $out"; exit 1; fi
if ! out="$(python3.13 "$DOCS" probe 2>&1)"; then log "STOP preflight: MCP server does not serve real docs: $out"; exit 1; fi

# The suites run in a clean environment: the campaign's own switches (CONTEXT_GRAPH*, CE60_DECISION_FILE) must not leak into
# tests that assert the defaults.
CLEAN=(env -u CONTEXT_GRAPH -u CONTEXT_GRAPH_MODE -u CONTEXT_GRAPH_KEEP_RECENT -u CONTEXT_GRAPH_KEEP_DIALOGUE -u CE60_DECISION_FILE)
log "PREFLIGHT offline test suites"
if ! (cd .opencode && "${CLEAN[@]}" bun test >/dev/null 2>&1); then log "STOP preflight: bun test failed"; exit 1; fi
if ! (cd validation && "${CLEAN[@]}" python3.13 -m pytest opencode_gate_a opencode_gate_d test_opencode_mcp_server.py -q >/dev/null 2>&1); then
  log "STOP preflight: Gate A / Gate D / MCP tests failed"; exit 1
fi
if ! "${CLEAN[@]}" context-core/.venv/bin/python -m pytest context-core/tests -q >/dev/null 2>&1; then
  log "STOP preflight: context-core tests failed"; exit 1
fi
log "PREFLIGHT real-OpenCode write-back probe (Haiku 5.5)"
if ! bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5 >"$RESULTS/probe-writeback-v9-preflight.json" 2>/dev/null; then
  log "STOP preflight: write-back probe failed"; exit 1
fi
log "PREFLIGHT ok"

if [ "$WHICH" = "sonnet" ] || [ "$WHICH" = "all" ]; then
  stage "sonnet55 R1" ce60-v9-sonnet55-r1 "$SONNET"
  stage "sonnet55 R2" ce60-v9-sonnet55-r2 "$SONNET"
  stage "sonnet55 R3" ce60-v9-sonnet55-r3 "$SONNET"
fi
if [ "$WHICH" = "opus" ] || [ "$WHICH" = "all" ]; then
  stage "opus55 R1" ce60-v9-opus55-r1 "$OPUS"
  stage "opus55 R2" ce60-v9-opus55-r2 "$OPUS"
  stage "opus55 R3" ce60-v9-opus55-r3 "$OPUS"
fi

log "CAMPAIGN v9 complete (scope=$WHICH)"
