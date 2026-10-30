#!/usr/bin/env bash
# CE-60 OpenCode campaign v8: practice D with the in-place policy, on top of A, plugin revision 2. Billable.
#
# This is NOT a Gate A campaign. Gate A judges practice A alone and it was decided in v3-v6 (1 of 3 repetitions `go` in v6:
# the hypothesis was not supported). The owner asked for practice D to be implemented and measured on OpenCode anyway and
# waived the requirement that D wait for a Gate A `go` (FR5.1 / BR5.1). That waiver is recorded here and in the results; it
# makes D `experimental`, never `supported`, and it does not change Gate A, which stays `no_go`. The in-place policy is a
# variant of D (closer to the "context curator" idea the design folded into D), reported as such and never as the reference
# algorithm.
#
# Why a second D campaign. v7 measured the REFERENCE policy of D: the total model input fell by 48.9 % and the bill rose by
# 215.7 % (3 repetitions, 30/30 correct turns each time). The cause is the provider's prompt cache, which is a prefix cache:
# the reference policy re-decides every turn which old turns to show whole, so a turn that flips back to whole content
# changes the request EARLY and everything after it is written to the cache again. The price list in use (fitted from
# OpenCode's own cost counters, exact to the cent) is $0.10 per million cache-read tokens, $2.50 per million cache-write
# tokens (25 times a read) and $10 per million output tokens. In v7 R1 the bill was $6.52 against $2.13 for the baseline, of
# which $5.40 was cache writes; about 82 % of the extra writes came from relevance flips and restores of old turns, and about
# 18 % from the summary block being rewritten at the end of every request.
#
# What changes. CONTEXT_GRAPH_MODE=inplace keeps the Cards and the two retrieval tools and replaces the policy. When the next
# user turn begins, the assistant messages of the turn that just closed that CALL TOOLS (the bulk of a turn: the tool inputs
# and results) are replaced by ONE short assistant message at the position the first of them had: the turn's Card (the title,
# the tools it used and the numeric lines it produced, copied literally, within 100 tokens). The user's question and the
# assistant's own text answers stay as written for one more turn (CONTEXT_GRAPH_KEEP_DIALOGUE=1); after that the whole
# assistant side of the turn is that one message (the same bytes). Nothing is kept whole beyond the turn in progress
# (CONTEXT_GRAPH_KEEP_RECENT=0). The decision is a pure function of the transcript: no score, no embedding call, no state, the
# same answer after a restart; and nothing is appended at the end of the request. `expand_card` returns an old turn's content
# as the tool result (it does not put the turn back where it was) and `find_context` is kept. What it gives up against the
# reference policy is the relevance decision: an old turn is never shown again whole unless the model asks for it. Everything
# else is the v7 setup (scenario, model, oracle v4, tool policy, English pin, GRAPH_TUNING).
#
# Why the tool results go at once and the dialogue stays. A provider caches the request up to the first byte that differs, so
# what a collapse rewrites is everything after it. A tool result is written to the cache once, when it is produced; if it is
# collapsed the next turn it is never read again and never written twice. The dialogue is small, so rewriting it is cheap, and
# it is what the next question most often leans on (the model's last answer, with its figures). The first form of this policy
# kept the whole previous turn as it was (keep 1, no dialogue tier), which writes that turn's tool results a second time one
# turn later, on every turn. It was probed on the first 21 turns (Sonnet 5.5, plugin rev 2, billed): total model input -57 %
# against the v6 baselines, but dollars only -7.7 % against them (A alone, same turns: -17 %), with cache writes 1.5x the
# baseline's; an offline cost model on the recorded v7 R1 session (calibrated to the real baseline bill; within 6 % of that
# probe's real bill) put that form at -17 % (-22 % on the baseline transcript) and the form below at -30 % (-35 %). The policy
# was changed to the form below BEFORE any repetition of this campaign ran. It is described here so the result cannot be read
# as if the first form had been the plan. The only other change made after the probe is not a result of it: a recall
# (`expand_card`) is capped at 40,000 characters and 1,500 lines, because OpenCode saves a tool result above 50 KB or 2,000
# lines to a file and shows the model a preview instead (read from the OpenCode 1.18.34 binary; the probe never hit it).
#
# Pre-registered reading (written before any v8 repetition ran). Three measures, each per repetition and as the mean of the
# paired differences (A+D against the baseline of the same repetition; both arms run in every repetition), reported next to
# the v6 figures for A alone (the mean of R1-R3 of v6, recomputed: total model input -13.9 %, dollars -11.9 %, correctness
# parity) and next to the v7 figures for the reference policy (tokens -48.9 %, dollars +215.7 %):
#   T  total model-input tokens, input + cache read + cache write, summed over the 60 turns;
#   C  dollars, from OpenCode's own message `cost` counter (the provider price list applied to input, cache read, cache
#      write and output tokens);
#   K  correct turns on the 30 eligible, oracle v4: A+D minus baseline.
# Labels, fixed now and applied mechanically to the three means (the same rule as v7, so the two campaigns read alike):
#   D_USEFUL        mean C is at least 5 points better than A alone in v6 (mean C <= -16.9 %) AND the minimum K over the
#                   repetitions is >= -1 (at most one correct turn lost);
#   D_TOKENS_ONLY   not D_USEFUL, and mean T is at least 5 points better than A alone in v6 (mean T <= -18.9 %): the tokens
#                   fall and the bill does not follow;
#   D_NOT_WORTH_IT  neither.
# Independently, D_COSTS_CORRECTNESS is added when the minimum K is < -1. The label never changes what D is: `experimental`
# under the owner waiver. A repetition that fails the acceptance below is not counted and is reported as it came out. The
# configuration is the default one (keep 0 whole, 1 turn of dialogue) and is not tuned between repetitions or after seeing
# them.
#
# Prediction, written before any v8 repetition ran. The offline cost model (chars/4, cache write at 25x a read, calibrated to
# the real v7 R1 baseline bill; it over-predicts the saving for the policies it has seen live) says, for this policy on the
# transcript of the A+D session: dollars about -30 % and total model input about -57 %. Variants it priced on the same
# transcript, for scale only: the whole previous turn kept as it was -17 %, nothing kept (every closed turn one summary at
# once) -36 %, two turns of dialogue -23 %. Tokens will fall; whether the bill follows by the 5 points that matter is the
# question, and whether the answers stay correct is what the model cannot say.
#
# Stages (fail closed: stop at the first stage that is not clean):
#   0. Preflight: docs cache, offline suites (Bun, Gate A, Gate D, MCP, context-core), real-OpenCode write-back probe.
#   1. Sonnet 5.5 R1-R3, full 60-turn paired runs. (Opus only on request: `opus` or `all`.)
#
# A repetition is accepted only if ALL hold: both arms 60/60, zero timeouts, a paired comparison that is go/no_go (never
# not_evaluable); arm A's persisted preview proven; oracle v4 on both arms; English answers; every arm-A turn has a retrieval
# count; the documented plugin switches including `graph:on,graph_mode:inplace,graph_keep:0,graph_dialogue:1`; the graph engaged and was
# request-only (`check_repetition.py graph`); docs cache unchanged; only a retryable provider error is retried (once).
#
# Never reads .env; AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_gate_a/campaigns/ce60-v8.sh [sonnet|opus|all]   (default: sonnet)
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2
RESULTS="validation/opencode_gate_a/results"
STATUS="$RESULTS/ce60-v8-campaign-status.log"
SONNET="amazon-bedrock/anthropic.claude-sonnet-5-5"
OPUS="amazon-bedrock/anthropic.claude-opus-5-5"
APPROVAL="user-approved-ce60-v8-5-5-2026"
PLAN="validation/opencode_gate_a/stat-plan-v8-5-5-3-reps.json"
POLICY="question,webfetch"
LANGUAGE_PIN="Always answer in English, even when tool results or data are in another language."
PLUGIN_OPTIONS="preview_style:scoped,retrieval_cleanup:on,plugin_rev:2,graph:on,graph_mode:inplace,graph_keep:0,graph_dialogue:1"
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
# Practice D, on top of A, with the in-place policy. The comparison file is named so it cannot be read as a Gate A decision.
export CONTEXT_GRAPH=on
export CONTEXT_GRAPH_MODE=inplace
export CONTEXT_GRAPH_KEEP_RECENT=0
export CONTEXT_GRAPH_KEEP_DIALOGUE=1
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
        log "STOP $label v8 conditions not met: $v4 | $summary"; exit 1
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

log "CAMPAIGN v8 start (scope=$WHICH policy=$POLICY plan=$PLAN oracle=$CE60_ORACLE_VERSION language=en-pinned plugin=$PLUGIN_OPTIONS graph_waiver=owner)"

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
if ! bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5 >"$RESULTS/probe-writeback-v8-preflight.json" 2>/dev/null; then
  log "STOP preflight: write-back probe failed"; exit 1
fi
log "PREFLIGHT ok"

if [ "$WHICH" = "sonnet" ] || [ "$WHICH" = "all" ]; then
  stage "sonnet55 R1" ce60-v8-sonnet55-r1 "$SONNET"
  stage "sonnet55 R2" ce60-v8-sonnet55-r2 "$SONNET"
  stage "sonnet55 R3" ce60-v8-sonnet55-r3 "$SONNET"
fi
if [ "$WHICH" = "opus" ] || [ "$WHICH" = "all" ]; then
  stage "opus55 R1" ce60-v8-opus55-r1 "$OPUS"
  stage "opus55 R2" ce60-v8-opus55-r2 "$OPUS"
  stage "opus55 R3" ce60-v8-opus55-r3 "$OPUS"
fi

log "CAMPAIGN v8 complete (scope=$WHICH)"
