#!/usr/bin/env bash
# Real coding session on OpenCode: baseline against the context-engineering plugin, in a throwaway copy of this repository.
# Billable (Bedrock). NOT Gate A and not a campaign: one scripted developer conversation per arm, reported, not judged.
#
# What it is. A developer asks OpenCode, step by step (prompts.json, 16 prompts), to write Claude Code plugins for this
# repository: explore the repo, read the Hermes contract and the OpenCode plugin, plan a layout, build four packages, run
# the tests, validate, answer two recall questions, edit the README, summarise. Each prompt is a user turn; the plugin's
# in-place context graph (practice D) only collapses a turn once it has closed, which is why this is a conversation and
# not one long prompt. The agent really reads, greps, edits, writes and runs commands.
#
# Why a copy. The agent has a shell and write access. prepare_sandbox.py copies the repository (no .git, .env, local
# agent or AI-DLC configuration, benchmark results, virtual environments), scans it for credential-looking content,
# makes it a fresh git repository and writes a permission policy: anything outside the sandbox, network and credential
# tools, package installs, git push and the interactive `question` tool are denied. The files the agent writes are
# throwaway: they are scored (workcheck.py) and deleted, and are NOT part of this repository's implementation.
#
# Arms. `baseline`: OpenCode --pure, no plugin. `plugin`: the plugin as it ships with the production defaults and the
# configuration measured in v9 (practice A with live admission over OpenCode's built-in tools read, grep, glob, list,
# bash above 8000 tokens; practice D in-place, KEEP_RECENT=0, KEEP_DIALOGUE=0). Note that A covers the built-in tools by
# default (CE-60 named only MCP tools), so here A is active on real file reads. CODE_PLUGIN_MODE=d-only or a-only runs one
# practice alone. Same model, same prompts, same tool policy (question, webfetch, task disabled) in both arms.
# Both arms also set the Bedrock provider's chunkTimeout to 180 s (OpenCode waits 15 minutes by default for a stream that
# went quiet; one did, mid tool call, in the first attempt). An arm that does not finish every turn cleanly is not counted:
# it is kept in attempt0-<arm>/ and rerun once from a fresh sandbox.
#
# Server. `opencode serve` on 127.0.0.1 with HTTP basic auth through OpenCode's own OPENCODE_SERVER_PASSWORD (random per
# run); the runner checks with lsof that the only listener is loopback and stops otherwise. The OpenCode process gets an
# allow-listed environment (PATH, HOME, locale, AWS_PROFILE, AWS_REGION), not the caller's.
#
# Reading, written before any run. This reports; it applies no threshold. For each arm: turns completed, model calls,
# input tokens (uncached + cache read + cache write), cache writes, output tokens, dollars from OpenCode's own message
# `cost` counter, per-call input (median, peak), wall time, tools used, denied calls; the deliverable check (expected
# files, JSON parses, Python compiles, marketplace lists four packages, README section, existing packages untouched);
# the two recall probes; and for the plugin arm what A previewed and what D collapsed and recalled, and whether the
# graph stayed request-only (stored session unchanged). One session per arm is an anecdote: the agent's path differs from
# run to run, so a difference is a hint, not a result, and nothing here changes what A or D are (A stays `no_go` under
# Gate A, D stays `experimental`).
#
# Never reads .env. AWS access comes only from AWS_PROFILE.
# Usage: bash validation/opencode_code_session/run.sh        (env: CODE_RUN_ID CODE_MODEL CODE_TURNS CODE_ARMS
#                                                              CODE_PLUGIN_MODE CODE_MAX_TURNS KEEP_SANDBOX)
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 2
RUN_ID="${CODE_RUN_ID:-code-v1}"
MODEL="${CODE_MODEL:-amazon-bedrock/anthropic.claude-sonnet-5-5}"
TURNS="${CODE_TURNS:-validation/opencode_code_session/prompts.json}"
ARMS="${CODE_ARMS:-baseline plugin}"
PLUGIN_MODE="${CODE_PLUGIN_MODE:-a+d}"
MAX_TURNS="${CODE_MAX_TURNS:-}"
WORK="/tmp/oc-code/$RUN_ID"
OUT="validation/opencode_code_session/results/$RUN_ID"
SESSION_DIR="validation/opencode_code_session"
STATUS="$OUT/status.log"

export AWS_PROFILE=claude-code
export AWS_REGION=us-west-2

mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"; }

log "RUN $RUN_ID start (model=$MODEL turns=$TURNS arms=[$ARMS] plugin_mode=$PLUGIN_MODE max_turns=${MAX_TURNS:-all})"

if [ -e "$WORK" ]; then log "STOP $WORK already exists; remove it or choose another CODE_RUN_ID"; exit 1; fi
if ! python3.13 "$SESSION_DIR/prepare_sandbox.py" --dest "$WORK" --arms $ARMS >"$OUT/sandbox-manifest.json" 2>"$OUT/prepare.err"; then
  log "STOP could not prepare the sandboxes: $(head -c 400 "$OUT/prepare.err")"; exit 1
fi
log "SANDBOX ready $(python3.13 -c "import json,sys; d=json.load(open('$OUT/sandbox-manifest.json')); print('tree', d['tree'], 'files', d['tracked_files'])")"

run_arm() {  # run_arm <arm> <sandbox root>
  local arm="$1" root="$2" extra=()
  [ -n "$MAX_TURNS" ] && extra+=(--max-turns "$MAX_TURNS")
  bun .opencode/runner/code-session.ts --arm "$arm" --plugin-mode "$PLUGIN_MODE" --sandbox "$root/$arm" \
    --model "$MODEL" --run-id "$RUN_ID" --output-dir "$OUT" --turns "$TURNS" ${extra[@]+"${extra[@]}"} >"$OUT/$arm.run.log" 2>&1
}

# An arm that does not finish every turn cleanly (a provider error, a stalled stream) is not counted: it leaves the session
# in a state the other arm never saw. It is kept as attempt0-<arm>/ and rerun ONCE from a fresh sandbox; a second failure
# stops the run.
rc_all=0
for arm in $ARMS; do
  root="$WORK"
  attempt=1
  while :; do
    log "START $arm (attempt $attempt)"
    run_arm "$arm" "$root"
    rc=$?
    log "DONE $arm rc=$rc $(tail -n 1 "$OUT/$arm.run.log" | head -c 200)"
    python3.13 "$SESSION_DIR/workcheck.py" --sandbox "$root/$arm" --out "$OUT/$arm-workcheck.json" >/dev/null 2>&1 \
      || log "WARN workcheck failed for $arm"
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$attempt" -ge 2 ]; then rc_all=$rc; break; fi
    log "RETRY $arm: attempt 1 kept in attempt0-$arm/, fresh sandbox"
    mkdir -p "$OUT/attempt0-$arm"
    mv "$OUT/$arm".run.log "$OUT/$arm"-*.json "$OUT/attempt0-$arm/" 2>/dev/null
    if [ "$arm" = "plugin" ]; then mv "$OUT"/A-metrics.jsonl "$OUT"/A-graph-evidence.jsonl "$OUT"/A-graph-durability.json "$OUT/attempt0-$arm/" 2>/dev/null; fi
    rm -rf "$WORK-retry"
    if ! python3.13 "$SESSION_DIR/prepare_sandbox.py" --dest "$WORK-retry" --arms "$arm" >"$OUT/sandbox-manifest-retry.json" 2>"$OUT/prepare.err"; then
      log "STOP could not prepare the retry sandbox: $(head -c 400 "$OUT/prepare.err")"; rc_all=1; break
    fi
    root="$WORK-retry"
    attempt=2
  done
  [ "$rc_all" -ne 0 ] && break
done

python3.13 "$SESSION_DIR/report.py" "$OUT" --json "$OUT/report.json" | tee "$OUT/report.txt"

if [ "${KEEP_SANDBOX:-0}" != "1" ]; then
  rm -rf "$WORK" "$WORK-retry"
  log "CLEANUP sandboxes removed"
else
  log "CLEANUP skipped (KEEP_SANDBOX=1): $WORK"
fi
log "RUN $RUN_ID complete rc=$rc_all"
exit "$rc_all"
