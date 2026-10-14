# CE-60 OpenCode Campaign v2 — F002 resolved, Gate A evaluable

## Summary

- The F002-filler timeout that blocked every earlier 5.5 run was the built-in OpenCode `question` tool waiting for a human reply in a headless session. It is fixed by disabling `question` through the public per-prompt `tools` map, identically in both arms.
- With that fix, **Sonnet 5.5 completed 3 of 3 paired repetitions at 60/60 turns in both arms with zero timeouts**, and Gate A became evaluable on 5.5 for the first time.
- **Those Gate A results are NOT valid evidence about practice A.** While analysing why arm A barely changed model input, the persisted tool outputs showed that the plugin's preview never reached the model for MCP tools (see "Second defect" below). Arm A in these runs was "plugin loaded, no effect". The `no_go` decisions are therefore an artefact of that defect, not a verdict on A.
- The defect is fixed and verified end to end on the real OpenCode. The corrected campaign is v3.
- Still valid from v2: the F002 root cause and fix, the baseline-arm results, the harness hardening, and the finding that the baseline has a 27.7k-token fixed context floor.

## Scope

- Host: OpenCode `1.18.34` (public SDK and plugin API only); provider `amazon-bedrock`; profile `claude-code`; region `us-west-2`.
- Fixture: deterministic `ce60-fixture` MCP server, shared by both arms. Baseline runs `opencode serve --pure`; arm A loads `.opencode/plugins/context-engineering.ts`.
- Models: `amazon-bedrock/anthropic.claude-sonnet-5-5` (complete) and `amazon-bedrock/anthropic.claude-opus-5-5` (see the Opus section).
- Reranker: `amazon.rerank-v1:0`, recorded separately from the execution model in every manifest.
- Declared tool policy for full runs, identical in both arms and recorded in the manifests: `question,webfetch` disabled via the per-prompt `tools` map. `webfetch` is a separate determinism change (the model was fetching live internet pages outside the fixture).
- Inherited `OPENCODE_*` environment overrides are stripped from both arms.
- StatPlan: `stat-plan-v2-5-5-3-reps.json`, approval `user-approved-ce60-v2-5-5-2026`. The budget figure in the manifests is an admission ceiling, not observed spend.
- Raw prompts, answers, tool arguments and tool results were not persisted.

## F002 root cause

| Evidence | Result |
|---|---|
| Position | F002-filler is the 19th turn in both arms (the SPEC's earlier "17 turns" was wrong and is corrected) |
| R1 live session read | The timed-out turn was an assistant message whose only tool part was `question`, no `step-finish`, lasting 600.47 s |
| Haiku 5.5 probe (`probe-question-haiku55.json`) | `question` blocked a headless session for the full window (`blocked_question`); abort returned the session to idle. `tools: { question: false }` removed the tool and the turn finished in ~2 s. MCP tools kept working with the map in place |
| D1 (`ce60-f002-d1-control`) | Baseline `--pure`, Sonnet 5.5, 19 turns, unchanged workload: F002-filler timed out at 600,027 ms and was classified `blocked_question` from the captured state (`question` / `running`, assistant incomplete), abort `ok`, idle `idle`, 0 residual processes |
| Variant (`ce60-f002-variant-question-off`) | `question` disabled in both arms, 19 turns: **0 timeouts in both arms**, F002 completed in 7.6 s (baseline) and 8.9 s (A), `question` never used |

Conclusion: the stall was a headless-session configuration issue, not a provider, MCP, bridge or reranker failure, and not a context-size or compaction effect.

## Sonnet 5.5 results (3 paired repetitions)

> **Void for practice A.** The `no_go` verdicts below were produced while the plugin's MCP preview never reached the model (see "Second defect"). They remain valid as completion, correctness, latency and timeout evidence for both arms, and as a record of baseline-vs-baseline-like variance, but they say nothing about the effect of A.

| Rep | Gate A | Completed (B / A) | Correctness (B / A) | Median model input (B / A) | Δ median | Δ total input | p95 latency (B / A) |
|---|---|---|---|---|---:|---:|---|
| R1 | `no_go` — input | 60 / 60 | 29/30 / 29/30 | 126,906 / 124,772.5 | −1.7% | −7.4% | 12.1 s / 8.7 s |
| R2 | `no_go` — input | 60 / 60 | 29/30 / 29/30 | 125,938 / 128,603 | +2.1% | +1.2% | 11.4 s / 10.4 s |
| R3 | `no_go` — input | 60 / 60 | 28/30 / 29/30 | 125,449.5 / 135,807 | +8.3% | +7.9% | 10.6 s / 13.7 s |

Mean change in median model input: **+2.9%** (required: at least −20%). Correctness never regressed: arm A scored equal to or higher than baseline in all three repetitions (29/30 vs 29/30, 29/30 vs 29/30, 29/30 vs 28/30). Retrieval was exposed in arm A and the evaluator recorded zero failures; the model made 0 `retrieve_context` calls, so the retrieval path is validated as available but was not exercised by the agent.

Per-repetition invariants (all verified from the artifacts): 0 timeouts, 0 residual processes, no SIGKILL needed, no payload strings in any artifact, identical host version / model / tool policy in both arms, compaction state `unavailable` (not "none observed").

## Second defect: MCP write-back (found while analysing the v2 numbers)

Symptom: in every Sonnet repetition the largest context jump (`A5-statement`, +24k tokens) was identical in both arms (+24,389 vs +24,385), although the bridge reported that it had replaced the 33,677 B result with a ~3.9 KB preview at exactly that turn.

Check: the persisted tool output of that call, read from the OpenCode session store with sizes only, was the full 33,677 B and carried no preview marker, in all four arm-A sessions (Sonnet R1-R3, Opus R1).

Cause (read from the OpenCode 1.18.34 runtime, not guessed): for a direct MCP tool the runtime runs `tool.execute.after` on the raw MCP result `{ content: [{type:"text", text}], isError }`, and only afterwards builds the model-facing text from `content[].text`. The plugin assigned the preview to `output.output`, a field MCP results do not have, so the replacement was dropped. Built-in tools (`{title, output}`) were never affected.

Fix: `writeOutputText()` in `.opencode/lib/tool-output.ts` rewrites `content[]` for MCP results (one text block, non-text blocks preserved) and `output.output` for built-ins; `readTextBlocks` no longer rejects results that mix text with an attachment. 80 offline tests pass.

Verification on the real OpenCode (`probe-writeback-haiku55.json`, Haiku 5.5): bridge `previewed` 44,960 → 4,002 B with a successful Bedrock rerank; the persisted tool output is 4,192 B with the retrieval hint; the model then called `retrieve_context` and received the full 44,960 B back. The contract (preview now, full content on request) works end to end.

Consequence for the numbers above: the Δ model-input values (−1.7%, +2.1%, +8.3%) measure session-to-session agent variance only. The earlier "agent variance swamps the signal" observation (whole-session tool parts ranged 40 to 55 across sessions; per-turn A−baseline averaged −5.5%, +1.2%, +7.6%) remains a real risk for the corrected campaign and must be handled in the analysis, but nothing can be concluded about A's own effect from these runs.

## Third defect: transient provider error counted as an arm failure

Opus R1 arm A completed 59/60. The failed turn was the first one (`A1-accounts`): `APIError`, HTTP 503, `isRetryable: true`, after 105.9 s (read from the session store: error name, status and retryable flag only). It is a transient Bedrock availability error, not a plugin, MCP, `question` or session fault, and the next turn recovered. Two harness gaps are fixed: the parser read `retryable` but the SDK field is `isRetryable` (so such errors could never classify as `provider_retry`), and non-timeout errors now persist their class, HTTP status and retryable flag (`error_*` fields). Because a transient 503 on turn 1 makes the pair non-comparable, a repetition with an unrecovered retryable provider error is repeated rather than scored.

## Decision

- Gate A for OpenCode A: **no decision**. The v2 `no_go` results are void for practice A because the preview never reached the model. The corrected campaign (v3) decides.
- Gate D remains deferred.
- Valid and kept: F002 resolved; evaluable, timeout-free baselines; payload-free, fail-closed harness.

## Provenance

- Per-repetition artifacts: `ce60-v2-sonnet55-r{1,2,3}/` (catalog, manifests, sanitized results, evaluations, evidence, cleanup, `gate-a-decision.json`).
- Diagnostics: `ce60-f002-d1-control/`, `ce60-f002-variant-question-off/`, `probe-question-haiku55.json`.
- Campaign driver and status log: `campaigns/ce60-v2.sh`, `ce60-v2-campaign-status.log`.
- Superseded: `ce60-5-5-REPORT.md` (its erratum records the corrected telemetry and the F002 cause).
- Write-back evidence: `probe-writeback-haiku55.json`; invalidation notes: `INVALID-ARM-A-PREFIX-WRITEBACK.txt` in each v2 run directory.
