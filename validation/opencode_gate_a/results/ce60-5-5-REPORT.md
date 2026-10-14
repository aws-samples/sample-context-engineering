# CE-60 OpenCode Sonnet 5.5 Clean R1

## Scope

- Host: OpenCode `1.18.34`
- Provider: `amazon-bedrock`
- Agent model: `amazon-bedrock/anthropic.claude-sonnet-5-5`
- AWS profile: `claude-code`
- AWS account: `<account-id>`
- Region: `us-west-2`
- Fixture: deterministic `ce60-fixture` MCP server, shared by baseline and A
- Arms: same-host `baseline` (`--pure`) and OpenCode `A` (plugin enabled)
- Turns: 60 sequential turns per arm in one persistent session
- StatPlan: `stat-plan-mcp-5-5-3-reps.json`
- Approval: `user-approved-ce60-mcp-5-5-2026`
- Budget ceiling: `100 USD` per arm admission; this is a configured ceiling and provenance field, not an observed billing total.
- Reranker: `amazon.rerank-v1:0`
- Bridge interpreter: `python3.13`
- MCP coverage: explicit allowlist of the 18 canonical CE-60 fixture tools
- Raw prompts, responses, tool arguments and tool payloads: not persisted by the runner; artifacts are sanitized and digest-linked.

Run directory:

```text
validation/opencode_gate_a/results/ce60-mcp-5-5-sonnet-clean-r1/
```

## Results

| Arm | Completed | Errors | Correctness | Latency p50 | Latency p95 | Model-input metric | Retrieval metric |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 59/60 | 1 | 30/30 eligible | 5,885 ms | 13,535 ms | `collection_failed` | `host_not_exposed` |
| A | 59/60 | 1 | 30/30 eligible | 5,577 ms | 10,512 ms | `collection_failed` | valid/exposed |

The single error in each arm is the same turn: `F002-filler`. The run-level Gate A decision is:

```json
{
  "gate_name": "Gate_A",
  "outcome": "not_evaluable",
  "reasons": ["GATE_METRIC_NOT_EVALUABLE"]
}
```

The complete decision is recorded at:

```text
validation/opencode_gate_a/results/ce60-mcp-5-5-sonnet-clean-r1/gate-a-decision.json
```

## Host and bridge evidence

The real integration path completed successfully for arm A:

- `ExecutionAdmission`: admitted with execution model `amazon-bedrock/anthropic.claude-sonnet-5-5`.
- Bridge: started with `python3.13`.
- Reranker: one attempt, one success, zero failures.
- Transform events: 54 total; 53 skipped and 1 previewed.
- The observed previewed transform reduced the bridge-observed tool-result bytes from `195,745` to `166,016`, a `15.19%` reduction within the observed transform stream.
- Fallbacks, refusals, source truncation and metrics-write failures: zero.
- The arm A manifest records the execution/rerank model separation and the MCP fixture catalog digest.

This byte reduction must not be reported as model-input reduction. OpenCode did not expose a comparable baseline transform stream, and the run-level model-input metric was invalidated by the timed-out turn.

## Accumulated-session diagnostic

The new payload-free telemetry confirms that the failure is associated with the accumulated CE-60 session rather than provider or startup availability:

| Signal | Baseline | A |
|---|---:|---:|
| MCP tool calls | 1,885 | 1,814 |
| Total tool calls | 2,258 | 2,232 |
| Maximum tool calls in a session observation | 56 | 55 |
| Maximum message count | 159 | 156 |
| Median message count | 92.5 | 90 |
| Compaction events observed | 0 | 0 |
| Timeout turn | `F002-filler` | `F002-filler` |

The isolated F002 diagnostic remains successful in a fresh session. The accumulated run therefore does not establish a Bedrock, MCP, bridge-startup or reranker-unavailability failure. It does establish a reproducible accumulated-session/tool-loop timeout at the current CE-60 configuration.

## Decision

Sonnet 5.5 CE-60 R1 is **not evaluable** for Gate A. The A path is not promoted.

Do not run the remaining Sonnet 5.5 repetitions or the Opus 5.5 matrix yet. A single 5.5 repetition with the same F002 timeout cannot support a stable promotion claim, and the required model-input metric is not valid at run scope.

Gate D remains deferred.

## Required next experiment

Run a targeted, paid-minimal diagnostic around the accumulated-session boundary before another full CE-60 matrix:

1. Preserve one persistent session and the canonical turn order through `F002-filler`.
2. Record a manifest-visible tool-loop boundary, context/message growth, MCP latency/error state, model request start/end, timeout boundary and compaction state.
3. If a controlled compaction or loop bound is introduced, apply it identically to baseline and A and record it in both manifests; do not silently reset the session.
4. Keep Gate A fail-closed until one full paired repetition reaches 60/60 with valid model-input metrics and no unexplained timeout.

After that diagnostic is clean, rerun Sonnet 5.5 R2/R3, then evaluate Opus 5.5.

## Provenance

- Baseline evidence: `ce60-mcp-5-5-sonnet-clean-r1/baseline-evidence.json`
- A evidence: `ce60-mcp-5-5-sonnet-clean-r1/A-evidence.json`
- Baseline evaluation: `ce60-mcp-5-5-sonnet-clean-r1/baseline-evaluation.json`
- A evaluation: `ce60-mcp-5-5-sonnet-clean-r1/A-evaluation.json`
- Gate A decision: `ce60-mcp-5-5-sonnet-clean-r1/gate-a-decision.json`
- F002 isolation history: `F002-DIAGNOSTIC.md`

## Erratum (added after the F002 investigation)

The Gate A decision above is unchanged (`not_evaluable`). Three statements in this report were wrong:

1. **Tool-call totals.** The figures 2,258 / 1,885 (baseline) and 2,232 / 1,814 (A) are sums of cumulative session snapshots, not call counts. The runner summed one whole-session count per turn. The real whole-session tool-part count was about 56 (baseline) and 55 (A); the "median message count" figures (92.5 / 90) were medians of the same snapshots. There was no heavy tool loop. The aggregation is fixed in observability schema 2 (snapshot deltas plus explicit final snapshot).
2. **Compaction.** "No compaction event was observed" is not supported: the per-turn model-input series contains unexplained context drops, so the correct state is `unavailable`, not "none observed".
3. **Cause of the F002 timeout.** F002-filler is the 19th turn in both arms; both timed out at ~600 s. During the run, the public session API showed the timed-out baseline turn as an assistant message whose only tool part was the built-in `question` tool, with no `step-finish`, lasting 600.47 s. `question` waits for a human reply that a headless `opencode serve` session never receives. A Haiku 5.5 probe reproduced the block in isolation and confirmed that the public per-prompt `tools: { question: false }` map removes the tool while MCP tools keep working (`probe-question-haiku55.json`). The F002 timeout is therefore classified as `blocked_question`, not a model or provider stall.
