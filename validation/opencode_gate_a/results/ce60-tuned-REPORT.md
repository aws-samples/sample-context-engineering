# CE-60 Calibration Attempt — Arm A Tuned

## Configuration

- `CONTEXT_MAX_RESULT_TOKENS=4000`
- `CONTEXT_PREVIEW_TOKENS=600`
- `CONTEXT_CHUNK_TOKENS=1000`
- `CONTEXT_RELEVANCE_THRESHOLD=0.5`
- StatPlan: `stat-plan-tuned-3-reps.json`
- Host: OpenCode `1.18.34`
- AWS profile: `claude-code`
- Region: `us-west-2`

## Runs executed

### Haiku R1

- Baseline: 60/60 completed; correctness 7/30; median input 60,816.
- Arm A: 60/60 completed; correctness 5/30; median input 58,888.5.
- Gate A: `no_go` — `GATE_CORRECTNESS_REGRESSION`, `GATE_MODEL_INPUT_INSUFFICIENT`.

### Sonnet R1

- Baseline: 60/60 completed; correctness 8/30; median input 60,095.
- Arm A: 56/60 completed; correctness 6/30; 4 errors; input and latency metrics `collection_failed`.
- Gate A: `not_evaluable` — `GATE_METRIC_NOT_EVALUABLE`.

## Decision

The tuned configuration is **rejected**. Lowering the result gate and preview budgets did not produce reliable input reduction and introduced correctness regression/turn failures. Opus was not run with this configuration because the calibration already failed the safety and quality criteria.

This is a fail-closed stop, not a missing result. The previous default campaign remains the valid comparison set in `ce60-heavy-REPORT.md`.

## Next improvement required

Do not reduce preview budgets further. Inspect the four Sonnet arm-A error turn IDs and the plugin/bridge lifecycle around those turns. The next experiment should first isolate whether failures are caused by bridge timeout, OpenCode tool replacement, Bedrock reranker latency, or context compaction. Only after a targeted fix should a new multi-model campaign be authorized.
