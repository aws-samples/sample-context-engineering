# CE-60 OpenCode Heavy Benchmark Campaign

## Scope

- Host: OpenCode `1.18.34`
- AWS profile: `claude-code`
- AWS account: `<account-id>`
- Region: `us-west-2`
- Arms: same-host baseline and OpenCode A
- Models: Haiku 4.5, Sonnet 4.5, Opus 4.5
- Repetitions: 3 paired repetitions per model
- Total: 18 CE-60 arms, 1,080 turns
- Turn execution: sequential, 60 turns per arm
- StatPlan: `stat-plan-heavy-3-reps.json`
- Approval: `user-approved-ce60-heavy-2026`
- Budget ceiling: `100 USD` per arm admission; this is a ceiling/provenance field, not an observed spend total.
- Reranker: `amazon.rerank-v1:0`
- Raw prompt/response payloads: not persisted; only sanitized artifacts were written.

## Aggregate results

| Model | Arm | Completion | Correctness | Median input tokens | Completed turns |
|---|---|---:|---:|---:|---:|
| Haiku 4.5 | baseline mean | 100% | 21.11% | 54,187.0 | 60/60 each run |
| Haiku 4.5 | A mean | 100% | 23.33% | 56,723.7 | 60/60 each run |
| Sonnet 4.5 | baseline mean | 100% | 27.78% | 58,395.0 | 60/60 each run |
| Sonnet 4.5 | A mean | 100% | 26.67% | 53,078.5 | 60/60 each run |
| Opus 4.5 | baseline mean | 100% | 27.78% | 58,229.3 | 60/60 each run |
| Opus 4.5 | A mean | 100% | 26.67% | 48,671.8 | 60/60 each run |

## Gate A decisions

| Model | R1 | R2 | R3 | Summary |
|---|---|---|---|---|
| Haiku 4.5 | `no_go` — input | `no_go` — input | `no_go` — input | No successful Gate A repetition |
| Sonnet 4.5 | `go` | `no_go` — input | `no_go` — correctness + input | 1 of 3 repetitions passed |
| Opus 4.5 | `go` | `no_go` — correctness | `no_go` — correctness + input | 1 of 3 repetitions passed |

All arms completed 60/60 without runner errors. Retrieval was valid for arm A and `host_not_exposed` for baseline, as expected from the baseline's `--pure` configuration. The Gate A evaluator therefore used the available A retrieval evidence while keeping baseline retrieval explicitly unavailable rather than fabricating zero.

## Interpretation

1. **Haiku:** A improved mean correctness from 21.11% to 23.33%, but increased median model input by approximately 4.7%; all three Gate A decisions were `no_go` for insufficient input reduction.
2. **Sonnet:** A reduced mean median input by approximately 9.1% but reduced mean correctness by approximately 1.11 percentage points; only R1 met all Gate A thresholds.
3. **Opus:** A reduced mean median input by approximately 16.4% but reduced mean correctness by approximately 1.11 percentage points; only R1 met all Gate A thresholds.
4. **Stability:** A single successful repetition is insufficient for a stable promotion claim. The three-repetition StatPlan exposed model-to-model and repetition-to-repetition variance.
5. **Gate A status:** No model has a stable three-of-three `go` result. The A path should not be promoted as generally validated from this campaign.

## Provenance

Each repetition has its own immutable artifacts under:

```text
validation/opencode_gate_a/results/ce60-heavy-{haiku,sonnet,opus}-r{1,2,3}/
```

Each run directory contains catalog, offline evaluation manifest, live provenance manifest, sanitized result, evaluation and Gate A evidence. The driver used the public OpenCode SDK, the real Bedrock model, the real Bedrock reranker and explicit `ExecutionAdmission` receipts.

## Cost note

The run manifests record a `100 USD` per-arm admission ceiling and operator-declared price provenance. This report does not claim an observed AWS spend total because Cost Explorer data was not queried during the campaign. The campaign consisted of 18 arms and therefore carried up to 18 declared arm-level ceilings under the admission metadata; actual usage must be reconciled separately against AWS billing data.

## Follow-up

- Inspect Cost Explorer for account `<account-id>`, region `us-west-2`, and the campaign time window.
- Investigate why A input reduction is below the 20% Gate A threshold in most repetitions.
- Investigate correctness regressions in Sonnet/Opus A repetitions.
- Keep Gate D deferred; the stable Gate A evidence requirement was not met across repetitions.
