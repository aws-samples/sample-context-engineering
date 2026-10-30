# OpenCode Gate A — offline runner

This package is the **offline, replay-only** slice of the CE-60 Gate A layer. It turns the pure
contracts, sanitizer and evaluator into something you can drive from Python or a command line:
build the canonical 60-turn assignments, assemble fail-closed run manifests, sanitize raw run
captures into payload-free results, evaluate them deterministically, and render the isolated
same-host Gate A go/no-go decision.

It never touches the network, AWS, a model provider or a live host. There is no benchmark execution
here — only the deterministic machinery around a run that already happened (or a replay of one).

## What ships

| File | Role |
|------|------|
| `canonical.py` | `canonical-json-v1` serialisation + SHA-256 digests (pre-existing) |
| `contracts.py` | Frozen, fail-closed entities (pre-existing) |
| `sanitize.py` | The trust boundary: raw turns → payload-free results (pre-existing) |
| `evaluator.py` | Deterministic evaluation + replay + Gate A decision (pre-existing) |
| `runner.py` | **This change** — assignments, manifests, file ops, CLI |
| `tests/test_runner.py` | **This change** — offline tests for the runner |

The runner only *orchestrates* the pre-existing pure layer. It does not reimplement classification,
metric derivation, or the Gate A ladder — those stay in `sanitize.py` and `evaluator.py`.

## Safety properties

Three guarantees are load-bearing and are covered by tests:

1. **Never truncate.** `build_assignments` produces exactly 60 assignments or raises
   `ContractViolation`. If no explicit turn ids are given and the CE-60 scenario is not importable, it
   fails closed and asks for the ids — it never pads, drops, or fabricates turns.
2. **Never persist a payload.** `sanitize_file` and `evaluate_file` write only the payload-free
   projections produced by the pure layer. A raw prompt, response, tool payload, credential or log
   never reaches an output file.
3. **Never start a live run by accident.** The CLI performs no model/host execution by default. Any
   `--execute-live` request fails closed with the stable code `LIVE_EXECUTION_REQUIRES_HOST_DRIVER`
   instead of beginning an incomplete benchmark. (No live host driver ships in this layer yet.)

Nothing in a manifest is invented. Identity, catalog digests, and the live-only admission / budget /
price / statistical-plan fields all come from the caller and are validated by the existing contracts.
Per-turn seeds are *derived deterministically* from the caller's `order_seed`, so the same inputs
always produce the same assignment digests.

## Python API

```python
from opencode_gate_a import runner

# 1. Canonical 60-turn assignments (explicit ids; CE-60 scenario is a fallback source when importable).
assignments = runner.build_assignments(turn_ids, order_seed="order-1")

# 2. A fail-closed manifest (offline/replay are local-only; live requires admission + budget + plan).
manifest = runner.build_manifest(
    run_id="run-base",
    arm="baseline",                 # baseline | A | A_plus_D | native_B
    host="opencode",
    host_version="1.4.0",
    model_id="model-x",
    capability_manifest_digest=cap_digest,
    fixture_catalog_digest=fix_digest,
    terminal_catalog_digest=catalog.digest,
    order_seed="order-1",
    scope="pilot",                  # pilot | full (traceability label; manifest shape is identical)
    turn_ids=turn_ids,
)

# 3. Sanitize a raw capture, then evaluate — both over files, both payload-free.
runner.write_catalog(catalog, "catalog.json")
runner.sanitize_file("raw.json", "base-manifest.json", "base-sanitized.json", catalog_path="catalog.json")
runner.evaluate_file("base-sanitized.json", "base-manifest.json", "catalog.json", "base-eval.json")

# 4. Decide Gate A from a baseline/treatment evaluation pair.
decision = runner.decide_gate_a_files(
    baseline_manifest_path="base-manifest.json",
    baseline_eval_path="base-eval.json",
    treatment_manifest_path="treat-manifest.json",
    treatment_eval_path="treat-eval.json",
    output_path="decision.json",
)
```

### Turn ids

`build_assignments(turn_ids=None, order_seed=...)` resolves turn ids in this order:

1. The explicit `turn_ids` argument (must be exactly 60 unique non-empty strings).
2. The CE-60 scenario (`validation/community-plugin-A-B-D/src/scenario.py`), **only if importable**.
   That module pulls optional heavy dependencies through its tool suite, so import failure is
   expected and handled: it simply means you must pass `turn_ids` explicitly.

There is no third path. The runner would rather refuse than guess.

## CLI

The CLI exposes three safe subcommands and performs no live execution.

```bash
# Build a manifest (offline by default). --pilot / --full selects the plan label.
python -m opencode_gate_a.runner manifest \
    --run-id run-base --arm baseline \
    --host opencode --host-version 1.4.0 --model-id model-x \
    --capability-digest sha256:... --fixture-digest sha256:... \
    --catalog catalog.json --order-seed order-1 \
    --turn-ids-file turn_ids.txt \
    --full --out base-manifest.json

# Sanitize a raw run then evaluate it (writes a payload-free sanitized file + an evaluation file).
python -m opencode_gate_a.runner evaluate \
    --raw raw.json --manifest base-manifest.json --catalog catalog.json \
    --out-sanitized base-sanitized.json --out-evaluation base-eval.json

# Decide Gate A from a baseline/treatment evaluation pair.
python -m opencode_gate_a.runner gate-a \
    --baseline-manifest base-manifest.json --baseline-eval base-eval.json \
    --treatment-manifest treat-manifest.json --treatment-eval treat-eval.json \
    --out decision.json
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Usage / validation error (e.g. `--execute-live` without confirmation) |
| `3` | Live guard fired — prints `LIVE_EXECUTION_REQUIRES_HOST_DRIVER` to stderr |
| `4` | Contract violation (prints the stable reason code and message to stderr) |

### The live guard

`--execute-live` is accepted but deliberately inert. Even when supplied with the required
`--i-understand-live-costs` confirmation and the `--admission-path` / `--stat-plan-path` files, it
fails closed with `LIVE_EXECUTION_REQUIRES_HOST_DRIVER` because no live host driver exists in this
layer. This prevents a half-wired runner from starting an unauthorised or incomplete benchmark.

Authoring a *live manifest* (data only) is still possible via `build_manifest(execution_mode="live",
...)` with a valid admission, configured budget, price provenance and an approved `StatPlan`. The
contracts enforce every live precondition; the runner adds nothing and omits nothing.

## Running the tests and linter

```bash
# From the repository root (conftest.py puts the package on sys.path):
python -m pytest validation/opencode_gate_a -q
ruff check validation/opencode_gate_a/runner.py validation/opencode_gate_a/tests/test_runner.py
```

All tests are offline and deterministic. They cover: the 60-assignment builder (count, determinism,
fail-closed truncation), pilot/full manifests, the local-only and live preconditions, file-level
sanitize/evaluate round trips, the no-payload persistence guarantee, the Gate A `go` path, and the
fail-closed live CLI guard.

## `host_driver.py` — payload-free CE-60 host bridge

The live CE-60 host driver itself lives in TypeScript (`.opencode/runner/ce60.ts`): it starts OpenCode,
sends the 60 prompts and keeps the raw answers only in memory. `host_driver.py` is the Python half it
shells out to — the parts that must be exact rather than driven by a live host. It still touches no
network, AWS or provider.

Two subcommands:

- `emit-turns [--turns-json <path>]` — resolve the 60 canonical CE-60 `(turn_id, prompt)` pairs from the
  community scenario (or an explicit file), printing only ids and prompts to stdout. Fails closed with
  `CE60_SCENARIO_TURNS_UNAVAILABLE` (exit 3) when neither source yields exactly 60 valid turns — it never
  truncates, pads or fabricates.
- `finalize` — read the driver's in-memory per-turn observations from stdin, score correctness through
  the existing oracle (`accuracy.py`), build the CE-60 terminal catalog (exactly 30 correctness-eligible
  turns), sanitize, build the fail-closed manifest, evaluate, and write **only** the payload-free
  catalog, manifest, sanitized result, evaluation and evidence. The raw answer text is used to score and
  then discarded; no prompt, answer, tool payload, credential or log is ever written.

A live run is evaluated as a **replay** of itself: `finalize` writes the offline/replay manifest the
evaluation binds to as `<arm>-manifest.json`, and a separate `<arm>-live-manifest.json` records the live
admission, budget, price and approved statistical plan. OpenCode itself exposes no CE-60 retrieval count, so the
baseline (which loads no plugin) is marked `host_not_exposed`; for arm A the TypeScript driver counts the plugin's
`retrieve_context` calls from the persisted session history and the bridge's own counters back that count up, so
retrieval is evaluable and Gate A can return `go` (campaign v6 R2 did). A failed retrieval makes the run invalid
(`GATE_RETRIEVAL_INVALID`).

The billable campaigns live in `campaigns/` (`ce60-v3.sh` to `ce60-v6.sh`, one stat plan and approval per
campaign). Each repetition is accepted only if both arms completed 60/60 turns with no timeout, arm A's preview
reached the model, the oracle version, English answers, per-turn retrieval counts and the plugin switches are as
declared, and the pinned documentation cache is unchanged (`campaigns/check_repetition.py`). Helpers:
`summarize_campaigns.py` (side-by-side table), `paired_series.py` (per-turn pairing of one run),
`preview_outcomes.py` and `analyze_matched_calls.py`. Results and the reading of them are in
`results/ce60-v3-v6-REPORT.md`.

`ce60-v7.sh` to `ce60-v9.sh` are not Gate A campaigns: they measure practice D on top of A (`CONTEXT_GRAPH=on`) against the
baseline of the same repetition, under an owner waiver of the requirement that D wait for a Gate A `go`. The runner writes
`d-slice-comparison.json` instead of `gate-a-decision.json` for them, `check_repetition.py graph` requires the graph to have been
request-only, and `d_slice_report.py` applies the labels fixed in each script header (D_USEFUL / D_TOKENS_ONLY /
D_NOT_WORTH_IT). Results, the cache finding and the cost model: `results/ce60-v7-v9-REPORT.md`. Gate D is in
`../opencode_gate_d/`.

```bash
python -m opencode_gate_a.host_driver emit-turns
# finalize is driven by the TypeScript driver over stdin; see .opencode/README.md
```

## Results kept in the repository

`results/` holds the campaign reports (`ce60-*-REPORT.md`, `ce60-v3-ANALYSIS-PLAN.md`), the campaign status logs, the
D-slice comparisons (`ce60-v{7,8,9}-d-slice-report.json`) and the write-back probes. The per-run directories
(`results/ce60-*/`, about 23 MB of manifests, sanitized turns and evidence) are not versioned: the campaign scripts in
`campaigns/` regenerate them, and the reports quote the numbers that matter. The AWS-documentation pages pinned for the
CE-60 `read_aws_documentation` fixture are not versioned either: `fixtures/docs-v4/MANIFEST.json` pins their URLs and
digests, `campaigns/docs_cache.py fetch --out fixtures/docs-v4` downloads the public pages again and `verify` reports any
page that changed since the pin.
