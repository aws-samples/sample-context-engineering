# Validation Harness — Community Plugins (A, B, D)

> **⚠️ Not for production use.** Provided for **experimentation and benchmark reproduction only**.
> The configurations below are minimal examples, not production-ready baselines. Do not use them in
> a production environment without an independent security review, least-privilege hardening, and an
> operational readiness assessment appropriate to your workload.

This directory benchmarks practices **A, B and D** as **community plugins** — three standalone
packages that attach to the SDK's extension surface and are installed alongside an **unmodified**
`strands-agents` from PyPI.

It is the sibling of [`../01-designA-B-D`](../01-designA-B-D), which measures the same three
strategies as **vended plugins of a forked SDK**. Same scenario, same tool suite, same model, same
60 turns — so the two runs' absolute token totals are directly comparable, which is the point of
having both.

| Strategy | Vended (`01-designA-B-D`) | Community (here) |
|---|---|---|
| **A** Relevance filtering | `ContextManager` + `Offload.relevance` | `strands_relevance_filter.RelevanceFilter` |
| **B** Progressive tool disclosure | `strands.vended_plugins.…` | `strands_progressive_tool_disclosure.ProgressiveToolDisclosure` |
| **D** Context graph | `ContextStrategy(strategy="graph")` | `strands_context_graph.ContextGraph` |

---

## What you need

- **Python 3.12** and [`uv`](https://github.com/astral-sh/uv).
- **AWS credentials** for an account with these Bedrock models enabled, in `us-east-1`:
  `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`.
  Credentials resolve through the standard AWS chain or a named profile in
  `VALIDATION_AWS_PROFILE`. Nothing is hardcoded; the account is discovered from STS. To pin the
  account a run must execute in, set `VALIDATION_ACCOUNT_ID`.
- Internet on the first run: it downloads a Chromium build (for the web tool) and a handful of AWS
  doc pages into `.cache/`, then reuses them so every configuration sees byte-identical payloads.

Re-rendering a recorded run's report needs **no AWS credentials at all**.

## The dependency: three packages, no SDK fork

`requirements.txt` installs the three community packages **editable from this repository**, so a run
measures the code a reader can read, plus the public SDK:

```
-e ../../community-plugins/strands-context-graph
-e ../../community-plugins/strands-progressive-tool-disclosure
-e ../../community-plugins/strands-relevance-filter
strands-agents>=1.44.0,<2.0.0
```

Verified: all three import and run against **unmodified `strands-agents` 1.56.0** from PyPI. The
private middleware seam they couple to (`strands._middleware.stages.InvokeModelStage`,
`strands.injection._message_injection`) is present there — which is what makes "community plugin"
a real claim rather than a repackaging of the fork.

---

## Layout

```
run.sh              create the venv on first use, then run the benchmark
report.sh           re-render an existing run's report — no Bedrock call
requirements.txt    the three community packages plus the public SDK
src/                the harness itself (a Python package, invoked by the two scripts)
results/            generated: one JSON, one Markdown report and one HTML page per run
```

`.venv/`, `.cache/`, `.artifacts/`, `results/` and `tmp/` are created here on first run and are
git-ignored.

Only two modules differ from the sibling harness: **`src/config.py`** (the configurations and the
model ids) and **`src/runner.py`** (how the three plugins are wired and metered). Everything
else — the scenario, the tool suite, accuracy scoring, metrics, the cost model, the report and the
HTML chart — is reused byte-for-byte, and the counter keys are deliberately the sibling's, so both
harnesses render through the same reporting path.

## Quick start

```bash
# all five configurations, 60 turns
./run.sh --total-turns 60 --tag cm60

# cheap wiring check (verifies engagement, not effect)
./run.sh --smoke

# one configuration
./run.sh --total-turns 60 --configs all --tag myrun
```

Valid configuration names: `baseline`, `relevance`, `disclosure`, `graph`, `all`.

Every run writes three artifacts to `results/`: `run-<tag>.json` (raw measurements, every answer
included), `report-<tag>.md`, and `curve-<tag>.html` (a self-contained page: result table plus
per-turn charts, no assets, no server).

---

## Three differences that are configuration, not detail

**1. The baseline installs no plugin at all.** The vended harness put its `ContextOffloader` in
every configuration, the baseline included, because without it the 60k–250k character payloads were
expected to overflow the window. Here the baseline is the unmodified agent: every payload enters the
history whole and stays there. That is the honest control — it measures the cost of doing nothing.

Measured: it does **not** overflow. All 60 turns completed on Opus 4.8, with zero errors, at
13.5M input tokens. So the comparison has a complete control rather than a truncated one.

**2. The graph is ephemeral.** The community plugin keeps its state in a weakly-keyed map and writes
nothing to `agent.state`, so there is no load path and no resume to measure. The sibling harness's
`--resume-at` and its `persist` variants have no counterpart here and are gone.

**3. The graph publishes no per-turn telemetry.** The vended plugin emitted one log record per turn
carrying the resolution ladder, which the sibling harness parses into curves. The community package
logs failures only, so the graph's evidence here is read off its end-of-run state
(`dialogue_at_end`, `evidence_at_end`, Cards, Links) and the per-turn ladder curves are unavailable.

---

## What the run measured

60 turns, five configurations, Opus 4.8, one replay each, zero errors.

| Configuration | Total tokens | Δ vs baseline | Accuracy | Materially correct | Cost (USD) |
|---|---:|---:|---:|:--:|---:|
| Baseline (no plugin) | 13,476,549 | — | 97.2% | 17/18 | $203.65 |
| Progressive Tool Disclosure only | 9,405,879 | −30.2% | 94.4% | 17/18 | $142.83 |
| Relevance Filtering only | 12,258,604 | −9.0% | 95.8% | 17/18 | $185.44 |
| Context Graph only | 9,663,413 | −28.3% | 91.5% | 15/18 | $146.17 |
| All three combined | 2,296,820 | **−83.0%** | 94.4% | 17/18 | **$35.88** |

All three engaged, and the evidence says they decided rather than merely ran: the graph ended with
44 Cards and 92 Links and a **three-rung ladder** in use (18 Cards at full content, 4 at
Description, 22 at Title), the relevance filter fired and spent 7 rerank search units, and the
disclosure catalog held the schema budget down.

The full stack sends **83% fewer tokens for the same 17 of 18 materially correct turns** as the
unmodified agent, at a sixth of the cost. It makes more model calls to do it — 104 against the
baseline's 87 — which is the trade the strategies make: a retrieval cycle is cheap next to resending
a 60k-character payload on every subsequent turn.

### Against the vended run

Same scenario and same model, so the absolute totals compare. The baselines do not — the vended one
carried an offloader — and neither do the single-strategy rows, since **every** vended configuration
carried that offloader and therefore a much smaller history than its community counterpart. The
honest comparison is the full stack, where both sides control payload, schema and history:

| | Vended (`graph-all`) | Community (`all`) |
|---|---:|---:|
| Total tokens | 2,255,712 | 2,296,820 (+1.8%) |
| Cost | $35.33 | $35.88 |
| Weighted accuracy | 95.8% | 94.4% |
| Materially correct | 17/18 | 17/18 |

**Parity.** The headline claim of the whole collection — roughly 80% fewer tokens at the same
answer quality — survives the move off the forked SDK onto three community packages and an
unmodified `strands-agents`. Getting there took one wiring fix, recorded next.

---

## The accuracy regression, and the fix

The first measurement of the full stack scored **84.5% weighted, 15 of 18 materially correct** —
clearly below the vended stack's 95.8% / 17 of 18. Of the three turns it got wrong, **two came from
one root cause**: `A5-statement` scored **0.00** where every other configuration scored 1.00, and
`R2-cross-reference` then failed because it asks about the fact A5 was supposed to establish.

The model said what went wrong in its own answer:

> "every export's artifact reference has come back **unreachable** … I can't read the stored
> artifacts."

Its tool calls on that turn were `expand_artifact` twice and `retrieve_context` once. The reason is
structural, and it belongs to the packaging rather than to the design:

- `RelevanceFilter` stores the sub-blocks it replaced and hands out references **its own**
  `retrieve_context` resolves.
- `ContextGraph` records addresses it saw in placeholder text and resolves them through **a store of
  its own**, via `expand_artifact`.
- **Nothing bridges the two.** The graph's README is explicit that its bridge to another plugin's
  stash is built entirely on private symbols and degrades to "answers as prose naming the miss".

The vended stack never hit this, because relevance lived *inside* the `ContextManager` whose stash
the graph bridged to — one store, one retrieval path. Split into two packages, there are two
plausible tools for one job and only one of them can resolve the reference.

**The fix, in the harness and not in the plugins:** when the relevance filter is installed, drop the
graph's `expand_artifact` tool, leaving exactly one artifact-retrieval path. Its other two tools,
`expand_card` and `find_context`, are untouched — they reach back into the conversation's own turns,
a different job the relevance filter does not do. See `_GRAPH_ARTIFACT_TOOL` in
[`src/runner.py`](src/runner.py); the run records `graph_artifact_tool_dropped` in its counters so
two runs stay distinguishable.

Re-measured with that one change, same 60 turns:

| `all` | Weighted accuracy | Materially correct | A5 | R2 |
|---|---:|:--:|:--:|:--:|
| Before | 84.5% | 15/18 | 0.00 | 0.70 |
| **After** | **94.4%** | **17/18** | **1.00** | **1.00** |

A5's tool calls became `list_investment_transactions`, `list_investment_transactions`,
`retrieve_context` — the right tool, and no `expand_artifact`. The one turn still wrong,
`C1-lambda-docs`, fails the same check in the graph-only arm and is a wording expectation about AWS
documentation ("retries the function **twice**"), not a context-engineering failure.

**Worth upstreaming:** the two packages should either share a reference store or agree on one
retrieval tool. Until they do, installing both means de-duplicating the overlap at the wiring site.

---

## Two other findings about the packages themselves

- **`strands-relevance-filter` ships no test suite.** Its `pyproject.toml` declares the dev
  dependencies and its README documents `hatch test`, but there is no `tests/` directory. The other
  two packages carry **641 passing tests** between them
  (`strands-context-graph` 560 passed / 1 skipped, `strands-progressive-tool-disclosure` 81 passed),
  so strategy A's only verification here is this harness.
- **The model does not search; it guesses and gets corrected.** Across 60 turns the disclosure
  plugin recorded **5 searches and 14 premature cancellations**: the catalog names a tool, the model
  calls it straight away with no arguments, the pre-call guard cancels and exposes the schema, and
  the model retries. The guard works — nothing is lost — but each occurrence costs one model cycle,
  which is why the full stack makes more model calls than the baseline (104 vs 87) while sending far
  fewer tokens.

---

## How the numbers are produced (reference)

- **Accuracy** — deterministic string checks against values computed from the mocked tools
  (`src/accuracy.py` + `src/ground_truth.py`). No judge model, so it adds no latency and no variance.
- **Token consumption** — measured **per model call** (a middleware captures the assembled messages
  and tool specs plus the provider's own `usage`), never from `accumulated_usage`, which reports a
  session running total and would make a per-call improvement look like a regression.
- **Cost** — one cost model (`src/compare.py` + the rates in `src/config.py`); the embedding and
  rerank calls a strategy makes on its own account are priced separately, not hidden.

A single replay is noise-dominated: the agent picks its own tool path and `temperature` cannot be
pinned on Opus 4.8. Treat the direction and the large moves as signal, and use `--repeats 3` before
trusting anything under ~20%.

### Verifying tokens against Bedrock

```bash
.venv/bin/python -m src.verify_logs results/run-cm60.json --per-call
```

Each request is stamped with metadata naming its configuration, turn and call, so the join against
Bedrock's invocation log is exact. Enabling that log is account-wide state; `src/run.py` checks it
during preflight and warns when a run will not be verifiable.

---

## Notes before sharing

- **`results/` runs carry the account id** (in each run's `meta`) and the full text of every answer.
  They are measurements, not code — scrub or omit them before publishing a specific run.
- **The tool fixtures are synthetic.** The institutions are fictional and every figure is computed
  by `src/ground_truth.py`. Changing a name means regenerating any recorded run, because the names
  are ground truth in `src/accuracy.py`.
