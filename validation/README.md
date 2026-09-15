# Validation Harness — Context Engineering

> **⚠️ Not for production use.** This deployment guide is provided for **experimentation and
> benchmark reproduction only**. The IAM roles, permissions, and configurations below are minimal
> examples — not production-ready baselines. Do not use them in a production environment without an
> independent security review, least-privilege hardening, and operational readiness assessment
> appropriate to your workload.

Reproducible benchmark for the context-engineering ideas in this repo. It replays the **same**
multi-turn conversation against a live Bedrock agent under two (or more) configurations and reports
**accuracy, token consumption, latency and cost** side by side, so the difference between them is the
strategy and nothing else.

One directory, one harness — it covers **one or more ideas** at once and can target **more than one
framework** (the token/cost path runs against Strands + Bedrock; the `agentcore/` path documents
deploying the strategies to AgentCore Runtime). It is not tied to a single idea.

The strategies under test are:

- **Relevance filtering** — score a tool result's chunks against the question, keep what answers it.
- **Progressive tool disclosure** — a lean tool catalog, full specs fetched on demand, forgotten after.
- **Context graph** — reorganizes the two above into one graph with remove/recover over an immutable
  log (it subsumes the earlier background-curator idea).

The `graph-all` configuration applies all three together.

---

## What you need

- **Python 3.12** and [`uv`](https://github.com/astral-sh/uv) (`brew install uv`).
- **AWS credentials** for an account with these Bedrock models enabled, in `us-east-1`:
  `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`.
  Credentials resolve through the standard AWS chain (an SSO session, exported keys, an instance
  role) or a named profile in `VALIDATION_AWS_PROFILE`. Nothing is hardcoded; the account is
  discovered from STS. To pin the account a run must execute in, set `VALIDATION_ACCOUNT_ID`.
- Internet on the first run: it downloads a Chromium build (for the web tool) and a handful of AWS
  doc pages into `.cache/`, then reuses them so every configuration sees byte-identical payloads.

Re-rendering a recorded run's report (Markdown + HTML) needs **no AWS credentials at all** — only a
live benchmark run does.

---

## The SDK dependency

The strategies are vended plugins of a **forked Strands SDK**, published on a public fork. You do
**not** need to clone anything — the harness installs it straight from git. `requirements.txt` pins the
exact **commit SHA** the results in `results/` were measured against (an immutable reference, not a
moving branch tip):

```
strands-agents @ git+https://github.com/scandura/harness-sdk.git@c4083a04d35170b0626cfbdbb5e4926d9b35af20#subdirectory=strands-py
```

Prefer to read or hack on the SDK code locally? Clone the fork at that commit and swap that line for an
editable install (there is a comment in `requirements.txt` showing exactly this):

```bash
git clone https://github.com/scandura/harness-sdk.git
git -C harness-sdk checkout c4083a04d35170b0626cfbdbb5e4926d9b35af20
pip install -e harness-sdk/strands-py
```

---

## Quick start — reproduce a run

From the repo root. The first invocation creates the venv, installs the SDK from git, and installs
Chromium; later invocations reuse them.

```bash
# baseline vs all three strategies, 60 turns, one replay
./validation/run.sh --configs baseline graph-all --total-turns 60 --tag myrun
```

Every run writes **three artifacts** to `validation/results/`, automatically:

| File | What it is |
|---|---|
| `run-<tag>.json` | raw measurements, including every answer |
| `report-<tag>.md` | the rendered report (accuracy → cost → latency) |
| `curve-<tag>.html` | **self-contained HTML page**: result table + four per-turn charts, no assets, no server |

Open the HTML:

```bash
open validation/results/curve-myrun.html
```

`report-latest.md` always points at the most recent run.

### Other run shapes

```bash
./validation/run.sh --smoke                          # cheap wiring check (verifies engagement, not effect)
./validation/run.sh --configs baseline graph-all ... # any subset of configurations
./validation/run.sh --repeats 3 --tag myrun          # averaged with spread — needed for deltas under ~20%
./validation/run.sh --sequential                     # clean latency numbers (~5x slower)
```

Valid configuration names: `baseline`, `disclosure`, `relevance`, `graph`, `all`, `graph-all`.

### Re-render without re-running (free)

Change a price in `config.py`, or just regenerate the report/HTML from an existing run — no Bedrock call:

```bash
./validation/report.sh myrun            # rebuilds comparison-*.md, curve-*.csv and curve-*.html
./validation/report.sh myrun --verify   # also reconciles the token counts against Bedrock's logs
```

> The live `run.sh` already emits the HTML; `report.sh` is for re-rendering an old run or adding `--verify`.

---

## What a result looks like

A representative run — `baseline` vs `graph-all`, 60 turns, Opus 4.8:

| Configuration | Agent input tokens | vs baseline | Correct turns | Tokens / correct |
|---|---:|---:|---:|---:|
| Baseline (offloader, prefix preview) | 11,225,270 | — | 15.0 | 748,351 |
| Graph + disclosure + relevance | 2,150,604 | **−80.8%** | 17.0 | **−83.1%** |

Same behaviour, ~80% fewer tokens. A single replay is noise-dominated (the agent picks its own tool
path and `temperature` cannot be pinned on Opus), so treat the **direction and the large moves** as
signal and use `--repeats 3` before trusting anything under ~20%.

---

## Deploying to AgentCore Runtime: `agentcore/`

The token harness above answers *what the strategies cost*. Running the same strategies as a real,
deployed agent — to observe behaviour end to end — is done on **Amazon Bedrock AgentCore Runtime**.

Deployment is a **manual, documented procedure** using the official `agentcore` CLI, not a scripted
build: the container installs the forked SDK straight from its public git branch (pinned to a commit
SHA), so there is no local wheel build. The full step-by-step — configure, create memory, deploy,
invoke, tear down — is in [`agentcore/DEPLOY.md`](agentcore/DEPLOY.md).

`agentcore/` therefore holds only what that path needs: the how-to (`DEPLOY.md`), a `.env.example`
for the runtime environment, and the container requirements. Everything is paid and account-changing,
and `DEPLOY.md` flags each step accordingly.

---

## How the numbers are produced (reference)

- **Accuracy** — deterministic string checks against values computed from the mocked tools
  (`accuracy.py` + `ground_truth.py`). No judge model, so it adds no latency and no variance.
- **Token consumption** — measured **per model call** (a middleware captures the assembled messages
  and tool specs plus the provider's own `usage`), never from `accumulated_usage`, which reports a
  session running total and would make a per-call improvement look like a regression.
- **Timing** — per model call, per turn, per run.
- **Cost** — one cost model (`compare._row` + the rates in `config.PRICING`); auxiliary embedding/rerank
  calls a strategy makes are priced separately, not hidden.

**Two calibrations decide whether this measures anything** (`run.py` prints them at startup and warns
on drift):

| | Value | Why |
|---|---|---|
| Schema budget | ~63k tokens/call, ~94 tools | Matches the motivating real session, where tool schema was ~85% of a call's floor. At 11k tokens progressive tool disclosure measured −18%; at 63k it measures −61%. |
| Topic-line length | 5 turns per line | The motivating case had 4 turns of connector debugging at 44.7% of session tokens. With 2-turn lines the history strategy measured +0.1%; with 5-turn lines it cuts history 18%. |

### Verifying tokens against Bedrock

Usage read off the model stream understates two cases: a call that raises yields no usage, and a
provider-side retry is one stream to the harness but two billed invocations. Bedrock's invocation log
has neither blind spot; each request is stamped with metadata naming its configuration/turn/call, so the
join is exact:

```bash
validation/.venv/bin/python -m validation.verify_logs validation/results/run-myrun.json --per-call
```

Enabling the invocation log is account-wide state (not something a run configures); `run.py` checks it
during preflight and warns when a run will not be verifiable. Entries take 2–3 minutes to deliver.

---

## Notes before sharing

- **`results/` runs carry the account id** (in each run's `meta`) and the full text of every answer.
  They are measurements, not code — scrub or omit them before publishing a specific run.
- **The mocked tool fixtures name real financial institutions.** They contain no personal data (all
  figures are synthetic, computed by `ground_truth.py`), but they are recognisable brands; a public demo
  reads cleaner with fictional ones. Changing them means regenerating any recorded run, because the names
  are ground truth in `accuracy.py`.
