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
- **AWS CLI v2**, only to configure and verify credentials.
- Internet on the first run: it downloads a Chromium build (for the web tool) and a handful of AWS
  doc pages into `.cache/`, then reuses them so every configuration sees byte-identical payloads.

Re-rendering a recorded run's report needs **no AWS credentials at all** — only a live benchmark run
does.

### The AWS account

A live run calls Bedrock on your own account and **spends real money**: the figures below cost roughly
$750 on Opus 4.8 and $46 on Haiku 4.5 for one pass of five configurations. Use an account you are
happy to bill, and read the cost column before launching.

These models must be **enabled** in the region you run in (Amazon Bedrock → Model access):

| Model id | Used by | Enabled for |
|---|---|---|
| `us.anthropic.claude-opus-4-8` | the agent under test | every configuration |
| `cohere.rerank-v3-5:0` | relevance filtering | `relevance`, `all` |
| `cohere.embed-multilingual-v3` | the graph's similarity matcher | `graph`, `all` |

The identity the run uses needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on
those model ids, `bedrock:Rerank` on `bedrock-agent-runtime`, `bedrock:GetModelInvocationLoggingConfiguration`
for the preflight check, and `sts:GetCallerIdentity`. Scope it to those resources rather than using a
wildcard. Token verification additionally reads CloudWatch Logs (`logs:FilterLogEvents`,
`logs:GetLogEvents`) on the invocation log group.

### Credentials, with the AWS CLI

Nothing is hardcoded and no credential is written to disk by this harness. It builds its boto3 session
from the **standard credential chain**, so whatever the AWS CLI is configured with is what the run
uses. The account is discovered from STS and printed at startup.

```bash
# interactive: paste an access key pair, choose a region
aws configure

# or, if your organisation uses IAM Identity Center (SSO)
aws configure sso
aws sso login --profile my-profile
export AWS_PROFILE=my-profile
```

Verify before launching — the run's own preflight checks the same things, but failing here costs
nothing:

```bash
export AWS_DEFAULT_REGION=us-east-1

aws sts get-caller-identity                      # account and ARN the run will use
aws bedrock list-inference-profiles \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'opus-4-8')].inferenceProfileId" \
  --output table                                 # the agent model is reachable
aws bedrock get-model-invocation-logging-configuration   # optional: makes tokens verifiable
```

Two environment variables change what the harness does with those credentials:

| Variable | Effect |
|---|---|
| `VALIDATION_AWS_PROFILE` | use this named profile instead of the default chain |
| `VALIDATION_ACCOUNT_ID` | refuse to run unless the credentials resolve to this account |

Two more change what is measured:

| Variable | Effect |
|---|---|
| `VALIDATION_SESSION` | `file` (default) attaches a `FileSessionManager` to the baseline agent; `off` removes it from the baseline too |
| `VALIDATION_RELEVANCE_RETRIEVAL_TOOL` | `1` (default) stores the raw content and registers the filter's `retrieve_all_context`; `0` measures the excerpt alone |

`VALIDATION_ACCOUNT_ID` is the guard worth setting when more than one account is in play: a run that
silently used the wrong one would produce numbers attributed to the wrong place. Both are read from the
environment rather than written into `src/config.py`, so nothing about one workstation's setup is
committed.

The last line above is optional but recommended. Bedrock's model invocation log is the only independent
check on the token numbers this harness reports; `src/run.py` reads its configuration during preflight
and warns when a run will not be verifiable afterwards.

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
./run.sh --total-turns 60 --tag op60

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

## What the script asks

Half of `--total-turns` is scored: at 60 turns that is 30 scored turns — the 18 hand-written spine,
the memory probes, and generated filler to fill the rest. The other half is unscored mass, and it is
where the heaviest payloads live (a 30-day statement export is the largest single result in the suite).

### Memory probes

The spine's return turns only test recall at the very end. The long script adds probes at several
depths, built so the answer can come from nowhere but the conversation.

`M0-seed` is a user turn stating two facts and asking for no lookup — an emergency-fund target and an
advisor's name. **No tool returns either**, so the only place they exist is the history (the
baseline's session) or the graph. Four probes follow, spread through the script:

| Probe | Asks for | Source of the fact |
|---|---|---|
| `M1-seed-target` | the emergency-fund target | the seed turn only |
| `M2-sync-job` | the job id of the connector sync, *without looking it up again* | a tool result in line B |
| `M3-seed-advisor` | the advisor's name | the seed turn only |
| `M4-case-id` | the support case number, *without looking it up again* | a tool result in line B |

Each probe is a scored turn with one critical literal check, and each is scored twice:

- **Correct** — the literal is in the answer.
- **Recalled** — correct **and** the turn called no domain tool. A right answer after re-calling
  `force_connector_sync` proves the tool works, not the memory. A retrieval tool does not disqualify
  it: the graph's `expand_card` and `find_context` *are* its memory, so they count as memory, as does
  `find_tools`.

The report renders both in a **Memory probes** table, one row per configuration, with the probe count.
`Correct` above `Recalled` on a plugin arm is the interesting reading: the fact survived, but the
answer paid a tool call to get it back. A short script carries no probes, and the table is omitted.

### The scored filler asks only questions that are true

Scored filler turns carry one static expectation each, keyed by the *kind* in the label, which works
only because the mocked tools return the same answer for every account. It also requires the premise
to hold. An earlier revision paired any account with a fixed premise — a savings account's yield, one
institution's connector on another institution's account, a role and bucket nobody had named — and a
careful model answered by correcting the premise without calling a tool, which scored as a critical
failure in **every** arm, so the arms differed by which run happened to push back.

What the prompts hold to:

- **projection** is asked only of investment accounts, the account types where a yield projection
  means anything.
- **connector** and **logs** name the account's **own** institution.
- **iam** and **objects** name the IAM role and the S3 bucket **in the question**, so the resource the
  tool needs exists.

Phrasing gives no hint about which tool to call, so tool selection stays the model's job and the
disclosure strategy is tested rather than bypassed.

---

## Three differences that are configuration, not detail

**1. The baseline installs no plugin at all.** The vended harness put its `ContextOffloader` in every
configuration, the baseline included, so nothing could overflow the window. Here the control is the
unmodified agent and every payload enters the history whole.

Measured: it completes, but only just, and only on a large-context model. All 60 turns finished at
14.4M input tokens with a **peak of 204,439 tokens on a single call** — above Haiku 4.5's entire
window. The baseline is what makes the comparison honest; it is also the arm that decides which models
this benchmark can run on.

The baseline is the only arm that gets a **session manager**. An agent as it ships persists its
messages, so the control does too: Strands' `FileSessionManager`, under `SESSIONS_DIR/<run tag>/`. The
plugin arms run without one, because what they measure is what a strategy changes on top of that
agent. The session id is **fresh for every agent** — configuration name plus a random suffix — since
an existing id is *restored*, and a repeat that reloaded the previous replay's history would be
measured on a context it did not build. `VALIDATION_SESSION=off` removes the manager from the baseline
as well; the id a run used is recorded in its counters as `session_id`.

**2. The graph is ephemeral.** The community plugin keeps its state in a weakly-keyed map and writes
nothing to `agent.state`, so there is no load path and no resume to measure. The sibling harness's
`--resume-at` and its `persist` variants have no counterpart here and are gone.

**3. The graph publishes no per-turn telemetry.** The vended plugin emitted one log record per turn
carrying the resolution ladder, which the sibling harness parses into curves. The community package
logs failures only, so the graph's evidence here is read off its end-of-run state
(`dialogue_at_end`, `evidence_at_end`, Cards, Links) and the per-turn ladder curves are unavailable.

---

## The graph has ONE tuning, and the split it used to have was dropped on a design argument

`src/config.py` carries a single set of graph thresholds, `GRAPH_TUNING`, used by every arm the graph
appears in. Each run records it under `graph_tuning`.

| | `GRAPH_TUNING` | package default |
|---|---:|---:|
| `expand_threshold` | 0.62 | 0.55 |
| `description_tokens` | 250 tight window / 100 large | 100 |
| `body_budget` | 40,000 | none |
| `neighbors_per_candidate` | 3 | 3 |

`description_tokens` is no longer a literal: it comes from the window regime (see `TIGHT_WINDOW_CEILING`),
which is a second change the unification made. The 250 that was measured for the graph-alone arm was
measured on Haiku 4.5 — a 200K window, so the tight regime, where the unified set still gives 250. On a
large-window model that arm now gets 100 instead, on the regime's reasoning that a larger budget there
buys a larger bill rather than a fact. That is a change to the graph-only arm's configuration on
large-window models and it is unmeasured; `VALIDATION_GRAPH_DESCRIPTION_TOKENS=250` restores it.

It used to be two sets, selected on whether the relevance filter was installed, on the reasoning that
*"when relevance has already compressed the evidence, the graph should fold less, not more"*. **That
reasoning treats the two plugins as rivals for one job, and they are not.** They act at different
moments, on different material:

- The relevance filter acts on `AfterToolCallEvent`, on a payload that has not entered the history yet.
  Its job is to decide what of that payload is worth keeping. Where the survivor goes next — the
  history, the graph, nowhere — is not its concern.
- The graph acts at delivery, on a history that already exists. It never sees a payload; it sees what
  was written down.

So the filter makes the graph's input *smaller*, not *different in kind*, and a knob deciding how
aggressively to fold a history has no business reading whether another plugin trimmed it first.

The measurement that justified the split is also confounded, which is what made it safe to drop. On
Haiku 4.5, one replay each:

| Arm | Tuning | Tokens | Peak call | Accuracy | Materially correct | Ladder full/desc/title |
|---|---|---:|---:|---:|:--:|:--:|
| graph | defaults | 9,045,017 | 139,917 | 85.0% | 23/30 | 17/10/6 |
| graph | **tuned** | **8,462,341** | **119,342** | **89.0%** | **24/30** | **12/15/6** |
| all | defaults | 2,406,570 | 49,863 | 81.9% | **21/30** | 21/7/5 |
| all | tuned | 2,377,270 | 40,622 | 74.8% | **16/30** | 13/15/7 |

*Tuned, the graph alone improves on both axes; the same values applied to all three lost five materially
correct turns while moving tokens 1.2% — which is what produced the split.*

That last row ran with `preview_tokens` at **800**, against payloads ten to thirty times that. The Cards
were starved by the FIRST cut in the chain, so a larger Description budget had nothing left to preserve:

    payload -> preview budget -> the message -> the Card's numeric lines -> Description budget

The preview is 2,000 now, so the condition that produced the result no longer holds. The unified set is
therefore **unmeasured as a unified set** — every knob is env-overridable, and `--repeats 3` on the
combined arm is what would settle it.

### What the tuning actually fixed

Not lost information. The graph was answering `42.1%` where the tool had emitted `42,1%` — the right
figures, reformatted — and this harness scores that as wrong on purpose: an assistant that restates a
value in its own format has introduced an error class.

`compose_description` copies `Card.numeric_lines` **verbatim** and the budget decides only *how many*
of those lines get in, appending `(+N numeric lines omitted)` for the rest. At 100 tokens a Card whose
turn carried a real payload keeps a handful, so a later turn answering from that Card re-renders the
figure from its own paraphrase, and that is where the separator flips. Measured: the graph answered one
allocation turn with **no tool call at all**, reading a truncated Card instead.

Caveat: the tuning was derived on Haiku and has not been isolated on Opus — the Opus run applies it,
but there is no Opus measurement with the defaults on this script to compare against. One replay each,
so treat the one-turn accuracy gain as inside the noise and the five-turn regression as the direction
it points. `--repeats 3` is what would settle either.

---

## What the run measured

60 turns, five configurations, Claude Opus 4.8, one replay each, zero errors, zero calls refused by
the context window. **Half the script is scored**: 30 of the 60 turns carry expectations — the 18
hand-written spine plus 12 generated — and the other 30 are unscored mass.

| Configuration | Total tokens | Δ vs baseline | Accuracy | Materially correct | Turn (s) | Cost (USD) |
|---|---:|---:|---:|:--:|---:|---:|
| Baseline (no plugin) | 14,377,382 | — | 96.9% | 28/30 | 11.7 | $72.50 |
| Progressive Tool Disclosure only | 9,466,084 | −34.2% | 94.5% | 27/30 | 12.8 | $48.10 |
| Relevance Filtering only | 12,766,237 | −11.2% | 96.9% | **29/30** | 11.7 | $64.45 |
| Context Graph only | 10,385,714 | −27.8% | 93.7% | 27/30 | 10.8 | $52.43 |
| **All three combined** | **2,569,888** | **−82.1%** | 96.1% | 28/30 | **8.7** | **$13.36** |

*The full stack sends 82% fewer tokens for the same 28 of 30 materially correct turns as the
unmodified agent, at a fifth of the cost and three seconds faster per turn — the turn times are
measured with all five configurations running concurrently, so read them as relative to each other
rather than as isolated latency.*

All three engaged, and the evidence says they decided rather than merely ran: the graph ended with a
**three-rung ladder** in use (12 Cards at full content, 14 at Description, 7 at Title when run alone),
the relevance filter fired and spent rerank search units, and the disclosure catalog held the schema
budget down.

It makes more model calls to do it — 101 against the baseline's 87 — which is the trade the practices
make: a retrieval cycle is cheap next to resending a 60k-character payload on every subsequent turn.
It is also the *fastest* per turn despite those extra calls, because each one carries far less.

### The same run on Claude Haiku 4.5

Worth stating because one strategy changes sign between the two models, which means no
single-strategy figure should be quoted without naming the model it came from.

| Configuration | Total tokens | Δ vs baseline | Accuracy | Materially correct | Turn (s) | Cost (USD) |
|---|---:|---:|---:|:--:|---:|---:|
| Baseline (no plugin) | 11,038,450 | — | 85.8% | 21/30 | 6.6 | $11.10 |
| Progressive Tool Disclosure only | 9,844,491 | −10.8% | 87.4% | 22/30 | 7.4 | $9.93 |
| Relevance Filtering only | 13,418,624 | **+21.6%** | 70.9% | 19/30 | 9.4 | $13.53 |
| Context Graph only | 8,498,825 | −23.0% | 89.0% | **24/30** | 7.3 | $8.58 |
| **All three combined** | **2,431,848** | **−78.0%** | 81.9% | 21/30 | **5.5** | **$2.52** |

*Haiku is ~15x cheaper per token and reaches the same conclusion about the full stack, but it is a
weaker model and it retrieves more — which is what flips relevance filtering from a saving to a cost.*

**Relevance filtering costs 21.6% *more* than doing nothing on Haiku and saves 11.2% on Opus.** Every
`retrieve_context` result became a conversation message and then rode along on every later call, so
a model that retrieves repeatedly paid for the same content many times. Haiku did exactly that and
exceeded the window outright on four calls (201,035 tokens against a 200,000 limit); Opus answered
from the preview more often and never overflowed.

*That is the mechanism as it was measured.* The tool is `retrieve_all_context` and the plugin
removes its exchanges from the history at the end of the invocation, which removes the re-send this
figure is made of. The sign on Haiku has not been re-measured since, so `+21.6%` should be quoted as
history, not as the current cost of the arm.

Peak input on a single call, Opus: baseline 204,439 — above Haiku's entire window, which is why the
baseline is only runnable here on a model with more than 200k of context; relevance 174,062;
disclosure 149,681; graph 147,948; all three 49,943.

### Against the vended run

**Not comparable any more, and the earlier claim of parity should not be repeated as a like-for-like.**
The vended figures in [`../../README-vended-plugins.md`](../../README-vended-plugins.md) were measured
on the previous version of this script — 18 scored turns, and a filler that asked about accounts the
fixture did not hold, so 36 of 42 filler turns made no tool call and contributed almost no payload
mass. This script grounds every filler turn, which is why its baseline carries 14.4M input tokens
where the vended one carried 11.3M.

The strategies' *direction* and *relative* ordering agree across both. Settling whether the two
packagings reach the same absolute number would mean re-running the vended harness on the corrected
script, which has not been done.

---

## The accuracy regression, and the fix

Measured when the two plugins were first combined, on the earlier 18-scored-turn version of this
script: the full stack scored **84.5% weighted, 15 of 18 materially correct**, below each strategy on
its own. Of the three turns it got wrong, **two came from one root cause**: `A5-statement` scored
**0.00** where every other configuration scored 1.00, and `R2-cross-reference` then failed because it
asks about the fact A5 was supposed to establish.

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

**The fix at the time, in the harness and not in the plugins:** when the relevance filter was
installed, drop the graph's `expand_artifact` tool, leaving exactly one artifact-retrieval path. Its
other two tools, `expand_card` and `find_context`, were untouched — they reach back into the
conversation's own turns, a different job the relevance filter does not do. See
`_GRAPH_ARTIFACT_TOOL` in [`src/runner.py`](src/runner.py); the run records
`graph_artifact_tool_dropped` in its counters so two runs stay distinguishable.

**That drop has since been removed, and not because the filter's tool went away.** The tool was
**renamed and narrowed**: `retrieve_context` is `retrieve_all_context`, scoped to the question an
excerpt cannot answer — one that needs every row — and the excerpt's disclaimer names it, with the
reference and the budgets to pass. Every `retrieve_context` on this page belongs to a run as it was
measured; read it as that tool's earlier name and earlier scope.

`include_retrieval_tool` defaults to `True`, and the harness follows it
(`VALIDATION_RELEVANCE_RETRIEVAL_TOOL=0` turns it off). So both tools are installed: every arm with
the graph runs `include_artifact_tool=True` and records `graph_artifact_tool_dropped` as `false`. The
disambiguation moved from the wiring into the tool — a distinct name, a stated scope, and the
disclaimer telling the model which call to make — and the filter's tool is kept **out of** the
disclosure arm's `always_available`, so it is loaded from the catalog only when a whole-result question
comes up. Whether the model still confuses the two is what the `all` arm measures; it is not assumed.
The figures immediately below were measured with the drop in place.

Re-measured with that one change, same 60 turns:

| `all` | Weighted accuracy | Materially correct | A5 | R2 |
|---|---:|:--:|:--:|:--:|
| Before | 84.5% | 15/18 | 0.00 | 0.70 |
| **After** | **94.4%** | **17/18** | **1.00** | **1.00** |

*One de-duplication at the wiring site recovered both turns; the figures are on the earlier
18-scored-turn script, which is where the regression was found.*

A5's tool calls became `list_investment_transactions`, `list_investment_transactions`,
`retrieve_context` — the right tool, and no `expand_artifact`.

**Worth upstreaming:** the two packages should either share a reference store or agree on one
retrieval tool. Until they do, installing both means de-duplicating the overlap at the wiring site.

---

## Two other findings about the packages themselves

- **Test coverage is uneven, and two documented contracts do not match the code.** The three packages
  carry **764 passing tests** between them — `strands-context-graph` 560 (plus 1 skipped),
  `strands-relevance-filter` 123, `strands-progressive-tool-disclosure` 81. The relevance filter's
  suite was written against its documented contract after the fact, and writing it surfaced two places
  where the docstring and the code disagree, neither of which is a defect in the behaviour:
  `retrieve_context` — since renamed `retrieve_all_context` — documents `ValueError` when a
  `line_range` "falls outside the content" but an over-large `end` is silently clamped, grep-style;
  and a truncated preview's closing gap marker can report the source's whole line count even when part
  of the first line was rendered.
- **The model does not search; it guesses and gets corrected.** Across 60 turns on Opus the disclosure
  plugin recorded **1 search against 15 premature cancellations** when run alone, and 3 against 14 in
  the full stack: the catalog names a tool, the model calls it straight away with no arguments, the
  pre-call guard cancels and exposes the schema, and the model retries. The guard works — nothing is
  lost — but each occurrence costs one model cycle, which is why disclosure makes 100 model calls
  against the baseline's 87 while sending 34% fewer tokens.

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

### Correctness fixes since the runs above were measured

Figures recorded before these landed are not comparable to figures recorded after, so a run tag is
worth keeping next to any number quoted from this page.

Scoring:

- **An empty answer scores 0.** A turn that produced no text — a context-window overflow, say — used
  to pass every `none_of` check, because nothing forbidden is present in nothing, so the arm that
  overflowed most collected the most free weight. It scores as a `no-answer` critical failure worth the
  full weight of the turn's checks.
- **`C2` and `R2` no longer accept the opposite needle or match inside a word.** `valid` is a
  substring of `invalid`, so the verdict is matched in phrase form (`is valid`, `a valid`) with the
  opposite verdict forbidden outright; and the comparison check no longer counts bare `less`/`more`,
  which matched inside `unless` and `moreover`.
- **The allocation is derived from the positions**, so `A3` and `R1` agree instead of expecting shares
  no account holds. There is no `allocation` kind in the filler for the same reason: each account's
  allocation differs, so no single static string can be its expectation.
- **The accuracy denominator is averaged like the numerator** across replays, and `turns_scored` per
  replay is reported alongside it.

Cost and tokens:

- **Cost is priced from the run's own model**, read out of its `meta`, so a `--report-only`
  re-render bills the model that produced the run rather than whatever the config currently names.
- **Cache read and write tokens are counted in the token column** and priced at their own rates,
  instead of being left out of the prompt mass.
- **Crashed replays are kept out of the averages**, with their errors still surfaced in the summary.
- **`model_seconds` stops at the provider's stop event**, so it measures the call rather than the
  consumer.

Counters:

- **`retrievals` counts the real retrieval tools** — the relevance filter's `retrieve_all_context`
  plus the graph's `expand_artifact`, `expand_card` and `find_context`. It used to key off
  `retrieve_offloaded_content`, a vended-SDK tool that never exists in this harness, so the column read
  0 on every run. The memory probes treat the same four, plus `find_tools` and `get_tool_details`, as
  not re-fetching a domain value.
- **The graph's rerank is read from `rerank_observed`** — the harness's own metered count of what the
  matcher sent — rather than inferred.
- **`neighbors_per_candidate` defaults to `0`** (`VALIDATION_GRAPH_NEIGHBORS` overrides).
- Report prose no longer says the baseline includes an offloader. It does not; it installs no plugin.

### Verifying tokens against Bedrock

```bash
.venv/bin/python -m src.verify_logs results/run-op60.json --per-call
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
