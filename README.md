# Context Engineering

> **⚠️ Not for production use.** This repository is a **reference implementation and validation
> harness** for context-engineering research. It is provided for experimentation, benchmarking, and
> learning purposes only. **Do not deploy to production** without an independent security review,
> infrastructure hardening, and testing appropriate to your workload and compliance requirements. IAM
> roles, permissions, and deployment configurations shown here are minimal examples — not
> production-ready baselines.

A collection of **context-engineering practices** for LLM agents — techniques that keep an agent's
working context small and relevant as a conversation grows, without losing the information the task
actually needs.

The practices ship as **three community plugins** for [Strands Agents](https://strandsagents.com):
ordinary installable packages that attach to the SDK's extension surface. No SDK fork, no pinned
commit — they run against an unmodified `strands-agents` from PyPI.

## Why

The collection starts from a measured diagnosis of a real agent session (13 turns, ~29 minutes), not
from intuition. Three findings shaped the ideas:

- **The context is dominated by what the task does *not* need.** In the measured session, four turns
  of tool-connector troubleshooting accounted for **44.7%** of all token consumption — legitimately
  discussed, but not the objective, and re-sent on every following turn.
- **A large fixed floor is paid on every call.** Roughly **63k tokens per call** were tool schema
  alone — about 85% of a call's floor — reprocessed on all 33 model calls with the prompt cache off.
- **Retrieved memory is accumulated, not transient.** Retrieved records are inserted *into* the
  conversation and re-sent every turn, making irrelevant retrieval a **quadratic** cost rather than a
  one-off.

The common thread: an agent's context grows with *everything that happened*, while a good answer needs
only *what the activity in progress requires*. The practices here separate those two — keeping the
task's **attention memory** resident and letting the **background** be reachable on demand instead of
resident.

This direction is consistent with published work: AWS's Strands benchmark for compaction-plus-offload
context management reports **cost −55%** with **accuracy rising 68% → 98%**
([reduced cost, better isolation, more resilience](https://strandsagents.com/blog/reduced-cost-better-isolation-more-resilience/)).
The practices below take that further and measure each one against a baseline in this repo.

## The practices

Each practice is documented in three layers: **the idea** (framework-agnostic), **an example** (real
framework code, in the same doc), and **results** (the reproducible benchmark under
`validation/community-plugin-A-B-D/`).

| | Practice | Idea | Package | Design |
|---|---|---|---|---|
| **A** | **Relevance filtering** | Score a tool result's chunks against the question and keep only what answers it, instead of a positional slice. | [`strands-relevance-filter`](community-plugins/strands-relevance-filter/) | [`design-a-relevance-filtering.md`](docs/design/design-a-relevance-filtering.md) |
| **B** | **Progressive tool disclosure** | Send a lean tool catalog; fetch a tool's full spec on demand, then forget it — attacking the ~63k schema floor. | [`strands-progressive-tool-disclosure`](community-plugins/strands-progressive-tool-disclosure/) | [`design-b-progressive-tool-disclosure.md`](docs/design/design-b-progressive-tool-disclosure.md) |
| **D** | **Context graph** | Reorganize the two above into one graph with remove/recover over an immutable log. It subsumes the earlier background-curator idea (C), which is why there is no standalone C. | [`strands-context-graph`](community-plugins/strands-context-graph/) | [`design-d-context-graph.md`](docs/design/design-d-context-graph.md) |

Each package is independent: install one, two, or all three.

## How to install and use the community plugins

**→ [`how-to/02-community-plugins-agent-sample.md`](how-to/02-community-plugins-agent-sample.md)**

That guide is the runnable path from nothing to a working agent: what you need, a minimal agent with
one oversized tool, then each plugin wired on its own, then all three together — with the exact
constructor arguments and what each one costs.

It also carries **four gotchas you should read before combining the plugins**, because two of them
cost a measured benchmark run its answers and neither fails loudly:

1. **Two retrieval tools, one job.** The relevance filter and the context graph each ship an
   artifact-retrieval tool over its own store, and nothing bridges them. Installed together, the model
   reaches for the wrong one and gets an unresolvable reference. Fixing it recovered two scored turns.
2. **`NullConversationManager` is a precondition of the graph**, not a suggestion — any other manager
   can physically drop what the graph only meant to fold.
3. **The relevance threshold is a position in a distribution, not a number.** The package default of
   `0.5` rejects every chunk with `cohere.rerank-v3-5`, whose strong matches score ~0.29.
4. **The graph should fold *less* when the relevance filter is present, not more.** They compete for
   the same job: the filter has already replaced the payload with a preview before the graph derives
   its Card. Thresholds that improved the graph alone, applied to all three together, lost five
   materially correct turns while moving tokens 1.2%.

Quick install, from a clone of this repository:

```bash
pip install -e community-plugins/strands-context-graph
pip install -e community-plugins/strands-progressive-tool-disclosure
pip install -e community-plugins/strands-relevance-filter
pip install "strands-agents>=1.44.0,<2.0.0"
```

Verified against **`strands-agents` 1.56.0** from PyPI.

## The result

**Total tokens** is the agent's own `usage`, input plus output, plus every auxiliary token the strategy
spent on its own account: the graph's embedding calls and the filter's rerank calls. **Cost** applies
the rates declared in `validation/community-plugin-A-B-D/src/config.py` to measured units — without
that, the strategies that buy their saving with a second model call would rank better than they are.

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 14,377,382 | — | 96.9% | 28/30 | 11.7s | $217.50 | — |
| Progressive Tool Disclosure only | 9,466,084 | −34.2% | 94.5% | 27/30 | 12.8s | $144.30 | −33.7% |
| Relevance Filtering only | 12,766,237 | −11.2% | 96.9% | 29/30 | 11.7s | $193.33 | −11.1% |
| Context Graph only | 10,385,714 | −27.8% | 93.7% | 27/30 | 10.8s | $157.30 | −27.7% |
| **All three combined** | **2,569,888** | **−82.1%** | **96.1%** | **28/30** | **8.7s** | **$40.07** | **−81.6%** |

*The full stack sends 82% fewer tokens for the same 28 of 30 materially correct turns as the
unmodified agent, at a sixth of the cost and three seconds faster per turn — turn times are measured
with all five configurations running concurrently, so they are relative to each other rather than
isolated latency.*

Agent `us.anthropic.claude-opus-4-8`, region `us-east-1`, one **replay** per configuration — the same
scripted conversation run start to finish, once for each row, 60 turns, zero errors.

The baseline here is a **bare agent**: no plugin at all, every tool payload entering the history whole
and staying there. That is the honest control — it measures the cost of doing nothing. It completes,
but only just: it peaked at **204,439 input tokens on a single call**, which is more than some models'
entire context window, so the baseline is what decides which models this benchmark can run on.

The large differences — **−82.1%** with correctness holding at 28 of 30 turns and cost dropping to a
sixth — support a decision; nothing under ~20% does. A single replay cannot be more precise than that:
the agent chooses its own tool path, so two replays of the *same* configuration already differ by
roughly that much. Full breakdown (per-token cost attribution, per-turn curves) is regenerated by the
harness under `validation/community-plugin-A-B-D/`.

The full stack makes **more model calls** to send fewer tokens — 101 against the baseline's 87 — which
is the trade the practices make: a retrieval cycle is cheap next to resending a 60k-character payload
on every subsequent turn. It is also the fastest per turn despite those extra calls, because each one
carries far less.

**One strategy changes sign with the model.** The same run on Claude Haiku 4.5 — ~15x cheaper per
token — reaches the same conclusion about the full stack (−78.0%, $2.52 against $11.10) but has
relevance filtering costing **21.6% more** than doing nothing instead of saving 11.2%. Every
`retrieve_context` result becomes a conversation message and rides along on every later call, so a
model that retrieves repeatedly pays for the same content many times. Do not quote a single-strategy
figure without naming the model it came from; see
[`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) for both
tables.

### How correctness is measured

The two correctness columns are different measures, not the same one twice. **Half the script is
scored**: 30 of the 60 turns carry weighted expectations — 18 hand-written turns plus 12 generated —
and the other 30 are unscored mass, there to lengthen the history. Expectations are strings the answer
must contain, may contain, or must *not* contain, the last catching the confidently wrong answers a
degraded context produces.

- **Accuracy** is the fraction of expectation *weight* met across the whole run — partial credit, so a
  turn that states the right figure but omits a secondary fact scores between 0 and 1.
- **Correct** counts turns with no *critical* failure — the fact the question was actually asking for.
  This is the stricter reading, and it is why a configuration can gain accuracy while losing a turn.

Scoring is deterministic, not an LLM judge: the tools are mocked, so every factual question has one
computable answer, and ground truth is derived from the tool implementations rather than written by hand.
Figures must match as the tools formatted them — a paraphrased or reformatted number fails even when the
arithmetic is right, which is deliberate. See `validation/community-plugin-A-B-D/src/accuracy.py`.

## Layout

```
community-plugins/
  strands-relevance-filter/               practice A, installable package
  strands-progressive-tool-disclosure/    practice B, installable package
  strands-context-graph/                  practice D, installable package
docs/design/
  design.md         the concepts, framework-agnostic (L100/L200 overview)
  design-a-*.md     idea A — relevance filtering (idea + example)
  design-b-*.md     idea B — progressive tool disclosure (idea + example)
  design-d-*.md     idea D — context graph (idea + example)
how-to/
  02-community-plugins-agent-sample.md    install and use the three packages  <- start here
  01-designA-B-D-agent-sample.md          the same practices on the forked SDK
validation/
  community-plugin-A-B-D/   the benchmark for the community packages
  01-designA-B-D/           the benchmark for the forked SDK's vended plugins
```

- **Read the ideas:** start at [`docs/design/design.md`](docs/design/design.md) for the concepts,
  then the per-practice docs (`design-a-*`, `design-b-*`, `design-d-*`).
- **Run an agent with the practices:** see
  [`how-to/02-community-plugins-agent-sample.md`](how-to/02-community-plugins-agent-sample.md).
- **Reproduce the numbers:** see
  [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) — it
  installs the packages, runs the benchmark, and generates JSON + Markdown + HTML reports.
  Re-rendering a recorded run's report needs no AWS credentials; only a live benchmark run does.

## The forked-SDK path

The same three practices also exist as **vended plugins of a forked SDK** — the form they were
originally measured in, where they are bundled by the SDK itself rather than installed beside it.

That path, its own benchmark and its results are documented in
**[`README-vended-plugins.md`](README-vended-plugins.md)**.

The two are **not directly comparable on absolute numbers**, and an earlier version of this page
claimed they were. The vended figures were measured on a previous version of the benchmark script,
whose filler turns asked about accounts the fixture did not hold — so most of them made no tool call
and contributed almost no payload mass. The corrected script grounds every filler turn, which is why
its baseline carries 14.4M input tokens where the vended one carried 11.3M.

What does hold across both: the direction and the relative ordering of the strategies, and a full stack
that cuts roughly 80% of tokens while holding answer quality. The community form needs no fork, which
is why it is the recommended path; the vended form keeps one structural advantage, described in
gotcha 1 of the how-to. Settling whether the two reach the same absolute figure would mean re-running
the vended harness on the corrected script, which has not been done.

## Scope and limitations

This content is a **research validation harness**, not a production application. Specifically:

- **IAM and permissions** — The execution roles and Bedrock model permissions shown in the validation
  directories are the minimum needed to run the benchmark. Production deployments must follow
  least-privilege principles tailored to your workload, including scoped resource ARNs, condition keys,
  and permission boundaries.
- **No operational hardening** — No availability, resilience, monitoring, alerting, or logging best
  practices are configured. The deployment path exists solely to reproduce benchmark measurements.
- **No input/output safeguards** — The harness invokes LLM models without Amazon Bedrock Guardrails.
  Customer-facing deployments should configure guardrails for content filtering, PII masking, and topic
  restrictions. See [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Synthetic data only** — All financial scenario data (FinBank, TestBank, NeoBank, account numbers,
  balances) is synthetic, computed by `validation/community-plugin-A-B-D/src/ground_truth.py`. No real
  customer or personal data is used.
- **One package ships without tests** — `strands-relevance-filter` carries no test suite in this
  branch; the other two carry 641 passing tests between them. Practice A's only verification here is
  the benchmark.

## Responsible AI considerations

This repository uses Amazon Bedrock foundation models (`us.anthropic.claude-opus-4-8`,
`cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`) for benchmarking context-engineering
strategies. Key considerations:

- **Intended use** — Measuring token consumption, accuracy, and latency of context-management
  strategies in a controlled benchmark environment. Not intended for real financial advice, customer
  interactions, or autonomous decision-making.
- **Model limitations** — LLM outputs are non-deterministic; accuracy figures in this benchmark
  reflect the specific scenario and model version tested and should not be generalized.
- **Guardrails** — No Amazon Bedrock Guardrails are configured in this harness because the benchmark
  requires unfiltered model output to measure token behavior faithfully. Production deployments **must**
  configure appropriate guardrails.
- **Data privacy** — All prompts and tool fixtures use synthetic data. No real personal, financial, or
  customer data is sent to Bedrock models.

## Status

Draft for discussion. The ideas are stable at A/B/D. The three packages are at `0.1.0` and are not yet
published to PyPI — install them from this repository. The benchmark runs against live Bedrock; a
recorded run's report re-renders with no credentials at all.
