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

It also carries **three gotchas you should read before combining the plugins**, because two of them
cost a measured benchmark run its answers and neither fails loudly:

1. **Two retrieval tools, one job.** The relevance filter and the context graph each ship an
   artifact-retrieval tool over its own store, and nothing bridges them. Installed together, the model
   reaches for the wrong one and gets an unresolvable reference. Fixing it moved the full stack from
   84.5% to 94.4% accuracy.
2. **`NullConversationManager` is a precondition of the graph**, not a suggestion — any other manager
   can physically drop what the graph only meant to fold.
3. **The relevance threshold is a position in a distribution, not a number.** The package default of
   `0.5` rejects every chunk with `cohere.rerank-v3-5`, whose strong matches score ~0.29.

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
| Baseline (no plugin) | 13,476,549 | — | 97.2% | 17/18 | 10.6s | $203.65 | — |
| Progressive Tool Disclosure only | 9,405,879 | −30.2% | 94.4% | 17/18 | 11.0s | $142.83 | −29.9% |
| Relevance Filtering only | 12,258,604 | −9.0% | 95.8% | 17/18 | 10.8s | $185.44 | −8.9% |
| Context Graph only | 9,663,413 | −28.3% | 91.5% | 15/18 | 9.0s | $146.17 | −28.2% |
| **All three combined** | **2,296,820** | **−83.0%** | **94.4%** | **17/18** | **8.1s** | **$35.88** | **−82.4%** |

Agent `us.anthropic.claude-opus-4-8`, region `us-east-1`, one **replay** per configuration — the same
scripted conversation run start to finish, once for each row, 60 turns, zero errors.

The baseline here is a **bare agent**: no plugin at all, every tool payload entering the history whole
and staying there. That is the honest control — it measures the cost of doing nothing — and it does not
overflow the window: all 60 turns completed.

The large differences — **−83.0%** with correctness holding at 17 of 18 turns and cost dropping to a
sixth — support a decision; nothing under ~20% does. A single replay cannot be more precise than that:
the agent chooses its own tool path, so two replays of the *same* configuration already differ by
roughly that much. Full breakdown (per-token cost attribution, per-turn curves) is regenerated by the
harness under `validation/community-plugin-A-B-D/`.

The full stack makes **more model calls** to send fewer tokens — 104 against the baseline's 87 — which
is the trade the practices make: a retrieval cycle is cheap next to resending a 60k-character payload
on every subsequent turn.

### How correctness is measured

The two correctness columns are different measures, not the same one twice. The script has **18 scored
turns** (the rest are unscored filler, there only to lengthen the history), and each scored turn carries
weighted expectations — strings the answer must contain, may contain, or must *not* contain, the last
catching the confidently wrong answers a degraded context produces.

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

Measured against each other on the same scenario, same model and same 60 turns, the two forms reach
**parity**: 2,296,820 tokens at $35.88 with 17 of 18 turns correct for the community packages, against
2,255,712 at $35.33 with 17 of 18 for the vended stack. The community form needs no fork, which is why
it is the recommended path; the vended form keeps one structural advantage, described in gotcha 1 of
the how-to.

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
