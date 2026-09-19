# Context Engineering — the vended-plugin path

> **This is not the repository's landing page.** Start at [`README.md`](README.md), which documents the
> three practices as **community plugins** that run against an unmodified `strands-agents`. This
> document covers the alternative form: the same three practices as **vended plugins of a forked
> SDK**, bundled by the SDK itself rather than installed beside it. It is the form they were originally
> measured in, and the results below are that measurement.

> **⚠️ Not for production use.** This repository is a **reference implementation and validation
> harness** for context-engineering research. It is provided for experimentation, benchmarking, and
> learning purposes only. **Do not deploy to production** without an independent security review,
> infrastructure hardening, and testing appropriate to your workload and compliance requirements. IAM
> roles, permissions, and deployment configurations shown here are minimal examples — not
> production-ready baselines.

A collection of **context-engineering practices** for LLM agents — techniques that keep an agent's
working context small and relevant as a conversation grows, without losing the information the task
actually needs.

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
framework code, in the same doc), and **results** (the reproducible benchmark under `validation/01-designA-B-D/`).

| | Practice | Idea | Design |
|---|---|---|---|
| **A** | **Relevance filtering** | Score a tool result's chunks against the question and keep only what answers it, instead of a positional slice. | [`design-a-relevance-filtering.md`](docs/design/design-a-relevance-filtering.md) |
| **B** | **Progressive tool disclosure** | Send a lean tool catalog; fetch a tool's full spec on demand, then forget it — attacking the ~63k schema floor. | [`design-b-progressive-tool-disclosure.md`](docs/design/design-b-progressive-tool-disclosure.md) |
| **D** | **Context graph** | Reorganize the two above into one graph with remove/recover over an immutable log. It subsumes the earlier background-curator idea (C), which is why there is no standalone C. | [`design-d-context-graph.md`](docs/design/design-d-context-graph.md) |

The `graph-all` configuration in the benchmark applies all three together.

## The result

**Total tokens** is the agent's own `usage`, input plus output, plus every auxiliary token the strategy
spent on its own account: the graph's embedding calls. **Cost** applies the rates declared in
`validation/01-designA-B-D/src/config.py` to measured units, embedding and rerank included — without that, the
strategies that buy their saving with a second model call would rank better than they are.

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (offloader, prefix preview) | 11,302,922 | — | 93.0% | 15/18 | 10.3s | $170.94 | — |
| Progressive Tool Disclosure only | 5,364,912 | −52.5% | 98.6% | 18/18 | 9.1s | $82.05 | −52.0% |
| Relevance Filtering only | 11,244,034 | −0.5% | 97.2% | 16/18 | 10.9s | $170.26 | −0.4% |
| Context Graph only | 8,384,258 | −25.8% | 94.4% | 16/18 | 9.5s | $126.93 | −25.7% |
| Disclosure + relevance combined | 6,002,228 | −46.9% | 97.2% | 18/18 | 8.2s | $91.42 | −46.5% |
| Graph + disclosure + relevance | 2,255,712 | −80.0% | 95.8% | 17/18 | 8.1s | $35.33 | −79.3% |

Agent `us.anthropic.claude-opus-4-8`, region `us-east-1`, one **replay** per configuration — the same
scripted conversation run start to finish, once for each row. The baseline is not a bare agent: it
carries the **offloader**, which parks an oversized tool result in storage and leaves a short prefix of
it in the conversation. That is the floor the three practices improve on.

The large differences — **−52.5%**, **−80.0%** and the accuracy holding at 95.8% while cost drops 79% —
support a decision; nothing under ~20% does. A single replay cannot be more precise than that: the agent
chooses its own tool path, so two replays of the *same* configuration already differ by roughly that
much, and a smaller gap says nothing about the strategy. Full breakdown (per-token cost attribution,
per-turn curves, plateau and slope) is regenerated by the harness under `validation/01-designA-B-D/`.

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
arithmetic is right, which is deliberate. See `validation/01-designA-B-D/src/accuracy.py`.

## Layout

```
docs/design/
  design.md         the concepts, framework-agnostic (L100/L200 overview)
  design-a-*.md     idea A — relevance filtering (idea + example)
  design-b-*.md     idea B — progressive tool disclosure (idea + example)
  design-d-*.md     idea D — context graph (idea + example)
how-to/             runnable guides: install the practices on an agent
validation/
  01-designA-B-D/   the benchmark for A, B and D: harness, results, HTML reports
```

- **Read the ideas:** start at [`docs/design/design.md`](docs/design/design.md) for the concepts,
  then the per-practice docs (`design-a-*`, `design-b-*`, `design-d-*`).
- **Run an agent with the practices:** see
  [`how-to/01-designA-B-D-agent-sample.md`](how-to/01-designA-B-D-agent-sample.md) — a minimal Strands
  agent with each plugin, and the three combined.
- **Reproduce the numbers:** see [`validation/01-designA-B-D/README.md`](validation/01-designA-B-D/README.md) — it installs the SDK
  from git, runs the benchmark, and generates JSON + Markdown + HTML reports. Re-rendering a recorded
  run's report needs no AWS credentials; only a live benchmark run does.

## Scope and limitations

This content is a **research validation harness**, not a production application. Specifically:

- **IAM and permissions** — The execution roles and Bedrock model permissions shown in
  `validation/01-designA-B-D/agentcore/DEPLOY.md` and `validation/01-designA-B-D/README.md` are the
  minimum needed to run the benchmark. Production deployments must
  follow least-privilege principles tailored to your workload, including scoped resource ARNs, condition
  keys, and permission boundaries.
- **No operational hardening** — No availability, resilience, monitoring, alerting, or logging best
  practices are configured. The deployment path exists solely to reproduce benchmark measurements.
- **No input/output safeguards** — The harness invokes LLM models without Amazon Bedrock Guardrails.
  Customer-facing deployments should configure guardrails for content filtering, PII masking, and topic
  restrictions. See [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Synthetic data only** — All financial scenario data (FinBank, TestBank, NeoBank, account numbers,
  balances) is synthetic, computed by `validation/01-designA-B-D/src/ground_truth.py`. No real customer
  or personal data is used.

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

Draft for discussion. The ideas are stable at A/B/D. The benchmark runs against live Bedrock; a
recorded run's report re-renders with no credentials at all.
