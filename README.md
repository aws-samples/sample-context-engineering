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
framework code, in the same doc), and **results** (the reproducible benchmark under `validation/`).

| | Practice | Idea | Design |
|---|---|---|---|
| **A** | **Relevance filtering** | Score a tool result's chunks against the question and keep only what answers it, instead of a positional slice. | [`design-a-relevance-filtering.md`](docs/design/design-a-relevance-filtering.md) |
| **B** | **Progressive tool disclosure** | Send a lean tool catalog; fetch a tool's full spec on demand, then forget it — attacking the ~63k schema floor. | [`design-b-progressive-tool-disclosure.md`](docs/design/design-b-progressive-tool-disclosure.md) |
| **D** | **Context graph** | Reorganize the two above into one graph with remove/recover over an immutable log. It subsumes the earlier background-curator idea (C), which is why there is no standalone C. | [`design-d-context-graph.md`](docs/design/design-d-context-graph.md) |

The `graph-all` configuration in the benchmark applies all three together.

## Layout

```
docs/design/
  design.md         the concepts, framework-agnostic (L100/L200 overview)
  design-a-*.md     idea A — relevance filtering (idea + example)
  design-b-*.md     idea B — progressive tool disclosure (idea + example)
  design-d-*.md     idea D — context graph (idea + example)
validation/         the benchmark: reproducible harness, results, HTML reports
```

- **Read the ideas:** start at [`docs/design/design.md`](docs/design/design.md) for the concepts,
  then the per-practice docs (`design-a-*`, `design-b-*`, `design-d-*`).
- **Reproduce the numbers:** see [`validation/README.md`](validation/README.md) — it installs the SDK
  from git, runs the benchmark, and generates JSON + Markdown + HTML reports. Re-rendering a recorded
  run's report needs no AWS credentials; only a live benchmark run does.

## Scope and limitations

This content is a **research validation harness**, not a production application. Specifically:

- **IAM and permissions** — The execution roles and Bedrock model permissions shown in `DEPLOY.md` and
  `validation/README.md` are the minimum needed to run the benchmark. Production deployments must
  follow least-privilege principles tailored to your workload, including scoped resource ARNs, condition
  keys, and permission boundaries.
- **No operational hardening** — No availability, resilience, monitoring, alerting, or logging best
  practices are configured. The deployment path exists solely to reproduce benchmark measurements.
- **No input/output safeguards** — The harness invokes LLM models without Amazon Bedrock Guardrails.
  Customer-facing deployments should configure guardrails for content filtering, PII masking, and topic
  restrictions. See [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Synthetic data only** — All financial scenario data (FinBank, TestBank, NeoBank, account numbers,
  balances) is synthetic, computed by `ground_truth.py`. No real customer or personal data is used.

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

Draft for discussion. The ideas are stable at A/B/D; the benchmark runs locally and offline today, and
the AgentCore cloud-deploy path is being migrated to the git-installed SDK.
