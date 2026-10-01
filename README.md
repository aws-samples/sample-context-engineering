# Context Engineering

> **⚠️ Not for production use.** This repository is a **reference implementation and validation
> harness** for context-engineering research. It is provided for experimentation, benchmarking, and
> learning purposes only. **Do not deploy to production** without an independent security review,
> infrastructure hardening, and testing appropriate to your workload and compliance requirements. IAM
> roles, permissions, and deployment configurations shown here are minimal examples — not
> production-ready baselines.

Three **context-engineering practices** for LLM agents: techniques that keep an agent's working context
small and relevant as a conversation grows, without losing what the task needs.

They ship for two frameworks, over one shared core:

- **[Strands Agents](https://strandsagents.com)** — three plugins that install next to an unmodified
  `strands-agents` from PyPI.
- **[LangChain / LangGraph](https://langchain-ai.github.io/langgraph/)** — three `create_agent`
  middlewares for LangChain v1.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** — four `ContextEngine` plugins (three
  practices + one composed engine, since Hermes is single-select).
- **[`context-core`](context-core/)** — the framework-agnostic logic both use. It imports no agent
  framework.

On a 60-turn benchmark with prompt caching off, the three practices together send **75–84% fewer
tokens** than a bare agent on large-window models, at the same accuracy. On models with a window of
256K tokens or less the bare agent cannot finish the conversation, and the three together answer every
turn. [The results](#the-results) have the numbers.

## Why

The practices come from a measured diagnosis of a real agent session (13 turns, ~29 minutes), not from
intuition. Three findings shaped them:

- **Most of the context is what the task does *not* need.** Four turns of tool-connector
  troubleshooting accounted for **44.7%** of all tokens in the session. They were legitimately
  discussed, but they were not the objective, and they were re-sent on every later turn.
- **A large fixed floor is paid on every call.** About **63k tokens per call** were tool schema alone,
  roughly 85% of a call's floor, re-processed on all 33 model calls.
- **Retrieved memory accumulates.** Retrieved records are inserted into the conversation and re-sent
  every turn, so irrelevant retrieval costs quadratically, not once.

An agent's context grows with everything that happened, while a good answer needs only what the current
activity requires. The practices keep the task's **attention memory** in the context and leave the
**background** reachable on demand.

This is consistent with published work: the Strands benchmark for compaction plus offload reports **cost
−55%** with **accuracy rising 68% → 98%**
([reduced cost, better isolation, more resilience](https://strandsagents.com/blog/reduced-cost-better-isolation-more-resilience/)).

## The practices

| | Practice | What it does | Strands | LangGraph | Hermes | Design |
|---|---|---|---|---|---|---|
| **A** | **Relevance filtering** | Scores a tool result's chunks against the question and keeps what answers it, before the result enters the history. | [`strands-relevance-filter`](community-plugins/strands-relevance-filter/) | [`langgraph-relevance-filter`](langgraph-plugins/langgraph-relevance-filter/) | [`hermes-relevance-filter`](hermes-plugins/hermes-relevance-filter/) | [design A](docs/design/design-a-relevance-filtering.md) |
| **B** | **Progressive tool disclosure** | Sends a lean tool catalog and a tool's full spec only on demand, attacking the ~63k schema floor. | [`strands-progressive-tool-disclosure`](community-plugins/strands-progressive-tool-disclosure/) | [`langgraph-progressive-tool-disclosure`](langgraph-plugins/langgraph-progressive-tool-disclosure/) | [`hermes-progressive-tool-disclosure`](hermes-plugins/hermes-progressive-tool-disclosure/) | [design B](docs/design/design-b-progressive-tool-disclosure.md) |
| **D** | **Context graph** | Turns the history into Cards and sends each at the resolution the question needs (full, Description or Title), recoverable on demand. It subsumes an earlier curator idea (C). | [`strands-context-graph`](community-plugins/strands-context-graph/) | [`langgraph-context-graph`](langgraph-plugins/langgraph-context-graph/) | [`hermes-context-graph`](hermes-plugins/hermes-context-graph/) | [design D](docs/design/design-d-context-graph.md) |

Each package is independent: install one, two or all three. They compose because they act at different
moments: the filter on a tool result before it enters the history, disclosure on the tool list, the
graph on a history that already exists.

## How to install and use them

| Framework | Guide | Benchmark |
|---|---|---|
| Strands Agents | **[`how-to/02-community-plugins-agent-sample.md`](how-to/02-community-plugins-agent-sample.md)** | [`validation/community-plugin-A-B-D/`](validation/community-plugin-A-B-D/README.md) |
| LangChain / LangGraph | **[`how-to/03-langgraph-plugins-agent-sample.md`](how-to/03-langgraph-plugins-agent-sample.md)** | [`validation/plugins-langgraph/`](validation/plugins-langgraph/README.md) |
| Hermes Agent | **[`how-to/04-hermes-plugins-agent-sample.md`](how-to/04-hermes-plugins-agent-sample.md)** | [`validation/plugins-hermes/`](validation/plugins-hermes/README.md) |

Each guide goes from nothing to a working agent: prerequisites, a minimal agent with one oversized tool,
each practice on its own, then all three together, with the constructor arguments the benchmark uses. Each
also lists the things that fail silently when the three are combined. **Read those before wiring all
three.** The ones that apply to both frameworks:

1. **The relevance threshold is a position in a distribution, not a number.** The package default of
   `0.5` rejects every chunk with `cohere.rerank-v3-5`, whose strong matches score ~0.29. The benchmark
   uses `0.02`.
2. **Nothing may delete from the history behind the graph.** In Strands that means
   `NullConversationManager`; in LangGraph, no summarization or trimming middleware. Either can drop
   what the graph only meant to fold.
3. **Two retrieval tools read two stores.** The filter's `retrieve_all_context` and the graph's
   `expand_artifact` must not look like the same job to the model; the guides show how they are scoped
   and, in LangGraph, how the graph reads the filter's store through `stash=`.

The packages are not on PyPI yet. Install them from a clone of this repository:

```bash
# Strands
pip install -e community-plugins/strands-relevance-filter \
            -e community-plugins/strands-progressive-tool-disclosure \
            -e community-plugins/strands-context-graph \
            "strands-agents>=1.44.0,<2.0.0"

# LangChain / LangGraph
pip install -e context-core \
            -e langgraph-plugins/langgraph-relevance-filter \
            -e langgraph-plugins/langgraph-progressive-tool-disclosure \
            -e langgraph-plugins/langgraph-context-graph \
            "langchain>=1.0,<2" "langgraph>=1.0,<2" "langchain-aws>=1.7,<2"

# Hermes Agent (hermes-agent is version 0.0.0 and not on PyPI; install the host from source)
pip install -e context-core \
            -e hermes-plugins/hermes-relevance-filter \
            -e hermes-plugins/hermes-progressive-tool-disclosure \
            -e hermes-plugins/hermes-context-graph \
            -e hermes-plugins/hermes-all-three \
            "hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git"
```

Verified against `strands-agents` 1.56.0 and `langchain` 1.4.2.

## The results

One scripted conversation of 60 turns, replayed once per configuration: a bare agent (no plugin), each
practice alone, and all three together. Several mocked tools return 40,000–120,000 characters in one
result, which is where the mass is. 30 turns are scored deterministically against ground truth computed
from the tools (no LLM judge); [how correctness is measured](#how-correctness-is-measured).

**Total tokens** is the agent's own input plus output, plus every token a strategy spends on its own
account (the graph's embeddings, the filter's reranks). **Cost** applies published Bedrock list prices to
those units, so read it as list cost, not a bill. Every model, every arm, the cache-on runs and the
parameters are in **[`BENCHMARK.md`](BENCHMARK.md)**.

### Large windows, caching off: the saving is cost

Strands plugins, Claude Opus 4.8 (1M-token window):

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 17,897,954 | — | 97.6% | 29/30 | 11.0s | $90.06 | — |
| Progressive Tool Disclosure only | 6,738,122 | −62.4% | 92.9% | 26/30 | 8.1s | $34.24 | −62.0% |
| Relevance Filtering only | 12,432,470 | −30.5% | 95.3% | 27/30 | 9.8s | $62.77 | −30.3% |
| Context Graph only | 10,511,391 | −41.3% | 94.5% | 27/30 | 10.0s | $53.13 | −41.0% |
| **All three combined** | **2,877,396** | **−83.9%** | **99.2%** | **30/30** | 10.0s | **$15.09** | **−83.2%** |

*Read Δ tokens and Correct: the full stack sends 84% fewer tokens, costs a sixth, and got every scored turn
right.*

The same comparison on the other large-window models, baseline against all three:

| Model | Framework | Baseline tokens | All three | Δ tokens | Correct (base → all) | Cost (base → all) |
|---|---|---:|---:|---:|:--:|---:|
| Claude Opus 4.8 | Strands | 17.9M | 2.9M | **−83.9%** | 29 → 30 | $90.06 → $15.09 |
| Claude Opus 4.8 | LangGraph | 15.2M | 2.5M | **−83.5%** | 29 → 29 | $76.57 → $13.18 |
| Claude Opus 5 | Strands | 18.8M | 4.7M | **−74.9%** | 28 → 27 | $95.34 → $24.94 |
| Claude Fable 5 | Strands | 19.2M | 4.5M | **−76.7%** | 29 → 29 | $193.09 → $47.01 |

*Read Δ tokens: on every large-window model the saving lands between −75% and −84% with accuracy held,
and the LangGraph port reproduces the Strands figure on the same model.* The LangGraph run's full table is
in [`langgraph-plugins/README.md`](langgraph-plugins/README.md#benchmark).

The full stack makes more model calls to send fewer tokens: a retrieval call is cheap next to re-sending a
60k-character payload on every later turn. Single-practice figures move with the model (disclosure alone
saves 62% on Opus 4.8 and spends 17% more on Fable 5), so do not quote one without naming its model.

### Small windows: the saving is completion

On models with a window of 256K tokens or less, the bare agent's history outgrows the window: the provider
rejects the call (`ContextWindowOverflowException`) and the turn is never answered. Strands plugins, same
script:

| Model | Window | Baseline answered | Refused calls | Baseline correct | All three answered | All three correct | Peak call (base → all) | Cost (base → all) |
|---|---:|:--:|---:|:--:|:--:|:--:|---:|---:|
| Claude Haiku 4.5 | 200K | 26/60 | 34 | 14/30 | **60/60** | **25/30** | 99% → 18% | $6.52 → $2.29 |
| GLM 5 | 200K | 26/60 | 34 | 15/30 | **60/60** | **25/30** | 97% → 17% | $8.36 → $2.42 |
| GLM 4.7 | 203K | 20/60 | 40 | 13/30 | **60/60** | **28/30** | 95% → 28% | $3.47 → $2.26 |
| GLM 4.7 Flash | 203K | 26/60 | 34 | 13/30 | **60/60** | **14/30** | 93% → 23% | $0.51 → $0.31 |
| Qwen3 Next 80B | 256K | 14/60 | 46 | 9/30 | **60/60** | **24/30** | 98% → 28% | $0.95 → $3.09 |
| Nemotron Nano 9B | 128K | 12/60 | 48 | 5/30 | **60/60** | **11/30** | 98% → 43% | $0.11 → $0.25 |

*Read Answered and Peak call: the bare agent never finishes, the full stack answers every turn on every
model, and its largest call stays under half the window.* A baseline that answered 12 of 60 turns is cheap
because it stopped working, so its cost is not a price for the workload. That is why cost rises on Qwen3
and Nemotron: the full stack answered four to five times as many turns.

Single practices do not get there alone: each one fails to keep the peak inside the window on at least one
of these models, and on Nemotron Nano 9B none of the three finishes by itself. They remove different mass
(A the payload, B the schema, D the history), which is why only the combination holds everywhere.

### With prompt caching on

Prompt caching attacks the same redundancy, and on a large-window model it makes the bare agent's re-sent
prefix cheap. There the compressing practices can cost more than doing nothing, because every edit to the
prompt is a new cache write. On Opus 4.8 with caching on, the baseline costs $12.07, relevance filtering
alone **$7.72**, and all three $12.50. Relevance filtering is the one to combine with caching: it compresses
once and then leaves the prompt alone.

Caching only pays when the same prefix comes back within its TTL. A system prompt per tenant, a tool set
per user permission, one-shot fan-out, an A/B prompt split, or a human who pauses longer than the TTL pay
the write premium and collect no read
([billing for cached tokens](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)).
In those shapes, and on models that publish no caching (GLM, Qwen3, Nemotron), the caching-off figures
above apply. The measured numbers are in [`BENCHMARK.md`](BENCHMARK.md) and the wiring rule is in
[the Strands guide](how-to/02-community-plugins-agent-sample.md#prompt-caching-and-these-plugins).

### How to read one replay

The agent chooses its own tool path, so two replays of identical code differ: four byte-identical arms
moved by up to six correct turns and 16% of their tokens between two runs. Nothing under about ±6 turns or
±20% of tokens is evidence. The figures above that decide something (−75% to −84% tokens, 60 of 60 turns
against 12 to 26) clear that band by a wide margin; the per-arm accuracy ranking within one model does not.

### How correctness is measured

The two correctness columns are different measures. 30 of the 60 turns carry weighted expectations (18
hand-written, 12 generated); the other 30 are grounded filler that lengthens the history. Expectations are
strings the answer must contain, may contain, or must *not* contain, the last catching the confidently
wrong answers a degraded context produces.

- **Accuracy** is the fraction of expectation *weight* met across the run, with partial credit.
- **Correct** counts turns with no *critical* failure, the fact the question asked for. It is the stricter
  reading, which is why an arm can gain accuracy while losing a turn.

Scoring is deterministic: the tools are mocked, so every factual question has one computable answer, and
ground truth is derived from the tool implementations. Figures must match as the tools formatted them — a
reformatted number fails even when the arithmetic is right, on purpose. See
`validation/community-plugin-A-B-D/src/accuracy.py`.

## Layout

```
BENCHMARK.md                              every model, every cell, and how to read them
context-core/                             the shared, framework-agnostic logic of A, B and D
community-plugins/
  strands-relevance-filter/               practice A for Strands
  strands-progressive-tool-disclosure/    practice B for Strands
  strands-context-graph/                  practice D for Strands
langgraph-plugins/
  langgraph-relevance-filter/             practice A for LangChain / LangGraph
  langgraph-progressive-tool-disclosure/  practice B for LangChain / LangGraph
  langgraph-context-graph/                practice D for LangChain / LangGraph
hermes-plugins/
  hermes-relevance-filter/                practice A for Hermes Agent
  hermes-progressive-tool-disclosure/     practice B for Hermes Agent
  hermes-context-graph/                   practice D for Hermes Agent
  hermes-all-three/                       the composed engine (Hermes is single-select)
docs/design/
  design.md         the concepts, framework-agnostic
  design-a-*.md     idea A — relevance filtering (idea + example)
  design-b-*.md     idea B — progressive tool disclosure (idea + example)
  design-d-*.md     idea D — context graph (idea + example)
  sequence/         sequence diagrams, Strands, langgraph/ and hermes/
how-to/
  02-community-plugins-agent-sample.md    Strands: install and use the three plugins
  03-langgraph-plugins-agent-sample.md    LangGraph: install and use the three middlewares
  04-hermes-plugins-agent-sample.md       Hermes: install and select the four engines
validation/
  community-plugin-A-B-D/   the benchmark for the Strands plugins
  plugins-langgraph/        the benchmark for the LangGraph middlewares
  plugins-hermes/           the benchmark for the Hermes engines
```

- **Read the ideas:** start at [`docs/design/design.md`](docs/design/design.md), then the per-practice
  docs.
- **Run an agent:** the [Strands guide](how-to/02-community-plugins-agent-sample.md) or the
  [LangGraph guide](how-to/03-langgraph-plugins-agent-sample.md).
- **Compare the models:** [`BENCHMARK.md`](BENCHMARK.md), one section per model.
- **Reproduce the numbers:** [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md)
  or [`validation/plugins-langgraph/README.md`](validation/plugins-langgraph/README.md). Re-rendering a
  recorded run's report needs no AWS credentials; only a live run does.

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
- **Test coverage is uneven.** The packages carry **1,588 passing tests**: `context-core` 449; Strands
  `strands-context-graph` 612, `strands-progressive-tool-disclosure` 169, `strands-relevance-filter` 145;
  LangGraph `langgraph-context-graph` 87, `langgraph-progressive-tool-disclosure` 82,
  `langgraph-relevance-filter` 44. The Strands relevance filter's suite was written against its documented
  contract after the fact, so it verifies what the docstrings promise rather than having driven the design.
- **One replay per cell.** Every benchmark figure comes from a single run per configuration; see
  [how to read one replay](#how-to-read-one-replay).

## Responsible AI considerations

This repository uses Amazon Bedrock foundation models for benchmarking context-engineering strategies —
the agent models listed in [`BENCHMARK.md`](BENCHMARK.md), plus `cohere.rerank-v3-5:0` and
`cohere.embed-multilingual-v3` for the practices themselves. Key considerations:

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

Draft for discussion. The ideas are stable at A/B/D. All seven packages are at `0.1.0` and are not yet
published to PyPI; install them from this repository. The benchmarks run against live Bedrock; a recorded
run's report re-renders with no credentials at all.
