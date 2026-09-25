# LangGraph A/B/D validation harness

Measures the same three context-engineering strategies as
[`validation/community-plugin-A-B-D`](../../../validation/community-plugin-A-B-D), on the same
script, against the same deterministic ground truth — but driving a LangChain v1 `create_agent`
graph with `AgentMiddleware` instead of a Strands `Agent` with plugins.

| Strategy | Strands harness | here |
|---|---|---|
| A — relevance filtering | `strands_relevance_filter.RelevanceFilter` | `langgraph_relevance_filter.RelevanceFilterMiddleware` |
| B — progressive tool disclosure | `strands_progressive_tool_disclosure.ProgressiveToolDisclosure` | `langgraph_progressive_tool_disclosure.ProgressiveToolDisclosureMiddleware` |
| D — context graph | `strands_context_graph.ContextGraph` | `langgraph_context_graph.ContextGraphMiddleware` |

Five arms, as before: `baseline` (no middleware), `relevance`, `disclosure`, `graph`, and `all`
(graph outermost, then disclosure, then relevance).

## Running it

Nothing here spends money by default. `python -m src.run` with no flags performs a **dry build**:
it constructs every arm, prints the middleware nesting and the tool-suite size, and invokes nothing.

```bash
cd validation/plugins-langgraph

# free: build every arm, invoke nothing
../../../.venv/bin/python -m src.run --dry-run

# free: re-render a recorded run from its JSON
../../../.venv/bin/python -m src.run --report-only results/run-lg01.json
```

The live benchmark needs credentials and **two** flags, neither with a short form:

```bash
isengardcli assume 045096474033        # us-east-1
export AWS_REGION=us-east-1

../../../.venv/bin/python -m src.run \
    --live --i-understand-this-spends-money \
    --total-turns 60 --tag lg01
```

A 60-turn five-arm run against `us.anthropic.claude-opus-4-8` is on the order of a few hundred
dollars at list price. `--configs baseline all --turn-limit 6` is the cheap way to see the pipeline
end to end.

Other flags: `--configs`, `--turn-limit`, `--total-turns`, `--phases`, `--repeats`, `--sequential`,
`--tag`, `--verbose`.

### A recorded run re-renders with no credentials

Every figure the table prints was measured during the run and written into
`results/run-<tag>.json`. `report.py` is arithmetic over that file — no AWS call, no model, no
credentials:

```bash
../../../.venv/bin/python -m src.report results/run-lg01.json
```

Correcting a rate is therefore an edit to `config.MODEL_PRICING` plus a re-render, never a re-run.

## What is copied and what is new

Copied **verbatim** from the Strands harness, so a difference between the two tables cannot come
from a re-tuned knob or a re-priced rate:

`scenario.py` · `ground_truth.py` · `accuracy.py` · `metrics.py` · `compare.py` · `corpus.py` ·
`config.py` (one paragraph of its module docstring rewritten to describe this binding)

`config.py` carrying over unchanged is the whole basis of comparability: `MODEL_PRICING`,
`THRESHOLDS`, `GRAPH_TUNING`, `TARGET_SCHEMA_TOKENS` and `estimate_tokens` are byte-identical to the
ones the published Strands numbers were produced with.

Two pieces of that copied code are **inert here**, and are left in place rather than deleted so the
files stay diffable against their originals: `metrics.RunCollector.middleware()` and
`metrics._absorb_usage()` are written against Strands' `InvokeModelStage` and its streamed usage
events. `runner.MetricsMiddleware` and `runner._absorb_usage` are their LangGraph counterparts, and
they fill the same `ModelCallRecord` — which is what keeps `compare.py` and the cost model reusable.
`config.SESSION_MANAGER` is inert for the same kind of reason: LangGraph's checkpointer takes that
role and the runner ignores the value.

Adapted:

- **`tools.py`** — `from strands import tool` became `from langchain_core.tools import tool`, and
  the filler budget, which used to be sized off Strands' `built.tool_spec`, now measures
  `_converse_spec()`: the same three-key `{name, description, inputSchema.json}` shape
  `langchain_aws` puts on the wire. **Every function body is byte-identical**, so every payload the
  model sees is byte-identical — which is why `ground_truth.py` and `accuracy.py` are reusable
  unchanged. `web.py` took the same one-line decorator swap.
- **`runner.py`** — new. Builds one `create_agent` per arm.
- **`report.py`** — new. Renders the benchmark table.
- **`run.py`** — new. CLI with the spend gate.

### One calibration differs, on purpose

LangChain's pydantic-generated JSON schema is a few percent fatter per tool than Strands'
`tool_spec`, so the same `TARGET_SCHEMA_TOKENS` budget fills with fewer tools:

| | Strands harness | here |
|---|---|---|
| core tools | 18 | 18 |
| web tool | 1 | 1 |
| filler tools | 74 | 68 |
| registered on the baseline arm | 93 | 87 |
| filler schema | ~59k tokens | ~59.9k tokens |
| whole suite, as sent | ~63k tokens | 63,758 tokens |

**Schema mass is matched; tool count is not.** Mass is the quantity the disclosure arm removes and
the thing `config.py` documents as the calibration, so budget-matching is the default. Tool count is
what the lexical index has to discriminate between, so a count-matched reading is legitimate too:
`VALIDATION_FILLER_TOOL_COUNT=74`. Both figures are recorded in every run's `meta`, so no table can
be read without knowing which produced it.

### Sync and async hooks

LangChain does **not** bridge a middleware's sync and async hooks — it raises `NotImplementedError`
naming the missing one. Every hook in the three packages ships both twins, so the harness uses the
packages as published. It drives `ainvoke` because the relevance reranker protocol is `async def score`.

Separately, the graph persists `context_core.graph.state` dataclasses through the checkpointer, which
LangGraph's msgpack serde warns about and says it "will be blocked in a future version" — a block
would silently lose the graph on every restore. `runner._checkpointer()` puts that module on the
serde's allowlist.

## Models

| role | model | note |
|---|---|---|
| agent | `us.anthropic.claude-opus-4-8` | default; override with `VALIDATION_AGENT_MODEL_ID` |
| reranker | `cohere.rerank-v3-5:0` | used by the relevance arm |
| embedder | `cohere.embed-multilingual-v3` | used by the graph's similarity matcher |
| region | `us-east-1` | |

Opus 4.8 is the default because it is the model the Strands harness's **published five-arm table**
was measured on (60 turns, cache off, zero errors — `README.md` and `BENCHMARK.md`), which makes a
LangGraph run against it directly comparable. A model with no entry in `config.MODEL_PRICING` is
refused at preflight rather than billed at another model's rate.

## Security and account notes

- **No credential is read from or written to disk by this harness.** It uses the default boto3
  credential chain, or `VALIDATION_AWS_PROFILE` when set; a profile backed by `credential_process`
  mints short-lived credentials on demand. Nothing is committed — `results/` holds measurements only.
- **No public resource is created.** Every AWS call is a read or an inference request:
  `bedrock-runtime` Converse, Rerank and Embed, plus one `sts:GetCallerIdentity` at preflight. No
  bucket, no queue, no endpoint, no security group.
- **HTTPS only.** The AWS SDK talks to Bedrock over TLS. `corpus.py` and `web.py` download public
  AWS documentation over `https://` from an allowlist of prefixes and refuse any other host without
  making a request.
- **No mutating tool.** Every tool in the suite is a fixture: `open_support_case`,
  `force_connector_sync` and `rotate_connector_credentials` return acknowledgement strings and touch
  nothing.
- Set `VALIDATION_ACCOUNT_ID` to have preflight refuse to run against any other account.

## Layout

```
src/
  config.py       copied, pricing and tuning of record
  scenario.py     copied, the 60-turn script
  tools.py        adapted, LangChain @tool with byte-identical payloads
  web.py          adapted, one decorator line
  corpus.py       copied, cached AWS doc pages
  ground_truth.py copied, computes the answers the tools imply
  accuracy.py     copied, deterministic scoring -- no LLM judge
  metrics.py      copied, the record and aggregation dataclasses
  compare.py      copied, the cost model
  runner.py       NEW, one create_agent per arm
  report.py       NEW, JSON -> the benchmark table
  run.py          NEW, CLI with the spend gate
results/          run JSON and its rendered markdown
```

## Status

Tasks 7.1–7.3 are done: the harness exists and its wiring is verified against a fake chat model.
**Task 7.4 — the live paid run — has not been executed.** No figure in this directory came from
Bedrock yet.
