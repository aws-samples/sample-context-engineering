# Hermes validation harness

The replay benchmark for the four Hermes `ContextEngine` plugins, mirroring
[`../plugins-langgraph/`](../plugins-langgraph/) and
[`../community-plugin-A-B-D/`](../community-plugin-A-B-D/): the **same scenario**, the **same mocked
tools**, the **same deterministic ground truth and accuracy** (no LLM judge), the **same rate table**,
and the **same five arms** — baseline / A / B / D / all-three — each non-baseline arm selecting the
corresponding Hermes engine.

The framework-agnostic modules (`scenario.py`, `corpus.py`, `ground_truth.py`, `accuracy.py`,
`metrics.py`, `config.py`, `report.py`, `compare.py`) are shared verbatim with the LangGraph harness, so
the three framework tables are read the same way. Only `runner.py` (drives Hermes engines) and `run.py`
(the CLI + engine factory) are framework-specific, plus `_toolshim.py` — a tiny neutral `tool` decorator
so the fat tool suite builds OpenAI schemas with **no LangChain** dependency.

## Install (offline — no AWS)

```bash
uv venv --python 3.12 .venv
uv pip install -e ../../context-core \
               -e ../../hermes-plugins/hermes-relevance-filter \
               -e ../../hermes-plugins/hermes-progressive-tool-disclosure \
               -e ../../hermes-plugins/hermes-context-graph \
               -e ../../hermes-plugins/hermes-all-three \
               pytest httpx
```

Hermes Agent itself is **not** required for the offline harness — the engines import the `ContextEngine`
ABC lazily and fall back to a faithful stub, and the offline runner uses a mocked model with fake
reranker/matcher. Install `hermes-agent` only for the live run.

## Run it

```bash
# Default: build every arm and invoke nothing.
.venv/bin/python -m src.run --dry-run

# Offline: run all five arms through a mocked model, no AWS credentials. Writes results/run-<tag>.json
# and results/run-<tag>.md.
.venv/bin/python -m src.run --offline --tag h01

# Re-render a stored run (no credentials, no model).
.venv/bin/python -m src.run --report-only results/run-h01.json
```

### What the offline run measures (and does not)

The offline run drives each arm's Hermes engine with a **deterministic mock model** that calls the
scenario's expected tool and answers from the result. It exercises the real engine seams —
`select_context`, `on_turn_complete`, `get_tool_schemas`/`handle_tool_call` — and measures the message
**shape** each engine produces (the estimated input size, the peak call, the tools added). It does **not**
produce live token totals, accuracy against a real LLM, or a real cost — those come from the live run.
This is the verification the spec requires: all five arms build and run with no credentials, and a
recorded run re-renders with none.

## Live run (paid, user-authorized only)

```bash
.venv/bin/python -m src.run --live --i-understand-this-spends-money --total-turns 60 --tag h01
```

The live path is **gated behind the acknowledgement flag and intentionally not wired to launch from
here** (Task 20.2). Wiring a real Hermes agent to Bedrock and spending on a 60-turn five-arm run against
`us.anthropic.claude-opus-4-8` in `us-east-1` is a deliberate, authorized step — the harness is left
ready and the comparative table templated so the real numbers can be filled in and read against the
Strands and LangGraph tables (with the one-replay variance caveat: a difference of one or two turns, or
some tens of percent in a single arm's tokens, is within run-to-run variance).

## Layout

```
src/
  run.py          CLI + engine factory (offline fakes) + the --live gate
  runner.py       offline mock-model runner driving the Hermes engines
  _toolshim.py    neutral tool decorator (OpenAI schemas, no LangChain)
  scenario.py     the 20-turn script (shared)
  tools.py        the fat tool suite (shared bodies, neutral decorator)
  corpus.py / ground_truth.py / accuracy.py   deterministic scoring (shared)
  config.py       the five arms, thresholds, pricing (shared)
  metrics.py / report.py / compare.py          run JSON + the comparative table (shared)
tests/
  test_offline.py       all five arms build and run offline; a run re-renders
  test_composition.py   the composed engine applies D -> B -> A
results/        run JSONs and reports (gitignored)
```
