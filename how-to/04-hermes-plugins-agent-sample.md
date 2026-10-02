# Running a Hermes Agent with the context-engineering engines

> **⚠️ Not for production use.** This is a minimal working sample for experimentation and learning.
> It has no guardrails, no error handling and no operational hardening.

A runnable example of the three practices — [relevance filtering](../docs/design/design-a-relevance-filtering.md),
[progressive tool disclosure](../docs/design/design-b-progressive-tool-disclosure.md) and the
[context graph](../docs/design/design-d-context-graph.md) — installed on a
[Hermes Agent](https://github.com/NousResearch/hermes-agent) as a **`ContextEngine`**, one at a time and
then all three together.

The logic is the same as in the Strands plugins
([`02-community-plugins-agent-sample.md`](02-community-plugins-agent-sample.md)) and the LangGraph
middlewares ([`03-langgraph-plugins-agent-sample.md`](03-langgraph-plugins-agent-sample.md)): all three
are thin bindings over one framework-agnostic package, [`context-core`](../context-core/). Same defaults,
same tool names, same model-facing text. The places where Hermes forces a difference are listed in
[`hermes-plugins/README.md`](../hermes-plugins/README.md#strands--hermes-mapping-and-per-practice-verdict).

**Hermes allows only one active context engine at a time**, so unlike LangGraph you do not stack three —
you select one, and "all three" is the dedicated composed engine `all-three`. **Read
[the things that will bite you](#things-that-will-bite-you) before using `all-three`.**

## Prerequisites

| | Requirement | Why |
|---|---|---|
| Python | **3.11 or newer** | the packages' `requires-python` (Hermes requires `>=3.11,<3.15`) |
| `hermes-agent` | from source | version `0.0.0`, not on PyPI; the `[hermes]` extra of each package pulls it from GitHub |
| AWS CLI | **v2** | only to configure and verify credentials |

Bedrock models that must be **enabled in your account**, in the region you use:

| Model id | Used by |
|---|---|
| `us.anthropic.claude-opus-4-8` | the agent itself (any Converse-capable model works) |
| `cohere.rerank-v3-5:0` | relevance filtering |
| `cohere.embed-multilingual-v3` | the context graph's similarity matcher |

Credentials come from the standard `boto3` chain, so whatever the AWS CLI is configured with is what the
engines use. Construction is inert — no reranker, matcher or AWS client is built until an engine actually
fires — so an engine you select but never trip needs no credentials.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install -e context-core \
               -e hermes-plugins/hermes-relevance-filter \
               -e hermes-plugins/hermes-progressive-tool-disclosure \
               -e hermes-plugins/hermes-context-graph \
               -e hermes-plugins/hermes-all-three
# plus the Hermes host from source:
uv pip install "hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git"
```

## Make an engine selectable in Hermes

A context engine is a **directory** under `$HERMES_HOME/plugins/<name>/` whose `__init__.py` exports a
`ContextEngine` (or a `register(ctx)` hook). Each installed package already provides this; copy or symlink
its module directory in:

```bash
ln -s "$(python -c 'import hermes_relevance_filter, os; print(os.path.dirname(hermes_relevance_filter.__file__))')" \
      "$HERMES_HOME/plugins/relevance-filter"
```

Then select it in `config.yaml`:

```yaml
context:
  engine: relevance-filter   # the plugin dir name / the engine's `name`
```

Discovery is a text heuristic — the dir's `__init__.py` must mention `ContextEngine` or
`register_context_engine` — and `context.engine` names the active one; no `plugins.enabled` entry is
needed.

## A — relevance filtering

Select `context.engine: relevance-filter`. When a tool returns more than `max_result_tokens` (default
8000) of text, the engine stores the full result and the model sees a marker + a disclaimer with the real
size + a **verbatim** relevance excerpt + a `[ref: …]`. For an answer that needs every row (a total, a
max, a count) the model calls `rf_retrieve_all_context` with that reference and a `pattern` or a
`max_chunks`/`max_tokens` budget. What it retrieves is dropped from the conversation once the turn ends.

```python
from hermes_relevance_filter import RelevanceFilterEngine
engine = RelevanceFilterEngine(max_result_tokens=8_000)
```

## B — progressive tool disclosure

Select `context.engine: progressive-tool-disclosure`. The model sees a one-line-per-tool **catalog** in
the system message instead of every full schema; it calls `ptd_find_tools(need)` to search and
`ptd_get_tool_details(names)` to expand a spec before using it. Expanded tools stay active (not re-summarized)
and closed `ptd_get_tool_details` exchanges are folded out of later requests.

```python
from hermes_progressive_tool_disclosure import ProgressiveToolDisclosureEngine
# the engine needs the agent's base tool schemas (OpenAI function-tool dicts) for the catalog:
engine = ProgressiveToolDisclosureEngine(tool_specs=my_agent_tools)
```

**Note the base-schema compromise:** the ABC gives no hook to strip the base tool schemas from the
provider request, so disclosure here steers the model to the catalog rather than removing the base schemas
(the portable path). See the package README.

## D — context graph

Select `context.engine: context-graph`. Each closed turn becomes a Card; every request is projected at the
resolution the current question needs (full / description / title). The model raises folded content back
with `cg_expand_card(titles)`, `cg_expand_artifact(reference)` and `cg_find_context(need)`. The persisted history
is never deleted.

```python
from hermes_context_graph import ContextGraphEngine
engine = ContextGraphEngine()
```

## All three — the composed engine

Select `context.engine: all-three`. One engine applies **D (project) → B (catalog + fold) → A (rewrite)**
each turn, over one shared relevance store, exposing the union of the six tools.

```python
from hermes_all_three import AllThreeEngine
engine = AllThreeEngine(
    tool_specs=my_agent_tools,     # for disclosure's catalog
    relevance_threshold=0.02,      # the benchmark default
    max_result_tokens=8_000,
)
```

These are the constructor arguments the benchmark uses (`validation/plugins-hermes/`).

## Things that will bite you

Three traps that lose state **silently** if you compose A/B/D by hand. `all-three` already handles all
three — this is why the composed engine exists rather than a "wire them yourself" recipe:

1. **Order matters.** D must project before B injects the catalog before A rewrites results. The composed
   engine fixes this order; a hand-rolled pipeline in the wrong order folds away context the next stage
   needed.
2. **The retrieval tools must share one store.** A's filter mints `[ref: …]` tokens; D's `cg_expand_artifact`
   can only resolve them if it reads **the same** store. `all-three` hands A's store to D as its `stash`.
   Two separate stores → a reference that resolves in neither.
3. **The two retrieval tools must stay distinct.** `rf_retrieve_all_context` (A) and `cg_expand_artifact` (D)
   do different jobs; if the model reads them as one it calls the wrong one. They are scoped by name and
   description so it does not.

## Benchmark

The replay harness is in [`validation/plugins-hermes/`](../validation/plugins-hermes/README.md): same
scenario, mocked tools, deterministic accuracy, five arms (baseline / A / B / D / all-three). Verified
offline with no AWS credentials; the `--live` run is gated behind an explicit acknowledgement flag.
