# Hermes Agent port of the three context-engineering practices

This directory ports the repository's three context-engineering practices — **relevance filtering** (A),
**progressive tool disclosure** (B), and the **context graph** (D) — to
[Hermes Agent](https://github.com/NousResearch/hermes-agent)'s `ContextEngine` plugin interface.

It follows the same **interface + core** architecture as [`../langgraph-plugins/`](../langgraph-plugins/):
all decision logic lives in the framework-agnostic [`context-core`](../context-core/) package (no
`hermes`/`langchain`/`strands` import), and each package here is a thin **`ContextEngine`** binding over
it. `context-core` is consumed **unchanged**.

## Packages

| | Practice | Package | Binding |
|---|---|---|---|
| **A** | Relevance filtering | [`hermes-relevance-filter`](hermes-relevance-filter/) | `on_turn_complete` + `select_context` + `rf_retrieve_all_context` |
| **B** | Progressive tool disclosure | [`hermes-progressive-tool-disclosure`](hermes-progressive-tool-disclosure/) | `select_context` (catalog + fold) + `ptd_find_tools` / `ptd_get_tool_details` |
| **D** | Context graph | [`hermes-context-graph`](hermes-context-graph/) | `select_context` (project) + `on_turn_complete` + `cg_expand_card` / `cg_expand_artifact` / `cg_find_context` |
| **A+B+D** | All three | [`hermes-all-three`](hermes-all-three/) | one composed engine (single-select mandates it) |

## Why four packages — Hermes is single-select

Hermes allows exactly **one** active context engine: `PluginManager.register_context_engine` refuses a
second one (`hermes_cli/plugins.py`). So the three practices cannot be three co-installed plugins the way
LangGraph stacks three middlewares. A/B/D ship as three selectable engines, and the "all three"
configuration — the one the benchmark's headline numbers come from — is a **dedicated composed engine**,
`hermes-all-three`, that applies A+B+D in one `select_context`/`get_tool_schemas`/`on_turn_complete`.

## Strands → Hermes mapping and per-practice verdict

Every practice ported. The three still act on different surfaces, which is why one composed engine can
run them together.

| Practice | Strands surface | Hermes `ContextEngine` member | Verdict |
|---|---|---|---|
| **A** Relevance | `AfterToolCallEvent` (rewrite result) + `AfterInvocationEvent` (drop retrieval exchanges) | `on_turn_complete` detects + stores the oversized result · `select_context` replaces it with marker + disclaimer + verbatim preview (+ `[ref]`) and drops closed exchanges · `rf_retrieve_all_context` tool | **Portable** |
| **B** Disclosure | `InvokeModelStage.Input` (rewrite `tool_specs` + `system_prompt` + fold messages) | `select_context` injects the catalog into the system message and folds closed `ptd_get_tool_details` exchanges · `ptd_find_tools` / `ptd_get_tool_details` tools | **Portable, with a documented base-schema compromise — see below** |
| **D** Context graph | `BeforeInvocation` + `MessageAdded` + `AfterToolCall` (project into Cards, record artifacts) | `select_context` runs `context_core.graph.project` · `on_turn_complete` records each tool return as an artifact · `cg_expand_card` / `cg_expand_artifact` / `cg_find_context` tools | **Portable — and the single-select bonus below** |

### Parity with the Strands plugins

Same defaults, same tool names and model-facing descriptions, same marker/disclaimer/catalog text. The
differences are the ones Hermes imposes, not design choices:

- **Synchronous.** Hermes's `ContextEngine` is synchronous, so each engine is a subclass with plain
  methods, not an async middleware with sync/async twins. The async `context-core` store and reranker are
  driven to completion on a private loop.
- **One turn-level observation hook, not per-result.** Hermes hands the engine the finished turn's
  messages via `on_turn_complete`, not each tool result at production time, so A and D do their indexing
  there.
- **A — token gate.** The sync engine exposes no model handle, so the gate uses the Strands default
  `count_tokens` heuristic (`ceil(chars/4)` text, `ceil(json chars/2)` JSON) recomputed locally. It flips
  at the same size as Strands unless a Strands model opts into native counting.

### B — the base-schema compromise (documented, non-blocking)

Strands and LangGraph implement disclosure by rewriting the model request's `tool_specs`/`tools`
directly, so the model literally cannot see a tool until it is disclosed. **Hermes's `ContextEngine`
exposes no hook to rewrite the agent's base tool catalog** — it owns only its own `get_tool_schemas()`
and the message list via `select_context()`. So B runs the **portable** path: it injects the catalog and
steers the model to `ptd_find_tools`/`ptd_get_tool_details`, but the base schemas Hermes assembled still reach the
provider. This matches the LangGraph README's "Portable (LangChain ships a subset of this natively)" note.
Stripping the base schemas (if a Hermes host hook is found) is a non-blocking follow-up, not a
requirement.

### D — the single-select bonus

Because `hermes-context-graph` *replaces* Hermes's default lossy `context_compressor`, the "nothing may
delete from the history behind the graph" precondition holds **structurally**: there is no co-active
summarizer to fight. In LangGraph the middleware can only *warn* about a co-installed pruning middleware;
single-select makes that conflict impossible here. The projection is request-only (`select_context`
returns a per-request list), so persisted history is never deleted — a mis-cut costs one recovery call,
never a lost fact (the repo's projection-not-destruction invariant).

### The three silent-failure traps `hermes-all-three` prevents

The composed engine wires the three so the traps the LangGraph port hit cannot recur:

1. **Composition order** is fixed at D (project) → B (catalog + fold) → A (rewrite), so each stage sees
   the output of the previous one.
2. **One shared relevance store** is handed to D as its `stash`, so a `[ref: …]` the filter mints
   resolves through `cg_expand_artifact` as well as `rf_retrieve_all_context` — not two stores the other cannot
   read.
3. **The two retrieval tools are scoped by name** (`rf_retrieve_all_context` vs `cg_expand_artifact`) so the
   model does not read them as the same job.

## Install

From a clone of this repository:

```bash
uv venv --python 3.12 .venv
uv pip install -e context-core
uv pip install -e hermes-plugins/hermes-relevance-filter \
               -e hermes-plugins/hermes-progressive-tool-disclosure \
               -e hermes-plugins/hermes-context-graph \
               -e hermes-plugins/hermes-all-three
```

`hermes-agent` is version `0.0.0` and not published to PyPI, so it is an optional `[hermes]` extra
installed from source (`pip install "hermes-relevance-filter[hermes]"`); the engines import the
`ContextEngine` ABC lazily, so the packages import, unit-test and build without it. Contract pinned from a
read-only clone — see [`CONTRACT.md`](CONTRACT.md).

## Select an engine in Hermes

A context engine ships as a directory under `$HERMES_HOME/plugins/<name>/` (an installed package exports
the engine and a `register(ctx)` hook; copy or symlink its `src/<module>/` as the plugin dir, or let
Hermes's discovery pick it up). Then name it in `config.yaml`:

```yaml
context:
  engine: all-three       # or: relevance-filter | progressive-tool-disclosure | context-graph
```

The composed engine is constructed with the agent's base tool schemas so disclosure has a catalog:

```python
from hermes_all_three import AllThreeEngine
engine = AllThreeEngine(tool_specs=my_agent_tools)  # relevance threshold defaults to 0.02
```

## Benchmark

The replay harness lives in [`validation/plugins-hermes/`](../validation/plugins-hermes/). It mirrors the
Strands and LangGraph harnesses: the same scenario, mocked tools, deterministic ground-truth and accuracy
(no LLM judge), the same rate table, and the same five arms (baseline / A / B / D / all-three), each
non-baseline arm selecting the corresponding Hermes engine via `context.engine`. It targets
`us.anthropic.claude-opus-4-8` so the three framework tables read side by side. It is verified offline
(all five arms run through a mocked model with no AWS credentials); a `--live` run is gated behind an
explicit acknowledgement flag.

## Provenance and scope

`context-core` is recreated from the Strands community plugins used as read-only reference; this work
never modifies them. See the spec at `.kiro/specs/vc-tem-criar-a-spec/`.
