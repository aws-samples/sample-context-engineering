# LangGraph port of the three context-engineering practices

This directory ports the repository's three context-engineering practices — **relevance filtering** (A),
**progressive tool disclosure** (B), and the **context graph** (D) — to
[LangGraph](https://langchain-ai.github.io/langgraph/) / LangChain v1's `create_agent` middleware.

It follows an **interface + core** architecture: all decision logic lives in the framework-agnostic
[`context-core`](../context-core/) package (no `strands`/`langchain` import), and each package here is a
thin LangChain-middleware binding over it. The existing Strands community plugins in
[`../community-plugins/`](../community-plugins/) are **read-only reference** — recreated in `context-core`,
never modified by this work.

## Packages

| | Practice | Package | Binding |
|---|---|---|---|
| **A** | Relevance filtering | [`langgraph-relevance-filter`](langgraph-relevance-filter/) | `wrap_tool_call` / `awrap_tool_call` + `after_agent` |
| **B** | Progressive tool disclosure | [`langgraph-progressive-tool-disclosure`](langgraph-progressive-tool-disclosure/) | `wrap_model_call` / `awrap_model_call` + `wrap_tool_call` / `awrap_tool_call` |
| **D** | Context graph | [`langgraph-context-graph`](langgraph-context-graph/) | `wrap_model_call` / `awrap_model_call` + `wrap_tool_call` / `awrap_tool_call` |

## Strands → LangGraph mapping and per-practice verdict

Every practice ported. The distinguishing property survives: **each acts on a different surface**, so the
three compose on one agent.

| Practice | Strands surface | LangGraph mechanism | Verdict |
|---|---|---|---|
| **A** Relevance | `AfterToolCallEvent` (rewrite result) + `AfterInvocationEvent` (drop retrieval exchanges) | `wrap_tool_call` / `awrap_tool_call` rewrites the returned `ToolMessage` to marker + disclaimer + verbatim preview (+ `[ref]`); `after_agent` drops closed `retrieve_all_context` exchanges; `retrieve_all_context` tool | **Portable** |
| **B** Disclosure | `InvokeModelStage.Input` (rewrite `tool_specs` + `system_prompt` + fold messages) | `wrap_model_call` + `request.override(tools=…, system_message=…+catalog, messages=…folded)`; `find_tools` / `get_tool_details` tools; `loaded_tools` state; catalog lines summarized by the agent's own model by default | **Portable** (LangChain ships a subset of this natively) |
| **D** Context graph | `BeforeInvocation` + `MessageAdded` + `AfterToolCall` (project into Cards, record artifacts) | `wrap_model_call` + `request.override(messages=projected)` from `context_core.graph.project`; `wrap_tool_call` records each tool return as an artifact; `expand_card` / `expand_artifact` / `find_context` tools; serialized-graph state | **Portable with caveats** — see below |

### Parity with the Strands plugins

Same defaults, same tool names and model-facing descriptions, same marker/disclaimer/catalog text. The
remaining differences are the ones LangChain imposes:

- **A — token gate.** `wrap_tool_call` exposes no model, so the gate uses the Strands default
  `count_tokens` heuristic (`ceil(chars/4)` for text, `ceil(json chars/2)` for JSON) computed locally.
  It flips at the same size as Strands unless a Strands model opts into native token counting.
- **D — artifact content.** Strands stores a reference name and reads the content back through the
  SDK's context-manager Stash, which LangGraph lacks; this binding stores the return's own text blocks
  under `<tool_call_id>_<index>` instead, so `expand_artifact` can answer, and takes an explicit `stash`
  (the relevance filter's store) as the second resolution layer. Line/pattern reads use
  `context_core.relevance.search` (registered in `HOST_SYMBOLS`).

### The D caveats (documented compromises, not silent losses)

1. **One projection point instead of three events.** LangGraph has a single natural hook
   (`wrap_model_call`). Card derivation runs lazily there from the message list about to be projected —
   same deterministic inputs (closed turns), computed one call later than Strands' `MessageAdded`.
2. **The `NullConversationManager` precondition is advisory.** LangGraph cannot forbid a co-installed
   summarization/pruning middleware, so the middleware **warns once at construction** when it detects one
   (matching `summariz`/`prun`/`trim`/`compact`) and still registers. Do not co-install a middleware that
   deletes from persisted `state["messages"]`.
3. **The note's TTL is aged by a turn ordinal**, not `event_loop_metrics.cycle_count` (there is no agent
   to read); same direction and semantics.
4. **`GraphState` is flattened before persistence** — `TurnChoice.by_title` is a `MappingProxyType`,
   which LangGraph's state-copy cannot pickle; the choice is recomputed each call, so nothing is lost.

The projection is transient to the model request (`override`), so persisted state is never deleted — a
mis-cut costs one recovery call, never a lost fact (the repo's projection-not-destruction invariant).

### One cross-cutting LangGraph fact

LangChain does **not** bridge a sync hook to an async run, or an async hook to a sync run (it raises
`NotImplementedError`). Every hook in the three packages therefore ships both twins, so the combined
stack runs under `invoke` and under `ainvoke`.

## Install

From PyPI:

```bash
pip install langgraph-relevance-filter langgraph-progressive-tool-disclosure langgraph-context-graph
```

Each binding pulls in the shared core, published as `agent-context-core` (import name `context_core`).

From a clone of this repository:

```bash
uv venv --python 3.12 .venv
uv pip install -e context-core
uv pip install -e langgraph-plugins/langgraph-relevance-filter \
               -e langgraph-plugins/langgraph-progressive-tool-disclosure \
               -e langgraph-plugins/langgraph-context-graph
uv pip install "langchain>=1.0,<2" "langgraph>=1.0,<2" langchain-aws
```

Verified against `langchain` 1.4.2.

## Compose all three on one agent

Install order is outermost-first (LangChain nests `wrap_*` hooks, first in the list is the outermost
layer). D wraps B (disjoint `ModelRequest` fields: D rewrites `messages`, B rewrites `tools` +
`system_message`); A is on the tool surface. Hand D the filter's `stash`, so a `[ref: mem_N_…]` the filter
mints resolves through `expand_artifact` as well as through `retrieve_all_context` — the role the
`ContextManager` Stash plays for the Strands graph plugin.

```python
from langchain.agents import create_agent
from langgraph_context_graph import ContextGraphMiddleware
from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware
from langgraph_relevance_filter import RelevanceFilterMiddleware

relevance = RelevanceFilterMiddleware()
agent = create_agent(
    model="bedrock:us.anthropic.claude-opus-4-8",
    tools=[...],  # all tools registered upfront; disclosure controls what the model *sees*
    middleware=[
        ContextGraphMiddleware(stash=relevance.stash),  # outermost: projects messages
        ProgressiveToolDisclosureMiddleware(),          # then rewrites tools + catalog + folds
        relevance,                                      # tool surface
    ],
)
result = await agent.ainvoke({"messages": [...]})  # or agent.invoke(...): every hook has both twins
```

## Benchmark

The replay harness lives in [`validation/plugins-langgraph/`](../validation/plugins-langgraph/). It mirrors the
Strands harness ([`../validation/community-plugin-A-B-D/`](../validation/community-plugin-A-B-D/)): the
same scenario, mocked tools, deterministic ground-truth and accuracy (no LLM judge), the same rate table,
and the same five arms (baseline / A / B / D / all-three). It targets `us.anthropic.claude-opus-4-8` — the
model the Strands published table was measured on — so the two tables can be read side by side.

The harness is verified offline (all five arms build and run through a mocked model; the relevance filter
engages, disclosure trims 87 → 3 tool specs) and a `--live` run is gated behind an explicit acknowledgement
flag. **The live benchmark figures are not yet recorded** — the paid Bedrock run is pending. Once run, its
table (same columns as the Strands `README`/`BENCHMARK.md`, one replay, labeled as such) will be added
here; Strands and LangGraph figures are only presented together with matched conditions stated.

## Provenance and scope

`context-core` is recreated from the Strands community plugins used as **read-only reference**; this work
never modifies, refactors, or ports them, and unifying the Strands packages onto `context-core` is out of
scope. See the spec at `.kiro/specs/analisa-os-3-plugins-de/`.
