# hermes-context-graph

Hermes Agent binding for **Practice D — context graph**. A thin
[`ContextEngine`](https://github.com/NousResearch/hermes-agent/blob/main/agent/context_engine.py)
over the unchanged [`agent-context-core`](https://github.com/aws-samples/sample-context-engineering/tree/main/context-core);
the Card model, scan, scoring, matcher and projection live in `context_core.graph`.

## What it does

Each closed turn becomes a **Card**. On every request the engine projects the conversation at the
resolution this question needs — Full Content, Description, or just Title — so the model sees a compact
graph instead of the whole transcript. The **persisted history is never deleted** (projection, not
destruction): a mis-cut costs one recovery call, not a lost fact. Three tools raise folded content back:
`expand_card(titles)`, `expand_artifact(reference)`, `find_context(need)`.

## How it maps onto the Hermes `ContextEngine`

| Seam | Role |
|---|---|
| `on_turn_complete(messages)` | store each tool return as an addressable artifact under `<tool_call_id>_0`; derive its artifact Cards |
| `select_context(request_messages)` | run `context_core.graph.project` over closed turns; return the projected list; request-only; **fail-open** |
| `get_tool_schemas()` / `handle_tool_call()` | `expand_card` + `expand_artifact` + `find_context` |

The graph state and reference store live on the engine instance (process-local, not Hermes persisted
state — the same rationale as the LangGraph binding's per-conversation store).

## Single-select bonus

Because this engine **replaces** Hermes's default lossy `context_compressor`, the "nothing may delete
from the history behind the graph" precondition is satisfied structurally — there is no co-active
summarizer to fight (in LangGraph the middleware can only *warn* about a co-installed pruning
middleware; here single-select makes the conflict impossible).

## Install & select

```bash
pip install hermes-context-graph
pip install "hermes-context-graph[hermes]"  # + Hermes host from source
```

```yaml
context:
  engine: context-graph
```

Pass `stash=` a relevance filter's store so a `[ref: …]` the filter minted also resolves through
`expand_artifact` (this is what `hermes-all-three` wires up).
