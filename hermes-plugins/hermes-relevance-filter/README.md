# hermes-relevance-filter

Hermes Agent binding for **Practice A — relevance filtering of oversized tool results**. A thin
[`ContextEngine`](https://github.com/NousResearch/hermes-agent/blob/main/agent/context_engine.py)
over the unchanged [`agent-context-core`](https://github.com/aws-samples/sample-context-engineering/tree/main/context-core);
all content decisions (chunking, reranking, preview assembly, search) live in `context_core.relevance`.

## What it does

When a tool result exceeds `max_result_tokens` (default 8000), the engine rewrites it in the request
into a marker + a disclaimer carrying the result's real size + a **verbatim** relevance preview scored
against the question in progress + a `[ref: …]` token. The full result is kept in a store and read back
on demand through the `rf_retrieve_all_context` tool. Selection is verbatim, so numeric, monetary and
tabular content stays exact.

## How it maps onto the Hermes `ContextEngine`

| Seam | Role |
|---|---|
| `on_turn_complete(messages)` | detect oversized `role:"tool"` results, store the full text, record the rewrite |
| `select_context(request_messages)` | request-only copy with oversized results replaced; closed `rf_retrieve_all_context` exchanges dropped; **fail-open** |
| `get_tool_schemas()` / `handle_tool_call()` | the `rf_retrieve_all_context` recovery tool |
| `compress()` | budget fallback only (A's saving is in the rewrite) |

Hermes hands the engine the finished turn's messages (`on_turn_complete`), not each tool result at
production time, so detection happens there rather than in a per-result hook. The token gate is the
Strands default heuristic (`ceil(chars/4)` text, `ceil(json chars/2)` JSON) recomputed here — Hermes's
synchronous engine exposes no model handle for native token counting.

## Install & select

```bash
# Not published to PyPI - install from a clone of this repository
pip install -e context-core -e "hermes-plugins/hermes-relevance-filter[hermes]"  # [hermes] = Hermes host from source
```

Copy/symlink the installed package into `$HERMES_HOME/plugins/relevance-filter/`, then in `config.yaml`:

```yaml
context:
  engine: relevance-filter
```

Construction is inert — no reranker and no AWS client is built until the first oversized result, so an
agent that never trips the gate needs no credentials merely to select the engine.
