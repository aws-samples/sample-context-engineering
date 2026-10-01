# hermes-progressive-tool-disclosure

Hermes Agent binding for **Practice B — progressive tool disclosure**. A thin
[`ContextEngine`](https://github.com/NousResearch/hermes-agent/blob/main/agent/context_engine.py)
over the unchanged [`agent-context-core`](https://github.com/aws-samples/sample-context-engineering/tree/main/context-core);
the lexical index and the catalog/fold logic live in `context_core.disclosure`.

## What it does

Instead of putting every tool's full schema in front of the model, the engine injects a **one-line-per-tool
catalog** into the system message and registers two tools: `find_tools(need)` searches the catalog
(lexical term-frequency index) and `get_tool_details(names)` returns a tool's full specification on
demand. A tool the model expands stays active, so its spec is not re-summarized; closed
`get_tool_details` exchanges are folded out of later requests.

## How it maps onto the Hermes `ContextEngine`

| Seam | Role |
|---|---|
| `select_context(request_messages)` | inject the catalog into the system message; fold closed `get_tool_details` exchanges; **fail-open** |
| `get_tool_schemas()` / `handle_tool_call()` | `find_tools` + `get_tool_details` |

The engine is constructed with the agent's base tool schemas (OpenAI function-tool dicts) because the
ABC gives the engine no access to the host tool set:

```python
from hermes_progressive_tool_disclosure import ProgressiveToolDisclosureEngine
engine = ProgressiveToolDisclosureEngine(tool_specs=my_agent_tools)
```

## Documented limitation — base schemas are not stripped (non-blocking follow-up)

Strands and LangGraph implement disclosure by rewriting the model request's `tool_specs`/`tools`
directly, so the model literally cannot see a tool until it is disclosed. **Hermes's `ContextEngine`
exposes no hook to rewrite the agent's base tool catalog** — it owns only its own `get_tool_schemas()`
and the message list via `select_context()`. This binding therefore delivers disclosure through the
**portable** path: it injects the catalog and steers the model to `find_tools`/`get_tool_details`, but
the base schemas Hermes assembled still technically reach the provider. This is consistent with the
LangGraph README's "Portable (LangChain ships a subset of this natively)" note. Stripping the base
schemas (if a Hermes host hook for the base tool set is found) is recorded as a **non-blocking
follow-up**, not a requirement.

## Install & select

```bash
pip install hermes-progressive-tool-disclosure
pip install "hermes-progressive-tool-disclosure[hermes]"  # + Hermes host from source
```

```yaml
context:
  engine: progressive-tool-disclosure
```
