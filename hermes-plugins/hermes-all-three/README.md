# hermes-all-three

The **composed** Hermes Agent context engine: all three context-engineering practices
(relevance filtering + progressive tool disclosure + context graph) in **one**
[`ContextEngine`](https://github.com/NousResearch/hermes-agent/blob/main/agent/context_engine.py).

## Why one engine (single-select)

Hermes allows only **one** active context engine — `register_context_engine` refuses a second. So the
"all three" configuration (the source of the benchmark's headline token savings) cannot be three
co-installed plugins the way LangGraph stacks three middlewares; it is this one composed engine. It
reuses the three sibling packages' engines unchanged over the unchanged `context-core`.

## How it composes

| Seam | Behaviour |
|---|---|
| `select_context` | pipeline **D projects** → **B injects catalog + folds** → **A rewrites oversized results** |
| shared store | A's store is handed to D as its `stash`, so a `[ref: …]` the filter mints resolves through `expand_artifact` too |
| `get_tool_schemas` | union of the six tools (`retrieve_all_context`, `find_tools`, `get_tool_details`, `expand_card`, `expand_artifact`, `find_context`) |
| `handle_tool_call` | dispatched by tool name to the owning practice |
| `on_turn_complete` | A's close, then D's indexing, in order |

The relevance threshold default here is **`0.02`** — the benchmark value (a distribution position, not
an absolute score).

## Install & select

```bash
pip install hermes-all-three          # pulls the three sibling packages
pip install "hermes-all-three[hermes]" # + Hermes host from source
```

```yaml
context:
  engine: all-three
```

Construct with the agent's base tool schemas so disclosure has a catalog:

```python
from hermes_all_three import AllThreeEngine
engine = AllThreeEngine(tool_specs=my_agent_tools)
```

The disclosure base-schema limitation and the relevance token-gate heuristic documented in the sibling
package READMEs apply here unchanged.
