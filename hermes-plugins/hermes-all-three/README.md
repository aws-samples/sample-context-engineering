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
| shared store | A's store is handed to D as its `stash`, so a `[ref: …]` the filter mints resolves through `cg_expand_artifact` too |
| `get_tool_schemas` | union of the six tools (`rf_retrieve_all_context`, `ptd_find_tools`, `ptd_get_tool_details`, `cg_expand_card`, `cg_expand_artifact`, `cg_find_context`) |
| `handle_tool_call` | dispatched by tool name to the owning practice |
| `on_turn_complete` | A's close, then D's indexing, in order |

The relevance threshold default here is **`0.02`** — the benchmark value (a distribution position, not
an absolute score).

## Install & select

```bash
# Not published to PyPI - install from a clone of this repository, siblings included
pip install -e context-core -e hermes-plugins/hermes-relevance-filter \
            -e hermes-plugins/hermes-progressive-tool-disclosure -e hermes-plugins/hermes-context-graph \
            -e "hermes-plugins/hermes-all-three[hermes]"   # [hermes] = Hermes host from source
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
