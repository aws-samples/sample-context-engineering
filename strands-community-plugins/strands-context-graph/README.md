# strands-context-graph

Short-term memory as a graph of Cards for Strands Agents. One Subject Card per closed turn, derived by deterministic
scan — no model call. Each Card enters a model call at one of three Resolutions: **Title** (always present),
**Description** (rule-derived, capped by `description_tokens`) or **Full Content**. Each turn a Note is computed per
Card from the similarity between the turn's question and the Card's Description, then propagated one jump along the
Links; the Note picks the Resolution. Resolution descends only for budget, never as a verdict.

The plugin reads the per-call copy the event loop hands to `InvokeModelStage` and never mutates `agent.messages`.
Every message stays readable in the history regardless of Resolution.

```python
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager

from strands_context_graph import ContextGraph

agent = Agent(
    conversation_manager=NullConversationManager(),
    plugins=[ContextGraph()],
)
```

## Read this before wiring it

### 1. `NullConversationManager` is a precondition, not a suggestion

Use `NullConversationManager`. Any other conversation manager edits the **live** message list before the call is
assembled, so it can physically drop what the graph only meant to fold — the folded Description block then describes
messages that no longer exist. The plugin emits one `warnings.warn` at wiring time when it sees a manager that is not
`NullConversationManager`, and still registers everything; the warning is the whole protection you get.

### 2. Ordering against a memory manager has no standalone escape hatch

The graph's `InvokeModelStage.Input` middleware must run its Removal **before** a memory manager folds its own block
into the call. A standalone plugin cannot assert same-stage middleware ordering, so when a memory manager is present the
plugin emits one wiring-time `warnings.warn` naming the requirement and the risk: a folded memory block may be projected
out. This fails **silently, as a worse answer** — not as an exception. Verify the order in your own wiring; the
ordering integration test is this package's merge gate.

### 3. The three thresholds are calibrated against the default matcher and are not portable

`expand_threshold` (0.55), `collapse_floor` (0.45) and `link_threshold` (0.50) are calibrated against the score
distribution of the default `EmbeddingSimilarityMatcher`. Supply your own `matcher` and those numbers mean nothing —
another matcher needs its own calibration. Nothing validates this for you: a mismatched matcher produces plausible
scores at the wrong scale and the graph quietly collapses or expands everything.

## Regression switch

`expand_threshold=0.0, collapse_floor=0.0` projects every Card at Full Content, producing a call identical field for
field to the one produced without the plugin. Use it to tell a graph problem apart from a pre-existing one.

## Where the logic lives

The Card model, the deterministic scan, the scoring, the similarity matcher, the projection and the bodies of the three
retrieval tools live in `agent-context-core` (`context_core.graph`), the same code the LangGraph and Hermes bindings run.
This package is the Strands interface: the `InvokeModelStage.Input` handler that calls `context_core.graph.project()` on
every model call, the `AfterToolCallEvent` hook that records artifact references, and the `@tool` wrappers
(`plugin.py`); every SDK import (`_compat.py`); and the conversion between Strands messages and the core's neutral shape,
which keeps each message's `tracking_id` (`_adapter.py`).

Because `project()` runs on every model call, the Turn Choice, the reuse TTL clock and the retrieval budget advance per
model call, as in the LangGraph binding.

## Install

From a clone of [sample-context-engineering](https://github.com/aws-samples/sample-context-engineering):

```bash
uv pip install -e context-core -e strands-community-plugins/strands-context-graph
```

## Dependencies

Two runtime dependencies: `agent-context-core` and `strands-agents` (range-pinned with a real upper bound). No graph
library, no similarity library. The only remote call in any configuration is the default matcher's embedding round; its
cache absorbs Descriptions that did not change between model calls.

## Private-API caveats

The plugin depends on private SDK symbols and is explicit about the cost. Read the **Degrades as** column first: it says
what a rename actually costs you, and it is the only column that varies.

| Private symbol | Used by | What it buys | Degrades as | What would remove the fragility |
| --- | --- | --- | --- | --- |
| `strands._middleware.stages.InvokeModelStage` + its `.Input` phase | `_compat.py`, `plugin.py` | The one seam where the messages sent to the provider change, without touching `agent.messages` | **Hard dependency.** `ImportError` at import — the plugin has no other delivery point | A public per-call input-transform stage |
| `dynamic_trailing_blocks` on `InvokeModelContext` | `plugin.py` | Telling downstream links how many blocks were appended, so they are not mistaken for history | **Hard dependency**, on the same context object | A public trailing-block API |
| Same-stage ordering against a memory manager's fold | `plugin.py` | The Removal running before any memory block is folded in | **Silently, as a worse answer.** No exception; one wiring-time `warnings.warn` is the whole protection | Declared middleware ordering (priority, or a documented phase order) |
| `agent._middleware_registry._handlers` | `plugin.py` | Moving the delivery to index zero of the stage, and reading back whether anything folds behind it | **No ordering guarantee.** The delivery stays registered in wiring order and the ordering notice fires | The same declared-ordering API |
| `agent._plugin_registry._plugins` | `plugin.py`, `context_core.graph.store` | Finding what else is wired to the agent — a memory manager to warn about, a `ContextManager` to bridge to | **Quietly.** Reads as "nothing else is wired": no ordering notice, no Stash bridge | A public read-only view of an agent's plugins |
| `_provide_memory_context` / `_injection_config` on a memory manager | `plugin.py` | Recognising a memory fold by member rather than by type | **No ordering notice.** The manager is not recognised, so the warning that protects the ordering is not emitted | A public capability marker for "this plugin folds into the model input" |
| `ContextManager` / `_stash` / `_extract_text` | `_compat.py` registers them; `context_core.graph.store` resolves them | The optional Stash fallback bridge for artifact references | **No Stash interop.** The plugin's own store still resolves every reference it recorded; a Stash-only reference answers as prose naming the miss | A public reference-store protocol on `ContextManager` |
| `context_offloader.search._search_content` | `_compat.py` registers it; `context_core.graph.store` resolves it | Line/pattern matching inside a resolved artifact without reimplementing its guards | **Whole reads only.** A `line_range`/`pattern` request answers as prose saying targeted reads are unavailable | A public artifact-search helper |

Three SDK rules the plugin used to import are no longer imported: the turn boundary (`_is_user_turn`), the tool-pair
index (`_get_tool_use_ids`, with `is_pinned`) and the fold of a trailing block into the last `user` message
(`_create_injection_middleware`). `context_core.graph` carries framework-free copies, and `tests/test_sdk_parity.py`
checks each copy against the SDK original on generated conversations, skipping with the symbol's name when the SDK no
longer ships it. The one deliberate difference: the core's fold keeps the folded message's `tracking_id`.

Two shapes of dependency, then. The **hard** ones fail at import: they are the seam the plugin is built on, and a rename
there is a version-range problem, which is why `strands-agents` is pinned with a real upper bound. Everything else is
resolved by name at call time and degrades — the Stash bridge in particular is built **entirely** on private symbols,
registered in `context_core.graph.store.HOST_SYMBOLS` by `_compat.py`, and turns into no-Stash-interop, not a crash, if
any of them is renamed or changes shape. No handler, tool or middleware propagates an exception out of the agent loop in
any of these cases.

## License

Apache-2.0
