# Sequence B — Progressive Tool Disclosure

| Band | Tree | Files |
|------|------|-------|
| **1. Strands interface** | `strands-community-plugins/strands-progressive-tool-disclosure/src/strands_progressive_tool_disclosure/` | `plugin.py`, `_compat.py` |
| **2. Message adapter** | the same package | `_adapter.py` |
| **3. `context-core`** | `context-core/src/context_core/` | `disclosure/catalog.py`, `disclosure/tool_index.py`, `message.py` |

All `file.py:LINE` references resolve into those three trees: bands 1 and 2 as `plugin.py`, `_compat.py`
and `_adapter.py`, band 3 as `catalog.py`, `tool_index.py` and `message.py`. Harness references are into
`validation/community-plugin-A-B-D/src/` (`runner.py`, `config.py`). The ordering section (§11)
additionally names the context-graph package and the installed Strands SDK by full path in prose,
because both sit outside the reference checker's roots. Every claim below carries a line reference, and
every literal string is quoted verbatim from source.

This is the Strands binding of the same practice the LangGraph document
[`langgraph/sequence-b-disclosure.md`](langgraph/sequence-b-disclosure.md) describes. The decision logic
is shared and lives in the core; what differs is everything that touches a framework — a per-agent
`WeakKeyDictionary` instead of graph state, `agent.event_loop_metrics.cycle_count` instead of a cycle
derived from the messages, a `replace` on the `InvokeModelStage.Input` context instead of
`request.override`, and a `BeforeToolCallEvent` hook instead of a wrapped tool call.

## 1. What the plugin does, mechanically

On every model call the plugin rewrites **three** fields of the invocation context, all three in one
`replace` (`plugin.py:634`).

`tool_specs` is reduced to the tools that are callable on this call, each one a **full verbatim spec**:
the two plugin tools `find_tools` and `get_tool_details`, the `always_available` names, and the tools
whose schema was **loaded** by a prior `get_tool_details` and is still live under a TTL measured in
event-loop cycles (`_compose_projection` `plugin.py:556`, driven from `_project` `plugin.py:599`). Every
other registered tool reaches the model as **one line of a catalog appended to the system prompt** — its
name and a summary of its description, at most `catalog_chars` characters (`catalog_prompt_block`
`catalog.py:370`, line built at `catalog.py:397`).

`system_prompt` receives that catalog block, appended (`_append_to_system_prompt` `plugin.py:208`, called
at `plugin.py:638`).

`messages` receives a folded copy: every **closed** exchange of a tool this call does not carry loses its
`toolUse` block, and its `toolResult` becomes one plain sentence — `The tool X was called and the result
was: Y` (`fold_closed_exchanges` `catalog.py:537`, `fold_note` `catalog.py:452`, driven from
`plugin.py:626`). The plugin's own two exchanges are dropped outright, with no sentence
(`catalog.py:598`). The agent's own history is never mutated — the fold produces a new list for this call
only, and when it folded nothing the context keeps the very list it arrived with (`plugin.py:628`).

**A loaded tool is released, not retained.** There is no history-retention block: the projection does
**not** keep a tool because the messages still mention it. A `toolUse` left in the history for a tool
absent from `tool_specs` is accepted by the Bedrock Converse API — checked on Claude Haiku 4.5,
Claude Opus 4.8, GLM 4.7 Flash and Qwen3 Next (module docstring `plugin.py:14`) — and retaining those
tools was what made `tool_specs` grow monotonically with every tool a conversation had touched
(`_compose_projection` docstring, `plugin.py:556`).

There is **one placement** and no mode flag. Nothing reduced is ever put in `tool_specs`, so nothing the
provider is shown asserts an empty `inputSchema`: a catalog name is not in `tool_specs` at all, and the
rule that governs it arrives in the same block as the name (`CATALOG_PROMPT_HEADER` `catalog.py:84`). The
projection and the catalog **partition** the registry between them — the block is rendered by skipping
exactly the projected names, so no tool appears in both and none is missing from both (`plugin.py:633`,
`catalog.py:394`).

**The common path is catalog → `get_tool_details([names])` → call.** The model reads a name in the
system prompt, calls `get_tool_details` with the names it wants, which loads them into `state.exposed`
(`_renew` `plugin.py:961`) and answers with a short confirmation; the full schemas arrive on the *next*
model call. `find_tools` is the **fallback** for a need no catalog name fits: it ranks the specs through
the `LexicalToolIndex` (`tool_index.py:177`) term-frequency `search` (`tool_index.py:214`) and lists
names plus summaries, and it **exposes nothing** — loading is `get_tool_details`' one job, so the model
takes the same road to a schema wherever it started (`find_tools` `plugin.py:867`, comment
`plugin.py:922`).

Two hooks maintain the TTL and the guard. `AfterToolCallEvent` (`_on_after_tool_call` `plugin.py:1008`)
renews a loaded tool that actually **ran**, so a stretch of work on one tool never reloads it.
`BeforeToolCallEvent` (`_on_before_tool_call` `plugin.py:972`) cancels a call to a tool the last
projection did not carry when that tool requires parameters — and loads **nothing** on the model's
behalf.

Every registered tool stays in the `ToolRegistry` and stays callable throughout. Release withdraws a
schema from the next projection, it does not unregister anything (`_expire` docstring, `plugin.py:404`).

## 2. The three bands

The binding is three layers with one rule each: the **interface** band touches the Strands SDK and
nothing else, through `_compat.py`; the **adapter** band is the only module that sees both worlds
(`_adapter.py:11`); the **core** band imports no framework at all (`catalog.py:26`).

```mermaid
flowchart TB
    subgraph band1["Band 1 · Strands interface — SDK only, through _compat.py"]
        direction TB
        IA["init_agent · one InvokeModelStage.Input handler per agent<br/>plugin.py:730 · 747"]
        PH["_projection_handler · expire, index, project<br/>plugin.py:749"]
        ENS["_ensure_index · fingerprint, then the missing summaries<br/>plugin.py:789 · 820"]
        MS["_model_summarizer · one plain model.stream call per tool<br/>plugin.py:173 · 195"]
        PJ["_project · replace of tool_specs, messages, system_prompt<br/>plugin.py:599 · 634"]
        CP["_compose_projection · three blocks of FULL specs<br/>plugin.py:556"]
        SYS["_append_to_system_prompt · keeps whichever shape arrived<br/>plugin.py:208"]
        TOOLS["find_tools and get_tool_details · auto-registered by Plugin<br/>plugin.py:867 · 927"]
        HOOKS["BeforeToolCallEvent guard · AfterToolCallEvent renewal<br/>plugin.py:972 · 1008"]
        ST["_DisclosureState in a WeakKeyDictionary keyed by agent<br/>plugin.py:333 · 369 · 383"]
    end
    subgraph band2["Band 2 · message adapter — the only module touching both worlds"]
        direction TB
        TN["messages_to_neutral · a new list over the same message dicts<br/>_adapter.py:28"]
        TL["messages_from_neutral · the same dicts back as Strands messages<br/>_adapter.py:33"]
        TS["specs_to_neutral · the same spec dicts, typed neutral<br/>_adapter.py:38"]
    end
    subgraph band3["Band 3 · context-core — context_core.disclosure, no framework import"]
        direction TB
        BC["catalog_prompt_block · the rule plus one line per hidden tool<br/>catalog.py:370"]
        FCE["fold_closed_exchanges · closed exchanges become sentences<br/>catalog.py:537"]
        GUARD["requires_parameters · only a non-empty required list decides<br/>catalog.py:203"]
        LTI["LexicalToolIndex · build and search, term frequency<br/>tool_index.py:197 · 214"]
        TRUNC["clamp_summary and truncate_description · the budgeted line<br/>catalog.py:231 · 128"]
    end
    IA --> PH
    PH --> ENS
    ENS --> MS
    ENS -->|"build over the incoming specs plugin.py:814"| LTI
    MS --> TRUNC
    PH --> PJ
    PJ --> CP
    PJ -->|"the names NOT projected plugin.py:633"| BC
    BC --> SYS
    PJ -->|"messages_to_neutral plugin.py:625"| TN
    TN --> FCE
    FCE -->|"a new list only when something folded plugin.py:628"| TL
    TL --> PJ
    PJ -.->|"writes state.projected after the projection plugin.py:782"| ST
    TOOLS -->|"search plugin.py:899"| LTI
    TOOLS -.->|"_renew writes state.exposed plugin.py:961"| ST
    HOOKS --> GUARD
    HOOKS -.->|"reads projected, renews exposed plugin.py:997 · 1024"| ST
    PJ -.->|"reads exposed, post-expiration plugin.py:778"| ST
```

Two facts the picture is worth reading for. First, **band 2 converts nothing.** The neutral shape was
modelled on the Strands SDK's own message dicts, so a Strands `Message` already *is* a neutral message
and the adapter only states that to the type checker while guaranteeing that unknown blocks —
`reasoningContent`, `image`, `document`, `cachePoint` — pass through untouched and nothing is mutated
(`_adapter.py:1`). That is why the fold's identity guarantee survives the round trip: the core returns
the received sequence itself when it folds nothing, and `plugin.py:628` tests exactly that.

Second, the summary path is the one decision that **cannot** move into band 3. The core's `SummaryCache`
(`catalog.py:271`) takes a synchronous summarizer (`catalog.py:282`), while this binding's default
summarizer is an async call to the agent's own model (`_model_summarizer` `plugin.py:173`), so the cache
here is the plugin instance's own dict keyed by `(name, description)` (`plugin.py:726`) and what it asks
of the core is only the clamping and the truncation fallback (`plugin.py:165`, `plugin.py:170`).

### 2a. What each band owns

| Step | Band | Why it sits there |
|------|------|-------------------|
| `init_agent`, the middleware handler, the two `@hook` methods, the two `@tool` methods | 1 | Pure SDK attachment. Every Strands symbol arrives through `_compat.py`, including the private middleware seam `InvokeModelStage` (`_compat.py:14`). |
| `_DisclosureState`, `_state_for`, `_expire`, `_renew` | 1 | TTL state is per agent and per process, and it is keyed by the framework's own agent object (`plugin.py:369`). |
| `_model_summarizer`, `_summarize_missing`, `self._summaries` | 1 | A model call is a framework concern, and so is the concurrency gate around a burst of them (`plugin.py:841`). |
| `messages_to_neutral`, `messages_from_neutral`, `specs_to_neutral` | 2 | The single place both worlds are touched (`_adapter.py:11`). No conversion, only identity-preserving re-typing (`_adapter.py:28`). |
| `catalog_prompt_block`, `CATALOG_PROMPT_HEADER`, the `- name: summary` line | 3 | Every decision about *what the model is told*, over the neutral spec shape (`catalog.py:370`). |
| `fold_closed_exchanges`, `fold_note`, `pairs_intact`, `_current_turn_start`, `_tidy` | 3 | The whole fold, including its own safety net (`catalog.py:518`). Shared verbatim with the other bindings. |
| `requires_parameters`, `clamp_summary`, `truncate_description`, `estimate_tokens` | 3 | Pure functions over plain dicts and strings (`catalog.py:203`, `catalog.py:109`). |
| `LexicalToolIndex`, `ToolIndex`, `ToolMatch` | 3 | Search is a protocol with a build phase and no network (`tool_index.py:53`), re-exported by the binding's `__init__.py` so a caller swaps it without importing the core. |

## 3. Integration table — every SDK attachment point

| # | SDK seam | `file.py:LINE` | Callback / order | Reads | Mutates |
|---|----------|----------------|------------------|-------|---------|
| 1 | `Plugin` base class (subclassed) | `plugin.py:642` `class ProgressiveToolDisclosure(Plugin)` | `Plugin` auto-registers `@hook` and `@tool` members but NOT middleware (per `init_agent` docstring `plugin.py:733`). The base class is reached only through `_compat.py` (`_compat.py:24`) | — | — |
| 2 | `init_agent(agent)` lifecycle callback | `plugin.py:730` | Called by the SDK when an agent is initialized | `agent._middleware_registry` | Adds one middleware to `InvokeModelStage.Input` (`plugin.py:747`) |
| 3 | `InvokeModelStage.Input` middleware — the projection | `plugin.py:747` (registration), `_projection_handler` `plugin.py:749` (handler) | Middleware stage `InvokeModelStage.Input`; the seam is reached only through `_compat.py`, which imports `InvokeModelContext` and `InvokeModelStage` from `strands._middleware.stages` (`_compat.py:14`). **No explicit order is set** — the handler is appended in wiring order, so it runs after any middleware that forced itself to the front (see §11). | `context.tool_specs`, `context.messages`, `context.system_prompt`, `context.agent`, `agent.tool_registry.registry`, `agent.event_loop_metrics.cycle_count`, `agent.model` (default summarizer) | Returns a NEW context via `replace(context, tool_specs=..., messages=..., system_prompt=...)` (`plugin.py:634`), or `tool_specs` and `messages` alone when the catalog is suppressed (`plugin.py:631`); writes `state.projected` (`plugin.py:782`) and mutates per-agent `_DisclosureState` (`_expire` `plugin.py:404`) plus the shared summary cache and index fingerprint (`_ensure_index` `plugin.py:789`). Does NOT touch the registry or `agent.messages`. |
| 4 | `@tool(context=True)` vended tool `get_tool_details` | `plugin.py:926` decorator, `plugin.py:927` method | Auto-registered by `Plugin`, by the `_PluginRegistry` AFTER `init_agent` (`plugin.py:730`) returns, as its docstring states, so early calls can arrive without it — covered by `_should_passthrough` (`plugin.py:520`). | `tool_context.agent`, `agent.event_loop_metrics.cycle_count`, `agent.tool_registry.registry`, the `names` argument | Increments `state.loads` (`plugin.py:944`); writes `state.exposed` via `_renew` for each known name (`plugin.py:961`). Returns a text result string. |
| 5 | `@tool(context=True)` vended tool `find_tools` | `plugin.py:866` decorator, `plugin.py:867` method | Auto-registered by `Plugin`, same post-`init_agent` window as row 4. | `tool_context.agent`, `agent.tool_registry.registry`, the `need` argument, the index | Increments `state.searches` (`_record_search` `plugin.py:481`, called `plugin.py:890`). **Writes no exposure** — it only reads the index and the registry. |
| 6 | `BeforeToolCallEvent` hook | `plugin.py:971` `@hook`, `_on_before_tool_call` `plugin.py:972` | Sync hook (the `# type: ignore` note at `plugin.py:971` says the `@hook` overloads only infer async). Auto-registered by `Plugin`. Fires before each tool call. No numeric order documented. | `event.tool_use["name"]` (`plugin.py:989`), `event.agent`, `agent.tool_registry.registry`, `state.projected`, `self._always_available`, `requires_parameters(spec)` (`catalog.py:203`) | On a premature call sets `event.cancel_tool` (`plugin.py:1004`) and increments `state.premature_cancellations` (`_record_premature_cancellation` `plugin.py:505`, called `plugin.py:1005`). **Writes no exposure at all** — it never calls `_renew`. |
| 7 | `AfterToolCallEvent` hook | `plugin.py:1007` `@hook`, `_on_after_tool_call` `plugin.py:1008` | Sync hook, same `# type: ignore` note. Auto-registered by `Plugin`. Fires after each tool call. | `event.cancel_message` (`plugin.py:1017`), `event.tool_use["name"]` (`plugin.py:1019`), `self._always_available` (`plugin.py:1020`), `state.exposed` | Renews an already-loaded tool via `_renew` (`plugin.py:1024`). A cancelled call returns first (`plugin.py:1017`), so it keeps nothing loaded, and a plugin tool or an always-available name is skipped outright — neither is ever in `exposed`. |

Supporting SDK type imports (not attachment points, but the surface used), all re-exported by `_compat`
and imported from there at `plugin.py:55`: `AfterToolCallEvent` and `BeforeToolCallEvent`
(`_compat.py:23`), `Plugin` and `hook` (`_compat.py:24`), `tool` (`_compat.py:25`), `Messages` and
`SystemPrompt` (`_compat.py:26`), `ToolContext` and `ToolSpec` (`_compat.py:27`), with `Agent` and
`Model` under `TYPE_CHECKING` (`_compat.py:30`, `_compat.py:31`). `SystemPrompt` is the type of the field
the catalog is appended to, `Messages` the type the fold rewrites. Everything the core contributes
arrives in one import at `plugin.py:38`, and the adapter in one at `plugin.py:54`.

Per-agent state is held in a `WeakKeyDictionary` keyed by agent (`_DisclosureStates` `plugin.py:369`,
`_new_disclosure_states` `plugin.py:374`, read through `_state_for` `plugin.py:383`), so one plugin
instance serves many agents without keeping any alive; state never reaches `agent.state` or session
storage (`_DisclosureState` docstring `plugin.py:337`). Summaries are the one thing shared across agents,
because a summary depends on the description alone (`self._summaries` `plugin.py:726`, `init_agent`
docstring `plugin.py:737`).

## 4. The projection model — three full-spec blocks, then the catalog

`_compose_projection` (`plugin.py:556`) visits three blocks in a **fixed order** and every name it emits
carries its full verbatim spec. A name emitted by an earlier block is added to `seen` and never
re-emitted (`plugin.py:592`, `plugin.py:593`). `_project` at `plugin.py:599` is the entry point that calls
it, folds the messages, and then places whatever is left into the system prompt.

| Order | Class | Membership source (`file.py:LINE`) | What the model receives |
|-------|-------|-------------------------------------|--------------------------|
| 1 | `{find_tools, get_tool_details}` | frozenset `PLUGIN_TOOL_NAMES` (`catalog.py:67`), first block of the tuple `plugin.py:589` (names from `FIND_TOOLS_NAME` `catalog.py:58` and `GET_TOOL_DETAILS_NAME` `catalog.py:61`) | **Full verbatim spec** — emitted first and unconditionally, which is what makes the projection non-empty on every projected path |
| 2 | `always_available` | `self._always_available` tuple, passed at `plugin.py:779`; set in ctor `plugin.py:719` | **Full verbatim spec** on every call |
| 3 | `exposed` | `state.exposed` map (post-`_expire` `plugin.py:404`), passed at `plugin.py:778`; written only by `_renew` (`plugin.py:427`), called from `get_tool_details` (`plugin.py:927`) at `plugin.py:961` and from the post-call hook (`plugin.py:1024`) | **Full verbatim spec** while the load is live under the TTL |
| 4 | catalog residue | every incoming name **not** in the projection, computed from the projection itself (`plugin.py:633`) and skipped while rendering (`catalog.py:394`); the whole block is skipped when `catalog_chars is None` (`plugin.py:631`) | **Nothing in `tool_specs`.** One line `- name: summary` in the system prompt (`catalog.py:397`), under the header at `catalog.py:401` |

There is deliberately **no history-referenced block**. A tool the conversation already used is a catalog
name again once its load is released, which is what keeps `tool_specs` at the mandatory set when the work
moves on (`_compose_projection` docstring, `plugin.py:556`).

Iteration inside every block follows `incoming` order, not the container's, for prompt-cache stability
(`_compose_projection` docstring, `plugin.py:556`; loop `plugin.py:590`).

`catalog_prompt_block` returns `""` when every incoming tool is already carrying a full specification
(`catalog.py:398`, `catalog.py:399`), and `_append_to_system_prompt` returns the prompt unchanged **by
identity** on an empty block (`plugin.py:226`) — an empty catalog adds nothing rather than a header
promising a list. Appending rather than prepending is deliberate: the operator's own text keeps the
opening position and, on a provider that caches by prefix, stays at a stable offset
(`_append_to_system_prompt` docstring, `plugin.py:208`). The three shapes of `SystemPrompt` are each
preserved: `None` becomes the block (`plugin.py:229`), a `str` is joined with a blank line
(`plugin.py:231`), and a list gains one text block rather than being flattened (`plugin.py:232`).

**Passthrough is structural, not a flag.** `_should_passthrough` (`plugin.py:520`) returns `True` when
any incoming name is absent from the registry — forced structured output swaps in a synthetic spec
(`plugin.py:548`) — and when **either** plugin tool is missing from the call: without `get_tool_details`
there is no way to load a hidden schema and without `find_tools` no way to find one, so there is nothing
to hide (`plugin.py:551`, `plugin.py:553`). It is the first thing the handler asks (`plugin.py:768`).

## 5. The message fold — every exchange the call cannot repeat

A `toolUse` block carries the tool's name **and its arguments**. Sitting next to a successful result it
reads as a template, and the model repeats it — which, once that tool has left `tool_specs`, is a call to
a tool it cannot see. So the fold removes the call shape and keeps the evidence
(`fold_closed_exchanges` docstring, `catalog.py:537`).

The decision is entirely the core's. What band 1 does is hand over the call's own messages and take back
whatever comes: `messages_to_neutral` (`_adapter.py:28`) at `plugin.py:625`, the fold at `plugin.py:626`,
and `messages_from_neutral` (`_adapter.py:33`) **only** when the returned sequence is not the one that
went in (`plugin.py:628`). Nothing folded therefore costs not even a list allocation, and the context
keeps its own `messages` object.

What is folded, and what is not:

- **Only closed turns.** `_current_turn_start` (`catalog.py:473`) returns the index of the last user
  message carrying no `toolResult` (`catalog.py:487`, `catalog.py:489`); everything from there on is the
  turn in flight, tool loop and latest assistant message included, and it passes through as **the very
  same objects** — the single exception being the user message that opened the turn, which takes the
  folded span's trailing content when there is any (below). That is what keeps a reasoning provider's
  latest assistant message byte-intact. `messages[0]` is likewise untouched (`catalog.py:587`), and a
  boundary of `0` or `1` means nothing is folded at all (`catalog.py:570`, `catalog.py:571`).
- **Only closed pairs of non-callable tools.** A `toolUse` is enrolled when its tool is a plugin tool or
  is not in this call's projection, and only if a `toolResult` answering it exists in the span
  (`catalog.py:579`, `catalog.py:584`).
- **The result becomes a sentence.** `fold_note` (`catalog.py:452`) renders `The tool X was called and
  the result was: Y`, or `... failed with: Y` when the result's status is an error (`catalog.py:469`);
  image and document parts of the result are kept as they are (`catalog.py:600`).
- **The plugin's own exchanges go entirely**, with no sentence: a `find_tools` or `get_tool_details`
  exchange matters on the call right after it and is dead weight past that (`catalog.py:598`). The two
  names come from the core's own `PLUGIN_TOOL_NAMES`, which is the `drop_names` default
  (`catalog.py:541`).

The rewrite then has to leave a shape Converse accepts. Emptied messages are dropped
(`catalog.py:608`), same-role neighbours are merged (`catalog.py:613`), and `_tidy` (`catalog.py:512`)
fixes each message for its role: an assistant message that was rewritten loses its `reasoningContent`,
because a modified message can no longer carry a valid signature (`_without_reasoning`
`catalog.py:494`, applied `catalog.py:607`), and a user message puts its `toolResult` blocks first
(`_results_first` `catalog.py:502`) — Converse rejects text placed ahead of the `toolResult` that answers
the previous assistant message. When the folded span now ends with a user message, its content is joined
into the **opening** user message of the turn in flight rather than into the turn's assistant messages
(`catalog.py:622`, `catalog.py:624`).

The safety net is the fold's own invariant, checked on its output. `pairs_intact` (`catalog.py:518`)
verifies that every `toolUse` is still answered in the very next message; when the fold broke an
adjacency the original messages are returned, with one warning (`catalog.py:628`, `catalog.py:629`). A
case the fold did not foresee therefore costs the saving on that call, never the call itself. The last
message is excluded from the check, since an assistant `toolUse` still waiting for its result is a
legitimate tail.

```mermaid
sequenceDiagram
    box rgb(222,235,255) Band 1 · Strands interface
        participant Project as _project [plugin.py:599]
    end
    box rgb(255,241,214) Band 2 · message adapter
        participant Ad as _adapter.py
    end
    box rgb(224,245,224) Band 3 · context-core
        participant Fold as fold_closed_exchanges [catalog.py:537]
        participant Turn as _current_turn_start [catalog.py:473]
        participant FoldNote as fold_note [catalog.py:452]
        participant Check as pairs_intact [catalog.py:518]
    end

    Project->>Ad: messages_to_neutral(context.messages) [plugin.py:625]
    Ad-->>Project: a new list over the SAME message dicts [_adapter.py:28]
    Project->>Fold: (neutral, names in the projection) [plugin.py:626]
    Fold->>Turn: index of the last user message with no toolResult [catalog.py:487]
    Turn-->>Fold: boundary · messages[boundary:] is the turn in flight
    alt boundary <= 1
        Fold-->>Project: the received sequence itself [catalog.py:571]
    else closed span exists
        Fold->>Fold: enrol closed pairs of tools NOT in the projection [catalog.py:579]
        alt nothing enrolled
            Fold-->>Project: the received sequence itself [catalog.py:585]
        else pairs to fold
            loop each message of messages[1:boundary]
                Fold->>Fold: drop the toolUse block [catalog.py:593]
                alt plugin tool
                    Fold->>Fold: drop the toolResult with no sentence [catalog.py:598]
                else domain tool
                    Fold->>FoldNote: (name, toolResult)
                    FoldNote-->>Fold: "The tool X was called and the result was: Y" [catalog.py:470]
                    Fold->>Fold: keep image and document parts [catalog.py:600]
                end
            end
            Fold->>Fold: drop emptied messages, merge same-role neighbours [catalog.py:608]
            Fold->>Fold: _tidy · assistant loses reasoningContent · user puts results first [catalog.py:512]
            Fold->>Fold: a trailing user message joins the turn's opening message [catalog.py:622]
            Fold->>Check: pairs_intact(folded minus the last message)
            alt an adjacency broke
                Check-->>Fold: False · warn and return the originals [catalog.py:629]
                Fold-->>Project: the received sequence itself [catalog.py:630]
            else intact
                Fold-->>Project: a NEW folded list [catalog.py:631]
            end
        end
    end
    Project->>Ad: messages_from_neutral(folded), only if the list is new [plugin.py:628]
    Ad-->>Project: the same dicts, typed as Strands messages [_adapter.py:33]
```

## 6. The first projection — index build and the catalog lines

Neither the index nor a summary can be produced at construction time: both need the specifications of a
call, and the first projection is what has them (`_ensure_index` docstring `plugin.py:792`). The incoming
`(name, description)` **pairs** are kept as a fingerprint and compared on every projection, so an MCP
tool or a `register_dynamic_tool` arriving at runtime — and equally a tool **re-registered with a new
description** — triggers exactly one rebuild (`plugin.py:806`, `plugin.py:807`, written at
`plugin.py:818` only after the build returns, so a build that raises is retried next call). Keying on the
description as well as the name is what makes the fingerprint agree with the summary cache, which is
keyed the same way: a changed description invalidates both together instead of leaving a stale line
behind a matching name (`plugin.py:724`).

A catalog line is produced in three tiers, in this order of preference (`_summarize` `plugin.py:142`):

1. **Verbatim.** A description that already fits `catalog_chars` is the best summary of itself and costs
   no call (`plugin.py:158`, `plugin.py:159`).
2. **Summarized.** A longer one goes to the summarizer, sync or async (`plugin.py:162`,
   `plugin.py:164`), and the answer is normalized by the core: whitespace collapsed, wrapping quotes
   dropped, and a boundary cut when it overran (`clamp_summary` `catalog.py:231`, applied
   `plugin.py:165`). The default summarizer is one plain call to the agent's own model with no tools and
   no history, so it passes through no agent middleware and cannot recurse into this projection
   (`_model_summarizer` `plugin.py:173`, `model.stream` `plugin.py:195`); its usage is accumulated into
   `state.summary_usage` (`plugin.py:194`, field `plugin.py:366`).
3. **Truncated.** A summarizer that raises or answers nothing falls back to a cut at a sentence or word
   boundary (`plugin.py:168`, `plugin.py:170`, `truncate_description` `catalog.py:128`). No tool is ever
   left without a line and no summarizer failure escapes the projection.

Lines are cached on the plugin instance keyed by `(name, description)` (`plugin.py:726`, written
`plugin.py:846`), so a tool is summarized once however many agents and calls read it, and a
re-registration with a changed description gets a new line. Missing lines are requested concurrently
under a semaphore of `_SUMMARY_CONCURRENCY` (`plugin.py:76`, gate `plugin.py:841`, gather
`plugin.py:848`), and only for catalog-eligible specs — the two plugin tools are excluded
(`plugin.py:834`).

```mermaid
sequenceDiagram
    box rgb(222,235,255) Band 1 · Strands interface
        participant Stage as InvokeModelStage.Input
        participant Handler as _projection_handler [plugin.py:749]
        participant Ensure as _ensure_index [plugin.py:789]
        participant Sum as _summarize [plugin.py:142]
        participant Project as _project [plugin.py:599]
    end
    box rgb(224,245,224) Band 3 · context-core
        participant Index as LexicalToolIndex [tool_index.py:177]
        participant Block as catalog_prompt_block [catalog.py:370]
    end
    participant Provider as Bedrock request

    Stage->>Handler: context · tool_specs = 93 full specs · system_prompt = operator's
    Handler->>Handler: _should_passthrough(...) returns False [plugin.py:768]
    Handler->>Handler: _expire(state, cycle_count, ttl_cycles) [plugin.py:772]
    Handler->>Ensure: (state, context.tool_specs, agent) [plugin.py:774]
    Ensure->>Ensure: the (name, description) fingerprint changed, so this call does the work [plugin.py:806]
    Ensure->>Sum: every spec with no cached line, bounded by the semaphore [plugin.py:834]
    Sum-->>Ensure: verbatim if it fits, else the summarizer, else a boundary cut [plugin.py:159]
    Ensure->>Index: build(specs_to_neutral(specs)) [plugin.py:814]
    Ensure->>Ensure: state.fingerprint = fingerprint, only now [plugin.py:818]
    Handler->>Project: (context, exposed, always_available, summaries) [plugin.py:776]
    Project->>Project: _compose_projection, three blocks of FULL specs [plugin.py:624]
    Project->>Project: the fold over the messages, through the adapter [plugin.py:626]
    Project->>Block: (incoming specs, the projected names, the cached lines) [plugin.py:633]
    Block-->>Project: the rule plus one line per tool NOT projected [catalog.py:401]
    Project-->>Handler: replace(context, tool_specs, messages, system_prompt) [plugin.py:634]
    Handler->>Handler: state.projected = the projected names [plugin.py:782]
    Handler->>Handler: _log_projection(projected.tool_specs) [plugin.py:783]
    Handler-->>Stage: the new context
    Note over Provider: toolConfig = find_tools + get_tool_details + always_available + loaded
    Note over Provider: system = operator's prompt + the catalog heading + one line per remaining tool
    Note over Provider: messages = the folded copy · no call shape for a tool absent from toolConfig
```

A projection that raises anywhere on that path degrades to the context received, unchanged: the call goes
out with the full `tool_specs`, which is today's behaviour without the plugin, and the next call tries
again (`plugin.py:786`). The log line is itself guarded — every counter update and every log call goes
through `_instrument` (`plugin.py:442`), so a logger that raises cannot fail a projection.

## 7. The common path — catalog, load, call

Two model calls and no guessing: the names live in the system prompt, the schemas arrive through
`get_tool_details`, and the call itself renews the TTL once it has returned.

`get_tool_details` tolerates the two shapes a model actually sends: a bare string instead of a list
(`plugin.py:948`) and duplicates, which are de-duplicated with order kept (`plugin.py:949`). A call that
named nothing usable answers with `_DETAILS_EMPTY_GUIDANCE` (`plugin.py:951`). A name the registry does
not have is collected and reported rather than silently dropped (`plugin.py:959`, `plugin.py:968`). The
specification itself is **not** in the result text — it travels in `tool_specs` on the next call, because
a tool result is resident in the history while a projection is per call and forgettable
(`_DETAILS_LOADED_HEADER` docstring, `plugin.py:104`).

```mermaid
sequenceDiagram
    participant Model
    box rgb(222,235,255) Band 1 · Strands interface
        participant Stage as InvokeModelStage.Input
        participant Plugin as ProgressiveToolDisclosure
        participant Reg as ToolRegistry
    end

    Note over Model: Call N — reads the catalog in the system prompt
    Stage->>Plugin: _projection_handler(context) [plugin.py:749]
    Plugin-->>Stage: tool_specs = callable tools · system_prompt += catalog · messages folded [plugin.py:634]
    Model->>Plugin: get_tool_details(["list_investment_transactions"]) [plugin.py:927]
    Plugin->>Plugin: state.loads += 1 [plugin.py:944]
    Plugin->>Plugin: normalize · bare string tolerated · duplicates dropped [plugin.py:948]
    loop each requested name
        Plugin->>Reg: registry.get(name) [plugin.py:957]
        Plugin->>Plugin: _renew(state, name, cycle) so exposed[name] = cycle [plugin.py:961]
    end
    Plugin-->>Model: "Loaded. These tools are callable with their full parameters on your next call..." [plugin.py:104]
    Note over Model: Call N+1 — the projection emits those specs FULL, from block 3
    Stage->>Plugin: _projection_handler(context) [plugin.py:749]
    Plugin-->>Stage: the loaded tool carried as a full verbatim spec [plugin.py:778]
    Note over Plugin: state.projected now holds its name [plugin.py:782]
    Model->>Plugin: list_investment_transactions(real args) · BeforeToolCallEvent
    Plugin->>Plugin: the name is in state.projected, so the guard returns [plugin.py:997]
    Reg-->>Model: tool executes normally
    Plugin->>Plugin: AfterToolCallEvent · _renew · the TTL window restarts here [plugin.py:1024]
```

`get_tool_details` takes a **list**, which is what keeps a step that needs three tools to one cycle
rather than three (docstring `plugin.py:930`).

## 8. The fallback — `find_tools` searches, it does not load

`find_tools` exists for a need the model cannot map to any listed name. It ranks specs and reports names
plus summaries, and then stops: the model still goes through `get_tool_details`, so there is one road to
a schema instead of two (comment `plugin.py:922`).

Every invocation is counted, including a blank need and a failed search, because each costs the cycle
just the same (`plugin.py:890`, comment `plugin.py:889`). A blank need is not searched at all and returns
guidance (`plugin.py:894`, `plugin.py:895`). A search that raises returns guidance too, worded the same
way as a no-match because from where the model stands the two cases are one (`plugin.py:901`,
`plugin.py:904`, `_SEARCH_FAILED_GUIDANCE` docstring `plugin.py:128`). A match the registry does not
have, and either plugin tool, are skipped (`plugin.py:910`, `plugin.py:912`).

```mermaid
sequenceDiagram
    participant Model
    box rgb(222,235,255) Band 1 · Strands interface
        participant Plugin as ProgressiveToolDisclosure
        participant Reg as ToolRegistry
    end
    box rgb(224,245,224) Band 3 · context-core
        participant Index as LexicalToolIndex
    end

    Note over Model: no catalog name fits the need
    Model->>Plugin: find_tools("list investment transactions") [plugin.py:867]
    Plugin->>Plugin: _record_search(state) · searches += 1 [plugin.py:890]
    alt need is blank
        Plugin-->>Model: _EMPTY_NEED_GUIDANCE [plugin.py:895]
    else need is usable
        Plugin->>Index: search(need, top_k) [plugin.py:899]
        alt search raised
            Plugin-->>Model: _SEARCH_FAILED_GUIDANCE [plugin.py:904]
        else matches returned
            loop each match
                Plugin->>Reg: registry.get(match.name) [plugin.py:910]
                Plugin->>Plugin: line = name + the cached summary [plugin.py:915]
            end
            Plugin->>Plugin: _log_search_outcome(need, names) [plugin.py:917]
            alt nothing usable
                Plugin-->>Model: _NO_MATCH_GUIDANCE [plugin.py:920]
            else names to report
                Plugin-->>Model: _MATCHES_HEADER plus one line per match · NOTHING exposed [plugin.py:924]
                Model->>Plugin: get_tool_details([names]) — then exactly as section 7
            end
        end
    end
```

The index is term-frequency and local, so a search needs no network (`LexicalToolIndex`
`tool_index.py:177`, `build` `tool_index.py:197`, `search` `tool_index.py:214`). Scoring sums, over the
*distinct* terms of the need, how often each occurs in the serialized specification text — name,
description and the parameter descriptions of the `inputSchema` (`_spec_text` `tool_index.py:135`) — and
ties break by indexing order, so repeated searches return the same matches in the same order
(`tool_index.py:239`). It is a structural protocol, so any object exposing a callable `build` and
`search` can replace it (`ToolIndex` `tool_index.py:53`, `_validate_index` `plugin.py:312`), and both
operations are allowed to be awaitable for a network-backed implementation (`plugin.py:815`,
`plugin.py:900`).

## 9. The safety net — premature-call cancellation

A premature call is a name the model read in the catalog and called without loading it. The guard reads
whether the schema was **visible**, not whether it is loaded: `state.projected` (`plugin.py:361`) holds
the names the last projection actually carried, written right after `_project` returns
(`plugin.py:782`). Reading `exposed` instead would mistake a tool that expired between the projection
and the call for a guessed call.

The exemptions are dealt with first, in one condition (`plugin.py:993`): a name the registry does not
have is left alone entirely, and the two plugin tools and everything in `always_available` are exempt —
they are carried in full on every call, so the question cannot arise for them. Then a name in
`state.projected` returns (`plugin.py:997`).

**The guard loads nothing on the model's behalf.** It cancels with `_PREMATURE_CALL_MESSAGE`
(`plugin.py:1004`), which names the tool and points at `get_tool_details`. A recovery that loaded the
tool and invited an immediate retry would teach the model that calling a catalog name directly works,
which is the very shortcut the catalog rule forbids (`_PREMATURE_CALL_MESSAGE` docstring
`plugin.py:138`).

The guard does not require an empty input. Any call to a tool whose schema was never projected is
cancelled when the tool requires parameters, because invented arguments against an unseen schema are the
*worse* failure — they can satisfy a permissive tool and return a confidently wrong answer that nothing
in the run marks as suspect (comment `plugin.py:1000`).

```mermaid
sequenceDiagram
    participant Model
    box rgb(222,235,255) Band 1 · Strands interface
        participant Plugin as ProgressiveToolDisclosure
        participant Reg as ToolRegistry
        participant EventLoop
    end
    box rgb(224,245,224) Band 3 · context-core
        participant Req as requires_parameters [catalog.py:203]
    end

    Note over Model: the model calls a catalog name without loading it first
    Model->>Plugin: BeforeToolCallEvent(tool_use.name = X) [plugin.py:972]
    Plugin->>Reg: X in registry · X a plugin tool · X in always_available? [plugin.py:993]
    alt any of those
        Plugin-->>EventLoop: return · no cancel · nothing written [plugin.py:994]
    else none of those
        Plugin->>Plugin: state = _state_for(states, agent) [plugin.py:996]
        alt X in state.projected
            Note right of Plugin: the schema was in the last projection, so the call is informed [plugin.py:997]
            Plugin-->>EventLoop: return · let the call through [plugin.py:998]
        else X was never projected
            Plugin->>Req: requires_parameters(registry[X].tool_spec) [plugin.py:1003]
            Req-->>Plugin: True only when the required list is non-empty [catalog.py:228]
            alt requires parameters
                Plugin->>Plugin: event.cancel_tool = _PREMATURE_CALL_MESSAGE.format(name=X) [plugin.py:1004]
                Plugin->>Plugin: _record_premature_cancellation · premature_cancellations += 1 [plugin.py:1005]
                Plugin-->>Model: "'X' did not run: it is not loaded... call get_tool_details with [X] first"
                Note over Model: X is NOT loaded by the guard · the model must call get_tool_details itself
                Model->>Plugin: get_tool_details(["X"]) · then exactly as section 7
            else no required parameter
                Plugin-->>EventLoop: return · an empty call is legitimate [plugin.py:1003]
            end
        end
    end
```

Only the `required` list decides — `requires_parameters` (`catalog.py:203`) answers `True` only when that
list is non-empty (`catalog.py:227`, `catalog.py:228`): a tool whose parameters are all optional is
callable with no arguments, so an empty call to it is not the symptom of a missing schema. The wrapped
`{"json": {...}}` shape and the unwrapped one are both tolerated (`catalog.py:224`), and any schema shape
the function cannot read is treated as requiring nothing (`catalog.py:220`), which errs towards letting
the call run.

The counter this path increments is the one worth watching. Under this design it should be near zero,
because the catalog names are not in `tool_specs` at all; a high `premature_cancellations`
(`plugin.py:365`) means the model is calling names it read in the prompt without loading them, which is a
catalog-wording problem rather than a mechanism failure.

## 10. One tool's lifecycle and release

TTL parameter: `ttl_cycles`, default `_DEFAULT_TTL_CYCLES = 3` (`plugin.py:93`; ctor `plugin.py:678`).
Age is measured against `agent.event_loop_metrics.cycle_count` only — no wall clock, no message count, no
tool-call count (`_expire` docstring, `plugin.py:404`). That is the one number this binding gets for free
and the LangGraph one has to derive by counting `AIMessage` objects. The boundary belongs to the live
side: `cycle - last_used == ttl_cycles` is KEPT, and release is `cycle - last_used > ttl_cycles`
(`plugin.py:423`).

Renewal happens on the **return** of a call, not before it: `_on_after_tool_call` (`plugin.py:1008`)
renews only a tool already in `exposed` (`plugin.py:1023`) and only when the call was not cancelled
(`plugin.py:1017`), so a cancelled call keeps nothing loaded.

```mermaid
stateDiagram-v2
    [*] --> Catalog: registered · one line in the system-prompt catalog [catalog.py:397]
    Catalog --> Loaded: get_tool_details([name]) so _renew writes the cycle [plugin.py:961]
    Catalog --> Cancelled: premature call · cancelled, and NOTHING is loaded [plugin.py:1004]
    Cancelled --> Catalog: the model must call get_tool_details itself
    Loaded --> Renewed: tool ran · _on_after_tool_call renews [plugin.py:1024]
    Renewed --> Renewed: used again within ttl_cycles, exposed[name] = cycle [plugin.py:439]
    Renewed --> Idle: idle · cycle - last_used <= ttl_cycles is kept [plugin.py:423]
    Loaded --> Idle: loaded but not yet used, still within the TTL
    Idle --> Catalog: cycle - last_used > ttl_cycles · _expire releases it [plugin.py:423]
    note right of Catalog
        ttl_cycles default = 3 (_DEFAULT_TTL_CYCLES, plugin.py:93)
        Release withdraws the schema from the next projection.
        The tool stays registered and callable (_expire docstring, plugin.py:404)
        Release also puts the name BACK into the system-prompt catalog,
        which is what makes the block's membership move (see section 12).
    end note
```

`find_tools` appears nowhere in this diagram on purpose: a search changes no exposure state, so a tool's
lifecycle is driven by `get_tool_details`, by use, and by inactivity.

## 11. Ordering guarantee — the context graph can never eat the catalog

The catalog is written into `system_prompt` from a handler on the *same* stage another plugin in this
stack rewrites: the context-graph plugin's delivery also registers on `InvokeModelStage.Input`. Since the
catalog is no longer optional, this guarantee is load-bearing on every projected call rather than on one
mode. It is **structural**, and it rests on three facts, each verified by reading source:

1. **The graph forces itself to index 0.** In
   `strands-community-plugins/strands-context-graph/src/strands_context_graph/plugin.py`, the delivery's
   `register` adds its handler (line 165) and then immediately moves it to the front (lines 167-168):

   ```python
   # strands_context_graph/plugin.py, lines 167-168
   handlers = registry._handlers[InvokeModelStage]
   handlers.insert(0, handlers.pop())
   ```

   The move is the only part of that registration allowed to fail: a rename upstream leaves the delivery
   registered at the back of the stage rather than crashing the wiring (same file, lines 169-175).

2. **The SDK builds the chain BACK TO FRONT, so index 0 is the OUTERMOST layer and runs FIRST.** In the
   SDK, `strands/_middleware/registry.py` in 1.56 (`strands/middleware/_registry.py` from 1.62, same loop), `compose` iterates the sorted handler list in
   reverse (line 128 seeds `current` with the terminal, line 129 walks downward), wrapping each handler
   *around* what was built so far:

   ```python
   # strands/_middleware/registry.py, lines 128-129
   current: MiddlewareNext = terminal
   for i in range(len(sorted_handlers) - 1, -1, -1):
   ```

   The last handler wrapped — index 0 — is the outermost layer, so it is entered first.

3. **The disclosure plugin sets no order, so it is appended and runs LAST.** `init_agent`
   (`plugin.py:730`) calls `add_middleware` plainly (`plugin.py:747`) with no reordering, so it sits
   after the graph's forced-front handler and is entered after it.

Net effect: by the time `_projection_handler` (`plugin.py:749`) appends the catalog block, the graph has
**already** projected the messages and returned. And even in the reverse arrangement the catalog would
survive, because the graph's delivery only ever rewrites `messages` and the trailing-block count: it
returns `replace(context, messages=..., dynamic_trailing_blocks=context.dynamic_trailing_blocks + ...)`
(`strands_context_graph/plugin.py`, lines 198-201). Neither field is `system_prompt`, so the field the
catalog is appended to is not a field the graph touches.

Both plugins now write `messages`, and the order settles that too rather than leaving it to chance. Under
the D-1 wiring the graph runs `context_core.graph.project` on **every model call**, not once per user
turn (`strands_context_graph/plugin.py`, lines 114-118), so its compaction runs first on every call and
hands its rewritten list down; this plugin's fold (§5) runs over that result, and the two compose in one
direction only. The fold's own guard (`pairs_intact` `catalog.py:518`) is checked against whatever
arrived, so a list the graph already reshaped is verified on the same terms as an untouched one.

This is worth stating explicitly because the intuition runs the wrong way twice over: "index 0" reads as
*innermost* until you read `compose`, and "the graph compacts the context" reads as *the graph might drop
my text* until you read which field it replaces.

## 12. Cache consequence of a system-prompt catalog

Two properties decide this, and they pull in opposite directions.

**The lines themselves do not drift.** A summary is computed once per `(name, description)` and cached
for the life of the plugin instance (`plugin.py:726`), so the same tool yields byte-identical text on
every call. Nothing in the block is re-derived per call, which is exactly what a per-call truncation or a
per-call model summary would have cost (module docstring `plugin.py:18`).

**The block's membership does move.** A tool that gets loaded leaves the listing, because the block is
rendered over the names that are *not* carrying a full specification (`catalog.py:394`, computed from the
projection at `plugin.py:633`), and it returns to the listing when its load is released (`_expire`
`plugin.py:404`).

On Bedrock the cacheable prefix is ordered **tools → system → messages**, and a change in an earlier
section invalidates the later ones (the AWS statement is quoted in
[`how-to/02-community-plugins-agent-sample.md`](../../../how-to/02-community-plugins-agent-sample.md)).
That ordering is what makes the placement cache-neutral rather than costly: the catalog's membership
changes **only** on an event that also changes `tool_specs` — a load adds a full spec, a release removes
one — and `tool_specs` sits ahead of `system` in the prefix, so the invalidation was already paid. The
three fields are written in one `replace` (`plugin.py:634`) precisely because they describe one decision.

What follows from that:

- The number of cache invalidations per session is driven by how often the exposure set changes, not by
  where the catalog is placed. Loading several tools in one `get_tool_details` call is therefore cheaper
  than loading them one at a time, which is the cache argument for the list-shaped signature
  (`plugin.py:927`).
- Release is the other side of that trade: a tool leaving `tool_specs` invalidates the prefix exactly as
  its arrival did, which is what `ttl_cycles` buys down by keeping a tool in use resident (`_expire`
  docstring, `plugin.py:404`).
- The `_compose_projection` (`plugin.py:556`) ordering discipline that exists to protect the cache
  still holds for `tool_specs`: two calls with the same state produce the same list in
  the same order.
- The fold rewrites `messages`, the last section of the prefix, so it never invalidates `tools` or
  `system`; what it changes is the size of the part that was going to be re-read anyway.
- The summary calls are an auxiliary cost that has to be read next to the saving, which is why they are
  counted rather than hidden (`state.summary_usage` `plugin.py:366`, accumulated at `plugin.py:194`).

## 13. Harness wiring

Every configuration runs with `NullConversationManager` (`runner.py:437`). For the graph that is a
documented precondition — any other manager edits the live message list before the call is assembled; for
this comparison it also keeps history handling from being a confounding variable between configurations
(`runner.py:26`).

The validation harness threads three knobs from environment variables into the constructor, with no
intermediate default of its own:

| Layer | `file.py:LINE` | What it does |
|-------|----------------|--------------|
| Threshold fields | `catalog_chars` (`config.py:398`), `ttl_cycles` (`config.py:401`), `top_k` (`config.py:404`) | `80`, `3`, `4` — the catalog line budget in characters, the load TTL in cycles, and the matches one search lists |
| Environment read | `config.py:420`, `config.py:421`, `config.py:422` | `_env_int("VALIDATION_CATALOG_CHARS", 80)` and its two siblings, inside the `THRESHOLDS` construction |
| Plugin construction | `runner.py:366` | `ProgressiveToolDisclosure(catalog_chars=..., ttl_cycles=..., top_k=...)` (`runner.py:367`, `runner.py:368`, `runner.py:369`) |
| `always_available` | `runner.py:379` | The graph's retrieval tools (`runner.py:380`) plus the one literal domain tool `list_accounts` (`runner.py:381`), derived from the plugins rather than hard-coded (comment `runner.py:376`). The relevance filter's `retrieve_all_context` is deliberately **not** here: it answers the rare question that needs a whole result, so it stays a catalog name and is loaded only then (comment `runner.py:373`) |

There is no environment switch for the placement, because there is only one placement, and **no
`referenced_source` bridge**: the plugin releases a loaded tool after `ttl_cycles` idle cycles and keeps
nothing because the history mentions it, so a Card that steps down has no schema to keep resident — its
tools are catalog names like any other (`runner.py:20`).

The harness reads five counters plus the summary usage off the per-agent state (`runner.py:490`):
`searches` (`runner.py:492`), `loads` (`runner.py:493`), `premature_cancellations` (`runner.py:495`),
`exposed_at_end` (`runner.py:496`), `exposed_count_at_end` (`runner.py:497`) and `summary_usage`
(`runner.py:499`), which `compare.py` bills at the agent's own rates (comment `runner.py:498`). All six
come off `_DisclosureState` (`plugin.py:333`), which is why this binding reports them directly where the
LangGraph one has to derive `searches` and `loads` from the transcript.

The graph's retrieval tools are put in `always_available` precisely so that a cycle spent loading them is
not charged to the strategy (comment `runner.py:370`). That is a measurement decision, not a workaround
for a defect in the projection.

## 14. Verbatim text the model sees

**Write targets.** `tool_specs`, `messages` and `system_prompt`, all three in one `replace`
(`plugin.py:634`), or `tool_specs` and `messages` alone when `catalog_chars is None` (`plugin.py:631`).
The two tool results are returned as **tool-result messages**, resident in message history, which is why
the schema itself is deliberately kept out of them (`_DETAILS_LOADED_HEADER` docstring, `plugin.py:104`) —
and why the fold drops those two exchanges once they are spent (`catalog.py:598`).

### 14.1 The system-prompt catalog header — `CATALOG_PROMPT_HEADER` (`catalog.py:84`)

Quoted literally, `{get_tool_details}` and `{find_tools}` being substituted with the two tool names at
`catalog.py:401`:

```text
# Tools available on request

The tools listed below are NOT in your tool list, and you MUST NOT call them directly: their parameters
are not loaded, and a direct call is rejected without running.

To use any of them, always follow these steps:
1. Call `get_tool_details` with the names you need, as a list, in one call.
2. On your next call they are in your tool list with their full parameters. Call them from there.
3. A tool left unused for a few calls is unloaded again. If a call to it is rejected, repeat step 1.

If no name below fits what you need, call `find_tools` with the need in your own words, then go to
step 1 with the names it returns.

The tools that ARE in your tool list for this call you call directly.

```

Immediately followed by one line per non-callable tool, built at `catalog.py:397`:

```python
lines.append(f"- {name}: {summary}" if summary else f"- {name}")
```

So the model sees, for example:

```text
- get_transactions: Transactions of an account over a date range, newest first.
- open_position: Opens a position on an instrument for an account and returns its id.
```

No sigil and no marker: the names are not in `tool_specs`, so nothing in the call claims they are
callable and nothing has to be walked back. Step 3 is what makes release intelligible from the model's
side — a rejected call is a documented state with a documented recovery, not a surprise.

### 14.2 `get_tool_details` docstring — exactly as the model receives it (`plugin.py:928`)

```text
Load the full parameters of one or more tools from the catalog, so you can call them.

Pass every tool you are about to need in one call. They arrive complete in your tool list on
your next call. A tool left unused for a few calls is unloaded; to call it after that, call this
again with its name.

Args:
    names: Exact tool names, as written in the catalog or in a `find_tools` result.
    tool_context: Injected by the framework. Not user-facing.

Returns:
    The tools that were loaded, and any requested name that is not a tool.
```

Its `inputSchema` is not a source literal: it is derived by the `@tool(context=True)` decorator
(`plugin.py:926`) from the signature `get_tool_details(self, names: list[str], tool_context:
ToolContext)` (`plugin.py:927`). The only model-facing parameter is `names`, a list of strings.

### 14.3 `find_tools` docstring — exactly as the model receives it (`plugin.py:868`)

```text
Search for tools that can do what you need, when no name in the tool catalog fits.

This only finds tools; it does not load them. It answers with matching tool names and one line
about each. To use any of them, call `get_tool_details` with their names, then call them.

If a name in the catalog already fits what you need, skip this and call `get_tool_details`
directly.

Args:
    need: What you are trying to do, described in your own words. A capability, not a tool
        name — "list the transactions of an investment account" works better than a guess at
        what the tool might be called.
    tool_context: Injected by the framework. Not user-facing.

Returns:
    The matching tool names with a one-line summary of each, or guidance to describe the need
    or to reword it when there is nothing to list.
```

Same derivation for its schema, from `find_tools(self, need: str, tool_context: ToolContext)`
(`plugin.py:867`): the only model-facing parameter is `need`, a string.

### 14.4 `get_tool_details` results

Header literal `_DETAILS_LOADED_HEADER` (`plugin.py:104`), which states the release rule in the same
breath as the load:

```python
_DETAILS_LOADED_HEADER = (
    "Loaded. These tools are callable with their full parameters on your next call. A tool left unused for "
    f"a few calls is unloaded; to call it after that, call `{GET_TOOL_DETAILS_NAME}` again:"
)
```

Assembled with one line per loaded tool at `plugin.py:966`, each line built at `plugin.py:962`, and
followed when needed by `_DETAILS_UNKNOWN` (`plugin.py:111`, formatted at `plugin.py:968`):

```python
_DETAILS_UNKNOWN = "Not a tool, ignored: {names}. Use names from the catalog or from `" + FIND_TOOLS_NAME + "`."
```

A call that named nothing usable gets `_DETAILS_EMPTY_GUIDANCE` (`plugin.py:114`, returned at
`plugin.py:951`):

```python
_DETAILS_EMPTY_GUIDANCE = (
    "Pass the names of the tools you want loaded, as a list. Take them from the catalog, or call `"
    + FIND_TOOLS_NAME
    + "` first."
)
```

### 14.5 `find_tools` results

Header literal `_MATCHES_HEADER` (`plugin.py:99`), which states in the same breath that nothing was
loaded:

```python
_MATCHES_HEADER = (
    f"Tools that match. Nothing is loaded yet: call `{GET_TOOL_DETAILS_NAME}` with the names you want, then call them."
)
```

The full result is assembled at `plugin.py:924` as `"\n".join([_MATCHES_HEADER, *lines])`, each line
built at `plugin.py:915`:

```python
lines.append(f"- {match.name}: {self._short_description(registered.tool_spec)}")
```

`_short_description` (`plugin.py:1026`) returns the cached catalog line, or a boundary truncation at the
default limit when the catalog is suppressed (`plugin.py:1041`, `plugin.py:1042`) — `catalog_chars=None`
drops the catalog from the prompt, it does not mean a search result should carry a full description. The
two tool names are the core's constants, so a result and a catalog line always name the same tools
(`FIND_TOOLS_NAME` `catalog.py:58`, `GET_TOOL_DETAILS_NAME` `catalog.py:61`).

### 14.6 `_PREMATURE_CALL_MESSAGE` (`plugin.py:132`)

```python
_PREMATURE_CALL_MESSAGE = (
    "'{name}' did not run: it is not loaded, so its parameters are unknown to you. Call `"
    + GET_TOOL_DETAILS_NAME
    + "` with [\"{name}\"] first, then call '{name}' with its real parameters."
)
```

Formatted with the tool name at `plugin.py:1004` before being assigned to `event.cancel_tool`. It names
the recovery instead of performing it: the tool is **not** loaded by the guard.

### 14.7 The fold sentence — `fold_note` (`catalog.py:452`)

Not a constant: the sentence is assembled in the core from the tool name and the rendered result, with
the verb chosen by the result's status (`catalog.py:469`, `catalog.py:470`):

```python
outcome = "failed with" if result.get("status") == "error" else "the result was"
return f"The tool {name} was called and {outcome}: {' '.join(parts)}"
```

So a closed exchange of a tool that is no longer callable reads, in the messages of the next call:

```text
The tool get_transactions was called and the result was: {"items": [...], "next": null}
```

It is model-facing text like any other, and it is the reason the evidence survives a release while the
call shape does not. Because it is the core's, every binding of this practice folds to the same sentence.

### 14.8 The remaining guidance strings

- `_EMPTY_NEED_GUIDANCE` (`plugin.py:121`), returned on a blank `need` (`plugin.py:895`):

  ```python
  _EMPTY_NEED_GUIDANCE = "Describe what you are trying to do, in your own words, then call this tool again."
  ```

- `_NO_MATCH_GUIDANCE` (`plugin.py:124`) — returned when nothing usable was found (`plugin.py:920`):

  ```python
  _NO_MATCH_GUIDANCE = "No tool matches that description. Try different wording, or answer directly."
  ```

- `_SEARCH_FAILED_GUIDANCE` (`plugin.py:127`) — returned when the search itself raised
  (`plugin.py:904`):

  ```python
  _SEARCH_FAILED_GUIDANCE = "Tool search is unavailable right now. Try a different description, or answer directly."
  ```

With the `CATALOG_PROMPT_HEADER` block and its lines, the two tool docstrings, `_MATCHES_HEADER`,
`_DETAILS_LOADED_HEADER`, `_DETAILS_UNKNOWN`, `_DETAILS_EMPTY_GUIDANCE`, `_PREMATURE_CALL_MESSAGE` and
the `fold_note` sentence, that is the complete set of model-facing text the plugin can produce. The
core's `_ELLIPSIS = "..."` literal (`catalog.py:78`) can appear inside a line that fell back to
truncation when the cut lands mid-sentence (`truncate_description` `catalog.py:128`).

One text is model-facing in a different sense: `_SUMMARY_SYSTEM_PROMPT` (`plugin.py:79`) is the
instruction the default summarizer sends, with `{max_chars}` substituted at `plugin.py:195`. It asks for
what tells a tool apart from its neighbours — the object it acts on and what it returns — because that is
what the model reading the catalog needs in order to decide whether to load it.

## 15. Configuration — every constructor parameter

Constructor: `ProgressiveToolDisclosure.__init__` (`plugin.py:673`). All keyword-only (`*` at
`plugin.py:675`). Every check runs before any state is set, so a construction that fails leaves no
handler, hook or tool registered (`plugin.py:708` through `plugin.py:713`).

| Parameter | Default (`file.py:LINE`) | Accepts `None`? | Meaning |
|-----------|--------------------------|-----------------|---------|
| `catalog_chars` | `DEFAULT_CATALOG_CHARS = 80` (`catalog.py:71`; ctor `plugin.py:676`) | **Yes** | Character limit of one catalog line's summary. `None` drops the catalog entirely (`plugin.py:631`), leaving the two plugin tools' descriptions as the only hint that other tools exist. Validated by `_validate_catalog_chars` (`plugin.py:257`) — `None` or int ≥ 1, `0` rejected because a zero-character line fits nothing. |
| `summarizer` | `None` (`plugin.py:677`) | **Yes** | What writes a line when a description does not fit. Receives `(spec, max_chars)`, sync or async (`ToolSummarizer` `plugin.py:87`). `None` uses the agent's own model, one plain call per tool, cached (`_model_summarizer` `plugin.py:173`, selected at `plugin.py:840`). `_validate_summarizer` (`plugin.py:275`) — `None` or callable. |
| `ttl_cycles` | `_DEFAULT_TTL_CYCLES = 3` (`plugin.py:93`; ctor `plugin.py:678`) | No | Cycles a loaded tool survives without a call; every call that runs renews it (`plugin.py:1024`). `_validate_positive_int` (`plugin.py:235`) — int ≥ 1, `bool` and `float` rejected. |
| `always_available` | `()` empty tuple (`plugin.py:679`) | No | Names carrying their full spec on every call, skipping the load cycle and exempt from the guard (`plugin.py:993`). `_validate_always_available` (`plugin.py:289`) — a sequence of non-empty strings; a bare string is rejected, since it would configure one name per character. Stored as a tuple (`plugin.py:719`). |
| `index` | `None` → `LexicalToolIndex()` (`plugin.py:680`; instantiated `plugin.py:723`) | **Yes** | Search implementation behind `find_tools`. `None` selects the core's term-frequency `LexicalToolIndex` (`tool_index.py:177`), which needs no network. `_validate_index` (`plugin.py:312`) — `None`, or an object with callable `build` and `search`. |
| `top_k` | `_DEFAULT_TOP_K = 3` (`plugin.py:96`; ctor `plugin.py:681`) | No | How many tools one search lists. `_validate_positive_int` (`plugin.py:235`) — int ≥ 1. |

Six parameters, none positional. There is no placement flag, no `referenced_source` and no
`find_tools_name` knob — the two tool names are the core's module constants
`FIND_TOOLS_NAME = "find_tools"` (`catalog.py:58`) and
`GET_TOOL_DETAILS_NAME = "get_tool_details"` (`catalog.py:61`), collected into `PLUGIN_TOOL_NAMES`
(`catalog.py:67`) and threaded through the projection, the fold, the passthrough test and the guard
rather than exposed on the constructor. The second name is deliberately not `get_details`: a verb-noun
that generic is one a domain tool can already hold, and a collision would silently shadow one of the two
in the registry (`catalog.py:64`).

The instance name is `name = "strands:progressive-tool-disclosure"` (`plugin.py:671`), and the package
re-exports the core's `LexicalToolIndex`, `ToolIndex` and `ToolMatch` alongside the plugin
(`__init__.py:9`), so a caller swapping the index never imports `context_core` itself.
