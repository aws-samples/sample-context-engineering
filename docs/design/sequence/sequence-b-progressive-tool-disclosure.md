# Sequence B — Progressive Tool Disclosure

All `file.py:LINE` references are into
`community-plugins/strands-progressive-tool-disclosure/src/strands_progressive_tool_disclosure/`
(`plugin.py`, `index.py`, `_compat.py`) unless the prose names another package. Harness references are
into `validation/community-plugin-A-B-D/src/` (`runner.py`, `config.py`); the ordering section
additionally cites the context-graph package (`projection.py`) and the installed Strands SDK, whose
paths are given in prose because they sit outside the reference checker's roots. Every claim below
carries a line reference; literal strings are quoted verbatim from source.

## 1. What the plugin does, mechanically

On every model call the plugin rewrites **three** fields of the invocation context.

`tool_specs` is reduced to the tools that are callable on this call, each one a **full verbatim spec**:
the two plugin tools `find_tools` and `get_tool_details`, the `always_available` names, and the tools
whose schema was **loaded** by a prior `get_tool_details` and is still live under a TTL measured in
event-loop cycles. Every other registered tool reaches the model as **one line of a catalog appended to
the system prompt** — its name and a summary of its description, at most `catalog_chars` characters
(`_compose_projection` `plugin.py:934`, `_project` `plugin.py:977`, `_catalog_prompt_block`
`plugin.py:187`, `_projection_handler` `plugin.py:1124`).

`system_prompt` receives that catalog block, appended (`_append_to_system_prompt` `plugin.py:308`).

`messages` receives a folded copy: every **closed** exchange of a tool this call does not carry loses
its `toolUse` block, and its `toolResult` becomes one plain sentence — `The tool X was called and the
result was: Y` (`_fold_tool_exchanges` `plugin.py:807`, `_fold_note` `plugin.py:723`, called at
`plugin.py:1003`). The plugin's own two exchanges are dropped outright, with no sentence. The agent's
own history is never mutated — the fold produces a new list for this call only.

**A loaded tool is released, not retained.** There is no history-retention block: the projection does
**not** keep a tool because the messages still mention it. A `toolUse` left in the history for a tool
absent from `tool_specs` is accepted by the Bedrock Converse API — checked on Claude Haiku 4.5,
Claude Opus 4.8, GLM 4.7 Flash and Qwen3 Next (module docstring `plugin.py:14`, `plugin.py:15`) — and
retaining those tools was what made `tool_specs` grow monotonically with every tool a conversation had
touched (`_compose_projection` docstring `plugin.py:934`).

There is **one placement** and no mode flag. Nothing reduced is ever put in `tool_specs`, so nothing
the provider is shown asserts an empty `inputSchema`: a catalog name is not in `tool_specs` at all, and
the rule that governs it arrives in the same block as the name (`_CATALOG_PROMPT_HEADER`
`plugin.py:162`). The projection and the catalog **partition** the registry between them — the block is
built from the projection's own names, so no tool appears in both and none is missing from both
(`plugin.py:1008`, `_catalog_prompt_block` docstring `plugin.py:187`).

**The common path is catalog → `get_tool_details([names])` → call.** The model reads a name in the
system prompt, calls `get_tool_details` with the names it wants, which loads them into `state.exposed`
(`_renew` `plugin.py:1336`) and answers with a short confirmation; the full schemas arrive on the *next*
model call. `find_tools` is the **fallback** for a need no catalog name fits: it ranks the specs through
the `LexicalToolIndex` (`index.py:168`) term-frequency `search` (`index.py:205`) and lists names plus
summaries, and it **exposes nothing** — loading is `get_tool_details`' one job, so the model takes the
same road to a schema wherever it started (`find_tools` `plugin.py:1242`, comment `plugin.py:1297`).

Two hooks maintain the TTL and the guard. `AfterToolCallEvent` (`_on_after_tool_call`
`plugin.py:1383`) renews a loaded tool that actually **ran**, so a stretch of work on one tool never
reloads it. `BeforeToolCallEvent` (`_on_before_tool_call` `plugin.py:1347`) cancels a call to a tool the
last projection did not carry when that tool requires parameters — and loads **nothing** on the model's
behalf.

Every registered tool stays in the `ToolRegistry` and stays callable throughout. Release withdraws a
schema from the next projection, it does not unregister anything (`_expire` docstring `plugin.py:607`).

## 2. Integration table — every SDK attachment point

| # | SDK seam | `file.py:LINE` | Callback / order | Reads | Mutates |
|---|----------|----------------|------------------|-------|---------|
| 1 | `Plugin` base class (subclassed) | `plugin.py:1017` `class ProgressiveToolDisclosure(Plugin)` | `Plugin` auto-registers `@hook` and `@tool` members but NOT middleware (per docstring `plugin.py:1108`) | — | — |
| 2 | `init_agent(agent)` lifecycle callback | `plugin.py:1105` | Called by the SDK when an agent is initialized | `agent._middleware_registry` | Adds one middleware to `InvokeModelStage.Input` (`plugin.py:1122`) |
| 3 | `InvokeModelStage.Input` middleware — the projection | `plugin.py:1122` (registration), `_projection_handler` `plugin.py:1124` (handler) | Middleware stage `InvokeModelStage.Input`; the seam is reached only through `_compat.py`, which imports `InvokeModelContext` and `InvokeModelStage` from `strands._middleware.stages` (`_compat.py:9`). **No explicit order is set** — the handler is appended in wiring order, so it runs after any middleware that forced itself to the front (see §10). | `context.tool_specs`, `context.messages`, `context.system_prompt`, `context.agent`, `agent.tool_registry.registry`, `agent.event_loop_metrics.cycle_count`, `agent.model` (default summarizer) | Returns a NEW context via `replace(context, tool_specs=..., messages=..., system_prompt=...)` (`plugin.py:1009`), or `tool_specs` and `messages` alone when the catalog is suppressed (`plugin.py:1006`); writes `state.projected` (`plugin.py:1157`) and mutates per-agent `_DisclosureState` (`_expire` `plugin.py:607`) plus the shared summary cache and index fingerprint (`_ensure_index` `plugin.py:1164`). Does NOT touch the registry or `agent.messages`. |
| 4 | `@tool(context=True)` vended tool `get_tool_details` | `plugin.py:1301` decorator, `plugin.py:1302` method | Auto-registered by `Plugin`, by the `_PluginRegistry` AFTER `init_agent` returns (`plugin.py:1115` docstring), so early calls can arrive without it — covered by `_should_passthrough` (`plugin.py:898`). | `tool_context.agent`, `agent.event_loop_metrics.cycle_count`, `agent.tool_registry.registry`, the `names` argument | Increments `state.loads` (`plugin.py:1319`); writes `state.exposed` via `_renew` for each known name (`plugin.py:1336`). Returns a text result string. |
| 5 | `@tool(context=True)` vended tool `find_tools` | `plugin.py:1241` decorator, `plugin.py:1242` method | Auto-registered by `Plugin`, same post-`init_agent` window as row 4. | `tool_context.agent`, `agent.tool_registry.registry`, the `need` argument, the index | Increments `state.searches` (`_record_search` `plugin.py:684`, called `plugin.py:1265`). **Writes no exposure** — it only reads the index and the registry. |
| 6 | `BeforeToolCallEvent` hook | `plugin.py:1346` `@hook`, `_on_before_tool_call` `plugin.py:1347` | Sync hook (the `# type: ignore` note at `plugin.py:1346` says the `@hook` overloads only infer async). Auto-registered by `Plugin`. Fires before each tool call. No numeric order documented. | `event.tool_use["name"]`, `event.agent`, `agent.tool_registry.registry`, `state.projected`, `self._always_available`, `_requires_parameters(spec)` (`plugin.py:335`) | On a premature call sets `event.cancel_tool` (`plugin.py:1379`) and increments `state.premature_cancellations` (`_record_premature_cancellation` `plugin.py:708`, called `plugin.py:1380`). **Writes no exposure at all** — it never calls `_renew`. |
| 7 | `AfterToolCallEvent` hook | `plugin.py:1382` `@hook`, `_on_after_tool_call` `plugin.py:1383` | Sync hook, same `# type: ignore` note. Auto-registered by `Plugin`. Fires after each tool call. | `event.cancel_message` (`plugin.py:1392`), `event.tool_use["name"]`, `event.agent`, `state.exposed` | Renews an already-loaded tool via `_renew` (`plugin.py:1399`). A cancelled call returns first (`plugin.py:1393`), so it keeps nothing loaded. |

Supporting SDK type imports (not attachment points, but the surface used): `AfterToolCallEvent` and
`BeforeToolCallEvent` (`plugin.py:39`), `Plugin`, `hook` (`plugin.py:40`), `tool` (`plugin.py:41`),
`Message`, `Messages`, `SystemPrompt` (`plugin.py:42`), `ToolContext`, `ToolSpec` (`plugin.py:43`);
`InvokeModelStage` via `_compat` (`plugin.py:45`), `InvokeModelContext` under `TYPE_CHECKING`
(`plugin.py:52`). `SystemPrompt` is the type of the field the catalog is appended to, `Messages` the
type the fold rewrites.

Per-agent state is held in a `WeakKeyDictionary` keyed by agent (`_DisclosureStates` `plugin.py:572`,
`_new_disclosure_states` `plugin.py:577`), so one plugin instance serves many agents without keeping any
alive; state never reaches `agent.state` or session storage (`_DisclosureState` docstring
`plugin.py:536`). Summaries are the one thing shared across agents, because a summary depends on the
description alone (`self._summaries` `plugin.py:1101`, `init_agent` docstring `plugin.py:1105`).

## 3. The projection model — three full-spec blocks, then the catalog

`_compose_projection` (`plugin.py:934`) visits three blocks in a **fixed order** and every name it emits
carries its full verbatim spec. A name emitted by an earlier block is added to `seen` and never
re-emitted (`plugin.py:967`, `plugin.py:970`). `_project` at `plugin.py:977` is the entry point that
calls it, folds the messages, and then places whatever is left into the system prompt.

| Order | Class | Membership source (`file.py:LINE`) | What the model receives |
|-------|-------|-------------------------------------|--------------------------|
| 1 | `{find_tools, get_tool_details}` | frozenset `_PLUGIN_TOOL_NAMES` (`plugin.py:65`), first block of the tuple `plugin.py:967` (names from `FIND_TOOLS_NAME` `plugin.py:56` and `GET_TOOL_DETAILS_NAME` `plugin.py:59`) | **Full verbatim spec** — emitted first and unconditionally, which is what makes the projection non-empty on every projected path |
| 2 | `always_available` | `self._always_available` tuple, passed at `plugin.py:1154`; set in ctor `plugin.py:1094` | **Full verbatim spec** on every call |
| 3 | `exposed` | `state.exposed` map (post-`_expire` `plugin.py:607`), passed at `plugin.py:1153`; written only by `_renew` (`plugin.py:630`), called from `get_tool_details` (`plugin.py:1302`) at `plugin.py:1336` and from the post-call hook (`plugin.py:1399`) | **Full verbatim spec** while the load is live under the TTL |
| 4 | catalog residue | every incoming name **not** in the projection, computed from the projection itself (`plugin.py:1008`); skipped entirely when `catalog_chars is None` (`plugin.py:1005`) | **Nothing in `tool_specs`.** One line `- name: summary` in the system prompt (`plugin.py:214`), under the header at `plugin.py:218` |

There is deliberately **no history-referenced block**. A tool the conversation already used is a catalog
name again once its load is released, which is what keeps `tool_specs` at the mandatory set when the work
moves on (`_compose_projection` docstring `plugin.py:934`).

Iteration inside every block follows `incoming` order, not the container's, for prompt-cache stability
(`_compose_projection` docstring, `plugin.py:934`; loop `plugin.py:968`).

`_catalog_prompt_block` returns `""` when every incoming tool is already carrying a full specification
(`plugin.py:215`), and `_append_to_system_prompt` returns the prompt unchanged **by identity** on an
empty block (`plugin.py:327`) — an empty catalog adds nothing rather than a header promising a list.
Appending rather than prepending is deliberate: the operator's own text keeps the opening position and,
on a provider that caches by prefix, stays at a stable offset (`_append_to_system_prompt` docstring,
`plugin.py:308`). The three shapes of `SystemPrompt` are each preserved: `None` becomes the block
(`plugin.py:329`), a `str` is joined with a blank line (`plugin.py:331`), and a list gains one text
block rather than being flattened (`plugin.py:332`).

**Passthrough is structural, not a flag.** `_should_passthrough` (`plugin.py:898`) returns `True` when
any incoming name is absent from the registry — forced structured output swaps in a synthetic spec —
and when **either** plugin tool is missing from the call: without `get_tool_details` there is no way to
load a hidden schema and without `find_tools` no way to find one, so there is nothing to hide
(`plugin.py:926`, `plugin.py:931`).

## 4. The message fold — every exchange the call cannot repeat

A `toolUse` block carries the tool's name **and its arguments**. Sitting next to a successful result it
reads as a template, and the model repeats it — which, once that tool has left `tool_specs`, is a call
to a tool it cannot see. So the fold removes the call shape and keeps the evidence
(`_fold_tool_exchanges` docstring `plugin.py:807`).

What is folded, and what is not:

- **Only closed turns.** `_current_turn_start` (`plugin.py:743`) returns the index of the last user
  message carrying no `toolResult` (`plugin.py:757`, `plugin.py:759`); everything from there on is the
  turn in flight, tool loop and latest assistant message included, and it passes through as **the very
  same objects** — the single exception being the user message that opened the turn, which takes the
  folded span's trailing content when there is any (below). That is what keeps a reasoning provider's
  latest assistant message byte-intact. `messages[0]` is likewise untouched (`plugin.py:851`), and a
  boundary of `0` or `1` means nothing is folded at all (`plugin.py:833`, `plugin.py:834`).
- **Only closed pairs of non-callable tools.** A `toolUse` is enrolled when its tool is a plugin tool or
  is not in this call's projection, and only if a `toolResult` answering it exists in the span
  (`plugin.py:847`, `plugin.py:848`).
- **The result becomes a sentence.** `_fold_note` (`plugin.py:723`) renders `The tool X was called and
  the result was: Y`, or `... failed with: Y` when the result's status is an error (`plugin.py:739`);
  image and document parts of the result are kept as they are (`plugin.py:864`).
- **The plugin's own exchanges go entirely**, with no sentence: a `find_tools` or `get_tool_details`
  exchange matters on the call right after it and is dead weight past that (`plugin.py:862`).

The rewrite then has to leave a shape Converse accepts. Emptied messages are dropped
(`plugin.py:872`), same-role neighbours are merged (`plugin.py:877`), and `_tidy` (`plugin.py:782`)
fixes each message for its role: an assistant message that was rewritten loses its `reasoningContent`,
because a modified message can no longer carry a valid signature (`_without_reasoning`
`plugin.py:764`, applied `plugin.py:871`), and a user message puts its `toolResult` blocks first
(`_results_first` `plugin.py:772`) — Converse rejects text placed ahead of the `toolResult` that answers
the previous assistant message. When the folded span now ends with a user message, its content is joined
into the **opening** user message of the turn in flight rather than into the turn's assistant messages
(`plugin.py:886`, `plugin.py:888`).

The safety net is the fold's own invariant, checked on its output. `_pairs_intact` (`plugin.py:788`)
verifies that every `toolUse` is still answered in the very next message; when the fold broke an
adjacency the original messages are sent, with one warning (`plugin.py:892`, `plugin.py:893`). A case
the fold did not foresee therefore costs the saving on that call, never the call itself. The last
message is excluded from the check, since an assistant `toolUse` still waiting for its result is a
legitimate tail.

```mermaid
sequenceDiagram
    participant Project as _project [plugin.py:977]
    participant Fold as _fold_tool_exchanges [plugin.py:807]
    participant Turn as _current_turn_start [plugin.py:743]
    participant Note as _fold_note [plugin.py:723]
    participant Check as _pairs_intact [plugin.py:788]

    Project->>Fold: (context.messages, names in the projection) [plugin.py:1003]
    Fold->>Turn: index of the last user message with no toolResult [plugin.py:757]
    Turn-->>Fold: boundary · messages[boundary:] is the turn in flight
    alt boundary <= 1
        Fold-->>Project: messages unchanged, same object [plugin.py:834]
    else closed span exists
        Fold->>Fold: enrol closed pairs of tools NOT in the projection [plugin.py:847]
        alt nothing enrolled
            Fold-->>Project: messages unchanged [plugin.py:848]
        else pairs to fold
            loop each message of messages[1:boundary]
                Fold->>Fold: drop the toolUse block [plugin.py:858]
                alt plugin tool
                    Fold->>Fold: drop the toolResult with no sentence [plugin.py:862]
                else domain tool
                    Fold->>Note: (name, toolResult)
                    Note-->>Fold: "The tool X was called and the result was: Y" [plugin.py:739]
                    Fold->>Fold: keep image and document parts [plugin.py:864]
                end
            end
            Fold->>Fold: drop emptied messages · merge same-role neighbours [plugin.py:872,877]
            Fold->>Fold: _tidy · assistant loses reasoningContent · user puts results first [plugin.py:782]
            Fold->>Fold: a trailing user message joins the turn's opening message [plugin.py:886]
            Fold->>Check: _pairs_intact(folded[:-1])
            alt an adjacency broke
                Check-->>Fold: False · warn and send the original [plugin.py:892,893]
                Fold-->>Project: messages unchanged
            else intact
                Fold-->>Project: the folded list [plugin.py:895]
            end
        end
    end
```

## 5. The first projection — index build and the catalog lines

Neither the index nor a summary can be produced at construction time: both need the specifications of a
call, and the first projection is what has them (`_ensure_index` docstring `plugin.py:1164`). The
incoming `(name, description)` **pairs** are kept as a fingerprint and compared on every projection, so
an MCP tool or a `register_dynamic_tool` arriving at runtime — and equally a tool **re-registered with a
new description** — triggers exactly one rebuild (`plugin.py:1181`, `plugin.py:1182`, written at
`plugin.py:1193` only after the build returns, so a build that raises is retried next call). Keying on
the description as well as the name is what makes the fingerprint agree with the summary cache, which is
keyed the same way: a changed description invalidates both together instead of leaving a stale line
behind a matching name (`_ensure_index` docstring `plugin.py:1164`).

A catalog line is produced in three tiers, in this order of preference (`_summarize` `plugin.py:242`):

1. **Verbatim.** A description that already fits `catalog_chars` is the best summary of itself and costs
   no call (`plugin.py:258`, `plugin.py:259`).
2. **Summarized.** A longer one goes to the summarizer, sync or async (`plugin.py:262`,
   `plugin.py:264`), and the answer is clamped to the limit with whitespace collapsed and wrapping
   quotes dropped (`_clamp_summary` `plugin.py:222`, `plugin.py:238`, `plugin.py:239`). The default
   summarizer is one plain call to the agent's own model with no tools and no history, so it passes
   through no agent middleware and cannot recurse into this projection (`_model_summarizer`
   `plugin.py:273`, `model.stream` `plugin.py:295`); its usage is accumulated into
   `state.summary_usage` (`plugin.py:294`, field `plugin.py:569`).
3. **Truncated.** A summarizer that raises or answers nothing falls back to a cut at a sentence or word
   boundary (`plugin.py:267`, `plugin.py:270`, `_truncate_description` `plugin.py:363`). No tool is ever
   left without a line and no summarizer failure escapes the projection.

Lines are cached on the plugin instance keyed by `(name, description)` (`plugin.py:1101`), so a tool is
summarized once however many agents and calls read it, and a re-registration with a changed description
gets a new line. Missing lines are requested concurrently under a semaphore of `_SUMMARY_CONCURRENCY`
(`plugin.py:72`, gate `plugin.py:1216`, gather `plugin.py:1223`), and only for catalog-eligible specs —
the two plugin tools are excluded (`plugin.py:1206`).

```mermaid
sequenceDiagram
    participant Stage as InvokeModelStage.Input
    participant Handler as _projection_handler [plugin.py:1124]
    participant Ensure as _ensure_index [plugin.py:1164]
    participant Sum as _summarize [plugin.py:242]
    participant Index as LexicalToolIndex [index.py:168]
    participant Project as _project [plugin.py:977]
    participant Provider as Bedrock request

    Stage->>Handler: context · tool_specs = 93 full specs · system_prompt = operator's
    Handler->>Handler: _should_passthrough(...) returns False [plugin.py:1143,898]
    Handler->>Handler: _expire(state, cycle, ttl_cycles) [plugin.py:1147,607]
    Handler->>Ensure: (state, context.tool_specs, agent) [plugin.py:1149]
    Ensure->>Ensure: (name, description) fingerprint changed, so this call does the work [plugin.py:1181,1182]
    Ensure->>Sum: every spec with no cached line, bounded by the semaphore [plugin.py:1206,1216]
    Sum-->>Ensure: verbatim if it fits, else the summarizer, else a boundary cut [plugin.py:258,265,270]
    Ensure->>Index: build(list(specs)) [plugin.py:1189 · index.py:188]
    Ensure->>Ensure: state.fingerprint = fingerprint, only now [plugin.py:1193]
    Handler->>Project: (context, exposed, always_available, summaries) [plugin.py:1151]
    Project->>Project: _compose_projection, three blocks of FULL specs [plugin.py:1002,934]
    Project->>Project: _fold_tool_exchanges over the messages [plugin.py:1003,807]
    Project->>Project: _catalog_prompt_block over the names NOT projected [plugin.py:1008,187]
    Project-->>Handler: replace(context, tool_specs=..., messages=..., system_prompt=...) [plugin.py:1009]
    Handler->>Handler: state.projected = the projected names [plugin.py:1157]
    Handler->>Handler: _log_projection(projected.tool_specs) [plugin.py:1158,666]
    Handler-->>Stage: the new context
    Note over Provider: toolConfig = find_tools + get_tool_details + always_available + loaded
    Note over Provider: system = operator's prompt + the catalog heading + one line per remaining tool
    Note over Provider: messages = the folded copy · no call shape for a tool that is not in toolConfig
```

## 6. The common path — catalog, load, call

Two model calls and no guessing: the names live in the system prompt, the schemas arrive through
`get_tool_details`, and the call itself renews the TTL once it has returned.

`get_tool_details` tolerates the two shapes a model actually sends: a bare string instead of a list
(`plugin.py:1323`) and duplicates, which are de-duplicated with order kept (`plugin.py:1324`). A call
that named nothing usable answers with `_DETAILS_EMPTY_GUIDANCE` (`plugin.py:1326`). A name the registry
does not have is collected and reported rather than silently dropped (`plugin.py:1334`,
`plugin.py:1343`). The specification itself is **not** in the result text — it travels in `tool_specs`
on the next call, because a tool result is resident in the history while a projection is per call and
forgettable (`_DETAILS_LOADED_HEADER` docstring `plugin.py:104`).

```mermaid
sequenceDiagram
    participant Model
    participant Stage as InvokeModelStage.Input
    participant Plugin as ProgressiveToolDisclosure
    participant Reg as ToolRegistry

    Note over Model: Call N — reads the catalog in the system prompt
    Stage->>Plugin: _projection_handler(context) [plugin.py:1124]
    Plugin-->>Stage: tool_specs = callable tools · system_prompt += catalog · messages folded [plugin.py:1009]
    Model->>Plugin: get_tool_details(["list_investment_transactions"]) [plugin.py:1302]
    Plugin->>Plugin: state.loads += 1 [plugin.py:1319]
    Plugin->>Plugin: normalize · bare string tolerated · duplicates dropped [plugin.py:1323,1324]
    loop each requested name
        Plugin->>Reg: registry.get(name) [plugin.py:1332]
        Plugin->>Plugin: _renew(state, name, cycle) so exposed[name] = cycle [plugin.py:1336,642]
    end
    Plugin-->>Model: "Loaded. These tools are callable with their full parameters on your next call..." [plugin.py:100,1341]
    Note over Model: Call N+1 — the projection emits those specs FULL, from block 3
    Stage->>Plugin: _projection_handler(context) [plugin.py:1124]
    Plugin-->>Stage: the loaded tool carried as a full verbatim spec [plugin.py:1153]
    Note over Plugin: state.projected now holds its name [plugin.py:1157]
    Model->>Plugin: list_investment_transactions(real args) · BeforeToolCallEvent
    Plugin->>Plugin: name is in state.projected, so the guard returns [plugin.py:1372]
    Reg-->>Model: tool executes normally
    Plugin->>Plugin: AfterToolCallEvent · _renew · the TTL window restarts here [plugin.py:1399]
```

`get_tool_details` takes a **list**, which is what keeps a step that needs three tools to one cycle
rather than three (docstring `plugin.py:1303`).

## 7. The fallback — `find_tools` searches, it does not load

`find_tools` exists for a need the model cannot map to any listed name. It ranks specs and reports
names plus summaries, and then stops: the model still goes through `get_tool_details`, so there is one
road to a schema instead of two (comment `plugin.py:1297`).

Every invocation is counted, including a blank need and a failed search, because each costs the cycle
just the same (`plugin.py:1265`, comment `plugin.py:1264`). A blank need is not searched at all and
returns guidance (`plugin.py:1269`, `plugin.py:1270`). A search that raises returns guidance too, worded
the same way as a no-match because from where the model stands the two cases are one
(`plugin.py:1276`, `plugin.py:1279`, `_SEARCH_FAILED_GUIDANCE` docstring `plugin.py:124`). A match the
registry does not have, and either plugin tool, are skipped (`plugin.py:1285`, `plugin.py:1287`).

```mermaid
sequenceDiagram
    participant Model
    participant Plugin as ProgressiveToolDisclosure
    participant Index as LexicalToolIndex
    participant Reg as ToolRegistry

    Note over Model: no catalog name fits the need
    Model->>Plugin: find_tools("list investment transactions") [plugin.py:1242]
    Plugin->>Plugin: _record_search(state) · searches += 1 [plugin.py:1265,684]
    alt need is blank
        Plugin-->>Model: _EMPTY_NEED_GUIDANCE [plugin.py:1270,117]
    else need is usable
        Plugin->>Index: search(need, top_k) [plugin.py:1274 · index.py:205]
        alt search raised
            Plugin-->>Model: _SEARCH_FAILED_GUIDANCE [plugin.py:1279,123]
        else matches returned
            loop each match
                Plugin->>Reg: registry.get(match.name) [plugin.py:1285]
                Plugin->>Plugin: line = name + cached summary [plugin.py:1290,1401]
            end
            Plugin->>Plugin: _log_search_outcome(need, names) [plugin.py:1292,694]
            alt nothing usable
                Plugin-->>Model: _NO_MATCH_GUIDANCE [plugin.py:1295,120]
            else names to report
                Plugin-->>Model: _MATCHES_HEADER + one line per match · NOTHING exposed [plugin.py:1299,95]
                Model->>Plugin: get_tool_details([names]) — then exactly as section 6
            end
        end
    end
```

The index is term-frequency and local, so a search needs no network (`LexicalToolIndex` `index.py:168`,
`build` `index.py:188`, `search` `index.py:205`). It is a structural protocol, so any object exposing a
callable `build` and `search` can replace it (`ToolIndex` `index.py:44`, `_validate_index`
`plugin.py:515`), and both operations are allowed to be awaitable for a network-backed implementation
(`plugin.py:1189`, `plugin.py:1274`).

## 8. The safety net — premature-call cancellation

A premature call is a name the model read in the catalog and called without loading it. The guard reads
whether the schema was **visible**, not whether it is loaded: `state.projected` (`plugin.py:564`) holds
the names the last projection actually carried, written right after `_project` returns
(`plugin.py:1157`). Reading `exposed` instead would mistake a tool that expired between the projection
and the call for a guessed call.

The exemptions are dealt with first, in one condition (`plugin.py:1368`): a name the registry does not
have is left alone entirely, and the two plugin tools and everything in `always_available` are exempt —
they are carried in full on every call, so the question cannot arise for them. Then a name in
`state.projected` returns (`plugin.py:1372`).

**The guard loads nothing on the model's behalf.** It cancels with `_PREMATURE_CALL_MESSAGE`
(`plugin.py:1379`), which names the tool and points at `get_tool_details`. A recovery that loaded the
tool and invited an immediate retry would teach the model that calling a catalog name directly works,
which is the very shortcut the catalog rule forbids (`_PREMATURE_CALL_MESSAGE` docstring
`plugin.py:133`).

The guard does not require an empty input. Any call to a tool whose schema was never projected is
cancelled when the tool requires parameters, because invented arguments against an unseen schema are the
*worse* failure — they can satisfy a permissive tool and return a confidently wrong answer that nothing
in the run marks as suspect (comment `plugin.py:1375`).

```mermaid
sequenceDiagram
    participant Model
    participant Plugin as ProgressiveToolDisclosure
    participant Reg as ToolRegistry
    participant EventLoop

    Note over Model: the model calls a catalog name without loading it first
    Model->>Plugin: BeforeToolCallEvent(tool_use.name = X) [plugin.py:1347]
    Plugin->>Reg: X in registry · X a plugin tool · X in always_available? [plugin.py:1368]
    alt any of those
        Plugin-->>EventLoop: return · no cancel · nothing written [plugin.py:1369]
    else none of those
        Plugin->>Plugin: state = _state_for(states, agent) [plugin.py:1371]
        alt X in state.projected
            Note right of Plugin: the schema was in the last projection, so the call is informed [plugin.py:1372]
            Plugin-->>EventLoop: return · let the call through [plugin.py:1373]
        else X was never projected
            Plugin->>Reg: _requires_parameters(spec)? [plugin.py:1378,335]
            alt requires parameters
                Plugin->>Plugin: event.cancel_tool = _PREMATURE_CALL_MESSAGE.format(name=X) [plugin.py:1379,128]
                Plugin->>Plugin: _record_premature_cancellation · premature_cancellations += 1 [plugin.py:1380,708]
                Plugin-->>Model: "'X' did not run: it is not loaded... call get_tool_details with [X] first"
                Note over Model: X is NOT loaded by the guard · the model must call get_tool_details itself
                Model->>Plugin: get_tool_details(["X"]) · then exactly as section 6
            else no required parameter
                Plugin-->>EventLoop: return · an empty call is legitimate [plugin.py:1378]
            end
        end
    end
```

Only the `required` list decides — `_requires_parameters` (`plugin.py:335`) answers `True` only when
that list is non-empty (`plugin.py:360`): a tool whose parameters are
all optional is callable with no arguments, so an empty call to it is not the symptom of a missing
schema. Any schema shape the function cannot read is treated as requiring nothing (`plugin.py:335`
docstring), which errs towards letting the call run.

The counter this path increments is the one worth watching. Under this design it should be near zero,
because the catalog names are not in `tool_specs` at all; a high `premature_cancellations`
(`plugin.py:568`) means the model is calling names it read in the prompt without loading them, which is
a catalog-wording problem rather than a mechanism failure.

## 9. One tool's lifecycle and release

TTL parameter: `ttl_cycles`, default `_DEFAULT_TTL_CYCLES = 3` (`plugin.py:89`; ctor `plugin.py:1053`).
Age is measured against `agent.event_loop_metrics.cycle_count` only — no wall clock, no message count,
no tool-call count (`_expire` docstring `plugin.py:607`). The boundary belongs to the live side:
`cycle - last_used == ttl_cycles` is KEPT, and release is `cycle - last_used > ttl_cycles`
(`plugin.py:626`).

Renewal happens on the **return** of a call, not before it: `_on_after_tool_call` (`plugin.py:1383`)
renews only a tool already in `exposed` (`plugin.py:1398`) and only when the call was not cancelled
(`plugin.py:1392`), so a cancelled call keeps nothing loaded.

```mermaid
stateDiagram-v2
    [*] --> Catalog: registered · one line in the system-prompt catalog [plugin.py:214,187]
    Catalog --> Loaded: get_tool_details([name]) so _renew writes the cycle [plugin.py:1336,630]
    Catalog --> Cancelled: premature call · cancelled, and NOTHING is loaded [plugin.py:1379]
    Cancelled --> Catalog: the model must call get_tool_details itself
    Loaded --> Renewed: tool ran · _on_after_tool_call renews [plugin.py:1399]
    Renewed --> Renewed: used again within ttl_cycles, exposed[name] = cycle [plugin.py:642]
    Renewed --> Idle: idle · cycle - last_used <= ttl_cycles is kept [plugin.py:626]
    Loaded --> Idle: loaded but not yet used, still within the TTL
    Idle --> Catalog: cycle - last_used > ttl_cycles · _expire releases it [plugin.py:626]
    note right of Catalog
        ttl_cycles default = 3 (_DEFAULT_TTL_CYCLES, plugin.py:89)
        Release withdraws the schema from the next projection.
        The tool stays registered and callable (_expire docstring, plugin.py:618)
        Release also puts the name BACK into the system-prompt catalog,
        which is what makes the block's membership move (see section 11).
    end note
```

`find_tools` appears nowhere in this diagram on purpose: a search changes no exposure state, so a tool's
lifecycle is driven by `get_tool_details`, by use, and by inactivity.

## 10. Ordering guarantee — the context graph can never eat the catalog

The catalog is written into `system_prompt` from a handler on the *same* stage another plugin in this
stack rewrites: the context-graph plugin's delivery also registers on `InvokeModelStage.Input`. Since
the catalog is no longer optional, this guarantee is load-bearing on every projected call rather than on
one mode. It is **structural**, and it rests on three facts, each verified by reading source:

1. **The graph forces itself to index 0.** In
   `community-plugins/strands-context-graph/src/strands_context_graph/projection.py`, `register`
   (`projection.py:150`) adds its handler (`projection.py:172`) and then immediately moves it to the
   front (`projection.py:175`):

   ```python
   # projection.py:175
   handlers.insert(0, handlers.pop())
   ```

2. **The SDK builds the chain BACK TO FRONT, so index 0 is the OUTERMOST layer and runs FIRST.** In the
   installed SDK, `strands/_middleware/registry.py`, `compose` iterates the sorted handler list in
   reverse (line 128 seeds `current` with the terminal, line 129 walks downward), wrapping each handler
   *around* what was built so far:

   ```python
   # strands/_middleware/registry.py, lines 128-129
   current: MiddlewareNext = terminal
   for i in range(len(sorted_handlers) - 1, -1, -1):
   ```

   The last handler wrapped — index 0 — is the outermost layer, so it is entered first.

3. **The disclosure plugin sets no order, so it is appended and runs LAST.** `init_agent`
   (`plugin.py:1105`) calls `add_middleware` plainly (`plugin.py:1122`) with no reordering, so it sits
   after the graph's forced-front handler and is entered after it.

Net effect: by the time `_projection_handler` (`plugin.py:1124`) appends the catalog block, the graph
has **already** folded the messages and returned. And even in the reverse arrangement the catalog would
survive, because the graph only ever rewrites `messages`: its delivery does
`replace(context, messages=removed)` (`projection.py:217`), and the SDK fold primitive it reuses does
`replace(context, messages=folded, dynamic_trailing_blocks=...)`
(`strands/injection/_message_injection.py`, line 95). Neither writes `system_prompt`, so the field the
catalog is appended to is not a field the graph touches.

Both plugins now write `messages`, and the order settles that too rather than leaving it to chance. The
graph's compaction runs first and hands its rewritten list down; this plugin's fold (§4) runs over that
result, so the two compose in one direction only and the graph never sees a message the fold produced.
The fold's own guard (`_pairs_intact` `plugin.py:788`) is checked against whatever arrived, so a list the
graph already reshaped is verified on the same terms as an untouched one.

This is worth stating explicitly because the intuition runs the wrong way twice over: "index 0" reads as
*innermost* until you read `compose`, and "the graph compacts the context" reads as *the graph might drop
my text* until you read which field it replaces.

## 11. Cache consequence of a system-prompt catalog

Two properties decide this, and they pull in opposite directions.

**The lines themselves do not drift.** A summary is computed once per `(name, description)` and cached
for the life of the plugin instance (`plugin.py:1101`), so the same tool yields byte-identical text on
every call. Nothing in the block is re-derived per call, which is exactly what a per-call truncation or
a per-call model summary would have cost (module docstring `plugin.py:18`).

**The block's membership does move.** A tool that gets loaded leaves the listing, because the block is
built over the names that are *not* carrying a full specification (`plugin.py:211`, computed from the
projection at `plugin.py:1008`), and it returns to the listing when its load is released
(`_expire` `plugin.py:607`).

On Bedrock the cacheable prefix is ordered **tools → system → messages**, and a change in an earlier
section invalidates the later ones (the AWS statement is quoted in
[`how-to/02-community-plugins-agent-sample.md`](../../../how-to/02-community-plugins-agent-sample.md)).
That ordering is what makes the placement cache-neutral rather than costly: the catalog's membership
changes **only** on an event that also changes `tool_specs` — a load adds a full spec, a release removes
one — and `tool_specs` sits ahead of `system` in the prefix, so the invalidation was already paid. The
three fields are written in one `replace` (`plugin.py:1009`) precisely because they describe one
decision.

What follows from that:

- The number of cache invalidations per session is driven by how often the exposure set changes, not by
  where the catalog is placed. Loading several tools in one `get_tool_details` call is therefore cheaper
  than loading them one at a time, which is the cache argument for the list-shaped signature
  (`plugin.py:1302`).
- Release is the other side of that trade: a tool leaving `tool_specs` invalidates the prefix exactly as
  its arrival did, which is what `ttl_cycles` buys down by keeping a tool in use resident
  (`_expire` docstring `plugin.py:607`).
- The `_compose_projection` ordering discipline that exists to protect the cache (docstring
  `plugin.py:951`) still holds for `tool_specs`: two calls with the same state produce the same list in
  the same order.
- The fold rewrites `messages`, the last section of the prefix, so it never invalidates `tools` or
  `system`; what it changes is the size of the part that was going to be re-read anyway.
- The summary calls are an auxiliary cost that has to be read next to the saving, which is why they are
  counted rather than hidden (`state.summary_usage` `plugin.py:569`, accumulated at `plugin.py:294`).

## 12. Harness wiring

The validation harness threads three knobs from environment variables into the constructor, with no
intermediate default of its own:

| Layer | `file.py:LINE` | What it does |
|-------|----------------|--------------|
| Threshold fields | `catalog_chars` (`config.py:398`), `ttl_cycles` (`config.py:401`), `top_k` (`config.py:404`) | `80`, `3`, `4` — the catalog line budget in characters, the load TTL in cycles, and the matches one search lists |
| Environment read | `config.py:420`, `config.py:421`, `config.py:422` | `_env_int("VALIDATION_CATALOG_CHARS", 80)` and its two siblings, inside the `THRESHOLDS` construction |
| Plugin construction | `runner.py:366` | `ProgressiveToolDisclosure(catalog_chars=..., ttl_cycles=..., top_k=...)` (`runner.py:367`, `runner.py:368`, `runner.py:369`) |
| `always_available` | `runner.py:379` | The graph's retrieval tools (`runner.py:380`) plus the one literal domain tool `list_accounts` (`runner.py:381`), derived from the plugins rather than hard-coded. The relevance filter's `retrieve_all_context` is deliberately **not** here: it answers the rare question that needs a whole result, so it stays a catalog name and is loaded only then (comment `runner.py:373`) |

There is no environment switch for the placement, because there is only one placement, and **no
`referenced_source` bridge**: the plugin releases a loaded tool after `ttl_cycles` idle cycles and keeps
nothing because the history mentions it, so a Card that steps down has no schema to keep resident — its
tools are catalog names like any other (`runner.py:20`).

The harness reads five counters plus the summary usage off the per-agent state (`runner.py:490`):
`searches` (`runner.py:492`), `loads` (`runner.py:493`), `premature_cancellations` (`runner.py:495`),
`exposed_at_end` (`runner.py:496`), `exposed_count_at_end` (`runner.py:497`) and `summary_usage`
(`runner.py:499`), which `compare.py` bills at the agent's own rates (comment `runner.py:498`).

The graph's retrieval tools are put in `always_available` precisely so that a cycle spent loading them
is not charged to the strategy (comment `runner.py:370`). That is a measurement decision, not a
workaround for a defect in the projection.

## 13. Verbatim text the model sees

**Write targets.** `tool_specs`, `messages` and `system_prompt`, all three in one `replace`
(`plugin.py:1009`), or `tool_specs` and `messages` alone when `catalog_chars is None`
(`plugin.py:1006`). The two tool results are returned as **tool-result messages**, resident in message
history, which is why the schema itself is deliberately kept out of them (`_DETAILS_LOADED_HEADER`
docstring `plugin.py:104`) — and why the fold drops those two exchanges once they are spent
(`plugin.py:862`).

### 13.1 The system-prompt catalog header — `_CATALOG_PROMPT_HEADER` (`plugin.py:162`)

Quoted literally, `{get_tool_details}` and `{find_tools}` being substituted with the two tool names at
`plugin.py:218`:

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

Immediately followed by one line per non-callable tool, built at `plugin.py:214`:

```python
f"- {name}: {summary}" if summary else f"- {name}"
```

So the model sees, for example:

```text
- get_transactions: Transactions of an account over a date range, newest first.
- open_position: Opens a position on an instrument for an account and returns its id.
```

No sigil and no marker: the names are not in `tool_specs`, so nothing in the call claims they are
callable and nothing has to be walked back. Step 3 is what makes release intelligible from the model's
side — a rejected call is a documented state with a documented recovery, not a surprise.

### 13.2 `get_tool_details` docstring — exactly as the model receives it (`plugin.py:1303`)

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
(`plugin.py:1301`) from the signature `get_tool_details(self, names: list[str], tool_context:
ToolContext)` (`plugin.py:1302`). The only model-facing parameter is `names`, a list of strings.

### 13.3 `find_tools` docstring — exactly as the model receives it (`plugin.py:1243`)

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
(`plugin.py:1242`): the only model-facing parameter is `need`, a string.

### 13.4 `get_tool_details` results

Header literal `_DETAILS_LOADED_HEADER` (`plugin.py:100`), which states the release rule in the same
breath as the load:

```python
_DETAILS_LOADED_HEADER = (
    "Loaded. These tools are callable with their full parameters on your next call. A tool left unused for "
    f"a few calls is unloaded; to call it after that, call `{GET_TOOL_DETAILS_NAME}` again:"
)
```

Assembled with one line per loaded tool at `plugin.py:1341`, each line built at `plugin.py:1337`, and
followed when needed by `_DETAILS_UNKNOWN` (`plugin.py:107`, formatted at `plugin.py:1343`):

```python
_DETAILS_UNKNOWN = "Not a tool, ignored: {names}. Use names from the catalog or from `" + FIND_TOOLS_NAME + "`."
```

A call that named nothing usable gets `_DETAILS_EMPTY_GUIDANCE` (`plugin.py:110`, returned at
`plugin.py:1326`):

```python
_DETAILS_EMPTY_GUIDANCE = (
    "Pass the names of the tools you want loaded, as a list. Take them from the catalog, or call `"
    + FIND_TOOLS_NAME
    + "` first."
)
```

### 13.5 `find_tools` results

Header literal `_MATCHES_HEADER` (`plugin.py:95`), which states in the same breath that nothing was
loaded:

```python
_MATCHES_HEADER = (
    f"Tools that match. Nothing is loaded yet: call `{GET_TOOL_DETAILS_NAME}` with the names you want, then call them."
)
```

The full result is assembled at `plugin.py:1299` as `"\n".join([_MATCHES_HEADER, *lines])`, each line
built at `plugin.py:1290`:

```python
lines.append(f"- {match.name}: {self._short_description(registered.tool_spec)}")
```

`_short_description` (`plugin.py:1401`) returns the cached catalog line, or a boundary truncation at the
default limit when the catalog is suppressed (`plugin.py:1416`) — `catalog_chars=None` drops the catalog
from the prompt, it does not mean a search result should carry a full description.

### 13.6 `_PREMATURE_CALL_MESSAGE` (`plugin.py:128`)

```python
_PREMATURE_CALL_MESSAGE = (
    "'{name}' did not run: it is not loaded, so its parameters are unknown to you. Call `"
    + GET_TOOL_DETAILS_NAME
    + "` with [\"{name}\"] first, then call '{name}' with its real parameters."
)
```

Formatted with the tool name at `plugin.py:1379` before being assigned to `event.cancel_tool`. It names
the recovery instead of performing it: the tool is **not** loaded by the guard.

### 13.7 The fold sentence — `_fold_note` (`plugin.py:723`)

Not a constant: the sentence is assembled from the tool name and the rendered result, with the verb
chosen by the result's status (`plugin.py:739`):

```python
outcome = "failed with" if result.get("status") == "error" else "the result was"
return f"The tool {name} was called and {outcome}: {' '.join(parts)}"
```

So a closed exchange of a tool that is no longer callable reads, in the messages of the next call:

```text
The tool get_transactions was called and the result was: {"items": [...], "next": null}
```

It is model-facing text like any other, and it is the reason the evidence survives a release while the
call shape does not.

### 13.8 The remaining guidance strings

- `_EMPTY_NEED_GUIDANCE` (`plugin.py:117`), returned on a blank `need` (`plugin.py:1270`):

  ```python
  _EMPTY_NEED_GUIDANCE = "Describe what you are trying to do, in your own words, then call this tool again."
  ```

- `_NO_MATCH_GUIDANCE` (`plugin.py:120`) — returned when nothing usable was found (`plugin.py:1295`):

  ```python
  _NO_MATCH_GUIDANCE = "No tool matches that description. Try different wording, or answer directly."
  ```

- `_SEARCH_FAILED_GUIDANCE` (`plugin.py:123`) — returned when the search itself raised
  (`plugin.py:1279`):

  ```python
  _SEARCH_FAILED_GUIDANCE = "Tool search is unavailable right now. Try a different description, or answer directly."
  ```

With the `_CATALOG_PROMPT_HEADER` block and its lines, the two tool docstrings, `_MATCHES_HEADER`,
`_DETAILS_LOADED_HEADER`, `_DETAILS_UNKNOWN`, `_DETAILS_EMPTY_GUIDANCE`, `_PREMATURE_CALL_MESSAGE` and
the `_fold_note` sentence, that is the complete set of model-facing text the plugin can produce. The
`_ELLIPSIS = "..."` literal (`plugin.py:140`) can appear inside a line that fell back to truncation when
the cut lands mid-sentence (`_truncate_description` `plugin.py:363`).

One text is model-facing in a different sense: `_SUMMARY_SYSTEM_PROMPT` (`plugin.py:75`) is the
instruction the default summarizer sends, with `{max_chars}` substituted at `plugin.py:295`. It asks for
what tells a tool apart from its neighbours — the object it acts on and what it returns — because that is
what the model reading the catalog needs in order to decide whether to load it.

## 14. Configuration — every constructor parameter

Constructor: `ProgressiveToolDisclosure.__init__` (`plugin.py:1048`). All keyword-only (`*` at
`plugin.py:1050`). Every check runs before any state is set, so a construction that fails leaves no
handler, hook or tool registered (`plugin.py:1083` onwards).

| Parameter | Default (`file.py:LINE`) | Accepts `None`? | Meaning |
|-----------|--------------------------|-----------------|---------|
| `catalog_chars` | `_DEFAULT_CATALOG_CHARS = 80` (`plugin.py:68`; ctor `plugin.py:1051`) | **Yes** | Character limit of one catalog line's summary. `None` drops the catalog entirely (`plugin.py:1005`), leaving the two plugin tools' descriptions as the only hint that other tools exist. Validated by `_validate_catalog_chars` (`plugin.py:460`) — `None` or int ≥ 1, `0` rejected because a zero-character line fits nothing. |
| `summarizer` | `None` (`plugin.py:1052`) | **Yes** | What writes a line when a description does not fit. Receives `(spec, max_chars)`, sync or async (`ToolSummarizer` `plugin.py:83`). `None` uses the agent's own model, one plain call per tool, cached (`_model_summarizer` `plugin.py:273`, selected at `plugin.py:1215`). `_validate_summarizer` (`plugin.py:478`) — `None` or callable. |
| `ttl_cycles` | `_DEFAULT_TTL_CYCLES = 3` (`plugin.py:89`; ctor `plugin.py:1053`) | No | Cycles a loaded tool survives without a call; every call that runs renews it (`plugin.py:1399`). `_validate_positive_int` (`plugin.py:438`) — int ≥ 1, `bool` and `float` rejected. |
| `always_available` | `()` empty tuple (`plugin.py:1054`) | No | Names carrying their full spec on every call, skipping the load cycle and exempt from the guard (`plugin.py:1368`). `_validate_always_available` (`plugin.py:492`) — a sequence of non-empty strings; a bare string is rejected, since it would configure one name per character. Stored as a tuple (`plugin.py:1094`). |
| `index` | `None` → `LexicalToolIndex()` (`plugin.py:1055`; instantiated `plugin.py:1098`) | **Yes** | Search implementation behind `find_tools`. `None` selects the default term-frequency `LexicalToolIndex` (`index.py:168`), which needs no network. `_validate_index` (`plugin.py:515`) — `None`, or an object with callable `build` and `search`. |
| `top_k` | `_DEFAULT_TOP_K = 3` (`plugin.py:92`; ctor `plugin.py:1056`) | No | How many tools one search lists. `_validate_positive_int` (`plugin.py:438`) — int ≥ 1. |

Six parameters, none positional. There is no placement flag, no `referenced_source` and no
`find_tools_name` knob — the two tool names are the module constants
`FIND_TOOLS_NAME = "find_tools"` (`plugin.py:56`) and
`GET_TOOL_DETAILS_NAME = "get_tool_details"` (`plugin.py:59`), collected into `_PLUGIN_TOOL_NAMES`
(`plugin.py:65`) and threaded through the projection, the fold, the passthrough test and the guard
rather than exposed on the constructor. The second name is deliberately not `get_details`: a verb-noun
that generic is one a domain tool can already hold, and a collision would silently shadow one of the two
in the registry (`plugin.py:60`).
