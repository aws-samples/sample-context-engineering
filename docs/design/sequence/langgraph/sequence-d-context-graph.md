# Sequence Design — Practice D on LangGraph: `langgraph-context-graph`

Authority: `langgraph-plugins/langgraph-context-graph/src/langgraph_context_graph/` for the binding and
`context-core/src/context_core/graph/` for the practice. Every claim carries a `file.py:LINE`. Literal strings are
quoted from source. Where a thing does not exist, it is stated as not existing rather than invented.

Companion document: `docs/design/sequence/sequence-d-context-graph.md`, the Strands binding of the same practice.
This one is **not** a restatement of it — the Card model, the scoring ladder and the compaction are shared code and
are described here only where the LangGraph binding changes what reaches the model.

---

## 1. What the binding does, mechanically

`ContextGraphMiddleware` (`middleware.py:277`) is a LangChain v1 `AgentMiddleware` with **two** engagement surfaces,
each in a sync and an async form. What the Strands plugin spreads across three hooks and a middleware stage, LangChain
admits in two places (`middleware.py:3`–`4` module header):

| Surface | Sync | Async | Strands analog |
|---|---|---|---|
| **Delivery** — project the call's message list | `wrap_model_call` (`middleware.py:436`) | `awrap_model_call` (`middleware.py:480`) | the `InvokeModelStage.Input` handler plus two event hooks |
| **Artifact** — address what a tool returned | `wrap_tool_call` (`middleware.py:520`) | `awrap_tool_call` (`middleware.py:548`) | `AfterToolCallEvent` (`middleware.py:527`) |

The body of the delivery surface is five statements, identical in both forms (sync line first, async second):

1. Read the graph carried in the agent state under `context_graph` (`middleware.py:462` / `501` → `_state_of`,
   `middleware.py:654`; key constant `middleware.py:117`).
2. Convert the call's LangChain messages to the neutral shape — `to_neutral_list(list(request.messages))`
   (`middleware.py:463` / `502`, `_adapter.py:159`).
3. Call `context_core.graph.project` (`middleware.py:465` / `504`, core at `projection.py:100`), which does the write
   half, the read half and the delivery in that order — the order the three Strands hooks imposed
   (`projection.py:134`, `projection.py:140`, `projection.py:146`).
4. Flatten the new state to something LangGraph can copy (`_persistable`, `middleware.py:667`) and write it into
   `request.state` for this same turn's retrieval tools (`middleware.py:474` / `513`, `_write_back` at
   `middleware.py:692`).
5. Send the call out, projected or not: `call = request if projected is neutral else request.override(messages=to_langchain_list(projected))`
   (`middleware.py:477` / `515`), then attach the durable state write as a `Command` on the response
   (`middleware.py:478` / `516` → `_with_state_update`, `middleware.py:705`).

The artifact surface is three statements and is §4.4.

**Three** retrieval tools are built at construction as ordinary LangChain tools (`_build_tools`,
`middleware.py:778`) and published on `self.tools` (`middleware.py:418`): `expand_card`, `expand_artifact`,
`find_context`, in that registration order (`middleware.py:832`–`836`), which is the Strands plugin's own order
(`middleware.py:779`, asserted at `tests/test_middleware.py:376`). `expand_artifact` is the one behind a switch —
`include_artifact_tool`, default `True` (`middleware.py:358`, built at `_build_artifact_tool`, `middleware.py:838`);
with it false the list is `["expand_card", "find_context"]` (`tests/test_artifacts.py:365`). See §3.4.

One module-level side effect completes the wiring: `_register_host_symbols` (`middleware.py:96`) is called at import
(`middleware.py:115`) and registers `"search_content"` with `context_core.graph.store.HOST_SYMBOLS`
(`middleware.py:112`), the optional bridge a targeted artifact read needs. `"context_manager"` and `"extract_text"`
are deliberately left unregistered, each for its own stated reason (`middleware.py:106`–`110`), and
`tests/test_artifacts.py:385`–`388` asserts exactly that set.

Three properties are carried over from the Strands plugin unchanged, because they are the practice rather than the
wiring (`middleware.py:21`–`35` header):

- **The persisted history is never touched.** `override` is per call, so `state["messages"]` comes out of a projection
  exactly as it went in. Nothing is deleted, which is why a Card the choice collapsed still has messages to be raised
  back to.
- **A full pass is the identity.** `expand_threshold=0.0` makes the core return the *received* neutral list by object
  identity (`scoring.py:175` → `projection.py:443`), and the middleware reads that identity and calls the handler with
  the **original** request, so no `override` is applied at all (`middleware.py:477`).
- **Nothing derives a Card with a model.** The only remote call in any configuration is the matcher's embedding round,
  one per turn (`projection.py:321`), and the default matcher is built on first need (`_matcher_for`,
  `middleware.py:736`, construction at `middleware.py:749`) so wiring reaches no network.

**Seven mechanics are documented below as first-class, not footnotes.** Each is a real finding already visible in the
code, not a hypothetical:

1. **`MappingProxyType` is not picklable, so `_persistable` flattens it.** `TurnChoice.by_title` is a
   `MappingProxyType` (`state.py:167`, docstring at `state.py:160`), which neither `copy.deepcopy` nor `pickle` can
   carry. LangGraph copies every state update as it applies it, so a proxy reaching the agent state fails the
   *superstep*, not merely the persistence. `_persistable` (`middleware.py:667`) rebuilds the choice around a plain
   dict (`middleware.py:683`–`688`). See §4.2.
2. **The adapter must carry `tracking_id`, or the graph is a silent no-op.** `msg.id` travels as the neutral
   `tracking_id` (`_adapter.py:155`), which is the Durable Identity every Card is addressed by. A history whose
   messages carry no id yields **no Card at all** and projects whole — machine-checked at
   `tests/test_middleware.py:178`. See §5.2.
3. **`expand_threshold=collapse_floor=0` is an identity call.** The constructor forbids a floor above the ceiling
   (`middleware.py:376`), so zeroing the threshold forces zeroing the floor; that pair short-circuits
   `warm_up_choice` before the matcher is reached (`scoring.py:175`) and the handler receives *the same request
   object* (`tests/test_middleware.py:259`–`281`). See §4.3.
4. **The elevation `expand_card` writes has a different lifetime here, because `project` runs per model call and not
   per invocation.** The Strands plugin computed the choice once per `BeforeInvocationEvent`; here every model call
   recomputes it at `projection.py:140`, which overwrites the `by_title` entry the tool wrote. What survives the
   boundary is the fed-back note, aged by **turn ordinal** rather than by a cycle counter (`projection.py:139` with
   the comment at `projection.py:136`–`138`). See §7.3.
5. **The artifact store is held on the middleware instance and is deliberately not agent state.** One
   `InMemoryReferenceStore` per conversation, keyed by thread id (`middleware.py:416`, `_store_for`,
   `middleware.py:635`), because LangGraph deep-copies every state update as it applies it — the very reason
   `_persistable` exists — and the blocks are the one part of this plugin that is content rather than addresses
   (`middleware.py:406`–`415`). See §3.4 and §4.4.
6. **The message the compaction's fold splits in two is marked, or an inner middleware reads it as a fresh user
   turn.** `to_langchain` renders one neutral tool-result-plus-text message as a `ToolMessage` followed by a
   `HumanMessage`, and that carrier is flagged `ATTACHED_TEXT_KEY` (`_adapter.py:111`, set at `_adapter.py:285`) so
   `to_neutral_list_with_sources` (`_adapter.py:164`) folds it back onto the preceding tool-result message
   (`_adapter.py:191`). Unmarked, the graph's own `<collapsed_turns>` block started a phantom user turn for the
   middleware nested inside it. See §5.3.
7. **`context_graph` needs a reducer, because two retrieval tools can answer in one step.** Each tool returns a
   `Command` carrying the graph, so two calls in one `AIMessage` are two writes to one state key, which LangGraph
   refuses with `InvalidUpdateError` and fails the whole turn. `_latest_graph` (`middleware.py:195`) takes the later
   write, which loses nothing because both writes are the same mutated object. See §3.2 and §7.1.

---

## 2. Integration table — where the binding attaches to LangChain v1

Every framework symbol is funnelled through one file, `_compat.py`, "so an API move touches one file", verified
against `langchain` 1.4.2 (`_compat.py:1`–`3`): `AgentMiddleware`, `AgentState`, `ExtendedModelResponse`,
`ModelRequest`, `ModelResponse` from `langchain.agents.middleware` (`_compat.py:8`–`14`); `ToolCallRequest` and
`ToolRuntime` from `langchain.tools.tool_node` (`_compat.py:15`); `Command` from `langgraph.types` (`_compat.py:16`).

| Attach point | Declared at | What it does | Reads | Mutates |
|---|---|---|---|---|
| `state_schema` | `middleware.py:342` (`state_schema = ContextGraphState`) | Adds the `context_graph` key to the agent state so the graph travels with the conversation, under the `_latest_graph` reducer (`middleware.py:195`) that admits two writes in one step | — | The agent's state schema |
| `wrap_model_call` | `middleware.py:436` | The delivery surface, sync form | `request.messages`, `request.state` | `request.state[…]` best-effort; returns a NEW call via `override`; never `messages` in state |
| `awrap_model_call` | `middleware.py:480` | Async twin; the model handler is awaited at `middleware.py:516`, everything else is the same synchronous helper | same | same |
| `wrap_tool_call` | `middleware.py:520` | The artifact surface, sync form: records the return, then hands it back untouched | `request.tool_call`, `request.state`, `request.runtime` | `self._stores[…]`; the graph object already in `request.state` |
| `awrap_tool_call` | `middleware.py:548` | Async twin; the tool handler is awaited at `middleware.py:565` and the recording is the same synchronous helper | same | same |
| `self.tools` | `middleware.py:418` | Publishes `expand_card`, `expand_artifact` and `find_context` for `create_agent` to register | — | The agent's tool registry |
| `self._thresholds` | `middleware.py:419` | Freezes the core's `Thresholds` for the instance's lifetime, including `retrieval_tools=tuple(each.name for each in self.tools)` (`middleware.py:429`) | `self.tools` | — |
| `self._stores` | `middleware.py:416` | One reference store per conversation, created on first tool call (`_store_for`, `middleware.py:635`) | the run's `thread_id` (`_thread_of`, `middleware.py:1139`) | Its own dict; never the agent state |
| `HOST_SYMBOLS` | `middleware.py:112`, called at `middleware.py:115` | Registers `"search_content"` so a `line_range`/`pattern` read is delegated rather than degraded | — | `context_core.graph.store.HOST_SYMBOLS`, by `setdefault` |
| Wiring notice | `middleware.py:432` → `_warn_on_pruning_middleware` (`middleware.py:755`) | One `warnings.warn` when the agent's middleware list holds a pruner | the passed `middleware` list | Nothing — "not removed, not reordered, not reconfigured" (`middleware.py:154` docstring) |

**There is no hook ordering problem to solve.** The Strands plugin had to move its delivery handler to index 0 of
`InvokeModelStage.Input`; here the nesting is the agent's middleware list and the first entry is the outermost
`wrap_model_call`. The core says the same from its side: the registry reordering, the `contextvars` hand-off and the
`WeakKeyDictionary` lookup "were all mechanism for delivering *inside* an event loop" and are not carried
(`projection.py:36`–`45`).

### 2.1 The async twins, and why they exist

Neither twin is symmetry for its own sake. `awrap_model_call` (`middleware.py:480`) states the gap it closes in its
docstring (`middleware.py:487`–`492`):

> LangChain does not bridge a sync ``wrap_model_call`` to an async run — it raises ``NotImplementedError`` — so a
> stack run under ``ainvoke`` (the harness does, since the relevance reranker protocol is async) needs this twin to
> exist, and every hook here has both. The projection (:func:`context_core.graph.project`) is pure and synchronous, so
> the only difference from the sync hook is that the model handler is awaited; the state writes and the response
> wrapping are the same synchronous helpers.

`awrap_tool_call` (`middleware.py:548`) closes the same gap on the other surface, for the same reason and with the
same division of labour: "LangChain does not bridge the two -- a sync ``wrap_tool_call`` raises
``NotImplementedError`` under an async run -- so both exist. The recording itself reaches no network and is shared"
(`middleware.py:555`–`556`).

The regression test carries the same reading in its module header (`tests/test_async_hook.py:1`–`5`) and asserts both
model hooks are present (`tests/test_async_hook.py:53`–`56`); the tool surface is asserted on both paths at
`tests/test_artifacts.py:97` and `tests/test_artifacts.py:112`. Because `project` is pure and synchronous, the model
twin differs from the sync form in exactly one token: `await handler(call)` (`middleware.py:516`) against
`handler(call)` (`middleware.py:478`). The tool twin differs in one token as well — `await handler(request)`
(`middleware.py:565`) against `handler(request)` (`middleware.py:544`).

**The split reaches one tool, not just the hooks.** `expand_artifact` is registered with **two** bodies, a sync
`expand_artifact` (`middleware.py:847`) and an async `aexpand_artifact` (`middleware.py:859`), handed to
`StructuredTool.from_function` as `func=` and `coroutine=` (`middleware.py:871`–`876`). The reason is stated at
`middleware.py:786`–`790`: LangChain bridges neither direction, "a coroutine-only tool raises
``NotImplementedError`` under ``invoke`` and a sync-only one would run in a worker thread under ``ainvoke``", and the
artifact read is the one awaitable step in the package because a store's `retrieve` may cross a process boundary. The
sync body therefore drives that single coroutine itself through `_driven` (`middleware.py:1177`, `asyncio.run` on the
ordinary path at `middleware.py:1188`, a one-worker pool when a loop is already running at `middleware.py:1190`).
`tests/test_artifacts.py:325`–`341` reaches the same answer through `invoke` and `ainvoke`. `expand_card` and
`find_context` need none of this: their bodies are synchronous throughout.

**What this cost the harness.** An earlier version of the benchmark harness carried an `AsyncContextGraphMiddleware`
subclass that supplied an `awrap_model_call` by running the sync hook on a worker thread (`asyncio.to_thread`) and
bridging the handler back onto the loop, because the package then implemented only the sync `wrap_model_call`. The
package carries the twin at `middleware.py:480`, so that subclass is gone and the harness wires the real
`ContextGraphMiddleware` — the removal and its reason are recorded where the subclass used to be
(`runner.py:277`–`283`). A benchmark run therefore measures the package's own `awrap_model_call` rather than a
thread-hop around the sync hook. The stack still runs under `ainvoke`, because the relevance filter's reranker
protocol is genuinely async (`runner.py:282`–`283`).

---

## 3. The data model

### 3.1 The neutral shape — the boundary the core never crosses

`context_core.message` defines the contract and forbids a framework import anywhere under `context_core`
(`message.py:1`–`13`). A message is `{"role": …, "content": [block, …]}` and a block carries exactly one of
`{"text"}`, `{"json"}`, `{"toolUse"}`, `{"toolResult"}` (`message.py:14`–`31`). `NeutralBlock` and `NeutralMessage`
are plain dict aliases rather than validated `TypedDict`s, "so an adapter can hand over the framework's own dict
without a copy" (`message.py:38`–`41`).

`_adapter.py` is "the ONLY module in the LangGraph binding that touches both worlds" (`_adapter.py:3`–`5`). §5 is
about that module.

### 3.2 `ContextGraphState` — the graph as ordinary agent state

```python
class ContextGraphState(AgentState):                              # middleware.py:208
    context_graph: NotRequired[Annotated[GraphState, _latest_graph]]  # middleware.py:216
```

`NotRequired` is load-bearing: "a first call has no graph yet, which the core reads as a fresh one: a conversation
with no Card is a full pass, so an agent whose state carries nothing here is delivered to exactly as one without the
middleware" (`middleware.py:211`–`213`). Machine-checked at `tests/test_middleware.py:354`.

**The `Annotated` reducer is load-bearing too, and for a reason that only shows up under parallel tool calls.**
`_latest_graph` (`middleware.py:195`) is `return left if right is None else right` (`middleware.py:205`) — the later
write wins. Without it LangGraph refuses the second write to a state key inside one step, and the refusal is fatal to
the turn rather than degrading it: "Parallel retrieval calls (``expand_card`` beside ``find_context``, say) each answer
with a ``Command`` carrying the graph, in the same step. Without a reducer LangGraph refuses the second write
(``InvalidUpdateError: Can receive only one value per step``) and the whole turn fails" (`middleware.py:198`–`200`).
Taking the later write loses nothing, because the writes are not rival values: "Every tool mutates the one graph object
the step's state holds, so the writes are the same object with every call's effect on it, and taking the later one
loses nothing -- the Strands plugin mutates its single per-agent graph the same way" (`middleware.py:201`–`203`). See
§7.1 for the retrieval path this closes and `tests/test_artifacts.py:551`–`577` for the regression it came from.

No codec is involved because `_GraphState` is plain data by design: "Every field is plain data (dicts, tuples, floats
and strings), so the state *is* the graph's serialized form — a binding may hand it back verbatim on the next call
without a codec" (`state.py:180`–`182`). Its fields are `cards`, `links`, `choice`, `reuse`, `turn`, `vectors`,
`retrieval_cycles`, `referenced` (`state.py:195`–`202`). The one field that breaks the plain-data promise in practice
is `choice`, and §4.2 is about that.

### 3.3 Card and the four `LinkKind` values

Shared with the Strands binding and unchanged: a `Card` (`state.py:76`) is a frozen dataclass holding **addresses and
derived text, never message content**; its fields are at `state.py:100`–`113`. `LinkKind = Literal["tool",
"artifact", "follows", "similar"]` (`state.py:51`); a `Link` is `(kind, target, weight)` (`state.py:129`–`131`).

| Link kind | Created | Target | Propagates a note? | Traversed by a retrieval path? |
|---|---|---|---|---|
| `tool` | `cards.py:422` on a subject Card, `cards.py:774` on an artifact Card | a tool name | Not through `_STRUCTURAL_WEIGHTS`; walked separately by `_spread_over_tool_hubs` (`scoring.py:289`) | No |
| `artifact` | `cards.py:425` | a reference | Yes, weight `_W_ARTIFACT` (`scoring.py:65`, listed `scoring.py:71`) | No — and see §3.4 |
| `follows` | `cards.py:428` | the prior Card's title | Yes, weight `_W_PREVIOUS` (`scoring.py:68`) | No |
| `similar` | `cards.py:435`–`436` (bidirectional, at registration) and `cards.py:1071`–`1072` (`link_newly_measurable`) | a Card title | **No** — `_STRUCTURAL_WEIGHTS` omits it (`scoring.py:71`), and `_spread_over_card_edges` skips any kind the mapping does not name (`scoring.py:283`–`286`) | **Yes** — `_similar_neighbors` (`middleware.py:1250`), its only reader anywhere |

`_STRUCTURAL_WEIGHTS` contains exactly `{"follows": _W_PREVIOUS, "artifact": _W_ARTIFACT}` (`scoring.py:71`–`76`) —
verified. Its docstring states the `similar` exclusion outright: "``similar`` targets a Card but inherits no Note at
all (Requirement 8.4): it is an edge a manual search traverses" (`scoring.py:80`–`81`).

### 3.4 Artifact Cards, the reference store, and the two resolution layers

The binding carries a full artifact path. `wrap_tool_call` (`middleware.py:520`) stores what a tool returned and
derives an artifact Card per reference the return names; `expand_artifact` reads it back. The three core entry points
are all reached: `derive_artifact_cards` (`cards.py:681`) and `register_artifact_cards` (`cards.py:739`) beneath
`derive_and_register_artifacts` (`cards.py:779`), which `_record_artifacts` calls at `middleware.py:621`. The
reference store (`InMemoryReferenceStore`, `store.py:135`) is imported at `middleware.py:67` and filled through
`record_references` (`store.py:191`) at `middleware.py:611` and `middleware.py:631`.

**Where the store lives, and why not in the state.** One store per conversation, held on the middleware instance
(`self._stores`, `middleware.py:416`) and created on first use (`_store_for`, `middleware.py:635`), keyed by the run's
`thread_id` (`_thread_of`, `middleware.py:1139`) with `""` for a run that has no checkpointer
(`middleware.py:1149`). The comment states the trade (`middleware.py:406`–`415`): LangGraph deep-copies every state
update as it applies it — the reason `_persistable` exists at all — so a store in the agent state "would be copied on
every superstep and checkpointed at its full size, and the blocks are the one part of this plugin that is content
rather than addresses". A process-local store trades durability across processes, "which the Strands store does not
have either", for not paying for the content twice. Per-thread isolation is asserted at
`tests/test_artifacts.py:124`–`131`; the address being minted from the call rather than from the graph is what lets a
tool call seen before any projection still store its return (`tests/test_artifacts.py:191`–`197`).

**What a Card holds is still an address.** An artifact `Card.kind` is `"artifact"` (`state.py:48`), it carries the
`reference` and the two facts the placeholder states about it, and never a copy of the content — so nothing in the
graph rots when the content behind the reference changes (`cards.py:692`–`696`;
`tests/test_artifacts.py:158`–`177`). This makes the `is_artifact` branches in `distribute` live rather than dead
code: an artifact Card is excluded from `in_progress` by kind (`scoring.py:375`), takes the artifact branch of the
dialogue ladder (`scoring.py:381`), and is Full Content on the evidence axis unconditionally (`scoring.py:396`).
`_artifact_header` (`describe.py:424`) is reached, so an artifact Description leads with the reference for textual
content and with the file name for non-textual content (`describe.py:441`–`455`):

```text
reference: mem_1_tc9_0
tool: query_ledger
turn: 3
```

**An artifact Card gets one link kind, not four.** `register_artifact_cards` writes only the `tool` edge
(`cards.py:774`); `follows` is absent because an artifact is not a turn, and `similar` is absent because it would
propagate Note into a Card automatic choice keeps out of full content (`cards.py:749`–`753`). The `artifact` edge
pointing *at* it is derived on the subject side from the reference string (`cards.py:425`), which is why that edge
resolves whether or not the artifact Card exists yet — and why, when it does exist, the edge now propagates a note to
a target that **is** in `note` rather than being skipped at `scoring.py:284`.

**The resolution order has two layers, and the second is passed in.** `expand_artifact` delegates the read to
`resolve_artifact` (`store.py:284`) and never reimplements it (`middleware.py:979`–`981`). The core asks the plugin's
own store first and the second layer only for what the own store does not hold (`store.py:306`–`319`). On a Strands
agent that second layer is discovered on the agent — the `ContextManager` Stash (`store.py:311`). LangGraph has no
equivalent, no `"context_manager"` host symbol is registered (`middleware.py:109`–`110`), so the layer is supplied
explicitly: the `stash=` constructor parameter, validated to have a callable `retrieve` (`middleware.py:390`) and
handed to the core as the keyword second layer (`middleware.py:1010`). The call comment names the substitution
(`middleware.py:1007`–`1009`):

> ``agent=None``: the core's own second-layer discovery is the Strands ``ContextManager`` Stash, which LangGraph has
> no equivalent of. The second layer is therefore the explicit ``stash`` -- typically the relevance filter's store --
> and with none a reference the own store does not hold is ``"absent"``.

The intended supplier is `RelevanceFilterMiddleware.stash`
(`langgraph-relevance-filter/src/langgraph_relevance_filter/middleware.py:408`), a read-only view of that filter's
store in the shape a resolver reads — `retrieve(reference) -> text`, with text and JSON decoded to a string and
anything else to `None` so the resolver reports it as non-textual
(`langgraph-relevance-filter/src/langgraph_relevance_filter/middleware.py:295`–`309`). Without it the filter's
`[ref: mem_N_<tool_call_id>_<index>]` addresses resolve to the `"absent"` prose, because each plugin then ships a
retrieval tool over a store the other cannot read (`middleware.py:312`–`316`). Both halves are machine-checked:
`tests/test_artifacts.py:539`–`548` resolves a reference only the stash holds, and
`tests/test_composition.py:93`–`126` mints one through the real filter, reads it back through `expand_artifact`, and
asserts the same reference is a miss on a graph built without the stash.

Three consequences of the layering, each a reading of the code:

- **With no stash, one layer.** `stash=None` (the default, `middleware.py:359`) leaves the conversation's own store as
  the only layer, "as in a Strands agent with no manager" (`middleware.py:321`–`322`), and every miss reads as
  `absent` rather than `unknown` (`store.py:310`–`313`; `tests/test_artifacts.py:258`–`275`).
- **What the hook can store, the Strands hook cannot.** Strands records a reference as *a name with nothing behind
  it*, because by the time its `AfterToolCallEvent` runs an offloader has already replaced the content; the
  `ContextManager` Stash then supplies the content. Wrapping the tool call is what changes here — the return is in
  hand, so it is stored rather than named (`middleware.py:575`–`581`).
- **The plugin's own answers are never stored.** A return from `expand_card`, `expand_artifact` or `find_context` is
  skipped (`middleware.py:601`–`605`), because a whole read echoes the artifact's entire text and storing it would
  keep a second copy under a second reference (`tests/test_artifacts.py:134`–`145`).

The tool side matches: the guidance can name all three tools, and does whenever all three are registered (§8.3).

---

## 4. Band 1 — the engagement surfaces

### 4.0 The delivery surface: `wrap_model_call` / `awrap_model_call`

```mermaid
sequenceDiagram
    autonumber
    participant LG as LangGraph model node
    participant MW as ContextGraphMiddleware<br/>wrap_model_call (middleware.py:436)<br/>awrap_model_call (middleware.py:480)
    participant AD as _adapter
    participant CORE as context_core.graph.project<br/>(projection.py:100)
    participant PERS as _persistable<br/>(middleware.py:667)
    participant ST as request.state
    participant H as handler (model call)
    participant WS as _with_state_update<br/>(middleware.py:705)

    LG->>MW: request (messages, state)
    MW->>ST: _state_of(request.state) (middleware.py:654)
    Note over ST,MW: state.get("context_graph"), isinstance-checked<br/>None for a conversation with no graph yet (middleware.py:661-664)
    MW->>AD: to_neutral_list(list(request.messages)) (middleware.py:463 / 502)
    AD-->>MW: neutral list, each message carrying tracking_id
    MW->>CORE: project(neutral, state=prior, matcher=self._matcher_for(), body_budget, thresholds) (middleware.py:465 / 504)
    CORE-->>MW: (projected, new_state)
    MW->>PERS: _persistable(new_state) (middleware.py:473 / 512)
    PERS->>PERS: by_title is a MappingProxyType -> rebuild TurnChoice around a plain dict (middleware.py:683-688)
    MW->>ST: state["context_graph"] = persisted (middleware.py:474 / 513 -> 700)
    Note over MW,ST: best effort · a mapping that refuses assignment is logged, not raised (middleware.py:701-702)

    alt projected IS neutral (full pass)
        MW->>H: handler(request) — the ORIGINAL request object, no override (middleware.py:477 / 515)
    else projected is a new list
        MW->>AD: to_langchain_list(projected) (_adapter.py:291)
        MW->>H: handler(request.override(messages=…)) (middleware.py:477 / 515)
    end
    Note over MW,H: sync: handler(call) (middleware.py:478) · async: await handler(call) (middleware.py:516)
    H-->>MW: ModelResponse | AIMessage | ExtendedModelResponse
    MW->>WS: _with_state_update(response, persisted)
    WS-->>LG: ExtendedModelResponse carrying Command(update={"context_graph": …})
```

### 4.1 `override` is the whole delivery mechanism

The Strands binding returned a new `InvokeModelContext` out of an `InvokeModelStage.Input` handler. Here the
equivalent is one call: `request.override(messages=to_langchain_list(projected))` (`middleware.py:477`). Because
`override` is per call, `state["messages"]` is untouched — asserted, by object identity of every persisted message,
at `tests/test_middleware.py:284`–`307`.

Nothing is written back to `messages` on either write path: `_write_back` touches only `_STATE_KEY`
(`middleware.py:700`) and `_with_state_update` writes only `_STATE_KEY` into the command
(`middleware.py:715`, `middleware.py:719`, `middleware.py:728`), which is what lets an inner middleware's command
keep its own keys (`middleware.py:712`–`720`, checked at `tests/test_middleware.py:339`).

### 4.2 The state write happens twice, on purpose — and `MappingProxyType` is why `_persistable` exists

Two writes, two different jobs (`middleware.py:451`–`453`):

| Write | Where | Purpose | Failure posture |
|---|---|---|---|
| In-place, into `request.state` | `middleware.py:474` / `513` → `_write_back` (`middleware.py:692`) | So `expand_card` / `expand_artifact` / `find_context` called **out of this very turn** read the graph the choice was taken from | Logged at debug, never raised: "A state that refuses the assignment costs the tools of *this* turn their fresh graph and nothing else" (`middleware.py:695`–`697`) |
| Durable, as a `Command` on the response | `middleware.py:478` / `516` → `_with_state_update` (`middleware.py:705`) | The write the checkpointer keeps | An unrecognised response shape is passed through untouched and logged (`middleware.py:731`–`733`) |

A third write exists on the artifact surface and is deliberately **not** a `Command`: see §4.4.

**The finding.** `TurnChoice.by_title` is declared `Mapping[str, CardChoice]` and is always built as a
`MappingProxyType` — `full_pass_choice` (`scoring.py:152`), `distribute`'s return (`scoring.py:407`), and the
default factory of a fresh state (`state.py:197`). The proxy is deliberate: "A ``MappingProxyType``, never a live
dict: the choice is frozen for the whole turn, so the context cannot shift mid-reasoning" (`state.py:160`–`162`).

That freeze is exactly what LangGraph cannot carry. `_persistable`'s docstring states the mechanism
(`middleware.py:670`–`674`):

> ...which is what keeps the turn's decision unwritable for the whole turn -- and which neither ``copy.deepcopy`` nor
> ``pickle`` can carry. LangGraph copies every state update as it applies it, so a proxy reaching the agent state
> fails the *superstep*, not just the persistence.

The fix flattens rather than unfreezes, guarded on the concrete type so an already-plain choice is returned untouched:

```python
if type(state.choice.by_title) is not dict:          # middleware.py:683
    state.choice = TurnChoice(                       # middleware.py:684
        by_title=dict(state.choice.by_title),
        full_pass=state.choice.full_pass,
        selected=state.choice.selected,
    )
return state
```

Nothing the freeze protects is lost, and the reason is specific to this binding: "the choice is recomputed by the
next projection rather than read back and extended, so what the proxy protects against — a decision shifting
mid-turn — cannot happen here" (`middleware.py:676`–`679`). The core independently notes the same incompatibility
from its own side, as the reason `_copy_state` is a container copy and not a `deepcopy` (`projection.py:152`–`155`).

`_persistable` mutates in place on purpose, and its docstring names the only two states that reach it: one this
binding just produced, or one a retrieval tool already mutates by design (`middleware.py:679`–`681`). The tool path
calls it too, at `middleware.py:892`.

The regression test is explicit about what it guards — `copy.deepcopy` over the state written both ways
(`tests/test_middleware.py:614`–`633`, docstring at `:615`).

### 4.3 The identity short circuit, end to end

`expand_threshold=0.0` is not a tuning value, it is a probe. The path:

1. The constructor rejects a floor above the ceiling (`middleware.py:376`–`380`), with the comment "a floor above the
   ceiling leaves the middle resolution unreachable" (`middleware.py:374`–`375`). So `expand_threshold=0.0` is
   admissible only together with `collapse_floor=0.0`.
2. `warm_up_choice` returns the full-pass choice before the matcher is reached: `if expand_threshold == 0.0 or
   len(state.cards) < min_cards` (`scoring.py:175`), reached from `projection.py:305`.
3. `_deliver` reads `state.choice.full_pass` first and returns the **received** list (`projection.py:443`–`444`).
4. `project` therefore returns its input object, and the middleware reads that by identity: `call = request if
   projected is neutral else …` (`middleware.py:477`), the comment at `middleware.py:476` reading it as "Identity, not
   equality".

So the provider sees "the call it would see without the middleware" — byte for byte, not merely equal
(`middleware.py:28`–`31` header, `middleware.py:447`–`449`). `tests/test_middleware.py:259`–`281` asserts
`handler.request is request`, `handler.request.messages is request.messages`, and `matcher.calls == []`.

Two other paths reach the same identity, and both matter operationally: `min_cards` not yet met (`scoring.py:175`;
`tests/test_middleware.py:202` asserts the matcher is never called), and an **empty removal request** —
`if not requested: return messages` (`projection.py:447`–`449`).

### 4.4 The artifact surface: `wrap_tool_call` / `awrap_tool_call`

The `AfterToolCallEvent` analog, at the same point in the lifecycle: the recording happens after the handler has
answered and never before, so what is recorded is the return an offloader behind this middleware has already had its
say on (`middleware.py:527`–`529`). The hook body is three statements — run the handler, record, return the handler's
own answer (`middleware.py:544`–`546`).

```mermaid
sequenceDiagram
    autonumber
    participant TN as LangGraph tool node
    participant MW as ContextGraphMiddleware<br/>wrap_tool_call (middleware.py:520)<br/>awrap_tool_call (middleware.py:548)
    participant H as handler (the tool)
    participant REC as _record_artifacts<br/>(middleware.py:569)
    participant AD as _adapter
    participant STORE as InMemoryReferenceStore<br/>_store_for (middleware.py:635)
    participant C as cards.derive_and_register_artifacts<br/>(cards.py:779)
    participant G as the graph in request.state

    TN->>MW: request (tool_call, state, runtime)
    MW->>H: handler(request) — called exactly once (middleware.py:544 / 565)
    H-->>MW: ToolMessage | Command | something else
    MW->>REC: _record_artifacts(request, result) (middleware.py:545 / 566)
    alt result is not a ToolMessage
        REC-->>MW: return — a Command is a state update, a cancelled call carries no return (middleware.py:594-598)
    else the call is one of this middleware's own tools
        REC-->>MW: return — a retrieval answer is not a return to be addressed (middleware.py:601-605)
    else
        REC->>AD: tool_message_to_result_block(result) (middleware.py:608, _adapter.py:71)
        REC->>STORE: record_references(store, ["<tool_call_id>_<index>", …], texts) (middleware.py:611)
        Note over REC,STORE: _stored_texts keeps text and json blocks only, index-aligned<br/>block n of the return is <tool_call_id>_n (middleware.py:1152)
        alt no graph in the state yet
            REC-->>MW: return — blocks stored all the same, the reference is minted from the call (middleware.py:613-618)
        else
            REC->>AD: to_neutral_list(_state_messages(request.state)) (middleware.py:620)
            REC->>C: derive_and_register_artifacts(state, messages, block, name, state.turn, thresholds…) (middleware.py:621)
            C-->>REC: one artifact Card per reference the return names, none when it names any
            REC->>STORE: record_references(store, [card.reference …]) — names only (middleware.py:631)
            REC->>G: the Cards are on the graph object already in request.state
        end
    end
    Note over REC: any failure logs ONE warning · the graph keeps no artifact Card,<br/>and the result the model receives is unaffected either way (middleware.py:632-633)
    MW-->>TN: the handler's answer, UNCHANGED — never wrapped in a Command (middleware.py:546 / 567)
```

**Why the hook returns the handler's answer and not a `Command`.** A `Command` would carry the artifact Cards into the
checkpoint, but it would also make an outer offloader's `isinstance(result, ToolMessage)` guard skip the result
entirely — silently turning that plugin off (`middleware.py:531`–`535`). So the Cards travel on the graph object
already in `request.state`, which is the same object the next projection reads, and the store they address is on the
instance. `tests/test_artifacts.py:502`–`529` is the end-to-end proof that this is enough: the Card written by the
tool hook is in the graph the next model call and the final state read. `tests/test_artifacts.py:97`–`108` asserts the
other half — the tool node receives the handler's own `ToolMessage`, "never a Command, never a copy".

Three writes happen inside `_record_artifacts`, in the order the Strands hook makes them plus one Strands cannot
(`middleware.py:572`–`585`):

1. **The return's blocks go into the conversation's store**, keyed `<tool_call_id>_<index>` — the same key format the
   relevance filter hands its own store. This is the write Strands cannot make; see §3.4.
2. **The artifact Cards are derived from the return**, off the references the placeholder text names, holding the
   address and never a copy of the content. A return naming none registers none, which is the ordinary
   nothing-offloaded path rather than a degradation (`tests/test_artifacts.py:180`–`188`).
3. **Those placeholder references are noted in the store**, names only, as Strands records them
   (`middleware.py:631`).

Failures do not propagate and nothing is half-written. Two guards stack: the core restores both `cards` and `links`
whole before logging (`cards.py:830`–`836`), and the hook wraps the lot in its own `try` that logs "artifact card
derivation failed | the graph keeps none" (`middleware.py:632`–`633`). A batch is restored whole rather than in part,
since half a batch would leave a reference with a Card and its sibling without one (`cards.py:793`–`794`).

---

## 5. Band 2 — the adapter: `to_neutral_list` / `to_langchain_list`

```mermaid
sequenceDiagram
    autonumber
    participant MW as middleware
    participant TN as to_neutral / to_neutral_list<br/>(_adapter.py:117 / 159)
    participant SRC as to_neutral_list_with_sources<br/>(_adapter.py:164)
    participant ID as _identified<br/>(_adapter.py:152)
    participant CORE as project
    participant TL as to_langchain / to_langchain_list<br/>(_adapter.py:221 / 291)

    MW->>TN: to_neutral_list([HumanMessage, AIMessage(tool_calls), ToolMessage, …])
    TN->>SRC: to_neutral_list_with_sources(messages)[0] (_adapter.py:161)
    loop per LangChain message
        opt HumanMessage flagged ATTACHED_TEXT_KEY, behind a user-role toolResult message
            SRC->>SRC: fold its text back onto that message, one neutral message for both (_adapter.py:184-193)
        end
        SRC->>TN: to_neutral(message) (_adapter.py:194)
        TN->>TN: role = _ROLE_BY_TYPE.get(msg.type, msg.type) (_adapter.py:51, 124)
        alt ToolMessage
            TN->>TN: tool_message_to_result_block (_adapter.py:71) -> {"role":"user","content":[{"toolResult":…}]} (_adapter.py:127)
        else AIMessage
            TN->>TN: drop provider call parts · _is_call_part (_adapter.py:105) filters tool_use, tool_call, function_call (_adapter.py:137)
            TN->>TN: remaining text blocks + one {"toolUse":…} per msg.tool_calls (_adapter.py:138-147)
        else Human / System
            TN->>TN: _content_to_text_blocks (_adapter.py:54)
        end
        TN->>ID: _identified(neutral, msg)
        ID->>ID: if msg.id: neutral["tracking_id"] = msg.id (_adapter.py:154-155)
    end
    SRC-->>MW: neutral list, plus the LangChain messages each one stands for (_adapter.py:196)
    MW->>CORE: project(neutral, …)
    CORE-->>MW: projected (removal applied, final block folded onto the last user message)
    MW->>TL: to_langchain_list(projected) (_adapter.py:291)
    loop per neutral message
        alt role == "assistant"
            TL-->>MW: [AIMessage(content, tool_calls, id=identity)] (_adapter.py:261)
        else role == "system"
            TL-->>MW: [SystemMessage(content, id=identity)] or [] (_adapter.py:264)
        else role == "tool" OR the message carries a toolResult block
            TL->>TL: one ToolMessage per toolResult · only position 0 keeps the identity (_adapter.py:271-281)
            opt the message ALSO carries text (the folded final block)
                TL->>TL: append HumanMessage(text) with NO id, marked ATTACHED_TEXT_KEY (_adapter.py:282-285)
            end
        else plain user
            TL-->>MW: [HumanMessage(content, id=identity)] or [] (_adapter.py:288)
        end
    end
    TL-->>MW: LangChain list, ready for request.override
```

### 5.0 A tool call reaches the adapter twice, and only one copy travels

A provider such as Bedrock returns each tool call on an `AIMessage` **twice**: in `msg.tool_calls` and again as a
`tool_use` part of `msg.content`. `_content_to_text_blocks` has no way to read the content part as anything but
structure, so it becomes a neutral `{"json": …}` block (`_adapter.py:67`). `to_neutral` drops exactly those blocks
before adding the canonical ones: `_CALL_PART_TYPES` is `{"tool_use", "tool_call", "function_call"}`
(`_adapter.py:101`), `_is_call_part` matches a `json` block whose `type` is one of them (`_adapter.py:105`–`108`), and
the AIMessage branch filters them out at `_adapter.py:137`.

The comment states what keeping them cost (`_adapter.py:132`–`136`): `tool_calls` is the canonical form and becomes
the `toolUse` block, so a surviving content copy is an **opaque** `json` block sitting beside it — and a core that
removes the `toolUse` cannot recognise the copy to remove it too. The disclosure fold does exactly that removal, which
left the provider receiving a tool use with no tool result. The rule is therefore one canonical source per call, and
it is applied identically in all three bindings' adapters.

This is a write-path fact rather than a graph fact: nothing in `context_core.graph` reads a `json` block, so the
projection behaves the same either way. What changes is what `to_langchain_list` can hand back — `_blocks_to_content`
renders a `json` block as a content part verbatim (`_adapter.py:216`–`217`), so a copy that got in would get out.

### 5.1 The asymmetry: one neutral message, two LangChain messages

`to_neutral` is a function; `to_langchain` returns a **list** (`_adapter.py:221`). The reason is stated in its
docstring (`_adapter.py:224`–`230`):

> A **list**, because the mapping is not one to one in this direction: a neutral ``user`` message is the carrier for
> tool results, and the projection's own compaction appends its final block as a text block to the last ``user``
> message — which in an autonomous tool loop is the message carrying a ``toolResult``. LangChain has no message type
> holding both, so that one neutral message renders as the ``ToolMessage`` followed by a ``HumanMessage`` with the
> appended text.

The two halves of the collision, each verifiable:

- **Inbound.** A `ToolMessage` becomes a **user-role** neutral message whose only block is a `toolResult`
  (`_adapter.py:126`–`127`; `tests/test_adapter.py:53`–`57`).
- **The fold.** `_fold_into_last_user_message` walks backwards for `messages[index]["role"] == "user"`
  (`projection.py:497`–`501`) and appends a text block to *that* message's content (`projection.py:506`–`507`). In a
  tool loop the newest user-role message is the one carrying the tool result, so the folded `<collapsed_turns>` block
  lands on a message that already holds a `toolResult`.

The split preserves the two invariants that matter. Ordering: "The tool result therefore stays immediately behind the
tool call, which is the ordering a provider requires, and the folded text lands behind it as trailing per-call
content" (`_adapter.py:228`–`230`). Identity: the first message out keeps the `tracking_id` and the synthetic carrier
gets none — "it is per-call content, not a persisted message" (`_adapter.py:232`–`234`, code at `_adapter.py:279` and
`_adapter.py:285`). Machine-checked at `tests/test_adapter.py:127`–`151`.

The carrier is not only id-less, it is **flagged**: `additional_kwargs={ATTACHED_TEXT_KEY: True}` (`_adapter.py:285`,
constant at `_adapter.py:111`). Nothing in this binding reads that flag on the way out — it is there for whatever reads
the list next, and §5.3 is about what that is.

`to_langchain_list` states the direction of the guarantee precisely: the round trip holds outward and back, and it is
"deliberately **not** injective the other way" (`_adapter.py:294`–`297`).

One more asymmetry, in content shape rather than message count: a single text block collapses to a plain string and
anything richer becomes a list of content parts (`_blocks_to_content`, `_adapter.py:202`, mirroring
`_content_to_text_blocks`, `_adapter.py:54`; same rule for a tool result at `result_block_to_content`,
`_adapter.py:83`). An assistant message with neither body nor tool calls renders as **no message at all**, because
"several providers reject an empty message outright" (`_adapter.py:258`–`260`).

### 5.2 `tracking_id` is the whole graph, and losing it is silent

`_identified` is four lines (`_adapter.py:152`–`156`) and is the binding's single point of contact with the Card
model. Its docstring in `to_neutral` states the consequence: "A message without one contributes no identity, so its
turn yields no Card and projects whole — the same quiet direction the core takes everywhere else"
(`_adapter.py:120`–`122`).

Followed through the core, the no-op is total:

1. `_identities_in` collects an identity only from a message that has one (`projection.py:275`), and `_close_turns`
   returns early when a closed turn yields none: "A turn whose messages all lack a Durable Identity: messages without
   a Card, projected whole" (`projection.py:230`–`233`).
2. With no Card, `warm_up_choice` short-circuits on `len(state.cards) < min_cards` (`scoring.py:175`) and the choice
   is a full pass.
3. `_deliver` returns the received list on `full_pass` (`projection.py:443`).

So an id-less history produces **no warning, no error and no measurable difference** — the middleware runs, the
matcher is never called, and every message goes whole. `tests/test_middleware.py:178`–`186` asserts exactly that:
`state["context_graph"].cards == {}` and `handler.messages == messages`.

This is the one failure mode of the binding that cannot be noticed from the model's side, which is why it is stated
here as a first-class mechanic. LangChain assigns message ids in normal operation; the exposure is a hand-assembled
history, a `model_copy(update={"id": None})`, or a middleware upstream that rebuilds messages without carrying ids.

### 5.3 The split is reversible, and an inner middleware depends on that

§5.1 is the outward half: one neutral message becomes two LangChain messages, because LangChain has no type holding a
tool result and trailing text together. The inward half is `to_neutral_list_with_sources` (`_adapter.py:164`), and
`to_neutral_list` is one line over it — `return to_neutral_list_with_sources(messages)[0]` (`_adapter.py:161`).

It does two things the plain loop did not. It **joins the split back**: a `HumanMessage` flagged `ATTACHED_TEXT_KEY`
(`_adapter.py:111`), sitting behind a user-role neutral message that already carries a `toolResult`, has its text
appended to that message's content rather than becoming a message of its own (`_adapter.py:184`–`193`). And it
**reports provenance**: the second element of its return is, per neutral message, the list of LangChain messages it
stands for (`_adapter.py:196`, docstring `_adapter.py:177`–`178`), so a caller that folds neutral messages can put the
originals back byte for byte instead of re-rendering them.

**What the flag is worth is stated in the docstring.** Read back naively, the
marked carrier "would be a fresh user turn, and an inner middleware would see the current turn start AT it -- the
disclosure fold then treats the turn's own ``get_tool_details`` exchanges as closed and folds them away, so the model
never sees its load and reloads forever" (`_adapter.py:171`–`175`). The graph is the outer middleware in the combined
stack (§10.2), so the message that starts that phantom turn is the graph's own doing: the compaction folds
`<collapsed_turns>` onto the latest user-role message (`projection.py:497`–`501`), which in a tool loop is a tool result
mid-turn (§5.1), and `to_langchain` renders it as a separate `HumanMessage` that unmarked reads as a user turn of its
own. Folding it back restores the exact neutral message the outer middleware produced, so the inner one reads what the
outer one wrote and not an artifact of the rendering between them.

The other side of the same fix is in the disclosure binding, which is where the fold and the provenance are consumed:
`_fold_messages` maps each neutral message to its **source group** and extends the output with the whole group
(`langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure/middleware.py:442`, used at
`langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure/middleware.py:447`), which a one-to-one
`zip` could not express once one neutral message could stand for two LangChain ones.

A second, independent contributor to the same loop sits in that binding's cycle accounting and is recorded there rather
than here: `_last_used` treats a load numbered on a cycle later than the one being decided as loaded just before it,
because "it was numbered on a longer history, before a middleware removed messages from state (the relevance filter
drops its closed retrieval exchanges at the end of a run)" and, left as it was, "the tool would stay uncallable until
the count caught up, and the model would reload it cycle after cycle"
(`langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure/middleware.py:307`–`311`). Both
contributors produce the same signature in the log, which the harness names outright: a turn over 20 tool calls logs
a warning with its top tool names, "a turn this long is a loop until shown otherwise" (`runner.py:885`–`890`).

---

## 6. Band 3 — the core: `project()` and its three phases

`project` (`projection.py:100`) is eight statements. The module header names what each phase used to be: the write
half "was `MessageAddedEvent`", the read half "was `BeforeInvocationEvent`", the delivery "was the
`InvokeModelStage.Input` handler" (`projection.py:8`–`24`).

```mermaid
sequenceDiagram
    autonumber
    participant MW as middleware
    participant P as project<br/>(projection.py:100)
    participant CP as _copy_state<br/>(projection.py:149)
    participant W as _close_turns<br/>(projection.py:198)
    participant C as cards.py
    participant D as describe.py
    participant R as _compute_choice<br/>(projection.py:281)
    participant S as scoring.py
    participant MA as matcher
    participant V as _cache_description_vectors<br/>(projection.py:368)
    participant DEL as _deliver<br/>(projection.py:422)
    participant RM as apply_removal<br/>(removal.py:173)
    participant RB as render_final_block<br/>(compaction.py:91)
    participant F as _fold_into_last_user_message<br/>(projection.py:470)

    MW->>P: project(messages, state, matcher, body_budget, thresholds)
    P->>CP: _copy_state(state) or a fresh _GraphState() (projection.py:132)
    Note over CP: container copy, NOT deepcopy — every value is already immutable<br/>and MappingProxyType cannot be pickled at all (projection.py:152-155)

    rect rgb(238,244,255)
    Note over P,D: PHASE 1 — the write half (was MessageAddedEvent)
    P->>W: _close_turns(new_state, messages, thresholds) (projection.py:134)
    W->>C: closed_turn_ranges(messages) (cards.py:161)
    alt no closed turn
        W-->>P: return — nothing to card (projection.py:220-222)
    else state.cards empty -> the restore case
        W->>C: rebuild_into(state, messages, **_card_config) (projection.py:225 -> cards.py:564)
    else incremental
        W->>C: turn_ids = _identities_in(messages[start:stop]) (projection.py:229 -> 261)
        alt no identity in the turn
            W-->>P: return — messages without a Card, projected whole (projection.py:230-233)
        else
            W->>C: derive_and_register(state, messages, turn_ids, len(closed)-1, **_card_config) (projection.py:236 -> cards.py:442)
            C->>C: derive_card (cards.py:306): partition_turn (189), tool_pairs_of (216), references, numeric_lines
            C->>D: compose_description (describe.py:189) · tag_candidates (241) · select_tags (277)
            C->>C: register_card (cards.py:375): tool (422) / artifact (425) / follows (428) · similar (435-436) only if vectors cached
            C->>C: retag(state, messages) (cards.py:439 -> 638) — rarity recount over the graph
        end
    end
    Note over W: any failure logs ONE warning and leaves the state as it was (projection.py:238)
    end

    rect rgb(240,255,243)
    Note over P,V: PHASE 2 — the read half (was BeforeInvocationEvent)
    P->>S: expire_reuse(new_state, new_state.turn) (projection.py:139 -> scoring.py:119)
    Note over P,S: aged by the TURN ORDINAL, not a cycle counter — the neutral<br/>equivalent of agent.event_loop_metrics.cycle_count (projection.py:136-138)
    P->>R: _compute_choice(new_state, messages, matcher, body_budget, thresholds) (projection.py:140)
    R->>S: warm_up_choice(state, expand_threshold, min_cards) (projection.py:305 -> scoring.py:155)
    alt skipped (expand_threshold == 0.0 OR cards < min_cards)
        S-->>R: full_pass_choice() — the matcher is never reached (scoring.py:175)
    else matcher is None
        R-->>P: full_pass_choice() — "score nothing, send everything" (projection.py:315-317)
    else
        R->>R: question = _question_of(messages) — texts of the LAST user message (projection.py:319 -> 344)
        R->>S: compute_notes(state, question, matcher) (projection.py:321 -> scoring.py:195)
        S->>MA: ONE score(question, descriptions) — the turn's only embedding round
        S->>S: Pass 1 = similarity + fed-back note, no factor (scoring.py:225-229)
        S->>S: Pass 2 = _propagate, exactly one jump over follows/artifact + tool hubs (scoring.py:235)
        alt notes empty (matcher failed or malformed)
            R-->>P: full_pass_choice() (projection.py:322-326)
        else
            R->>V: _cache_description_vectors(state, matcher, link_threshold) (projection.py:330)
            V->>MA: matcher.vectors(descriptions) — free, already in the cache under the document purpose
            V->>C: link_newly_measurable(state, newly_measurable, link_threshold) (projection.py:416 -> cards.py:1029)
            C->>C: writes ONLY 'similar' edges (cards.py:1071-1072)
            R->>S: distribute(notes, state, expand_threshold, collapse_floor, body_budget) (projection.py:332 -> scoring.py:320)
            S-->>R: TurnChoice.by_title — a rung per Card, frozen behind MappingProxyType (scoring.py:407)
        end
    end
    Note over R: any failure logs ONE warning and degrades to the full pass (projection.py:340)
    P->>P: new_state.turn += 1 (projection.py:142) · new_state.retrieval_cycles = 0 (projection.py:144)
    end

    rect rgb(255,248,240)
    Note over P,F: PHASE 3 — the delivery (was InvokeModelStage.Input)
    P->>DEL: _deliver(new_state, messages, thresholds) (projection.py:146)
    alt choice.full_pass
        DEL-->>MW: the RECEIVED messages object, by identity (projection.py:443-444)
    else
        DEL->>RM: apply_removal(messages, state, choice, current_turn_ids(messages)) (projection.py:446)
        RM->>RM: removal_ids skips any part at 'full' and subtracts the open turn (removal.py:124-128)
        RM->>RM: remove_messages preserves by pin / first user / tool-pair reconciliation (removal.py:131)
        RM-->>DEL: (removed list, requested ids)
        alt requested empty
            DEL-->>MW: the same list object (projection.py:447-449)
        else
            DEL->>RB: render_final_block(removed, state, requested, description_tokens, retrieval_tools) (projection.py:451)
            RB->>RB: retained read off the removed list · dropped = requested - retained (compaction.py:129)
            RB->>RB: one _entry per Card in ascending (turn, title) (compaction.py:133 -> 196)
            RB-->>DEL: "<collapsed_turns>…</collapsed_turns>\n\n<trailer>" or None
            alt final_block is None
                DEL-->>MW: the removed list as it is — no description to lose (projection.py:458-461)
            else
                DEL->>F: _fold_into_last_user_message(removed, final_block) (projection.py:463)
                F->>F: walk backwards to the last role == "user" (projection.py:497-501)
                F->>F: APPEND a text block, separator "\n\n" (projection.py:504-507)
                F->>F: carry metadata + tracking_id onto the rebuilt message (projection.py:510-513)
                F-->>DEL: NEW list, input never mutated
                DEL-->>MW: folded (projection.py:464)
            end
        end
    end
    Note over DEL: any failure returns the RECEIVED messages, one warning (projection.py:466)
    end
```

### 6.1 The write half — what closes a turn, and the restore case

A turn is closed by what comes after it, so the Card of the last closed turn is derived off the messages themselves —
"Title, Description, Tags and Links together, with no model, no disk and no network" (`projection.py:9`–`11`).

`_close_turns` has three branches and each is a distinct situation (`projection.py:216`–`237`):

| Branch | Condition | Action |
|---|---|---|
| Nothing closed | `not closed` | Return. "Nothing has closed yet: no turn to card, and nothing to rebuild from" (`projection.py:220`) |
| Restore | `not state.cards` | `rebuild_into` — one scan derives the whole graph (`projection.py:225`) |
| Incremental | otherwise | `derive_and_register` on the last closed range (`projection.py:236`) |

The ordinal passed to `derive_and_register` is `len(closed) - 1` and **not** `state.turn`, "so a turn's Card is the
same whether it was derived turn by turn or by one scan" (`projection.py:234`–`235`). `_card_config`
(`projection.py:241`) is the single source of the derivation parameters for both entry points, which is what makes the
two comparable (`projection.py:246`–`248`).

In LangGraph the restore branch is not hypothetical: the graph travels in checkpointed state, so anything the
checkpointer drops on the way back in arrives as a state holding no Card in front of a history that already has a
closed boundary — which is precisely the restore branch, rebuilding the whole graph on every resume. The harness
therefore puts the graph's state types on the checkpointer's msgpack allowlist (`_graph_state_types`,
`runner.py:237`, applied at `runner.py:271`), and §10.2 is about what that allowlist has to be given.

`derive_and_register` absorbs a per-Card failure itself — one warning, nothing registered — and `_close_turns` wraps
the lot in a second `try` that logs "closing the turn's card failed | the turn's messages go whole"
(`projection.py:238`). A late write half is a turn with no Card yet, which projects Full Content, i.e. the behaviour
without the feature (`projection.py:207`–`209`).

### 6.2 The read half — one embedding round, and where the `similar` edge gets paid for

`compute_notes` (`scoring.py:195`) invokes the matcher **exactly once** (`scoring.py:203`), over
`titles_in_turn_order` (`scoring.py:180`) so the similarities come back aligned and every later sum walks the same
order — which is where determinism comes from (`scoring.py:186`–`188`). Asserted at
`tests/test_middleware.py:189`–`200`: one call, four descriptions, the question being the last user text.

`_cache_description_vectors` (`projection.py:368`) runs **only** on this path and only after the scoring round, so
the document vectors are already in the matcher's cache and the index fill sends nothing (`projection.py:378`–`380`).
Its docstring states what an unfilled index actually costs, which is not latency:

> Without this the index stays empty, and an empty index is not a slow path but a missing feature: the similarity Link
> is measured from this index alone, by a step Requirement 3.2 keeps free of network calls, so an unfilled index makes
> every pair unmeasurable and the ``similar`` Link never forms at all. The three structural Link kinds still form,
> which is why a graph with no ``similar`` edge looks like a working graph in the counters. (`projection.py:371`–`376`)

This is the sequencing fact behind §7.4: the `similar` edge for a Card cannot form on the turn the Card is registered
(the write half is network-free, so its own vector is not cached yet), and `link_newly_measurable`
(`projection.py:416` → `cards.py:1029`) closes the gap on the **next** projection, writing `similar` edges and
nothing else (`cards.py:1071`–`1072`).

### 6.3 `distribute` — the two axes read different things

Unchanged from the Strands binding; restated because it is what the model's context is made of.
`distribute`'s docstring gives both axes as two bullets (`scoring.py:330`–`344`).

| | Dialogue axis | Evidence axis |
|---|---|---|
| Rungs | **Three** — full / description / title | **Two** — full / description |
| Decided by | the Note (cosine similarity plus the fed-back bonus) | **message order alone** |
| Code | thresholds at `scoring.py:379`, `scoring.py:389`, `scoring.py:392` | `all(pair.consumed for pair in card.pairs)` at `scoring.py:396` |
| Model calls for the decision | zero; one embedding round per turn feeds the Note | **zero**, and zero embeddings — `scoring.py:342` states it |
| Knob that moves it | `expand_threshold`, `collapse_floor`, `body_budget` | **none exists** |

`body_budget=None` is not "no ceiling", it is **the step down turned off**: the branch is
`elif remaining is None or cost <= remaining or in_progress:` (`scoring.py:383`), so with `None` the first disjunct is
permanently true and every Card at or above `expand_threshold` travels whole, however many there are. With a finite
budget a Card that does not fit takes the `else` at `scoring.py:386` and drops **one rung, to Description, never to
Title** — the comment is literally "Budget exhausted: one rung down, never to Title" (`scoring.py:387`).
`tests/test_middleware.py:217` is named for that invariant.

The evidence axis debits the budget but clamps at zero rather than denying the rung: "an unconsumed pair travels whole
regardless, so what would have gone negative is clamped instead of denied" (`scoring.py:400`–`403`).

A Card of the turn in progress is Full Content on both axes whatever its Note (`in_progress`, `scoring.py:375`).

### 6.4 The delivery is one step, or it is nothing

Removal and fold are not two stages that can half-happen. The core states the constraint: Requirement 16.2 "does not
admit the state in between: a removal applied with no block folded is a call that lost content and says nothing about
the loss", so any failure returns the received list by object identity (`projection.py:16`–`21`,
`projection.py:437`–`439`, `projection.py:466`).

The fold **appends**, and the reason is a provider-cache argument rather than a formatting preference: "a trailing run
is the only placement a provider can keep out of its cached prefix — text ahead of the stable conversation would
invalidate the cache from the first block onward" (`projection.py:483`–`486`). The separator is `"\n\n"` unless the
text already starts with a newline, "Some providers concatenate adjacent text blocks, which would run this onto the
user's own words" (`projection.py:504`–`506`). `tracking_id` is carried onto the rebuilt message, because "dropping it
here would make the folded message invisible to the next projection's scan, which reads Cards off exactly these
identities" (`projection.py:511`–`512`).

`tests/test_middleware.py:235`–`257` asserts both halves against the LangChain output: a full Card's `HumanMessage`
survives the `override`, and the folded block arrives on a `HumanMessage`.

---

## 7. Retrieval path — the model calling back in

### 7.1 Three tools, and a `Command` instead of a mutation

```mermaid
sequenceDiagram
    autonumber
    participant Model
    participant TN as @tool expand_card / find_context<br/>(middleware.py:793 / 814)
    participant G as _graph_of<br/>(middleware.py:878)
    participant B as body: self.expand_card / self.find_context<br/>(middleware.py:897 / 1038)
    participant EX as _exhausted<br/>(middleware.py:1087)
    participant MA as matcher.score
    participant SC as record_reuse<br/>(scoring.py:97)
    participant A as _answer<br/>(middleware.py:888)
    participant LG as LangGraph state

    Model->>TN: tool call (titles=[…]) or (need=…, tag=…)
    TN->>G: _graph_of(runtime) — runtime.state's graph, or a FRESH GraphState()
    Note over G: a fresh graph holds no Card, so every title misses and the model is told so,<br/>rather than being answered from a graph that does not describe this conversation (middleware.py:881-883)
    TN->>B: body(state, …)
    B->>EX: _exhausted(name, state)
    Note over EX: checked BEFORE the increment, so a ceiling of n admits exactly n calls (middleware.py:1090)
    alt budget spent
        EX-->>B: _EXHAUSTED refusal text (middleware.py:1094)
    else
        B->>B: state.retrieval_cycles += 1 (middleware.py:924 / 1060)
        alt expand_card
            B->>B: look up each title · kind must be "subject" (middleware.py:934)
            B->>B: rewrite state.choice with CardChoice(dialogue="full", evidence="full") for the found (middleware.py:940-949)
            Note over B: skipped entirely on a full pass — an entry would flip full_pass false<br/>and cost the delivery its identity short circuit (middleware.py:939, 902-904)
            B->>SC: record_reuse per found title (middleware.py:952)
        else find_context
            B->>B: titles_in_turn_order, optionally narrowed by normalize(tag) (middleware.py:1066-1069)
            B->>MA: ONE score(need, every candidate Description) (middleware.py:1107)
            B->>B: keep >= collapse_floor, sort by (-sim, turn, title), cap at 5 (middleware.py:1075-1077)
            B->>SC: record_reuse per chosen title (middleware.py:1083)
            B->>B: _render_candidates -> per-candidate 'related turns:' line (middleware.py:1085 -> 1271)
        end
    end
    B-->>TN: answer text
    TN->>A: _answer(runtime, state, text)
    A->>A: _persistable(state) — flatten the MappingProxyType again (middleware.py:892)
    A-->>LG: Command(update={"context_graph": …, "messages": [ToolMessage(text, tool_call_id)]}) (middleware.py:890-895)
    Note over A,LG: two tools answering in ONE step write this key twice · the _latest_graph reducer<br/>takes the later write, which is the same mutated object (middleware.py:195, 216)
```

`expand_artifact` shares the budget, the `_graph_of` read and the `_answer` wrapper with these two, and differs in
being asynchronous and in changing no Resolution: §7.5.

Three binding-specific facts:

- **The tools are closures, not bound methods.** `_build_tools` (`middleware.py:778`) closes over `self` "so the
  schema the model sees carries the tool's own arguments and nothing else" (`middleware.py:781`–`782`). The bodies are
  published as `self.expand_card` (`middleware.py:897`), `self.expand_artifact` (`middleware.py:969`) and
  `self.find_context` (`middleware.py:1038`) precisely so they are testable without an agent: "a body reachable
  without a ``ToolRuntime`` is a body that can be tested without an agent" (`middleware.py:906`–`907`).
  `tests/test_middleware.py:368`–`382` checks the model-facing schema of all three carries the tool arguments only,
  the injected `runtime` never appearing in it.
- **A tool persists state with a `Command`, never by mutating.** `_answer` (`middleware.py:888`) returns one
  `Command` carrying both the graph update and the `ToolMessage`, "which is how a tool persists a state change in
  LangGraph: the elevation and the fed-back note have to outlive the tool call to be read by the next projection"
  (`middleware.py:782`–`784`). `tests/test_middleware.py:508`–`525` asserts the shape. The artifact *hook* is the one
  place that deliberately does not do this, for the reason in §4.4.
- **Two tools in one `AIMessage` are two writes to one state key, and that needs the reducer.** A model may call
  `expand_artifact` beside `find_context` in a single batch; both bodies reach `_answer`, so both `Command` objects
  carry `_STATE_KEY` in the same step. LangGraph rejects the second with `InvalidUpdateError: Can receive only one
  value per step` and the turn fails outright, so `context_graph` is annotated with `_latest_graph`
  (`middleware.py:195`, schema at `middleware.py:216`). Both writes are the same graph object, each carrying every
  call's mutation of it, so the later one is a complete value rather than half of a merge — §3.2 has the argument, and
  `tests/test_artifacts.py:551`–`577` is the regression, taken from the live combined arm.
- **`expand_card` accepts a scalar.** "the schema says array, and a model that sends the scalar anyway should be
  answered rather than corrected" (`middleware.py:926`, docstring `middleware.py:912`–`914`;
  `tests/test_middleware.py:426`).

### 7.2 The retrieval budget

`_exhausted` (`middleware.py:1087`) refuses when `state.retrieval_cycles >= self._max_retrieval_cycles`
(`middleware.py:1092`), and `state.retrieval_cycles` is reset to `0` by **every projection**
(`projection.py:144`) — "Counted per turn, so it starts each turn at zero" (`projection.py:143`).
`tests/test_middleware.py:490` is named `test_the_retrieval_budget_is_spent_per_turn`.

One ceiling covers all three tools, each incrementing the same counter (`middleware.py:924`, `middleware.py:1005`,
`middleware.py:1060`) and each checking it before its own increment. `tests/test_artifacts.py:307`–`322` asserts that
for `expand_artifact` specifically, including that the refusal itself costs no budget.

Note the interaction with §7.3: because `project` runs per model call, the budget resets on every model call of an
agent loop, not once per `invoke`. A ceiling of 8 is therefore 8 calls per **model call**, which is a looser bound in
LangGraph than the same number was in Strands.

### 7.3 The elevation's lifetime — the finding

`expand_card`'s docstring says the elevation "ends with the turn: the next projection recomputes the choice from the
graph, and the fed-back note is what carries the request across that boundary" (`middleware.py:900`–`902`). Followed
through the LangGraph control flow, that sentence is precise and its consequence is sharper than it first reads.

The sequence, in code:

1. The tool rewrites `state.choice.by_title` with `dialogue="full", evidence="full"` (`middleware.py:940`–`949`) and
   returns it in a `Command` (`middleware.py:890`–`895`).
2. LangGraph applies the update, so `state["context_graph"]` now holds the elevated choice.
3. The agent loops back to the model node, which calls `wrap_model_call` again → `project` → `_compute_choice`
   (`projection.py:140`), which **assigns** `new_state.choice` unconditionally.

So no delivery ever reads the `by_title` entry the tool wrote: the elevation is overwritten before the next
`_deliver` runs. The whole graph has exactly two readers of `choice` — `_deliver` (`projection.py:443`,
`projection.py:446`) and `render_final_block` beneath it (`compaction.py:131`, `compaction.py:141`) — and both sit
*downstream* of the assignment at `projection.py:140`. Nothing in the read half reads the prior choice: `_copy_state`
carries it across (`projection.py:166`) and `_compute_choice` replaces it unread.

What crosses the boundary is `record_reuse` (`middleware.py:952`, `middleware.py:1034`, `middleware.py:1083` →
`scoring.py:97`), which stores `(_REUSE_BONUS, cycle + reuse_ttl_cycles)` (`scoring.py:116`) with `_REUSE_BONUS = 1.0`
undecayed (`scoring.py:92`). That bonus is added in Pass 1 of the next `compute_notes` (`scoring.py:229`), and since
`expand_threshold` is validated into `[0.0, 1.0]` (`_validate_ratio`, `middleware.py:220`), a note of at least `1.0`
clears any admissible threshold — the constant's docstring says so: "it clears any admissible ``expand_threshold`` on
the turn the model asked: the request the model made is not a hint to be outvoted" (`scoring.py:92`–`94`).

Two differences follow, and both are readings of the code rather than measurements:

- **The bonus lifts the dialogue axis only.** The evidence axis is decided by `pair.consumed`
  (`scoring.py:396`) and no note touches it. `expand_card`'s confirmation promises "its messages and its tool results
  together" (`middleware.py:960`–`963`); the dialogue half of that promise is delivered by the bonus, and the evidence
  half was delivered by the elevation the next projection discards. A Card whose pairs are all consumed therefore
  returns with its dialogue whole and its evidence at Description.
- **`reuse_ttl_cycles=0` makes the tool a text-only no-op.** `record_reuse` returns before writing when the TTL is
  zero (`scoring.py:113`), so nothing crosses the boundary and the elevation is discarded — the model gets the
  confirmation string and no content. In Strands the same setting still left the direct elevation working for the rest
  of the invocation.

**The clock is the turn ordinal.** `expire_reuse` is called as `expire_reuse(new_state, new_state.turn)`
(`projection.py:139`) with the comment naming the substitution: "The plugin reads
``agent.event_loop_metrics.cycle_count``; with no agent to read, the turn ordinal is the neutral equivalent —
monotonic, incremented once per projection, and never advanced by a burst of messages inside one turn"
(`projection.py:136`–`138`). The ordinal is incremented after the choice (`projection.py:142`), and `expire_reuse`
drops an entry only when `cycle > expiry_cycle` (`scoring.py:141`–`143`). Since the ordinal advances once per
projection and a projection is one model call, a TTL of 5 is five **model calls** here, not five agent invocations.

### 7.4 `find_context` and the `similar` edge

`_similar_neighbors` (`middleware.py:1250`) is "The only reader of the ``similar`` edge: it is measured on the write
path and stored with its similarity as the weight, propagates no note by design, and without this traversal is paid
for and read by nothing" (`middleware.py:1253`–`1254`).

```mermaid
sequenceDiagram
    autonumber
    participant Model
    participant FC as find_context<br/>(middleware.py:1038)
    participant SIM as _similarities<br/>(middleware.py:1096)
    participant MA as matcher.score
    participant RC as _render_candidates<br/>(middleware.py:1271)
    participant SN as _similar_neighbors<br/>(middleware.py:1250)
    participant L as state.links

    Model->>FC: find_context(need="allocation split by segment")
    FC->>FC: empty need -> _nothing_found, no score at all (middleware.py:1062-1064)
    FC->>SIM: _similarities(state, titles, need)
    SIM->>MA: ONE score(need, every candidate Description) (middleware.py:1107)
    Note over SIM,MA: a length mismatch raises internally and reads as "no candidate",<br/>never as an exception the model must interpret (middleware.py:1108-1114)
    MA-->>FC: one similarity per candidate
    FC->>FC: >= collapse_floor (middleware.py:1075) · sort (-sim, turn, title) (1076) · cap at _MAX_CANDIDATES = 5 (1077, constant middleware.py:132)
    FC->>RC: _render_candidates(state, need, chosen, neighbors_per_candidate) (middleware.py:1085)
    RC->>RC: already = frozenset(chosen) (middleware.py:1282)
    loop per candidate in rank order
        RC->>RC: "- title: X" (1272) · optional "  tags: …" (1274) · Description lines (1277)
        RC->>SN: _similar_neighbors(state, title, already, limit) (middleware.py:1291)
        alt limit <= 0
            SN-->>RC: [] — byte-for-byte the pre-neighbour answer (middleware.py:1259-1260)
        else
            SN->>L: links of this title where kind == "similar", target not excluded, target still in state.cards (middleware.py:1262-1266)
            SN->>SN: sort by (-weight, title), never by turn (middleware.py:1267)
            SN-->>RC: neighbors[:limit] (middleware.py:1268)
        end
        opt neighbours exist
            RC->>RC: "  related turns: " + ", ".join(f"{t} ({w:.2f})") (middleware.py:1293-1294)
        end
    end
    RC-->>Model: block ending "call expand_card with one of these titles…" (middleware.py:1295)
```

Three deliberate decisions, each verified:

1. **A candidate is never listed as another candidate's neighbour.** `already = frozenset(chosen)`
   (`middleware.py:1282`) is computed once over the **whole** chosen list and passed on every call
   (`middleware.py:1291`), filtered at `middleware.py:1265`. The reason is at `middleware.py:1280`–`1281`: a candidate
   is already being rendered in full, so offering it again would spend tokens to say nothing. The exclusion is
   therefore symmetric and independent of rank order.
2. **A neighbour gets no fed-back note.** `record_reuse` is called only over `chosen` (`middleware.py:1082`–`1083`),
   *before* `_render_candidates` is reached (`middleware.py:1085`). Nothing in `_similar_neighbors` or
   `_render_candidates` touches `state.reuse` or `state.choice`. A neighbour is a **hint, not evidence**: it does not
   raise, it does not persist, and its content does not arrive. The model must call `expand_card` with that title,
   which is what the closing line tells it to do (`middleware.py:1295`).
3. **`0` reproduces the pre-neighbour answer byte for byte.** At `0`, `_similar_neighbors` returns `[]` on its first
   guard (`middleware.py:1259`), `neighbors` is falsy and the append is skipped (`middleware.py:1292`). This is why the
   parameter is validated with `floor=0` (`middleware.py:383`) rather than by the default `floor=1`.

**Why the edge answers something the ranking cannot.** `find_context` scores each Description against the *question*
and never against another Description — one `score` call at `middleware.py:1107`. Two turns covering the same ground in
different words are invisible to each other in that ranking; the `similar` edge holds exactly that
Description-to-Description relation, already measured. The argument is recorded at `middleware.py:1274`–`1277`.

**And the one-turn delay.** Per §6.2, a Card's `similar` edges cannot form on the turn it is registered, so the first
projection after a Card appears renders no `related turns:` line for it. Not a failure path — the list is empty and
the line is skipped.

### 7.5 `expand_artifact` — the read path

The third tool differs from the other two in three ways, and in nothing else: it is asynchronous, it changes no
Resolution on any path, and its answer is content rather than a confirmation.

```mermaid
sequenceDiagram
    autonumber
    participant Model
    participant TW as expand_artifact / aexpand_artifact<br/>(middleware.py:847 / 859)
    participant DR as _driven<br/>(middleware.py:1177)
    participant B as self.expand_artifact<br/>(middleware.py:969)
    participant EX as _exhausted<br/>(middleware.py:1087)
    participant RA as store.resolve_artifact<br/>(store.py:284)
    participant OWN as the conversation's own store
    participant STASH as the explicit stash
    participant RD as store.read_artifact<br/>(store.py:322)
    participant SC as record_reuse<br/>(scoring.py:97)
    participant A as _answer<br/>(middleware.py:888)

    Model->>TW: expand_artifact(reference, line_range?, pattern?)
    Note over TW: sync body drives the one coroutine itself (middleware.py:856)<br/>async body awaits it (middleware.py:868)
    TW->>DR: _driven(coroutine) — asyncio.run, or a one-worker pool inside a running loop
    DR->>B: await self.expand_artifact(state, store, reference, line_range, pattern)
    B->>EX: _exhausted("expand_artifact", state) (middleware.py:1001)
    alt budget spent
        EX-->>B: the same refusal the other two get (middleware.py:1094)
    else
        B->>B: state.retrieval_cycles += 1 (middleware.py:1005)
        B->>RA: resolve_artifact(store, None, reference, stash=self._stash) (middleware.py:1010)
        RA->>OWN: retrieve(reference) — always asked, always first (store.py:306)
        alt the own store holds it
            OWN-->>RA: block -> outcome "text" or "non_textual" (store.py:308)
        else no stash was given
            RA-->>B: outcome "absent" (store.py:310-313)
        else
            RA->>STASH: retrieve(reference) (store.py:315)
            STASH-->>RA: text -> "text", or nothing -> "unknown" (store.py:316-319)
        end
        alt outcome is not "text"
            B-->>TW: absent_message / unknown_message / non_textual_message (middleware.py:1011-1016)
        else no line_range and no pattern
            B->>B: _whole_artifact — the text verbatim behind the cost notice (middleware.py:1019 -> 1194)
        else
            B->>B: _span_of(line_range) (middleware.py:1021 -> 1209)
            alt a line_range that is not a pair of integers
                B-->>TW: named back, with the shape to pass instead (middleware.py:1023-1026)
            else
                B->>RD: read_artifact(text, line_range=span, pattern=pattern) (middleware.py:1028)
                Note over RD: delegated · this module opens no file, resolves no path, builds no URI (middleware.py:979-981)
                RD-->>B: the requested part, or a ValueError named back with the reference (middleware.py:1029-1030)
            end
        end
        B->>SC: record_reuse on the artifact Card's title, when the graph holds one (middleware.py:1032-1034)
        Note over B,SC: _artifact_title (middleware.py:1222) · a reference the graph never carded<br/>is still readable, so the note simply has no Card to land on (middleware.py:1225-1227)
    end
    B-->>TW: answer text
    TW->>A: _answer(runtime, state, answer) (middleware.py:857 / 869)
    A-->>Model: Command(update={"context_graph": …, "messages": [ToolMessage(answer, tool_call_id)]})
```

Three properties, each stated in the code:

- **No Resolution changes, success included** (`middleware.py:983`–`984`). The content asked for is in the answer
  itself, and what crosses into the next turn is the fed-back note on the artifact's Card. A miss records nothing at
  all — `tests/test_artifacts.py:274`–`275` asserts `graph.reuse == {}` after one.
- **The read is delegated, never reimplemented.** `resolve_artifact` consults the stores and `read_artifact` bounds a
  targeted read; a range outside the content therefore carries the core helper's own refusal with only the reference
  prefixed (`tests/test_artifacts.py:248`–`255`). The registration at `middleware.py:112` is what makes the targeted
  read possible at all: without `"search_content"` every `line_range`/`pattern` request degrades to the prose
  "targeted reads are unavailable" (`middleware.py:103`–`105`).
- **A whole read states its own cost.** `_whole_artifact` (`middleware.py:1194`) returns the notice and then the text,
  separated by a blank line (`middleware.py:1206`), the text "untouched -- no truncation, no reformatting -- so the
  answer contains it character for character" (`middleware.py:1198`–`1199`). Asserted by `endswith` at
  `tests/test_artifacts.py:209`.

---

## 8. Verbatim text the model sees

**System prompt:** the middleware writes **nothing** to it. There is no system-prompt hook in the class; the only
text placed in front of the model is the folded `<collapsed_turns>` block on the last user message, plus the three
tool schemas and their return values.

### 8.1 The three tool descriptions — copied exactly

For the two `@tool`-decorated tools the decorator derives the description from the docstring and the schema from the
typed signature. Signatures: `expand_card(titles: list[str], runtime: ToolRuntime)` (`middleware.py:794`) and
`find_context(need: str, runtime: ToolRuntime, tag: str | None = None)` (`middleware.py:815`). `runtime` is
framework-injected and is **not** documented in the `Args:` block, unlike the Strands version which documented
`tool_context` as "Injected by the framework. Not user-facing."

`expand_card` (`middleware.py:795`–`810`):

```text
Bring back the full content of one or more earlier turns, by their titles.

Earlier turns may reach you as a title and a short description instead of their messages. When
a description is not enough to answer, call this with the titles exactly as they were shown and
those turns arrive in full for the rest of this turn.

Ask for every turn you need in ONE call: a list of titles costs one retrieval where the same
titles one at a time cost one each, and each extra call grows the conversation you are about to
reason over.

Args:
    titles: Titles of the turns you want back, copied as they were shown to you.

Returns:
    Confirmation that the turns will arrive in full, or an error naming the title asked for.
```

`find_context` (`middleware.py:816`–`827`):

```text
Find earlier turns of this conversation that match what you need, described in your words.

Use this when you suspect the conversation already covered something but you cannot see it in
what reached you. Describe the need, not a title.

Args:
    need: What you are looking for, in your own words.
    tag: Restrict the search to turns carrying this tag.

Returns:
    Up to five candidate turns with their title, tags and description, or an empty result
    naming the need received.
```

`expand_artifact` is the exception: its description is the module constant `_EXPAND_ARTIFACT_DESCRIPTION`
(`middleware.py:166`–`187`), passed to `StructuredTool.from_function` as `description=` (`middleware.py:875`) rather
than derived from a docstring. The reason is that the tool carries two bodies, "and the text the model sees must not
depend on which of them a run reaches" (`middleware.py:191`–`192`); the two body docstrings say as much and are read by
nobody (`middleware.py:853`, `middleware.py:865`). It is the Strands plugin's docstring verbatim with `tool_context`
renamed to `runtime` (`middleware.py:188`–`189`), asserted at `tests/test_artifacts.py:347`–`358`, which also asserts
the string `tool_context` does not survive anywhere in it:

```text
Read a stored artifact that an earlier turn of THIS conversation referred to by address.

Use this for a reference that appeared in the conversation as a placeholder standing in for content
too large to keep -- an image, a document, an export. The reference is the address that placeholder
carried.

If another tool told you it had replaced a tool result with a preview and handed you a reference,
that reference belongs to that tool, not to this one: use the tool that minted it. This one resolves
only addresses this plugin recorded, and answers by naming the miss when handed any other.

Prefer a line range or a pattern: without either, the whole artifact comes back and costs its
full token count again.

Args:
    reference: The artifact reference, copied as it was shown to you.
    runtime: Injected by the framework. Not user-facing.
    line_range: ``{"start": int, "end": int}`` to read only those lines.
    pattern: Return only the lines matching this pattern.

Returns:
    The requested part of the artifact, or an error naming what was missing.
```

Note the third paragraph against §3.4: the text tells the model that a reference another tool minted "belongs to that
tool", which is the behaviour with no `stash` wired. With the relevance filter's stash passed, such a reference
resolves here too — the description is the conservative instruction, not a statement about the stash.

`_build_tools` returns `[expand_card, expand_artifact, find_context]` in that order (`middleware.py:832`–`836`), the
artifact tool appended between the two that read the conversation's turns so the set a model is shown matches the
Strands binding's order (`tests/test_middleware.py:371`–`376`). With `include_artifact_tool=False` the list is the
other two (`tests/test_artifacts.py:361`–`365`).

### 8.2 A rendered Card — the literal shape

A subject Card's Description is assembled by `compose_description` (`describe.py:189`) as a header
(`_subject_header`, `describe.py:401`) followed by the Card's numeric lines copied verbatim
(`describe.py:220`–`222`). The header is the Title, then a tools line when there are tools, then a references line
when there are references — an absent line is left out rather than rendered empty, because "``tools:`` with nothing
after it spends budget to say nothing" (`describe.py:407`–`408`, code at `describe.py:413`–`419`):

```text
Where does the S000 allocation land
tools: query_ledger (2), fetch_segment (1)
references: ref-7f21
S000 allocation 12,400.00 reais
segment split 61.0% / 39.0%
(+3 numeric lines omitted)
```

The `tools:` line is `"tools: " + ", ".join(f"{name} ({count})" …)` (`describe.py:415`), the `references:` line is
`"references: " + ", ".join(references)` (`describe.py:419`), and the omission line is
`_OMISSION = "(+{count} numeric lines omitted)"` (`describe.py:111`), emitted only when the budget cut lines
(`describe.py:235`).

The contract on that cut is `startswith`-verifiable: strip the omission line and what remains is a literal prefix of
the Description the same Card yields with an unbounded budget (`describe.py:207`–`209`). Only when the header alone
overruns does the cut fall back to sentence, then word, then character boundary — and with no ellipsis, "which would
cost the prefix property" (`describe.py:204`–`206`).

Inside the folded block, `_entry` (`compaction.py:196`) renders a Card as `- <title>` followed by its fragments
indented two spaces (`compaction.py:234`), with the Card's own title removed from the fragments because "The title
line already carries the title, so the description's own first line is a duplicate"
(`compaction.py:223`–`224`). Constants: `_ENTRY_PREFIX = "- "` (`compaction.py:84`),
`_FRAGMENT_INDENT = "  "` (`compaction.py:87`). So the same Card, at dialogue `description`:

```text
- Where does the S000 allocation land
  tools: query_ledger (2), fetch_segment (1)
  references: ref-7f21
  S000 allocation 12,400.00 reais
  segment split 61.0% / 39.0%
```

At dialogue `title` the entry is the `- <title>` line alone, no fragment — and the line is still emitted, because
"omitting it would drop the Card's address from the call" (`compaction.py:206`–`208`).

Evidence at `description` contributes its own fragments (`_evidence_fragments`, `compaction.py:253`): a
`tools: name (count), …` line (`compaction.py:276`), a `references: …` line (`compaction.py:279`), the numeric lines,
then an omission line if any were cut (`compaction.py:289`). A part that stayed contributes **nothing**, because a
part contributes only when all of its identities left (`_part_left`, `compaction.py:237`; `dropped` at
`compaction.py:129`).

### 8.3 The block markers and the guidance trailer

Markers: `_HEADER = "<collapsed_turns>"` (`compaction.py:49`), `_FOOTER = "</collapsed_turns>"`
(`compaction.py:53`). A marker and not a sentence, because "the block is appended to the user's own words, so the
model must be able to tell where its message ends and the graph's summary begins" (`compaction.py:50`–`51`).

Preamble (`compaction.py:56`):

```text
The turns above left this call in collapsed form; their numeric lines are copied literally.
```

`_RETRIEVAL_PHRASES` (`compaction.py:59`–`63`) holds three clauses, keyed by registered tool name:

```text
expand_card: call expand_card with a title to get that turn's messages back
expand_artifact: call expand_artifact with a reference to read an artifact
find_context: call find_context with what you need to search the turns by description
```

**All three clauses are reachable in this binding, and by default all three are emitted.** `guidance` filters on
`name in retrieval_tools` (`compaction.py:167`), and `retrieval_tools` is
`tuple(each.name for each in self.tools)` (`middleware.py:429`) — `("expand_card", "expand_artifact",
"find_context")` with the default configuration, and `("expand_card", "find_context")` with
`include_artifact_tool=False`. The mechanism matters and is not incidental: the mapping is keyed rather than
concatenated because "a tool can be de-registered after this plugin is built, and a guidance block that names a tool
the agent does not have sends the model after something it cannot call", with the measured cost recorded — "with the
artifact tool removed to avoid a two-store collision, the guidance still advertised it, and the model spent a whole
turn alternating between the remaining paths — 31 tool calls, no answer" (`compaction.py:64`–`70`). Here that filter
is what makes `include_artifact_tool` a complete switch with no second one beside it: excluding the tool stops it being
advertised in the same movement (`middleware.py:842`–`844`, asserted both ways at
`tests/test_artifacts.py:374`–`380`).

Assembly (`compaction.py:170`–`176`): one clause → `f"{_PREAMBLE} To close the gap, {clauses}."`; several → joined
with `", "` and a final `", or "`. With all three tools registered, the literal trailer is therefore:

```text
The turns above left this call in collapsed form; their numeric lines are copied literally. To close the gap, call expand_card with a title to get that turn's messages back, call expand_artifact with a reference to read an artifact, or call find_context with what you need to search the turns by description.
```

and with `include_artifact_tool=False`:

```text
The turns above left this call in collapsed form; their numeric lines are copied literally. To close the gap, call expand_card with a title to get that turn's messages back, or call find_context with what you need to search the turns by description.
```

When no retrieval tool is registered, `_NOTHING_TO_CALL` (`compaction.py:72`) is used instead:

```text
No retrieval tool is registered, so the summary above is all that is available -- answer from it and say what it does not cover.
```

Naming the dead end is deliberate: "Silence would read as 'the evidence is somewhere', which is the state that
produces an unbounded hunt" (`compaction.py:76`–`78`).

`_SEARCHABLE` (`compaction.py:79`) announces a selection gap:

```text
{count} earlier turn(s) of this conversation are not shown above.
```

rendered by `_trailer` as `_SEARCHABLE.format(count=…) + " " + guidance(...)` (`compaction.py:193`). Note that
`TurnChoice.selected` is `None` on every choice this binding produces — `full_pass_choice` (`scoring.py:152`) and
`distribute` (`scoring.py:407`) both leave it defaulted (`state.py:169`) — so `unaddressed` is `0`
(`compaction.py:147`) and `_trailer` returns the guidance alone (`compaction.py:190`–`191`), so this line is never
rendered. The whole block is joined as
`"\n".join((*body, "", trailer)).lstrip("\n")` (`compaction.py:154`).

### 8.4 `find_context`'s answer

First line (`middleware.py:1279`):

```text
find_context | {len(chosen)} earlier turn(s) match '{need}', best first:
```

Per candidate: `- title: {title}` (`middleware.py:1285`), an optional `  tags: {', '.join(card.tags)}`
(`middleware.py:1287`), the Description lines indented two spaces with blank lines dropped
(`middleware.py:1288`–`1290`), then the neighbour line when there is one (`middleware.py:1294`):

```text
  related turns: {title} ({weight:.2f}), {title} ({weight:.2f})
```

Two spaces, the literal words `related turns: `, then `", "`-joined `TITLE (0.71)` pairs, strongest first
(ordering at `middleware.py:1267`), weights to exactly two decimals. One candidate renders as:

```text
- title: Reconcile the segment totals
  tags: allocation, segment, s000
  tools: query_ledger (2)
  references: ref-7f21
  related turns: Where does the S000 allocation land (0.71), Segment split for Q3 (0.64)
```

Closing line (`middleware.py:1295`):

```text
call expand_card with one of these titles to bring that turn back in full
```

Empty result (`_nothing_found`, `middleware.py:1236`, assembled `middleware.py:1243`–`1247`), with `{narrowed}` =
`", among the turns tagged '{tag}'"` when a tag was given, else empty (`middleware.py:1242`):

```text
find_context | nothing in this conversation matches '{need}'{narrowed} | the titles already in front of you are the whole conversation, so what you need was either never discussed or is in a turn you can name directly with expand_card
```

Naming both is deliberate: it "lets the model tell 'nothing in this conversation is about that' from 'nothing
carrying that tag is about that', and only the second has an obvious next move" (`middleware.py:1239`–`1240`).
Reached on three paths — a blank `need` (`middleware.py:1062`), an unusable matcher (`middleware.py:1072`), and nothing
clearing the floor (`middleware.py:1079`–`1080`).

### 8.5 `expand_card`'s answers

No title given (`middleware.py:928`):

```text
expand_card | no title given | pass the titles you need, copied exactly as they were shown to you
```

No match (`middleware.py:955`–`958`):

```text
expand_card | no earlier turn of this conversation is titled {_quoted(missing)} | copy a title exactly as it was shown to you, or use find_context to describe what you need
```

Success (`middleware.py:960`–`963`):

```text
expand_card | {_quoted(found)} arrives in full for the rest of this turn, its messages and its tool results together
```

with, when some titles missed (`middleware.py:965`): ` | no turn is titled {_quoted(missing)}, so nothing was raised
for it`. `_quoted` wraps each title in single quotes, comma-separated (`middleware.py:1117`–`1119`). The partial batch
is checked at `tests/test_middleware.py:415`.

### 8.6 `expand_artifact`'s answers

A whole read is the cost notice, a blank line, then the artifact verbatim (`_whole_artifact`, `middleware.py:1201`–
`1193`), the token figure coming from the core's `estimate_tokens` (`store.py:375`):

```text
expand_artifact | whole artifact '{reference}' | this call re-injects the artifact's entire token count, about {N} tokens, and it stays in the conversation for the rest of the turn | next time pass line_range or pattern to read only the part you need
```

A targeted read is the helper's output with no notice at all (`middleware.py:1028`). The three miss messages are the
core's, shared with the Strands binding and worded there. Nothing holds the reference and no second layer answered —
`absent_message` (`store.py:387`, returned at `middleware.py:1012`), which is the message every miss gets when no
`stash` is wired (§3.4):

```text
expand_artifact | no artifact storage holds reference '{reference}' on this agent | nothing was ever offloaded under that reference, which means the full results are already in the conversation
```

A second layer was asked and did not hold it — `unknown_message` (`store.py:405`, at `middleware.py:1014`):

```text
expand_artifact | unknown reference '{reference}' | copy a reference exactly as it was shown to you in a turn's title or preview
```

The block resolved but yields no text — `non_textual_message` (`store.py:420`, at `middleware.py:1016`), deliberately
**without** naming a media type, because a decoded block carries none and inventing one would be a guess the model
would repeat (`store.py:421`–`424`; `tests/test_artifacts.py:278`–`288`):

```text
expand_artifact | reference '{reference}' holds non-textual content | line_range and pattern do not apply to it, and it cannot be returned as text
```

Two are the binding's own. A `line_range` that is not a pair of integers (`middleware.py:1023`–`1026`), the Strands
plugin's wording character for character (`tests/test_artifacts.py:235`–`245`):

```text
expand_artifact | line_range=<{line_range!r}> is not a pair of integers | pass {"start": <int>, "end": <int>}, 1-indexed and inclusive
```

And a `ValueError` out of the read helper, named back with the reference prefixed and the helper's own sentence kept
(`middleware.py:1030`; `tests/test_artifacts.py:248`–`255`):

```text
expand_artifact | reference '{reference}' | {error}
```

### 8.7 The retrieval-budget refusal

`_EXHAUSTED` (`middleware.py:156`–`159`):

```text
{tool} | this turn has already spent its {spent} retrieval calls | no further recovery is available on this turn: answer from what the summary and the messages already give you, and state plainly which part you could not verify
```

Worded as an instruction rather than an error "because the failure mode it exists to stop is a model that keeps
asking: every retrieval miss on these tools answers with text, so 'not found' reads as 'try differently'"
(`middleware.py:160`–`164`). `{tool}` is the calling tool's own name, so the same sentence covers all three
(`middleware.py:1094`).

### 8.8 The one string that does NOT reach the model

`_PRUNING_WARNING` (`middleware.py:144`–`149`) goes to the **developer** through `warnings.warn`
(`middleware.py:767`), not to the model:

```text
context_graph=<pruning_middleware> | middleware=<{middleware}> prunes or summarizes the persisted message list | a middleware that rewrites state['messages'] removes messages this middleware only meant to fold, so raising a Card's Resolution back up then recovers nothing | drop it, or accept that a collapsed turn may be unrecoverable
```

It is the LangGraph analog of the Strands `NullConversationManager` precondition, "matched by name because a
middleware that returns ``RemoveMessages`` declares nothing this could be checked by" — markers
`("summariz", "summaris", "prun", "trim", "compact")` (`middleware.py:136`), matched case-folded against a candidate's
`name` or class name (`_name_of`, `middleware.py:771`; match at `middleware.py:766`).

`ContextEditingMiddleware` is **deliberately absent** from the markers, "because it edits the call through
``override`` like this one, so it removes nothing from ``state["messages"]`` and breaks no recovery"
(`middleware.py:140`–`142`).

Three properties of the notice: at most one per construction, naming the first offender (`middleware.py:768` returns);
the middleware never warns about itself (`middleware.py:763`, checked at `tests/test_middleware.py:579`); and passing
no list warns about nothing — "silence here is 'the wiring was not described', not 'the wiring is safe'"
(`middleware.py:759`–`760`, checked at `tests/test_middleware.py:570`). Fired exactly once:
`tests/test_middleware.py:554`.

---

## 9. Configuration — every constructor parameter

`__init__` (`middleware.py:344`). Every check runs before the first attribute is assigned, "so a ``ValueError``
leaves an instance that was never handed to an agent", and nothing is built beyond plain attributes and the tool
objects — "no network call, no model client, no matcher, and no reference store -- the store of a conversation is
created the first time a tool call is seen on it" (`middleware.py:365`–`368`).

| Parameter | Default (source) | Validation | Accepts `None`? |
|---|---|---|---|
| `expand_threshold` | `0.55` (`middleware.py:120`) | `_validate_ratio` (`middleware.py:220`, `:370`) — finite real in `[0.0, 1.0]`, `bool` rejected explicitly | No |
| `collapse_floor` | `0.45` (`middleware.py:121`) | ratio (`:371`), plus `collapse_floor <= expand_threshold` (`:376`) | No |
| `description_tokens` | `100` (`middleware.py:122`) | `_validate_count` (`middleware.py:238`, `:381`) — int ≥ 1 | No |
| `tags_per_card` | `5` (`middleware.py:123`) | int ≥ 1 (`:382`) | No |
| `neighbors_per_candidate` | `3` (`middleware.py:124`) | int ≥ **0** (`:383`, `floor=0`) — `0` must be admissible so the pre-neighbour answer is reproducible | No |
| `body_budget` | `None` (`middleware.py:125`) | `_validate_optional_count` (`middleware.py:248`, `:386`) | **Yes** — and `None` disables the step down (§6.3) |
| `min_cards` | `3` (`middleware.py:126`) | int ≥ 1 (`:385`) | No |
| `link_threshold` | `0.50` (`middleware.py:127`) | ratio (`:372`) — governs which `similar` edges exist at all | No |
| `reuse_ttl_cycles` | `5` (`middleware.py:128`) | int ≥ **0** (`:384`) — `0` writes no fed-back note at all (`scoring.py:113`), see §7.3 | No |
| `max_retrieval_cycles` | `8` (`middleware.py:129`) | optional count (`:387`) | **Yes** — unbounded (`middleware.py:1092`) |
| `rarity_weight` | `0.70` (`middleware.py:130`) | ratio (`:373`) | No |
| `include_artifact_tool` | `True` (`middleware.py:358`) | `isinstance(…, bool)` (`:388`) — refused for anything else, including `None` (`tests/test_artifacts.py:368`–`371`) | No |
| `stash` | `None` (`middleware.py:359`) | callable `retrieve` member (`:390`) — checked by member for the same reason the matcher is (`tests/test_artifacts.py:532`–`536`) | **Yes** — one resolution layer instead of two (§3.4) |
| `matcher` | `None` → `EmbeddingSimilarityMatcher` on first need (`middleware.py:736`, built `:749`) | `_validate_matcher` (`middleware.py:262`, `:392`) — checked by **member**, not `isinstance`: "the matcher contract is structural… which is what lets a test pass a mock and reach no network" (`middleware.py:265`–`266`) | **Yes** |
| `middleware` | `None` | none; only read by `_warn_on_pruning_middleware` (`:419`) | **Yes** |

One absence against the Strands plugin's constructor remains real: there is **no `name`**. The Strands plugin published
`"strands:context-graph"`; here the middleware's identity is its class, which is also what `_name_of`
(`middleware.py:771`) falls back to when it inspects a candidate middleware.

`_validate_ratio` rejects `bool` explicitly (`middleware.py:230`), and the reason is stated: it "passes as a number in
Python, and ``True`` silently meaning ``1.0`` is configuration that looks like it works"
(`middleware.py:223`–`224`). Refusals are parameterized at `tests/test_middleware.py:591`–`611`.

The default matcher's knobs are the core's: `model_id = "cohere.embed-multilingual-v3"` (`matcher.py:37`),
`cache_size = 512` (`matcher.py:49`), timeout 10s (`matcher.py:56`), Cohere batch cap 96 (`matcher.py:46`). Its
`bedrock-runtime` client is built lazily on first `score` (`matcher.py:284`), so `EmbeddingSimilarityMatcher()` at
`middleware.py:749` still performs no I/O — "Its client is built lazily too, so this construction is still free of
I/O" (`middleware.py:748`). `tests/test_middleware.py:636` asserts the default is never built when one is supplied.

---

## 10. How the validation harness configures the graph

Authority: `validation/plugins-langgraph/src/`. Its `config.py` is "a verbatim copy of the Strands
harness's ``config.py``" apart from one paragraph (`config.py:20`), so the tuning argument is the same one; what
differs is the wiring in `runner.py`.

### 10.1 One tuning, not two

`GRAPH_TUNING` (`config.py:505`) is a single set for every arm the graph appears in. `build_middleware` reads it
unconditionally — `tuning = GRAPH_TUNING` (`runner.py:530`) — and records
`config.extra["_graph_tuning"] = "unified"` (`runner.py:531`). The ruling is a design argument, recorded in
`GraphTuning`'s docstring (`config.py:437`–`453`) and restated at the call site (`runner.py:526`–`529`): the relevance
filter acts on a tool result *before* it enters the history, the graph acts at delivery on a history that *already
exists*, so the filter makes the graph's input smaller rather than different in kind, and a knob deciding how
aggressively to fold a history has no business reading whether another plugin trimmed it first.

| Knob | Value | Env override | Note |
|---|---|---|---|
| `expand_threshold` | 0.62 | `VALIDATION_GRAPH_EXPAND` (`config.py:506`) | above the package's 0.55 — moves mass off Full Content onto a richer Description |
| `collapse_floor` | 0.45 | `VALIDATION_GRAPH_COLLAPSE` (`config.py:507`) | package default |
| `link_threshold` | 0.50 | `VALIDATION_GRAPH_LINK` (`config.py:508`) | package default; decides which `similar` edges exist |
| `description_tokens` | 250 tight / 100 large | `VALIDATION_GRAPH_DESCRIPTION_TOKENS` (`config.py:509`) | from the regime: `TIGHT_WINDOW` (`config.py:328`, `:330`) vs `LARGE_WINDOW` (`config.py:305`, `:307`), selected into `BUDGETS` (`config.py:363`) |
| `body_budget` | **40000** (package default `None`) | `VALIDATION_GRAPH_BODY_BUDGET` (`config.py:510`), `=none` restores the measured configuration (`config.py:552`) | this is what makes the step down (`scoring.py:386`) an exercised path rather than dead code |
| `max_retrieval_cycles` | 4 tight / 8 large | `VALIDATION_GRAPH_MAX_RETRIEVAL_CYCLES` (`config.py:511`) | `config.py:331` / `config.py:308` |
| `reuse_ttl_cycles` | 5 | `VALIDATION_GRAPH_REUSE_TTL` (`config.py:512`) | package default, restated so a sweep can reach it |
| `tags_per_card` | 5 | `VALIDATION_GRAPH_TAGS` (`config.py:513`) | what `find_context` filters on |
| `neighbors_per_candidate` | **0** (package default 3) | `VALIDATION_GRAPH_NEIGHBORS` (`config.py:514`), `=3` turns the edge on for a sweep (`config.py:502`) | held at `0` so figures stay comparable with every published one, all produced with the edge unread |

These reach the middleware one-for-one at `runner.py:533`–`543`, with `min_cards` coming from `THRESHOLDS`
(`config.py:424`) rather than from the graph tuning, and the matcher being a metered subclass
(`_MeteredMatcher`, `runner.py:523`; embedding model `EMBED_MODEL_ID` at `config.py:207`).

### 10.2 Three things the harness has to do that the Strands harness did not

1. **Put the graph's state *classes* on the checkpointer's allowlist — the classes, not the module name.** The graph
   travels in checkpointed state as `context_core.graph.state` dataclasses — `_GraphState`, `Card`, `Link`, `ToolPair`,
   `TurnChoice` — and LangGraph's msgpack serde warns on deserializing a type it was not told about, "stating plainly
   that it 'will be blocked in a future version'" (`runner.py:225`–`230`). The allowlist belongs to whoever constructs
   the saver, which is the harness and not the middleware package (`runner.py:232`–`233`).

   The shape of what is passed is load-bearing. The serde's allowlist is keyed by `(module, class name)`, so a bare
   module name matches nothing — it makes the allowlist strict **and** empty, and every graph type is then blocked on
   restore (`runner.py:240`–`243`). `_graph_state_types` (`runner.py:237`) therefore collects the classes the module
   defines, by `__module__` identity (`runner.py:249`–`253`), and hands those to the serde
   (`runner.py:271`). `_GRAPH_STATE_MODULES` (`runner.py:222`) remains as the named subject of that docstring.

   What a blocked type costs is the same silence as §5.2 reached from the other end: a dropped type does not raise, it
   is dropped with a log warning, so the graph arm starts every turn from an empty graph and a lost graph looks exactly
   like a working one. The regression test therefore round-trips a `GraphState` through the harness's own serde and
   asserts both halves — no `Blocked deserialization` warning, and the restored value being a `GraphState` carrying its
   turn ordinal (`tests/test_checkpointer.py:20`–`32`). Note what it mirrors at `tests/test_checkpointer.py:24`: the
   state it round-trips carries a **flattened** choice, because that is what `_persistable` puts in the state (§4.2).
2. **Register `expand_artifact` explicitly and hand the graph the filter's stash.** `include_artifact_tool=True` is
   passed rather than left to the default, "as in the Strands harness (runner.py there passes True explicitly)"
   (`runner.py:545`–`546`), and `stash=relevance.stash if relevance is not None else None` (`runner.py:550`) supplies
   the second resolution layer, "so a [ref: mem_N_...] the filter minted resolves through expand_artifact too. Strands
   gets this from the ContextManager Stash when one is installed; LangGraph has none, so it is wired explicitly"
   (`runner.py:547`–`549`). Both facts are recorded on the run:
   `config.extra["_graph_artifact_tool_dropped"] = False` (`runner.py:553`), which keeps the key the Strands JSON
   carries present with its true value, and `config.extra["_graph_stash"] = relevance is not None`
   (`runner.py:555`). The filter's own retrieval tool stays on in every arm —
   `include_retrieval_tool = RELEVANCE_RETRIEVAL_TOOL` (`runner.py:505`), recorded at `runner.py:506` — because with
   both stores readable through one resolution order the two tools are not two paths over two stores
   (`runner.py:502`–`504`). `tests/test_composition.py:81`–`89` asserts the combined arm carries both.
3. **Nothing for the sync/async gap: it is closed in the package.** The graph arm wires the real
   `ContextGraphMiddleware`, whose native `awrap_model_call` (`middleware.py:480`) lets the combined stack run under
   `ainvoke`. The harness subclass that once bridged the sync hook onto a worker thread is gone, and the note left in
   its place records why (`runner.py:277`–`283`) — see §2.1.

Ordering is a list rather than a handler index: `build_middleware` returns the middleware "outermost first"
(`runner.py:463`, `:574`–`575`), and the `all` arm reads graph, disclosure, relevance — "the graph folds the history
first, disclosure then folds the exchanges of every tool the call does not carry, and relevance acts on a tool result
before either of them sees it" (`runner.py:468`–`472`). That is the same nesting the Strands stack reached by moving
its own handler to index zero, and `tests/test_composition.py:44`–`50` builds it that way.

### 10.3 How to read the numbers in `config.py`

Read every quantity in those tuning docstrings as a **direction, not a settled number** — one replay each. The file
says so itself, and the comparison it reports has a baseline row that drifted with no plugin installed at all, which
is the floor below which nothing there is attributable. The `body_budget` change in particular is justified by a peak
call moving 49,943 → 81,065 tokens "while the graph's resolution ladder barely moved (full 20/9/4 → 19/8/7)"
(`config.py:317`–`318`) — i.e. every rung was carrying more, not a different rung being chosen.
