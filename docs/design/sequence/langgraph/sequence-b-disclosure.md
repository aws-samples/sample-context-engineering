# Sequence B (LangGraph) — Progressive Tool Disclosure

All `file.py:LINE` references resolve into two roots. The **binding** —
`langgraph-plugins/langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure/` —
is cited as `middleware.py`, `_adapter.py` and `_compat.py`. The **core** —
`context-core/src/context_core/` — is cited as `catalog.py` and `tool_index.py` (both under
`disclosure/`) and `message.py`. LangChain and LangGraph symbols are named in prose, because they sit
outside both roots. Every claim below carries a line reference, and every literal string is quoted
verbatim from source.

This is the LangGraph binding of the same practice the Strands document
[`sequence-b-progressive-tool-disclosure.md`](../sequence-b-progressive-tool-disclosure.md) describes.
The decision logic is shared and lives in the core; what differs is everything that touches a framework,
and the differences are structural rather than cosmetic — graph state instead of a per-agent
`WeakKeyDictionary`, a cycle counter derived from the messages instead of read off an event-loop metric,
`request.override` instead of a `replace` on a middleware context, and a wrapped tool call instead of a
`BeforeToolCallEvent` hook.

## 1. What the middleware does, mechanically

On every model call the middleware rewrites **three** fields of the request, all through one
`request.override` (`middleware.py:689`).

`tools` is reduced to the entries callable on this call: the two disclosure tools `find_tools` and
`get_tool_details`, the `always_available` names, and the tools whose schema was **loaded** by a prior
`get_tool_details` and is still live under a TTL measured in cycles (`_active_names`
`middleware.py:308`, applied by `_keep_active` `middleware.py:377`). Every entry kept is the caller's own
object, verbatim — there is no reduced form of a tool in the list, only presence or absence
(`_keep_active` docstring `middleware.py:382`). Every other bound tool reaches the model as **one line of
a catalog appended to the system message** — its name and a summary of its description, at most
`catalog_chars` characters (`build_catalog` `catalog.py:376`, line built at `catalog.py:368`).

`system_message` receives that catalog block, appended as one more text block on a **new**
`SystemMessage` (`_with_catalog` `middleware.py:352`, built at `middleware.py:374`). The override is set
only when the block is non-empty (`middleware.py:679`).

`messages` receives a folded copy: every **closed** exchange of a tool this call does not carry loses its
`toolUse` block, and its `toolResult` becomes one plain sentence — `The tool X was called and the result
was: Y` (`fold_closed_exchanges` `catalog.py:508`, `fold_note` `catalog.py:423`, driven from
`_fold_messages` `middleware.py:400`). The two disclosure tools' own exchanges are dropped outright, with
no sentence (`catalog.py:569`). The graph's own message history is never mutated — the fold produces a
new list for this call only (`middleware.py:424`).

**A loaded tool is released, not retained.** Nothing keeps a tool callable because the messages still
mention it: `_active_names` (`middleware.py:308`) is computed from the load map and the TTL alone, and a
`toolUse` left in the history for a tool absent from `tools` is what the fold exists to remove rather
than what keeps the tool resident (`catalog.py:516`).

There is **one placement** and no mode flag. Nothing reduced is ever put in `tools`, so nothing the
provider is shown asserts an empty parameter schema: a catalog name is not in `tools` at all, and the
rule that governs it arrives in the same block as the name (`CATALOG_PROMPT_HEADER` `catalog.py:83`). The
active tool list and the catalog **partition** the bound set between them — the block is built by
skipping exactly the active names, so no tool appears in both and none is missing from both
(`catalog_prompt_block` `catalog.py:365`, docstring `catalog.py:348`).

**The common path is catalog → `get_tool_details([names])` → call.** The model reads a name in the system
message, calls `get_tool_details` with the names it wants, which records them in `loaded_tools` through a
`Command` state update (`_load` `middleware.py:893`, `Command` at `middleware.py:943`) and answers with a
short confirmation; the full schemas arrive on the *next* model call. `find_tools` is the **fallback**
for a need no catalog name fits: it ranks the bound specifications through the `LexicalToolIndex`
(`tool_index.py:177`) term-frequency `search` (`tool_index.py:214`) and lists names plus summaries, and
it **writes no state** — loading is `get_tool_details`' one job, so the model takes the same road to a
schema wherever it started (`_search` `middleware.py:857`, docstring `middleware.py:860`).

The guard lives in the tool-call seam rather than in a hook. `wrap_tool_call` (`middleware.py:712`)
cancels a call to a tool the model could not see when that tool requires parameters, and loads
**nothing** on the model's behalf (`_cancellation` `middleware.py:749`, cancellation built at
`middleware.py:790`).

Every tool stays bound to the agent and stays callable throughout. Release withdraws a schema from the
next call's tool list, it does not unbind anything (class docstring `middleware.py:525`).

## 2. The three bands

The binding is three layers with one rule each: the **interface** band touches LangChain and LangGraph
and nothing else; the **adapter** band is the only module that sees both worlds; the **core** band
imports no framework at all (`catalog.py:23`, `_adapter.py:4`).

```mermaid
flowchart TB
    subgraph band1["Band 1 · interface — LangChain and LangGraph only"]
        direction TB
        WMC["wrap_model_call · awrap_model_call<br/>middleware.py:612 · 636"]
        RW["_rewrite · request.override of tools, messages, system_message<br/>middleware.py:648 · 689"]
        KEEP["_keep_active · caller objects verbatim, presence or absence<br/>middleware.py:377"]
        SYS["_with_catalog · a replacement SystemMessage<br/>middleware.py:352 · 374"]
        TOOLS["find_tools and get_tool_details as closures · self.tools<br/>middleware.py:815 · 837 · 605"]
        WTC["wrap_tool_call · awrap_tool_call · _cancellation<br/>middleware.py:712 · 740 · 749"]
        ST["loaded_tools in graph state · _merge_loads reducer<br/>middleware.py:156 · 124 · state_schema 553"]
    end
    subgraph band2["Band 2 · adapter — the only module touching both worlds"]
        direction TB
        TN["to_neutral_list · BaseMessage to neutral dicts<br/>_adapter.py:131"]
        TL["to_langchain · one neutral message to a LIST of messages<br/>_adapter.py:153"]
    end
    subgraph band3["Band 3 · core — context_core.disclosure, no framework import"]
        direction TB
        BC["build_catalog · the rule plus one line per hidden tool<br/>catalog.py:376"]
        FCE["fold_closed_exchanges · closed exchanges become sentences<br/>catalog.py:508"]
        LTI["LexicalToolIndex · build and search, term frequency<br/>tool_index.py:197 · 214"]
    end
    WMC --> RW
    RW --> KEEP
    RW --> BC
    BC --> SYS
    RW -->|"_fold_messages middleware.py:400"| TN
    TN --> FCE
    FCE -->|"rewritten messages only · unchanged ones keep identity 423"| TL
    TL --> RW
    RW -.->|"reads loaded_tools middleware.py:669"| ST
    RW -->|"_ensure_index middleware.py:665 · 691"| LTI
    TOOLS -->|"_search middleware.py:857 · 878"| LTI
    TOOLS -->|"_load middleware.py:893 · Command update 943"| ST
    WTC -.->|"recomputes the active set for the chosen cycle 776 · 779"| ST
```

Two facts the picture is worth reading for. First, **only `get_tool_details` writes state** — the arrow
into `loaded_tools` has exactly one source, and both the rewrite and the guard are readers
(`middleware.py:669`, `middleware.py:777`). Second, the adapter's return path is **not** symmetric with
its outbound path: `to_neutral_list` converts every message, but only the messages the core actually
rewrote go back through `to_langchain`, because the unchanged ones are mapped back to the original
`BaseMessage` by object identity (`middleware.py:423`, `middleware.py:427`).

`to_langchain_list` (`_adapter.py:221`) exists and is exported (`_adapter.py:37`), but the fold path does
not use it: it would rebuild every message, which is exactly what the identity map exists to avoid. The
middleware imports the singular `to_langchain` and `to_neutral_list` only (`middleware.py:57`).

## 3. Integration table — every LangChain and LangGraph attachment point

| # | Framework seam | `file.py:LINE` | Callback / order | Reads | Mutates |
|---|----------------|----------------|------------------|-------|---------|
| 1 | `AgentMiddleware` base class (subclassed) | `middleware.py:522` `class ProgressiveToolDisclosureMiddleware(AgentMiddleware)` | The base class is reached only through `_compat.py`, which imports `AgentMiddleware`, `AgentState`, `ModelRequest` and `ModelResponse` from `langchain.agents.middleware` (`_compat.py:8`) | — | — |
| 2 | `state_schema` class attribute | `middleware.py:553` | Read by `create_agent` when it builds the graph, so `loaded_tools` becomes part of the thread's checkpointed state | — | Extends the agent state with `DisclosureState` (`middleware.py:148`) |
| 3 | `tools` instance attribute | `middleware.py:605`, set before `super().__init__()` (`middleware.py:606`) | The middleware contributes its two tools to the agent's bound set | — | Adds `find_tools` and `get_tool_details` to the agent |
| 4 | `wrap_model_call` | `middleware.py:612`, rewrite in `_rewrite` (`middleware.py:648`) | Wraps the model call. No explicit order — the position in the chain is the order the caller lists the middleware in | `request.tools`, `request.messages`, `request.state`, `request.system_message` | Returns `request.override(**overrides)` (`middleware.py:689`) carrying new `tools` and `messages`, plus `system_message` when the catalog is non-empty (`middleware.py:679`). Mutates the instance's index and summary cache through `_ensure_index` (`middleware.py:691`) and `build_catalog` (`middleware.py:678`). Writes **no** graph state. |
| 5 | `awrap_model_call` | `middleware.py:636` | Async twin. The rewrite does no I/O, so the body is the same call to `_rewrite` (`middleware.py:643`) | Same as row 4 | Same as row 4 |
| 6 | `wrap_tool_call` | `middleware.py:712`, decision in `_cancellation` (`middleware.py:749`) | Wraps each tool call. Returns the cancellation instead of calling `handler` when the call was a guess (`middleware.py:737`) | `request.tool_call`, `request.runtime.tools`, `request.state["messages"]`, `request.state["loaded_tools"]`, `self._always_available` | Returns a `ToolMessage` with `status="error"` (`middleware.py:790`). Writes nothing — not even a counter. |
| 7 | `awrap_tool_call` | `middleware.py:740` | Async twin, same decision (`middleware.py:746`) | Same as row 6 | Same as row 6 |
| 8 | `@tool` closure `find_tools` | `middleware.py:815` decorator, `middleware.py:816` function | Built in `_build_tools` (`middleware.py:804`) and handed to the agent through `self.tools`. Receives a `ToolRuntime` injected by the framework | `runtime.tools` (`middleware.py:874`), the `need` argument, the index | **Nothing.** It returns a plain `str` (`middleware.py:891`), so no state update can travel with it. |
| 9 | `@tool` closure `get_tool_details` | `middleware.py:837` decorator, `middleware.py:838` function | Same construction as row 8. Returns a `Command`, which is how a LangGraph tool writes state | `runtime.tools`, `runtime.state["messages"]`, `runtime.tool_call_id` (`middleware.py:937`), the `names` argument | Returns `Command(update=...)` (`middleware.py:943`) carrying the answer `ToolMessage` (`middleware.py:935`) and, only when something was loaded (`middleware.py:941`), the `loaded_tools` delta. |
| 10 | `Annotated` state reducer | `middleware.py:156`, reducer `_merge_loads` (`middleware.py:124`) | Applied by LangGraph when it merges channel writes at the end of a superstep | The map already in state and the incoming update | Produces a new merged map, keeping the **later** cycle per name (`middleware.py:144`). Neither argument is mutated (`middleware.py:142`). |

Supporting framework imports (the surface used, not attachment points): `AIMessage`, `BaseMessage`,
`SystemMessage` and `ToolMessage` from `langchain_core.messages` (`middleware.py:37`), `BaseTool` and
`tool` from `langchain_core.tools` (`middleware.py:38`), `ToolRuntime` from `langchain.tools`
(`middleware.py:39`), `Command` from `langgraph.types` (`middleware.py:40`), and `ToolCallRequest` from
`langchain.tools.tool_node` via `_compat` (`_compat.py:9`). `_compat.py` exists so that an API move
touches one file, and it records the version it was verified against — `langchain` 1.4.2
(`_compat.py:3`).

**There is no per-agent state object and no `WeakKeyDictionary`.** A LangGraph agent is a graph and its
per-thread facts belong in its state so they survive a checkpoint (module docstring
`middleware.py:23`). The instance therefore holds configuration only: the four thresholds, the index, the
fingerprint and the summary cache (`middleware.py:593` through `middleware.py:604`). One consequence is
that the binding keeps **no counters** — no `searches`, `loads` or `premature_cancellations` field
exists; what the Strands reference counts, this binding logs (`middleware.py:888`, `middleware.py:934`,
`middleware.py:789`).

## 4. The active set — who is callable on this call

`_active_names` (`middleware.py:308`) is the whole membership decision, and it is a set union of three
terms, every one of them intersected with the names the agent actually has:

| Order | Class | Membership source (`file.py:LINE`) | What the model receives |
|-------|-------|-------------------------------------|--------------------------|
| 1 | `{find_tools, get_tool_details}` | `PLUGIN_TOOL_NAMES` (`catalog.py:66`), first term at `middleware.py:337` | The caller's own tool objects, verbatim — emitted unconditionally, which is what makes the tool list non-empty on every rewritten call |
| 2 | `always_available` | `self._always_available` tuple (`middleware.py:596`), second term at `middleware.py:338` | The caller's own objects on every call |
| 3 | live loads | `loaded_tools` from state (`middleware.py:669`), aged by `_last_used` (`middleware.py:269`), third term at `middleware.py:339` | The caller's own objects while the load is live under the TTL |
| 4 | catalog residue | every bound name **not** in the active set, skipped while rendering (`catalog.py:365`, `catalog.py:410`) | **Nothing in `tools`.** One line `- name: summary` in the system message (`catalog.py:368`), under the header at `catalog.py:372` |

The `catalog_names` intersection is not incidental: a name outside the bound set is ignored, so a tool
unbound since it was loaded cannot resurrect from a stale `loaded_tools` entry (docstring
`middleware.py:331`, applied at `middleware.py:337`, `middleware.py:338` and `middleware.py:342`).

**Passthrough is structural, not a flag.** `_rewrite` returns the request untouched when the disclosure
tools are not a subset of the bound names (`middleware.py:662`): without `get_tool_details` there is no
way to load a hidden schema and without `find_tools` no way to find one, so there is nothing to hide
(docstring `middleware.py:655`). That single subset test is also what covers the no-tools case, since an
empty bound set cannot contain either name.

Iteration order for the tool list comes from `bound`, the arrival order, not from the active set — two
calls with the same disclosure state then produce the same list and a provider's prompt cache is not
invalidated by a reordering alone (`_keep_active` docstring `middleware.py:380`, loop
`middleware.py:393`). The catalog is rendered over `tool_specs` in the same arrival order
(`catalog.py:363`, `catalog.py:408`).

`_spec_of` (`middleware.py:186`) is what lets both work over a heterogeneous list: `request.tools` is
`list[BaseTool | dict]`, so a `BaseTool` is read through its `tool_call_schema`
(`_schema_of` `middleware.py:164`, `middleware.py:201`) and a provider-native declaration is accepted in
either the OpenAI `{"type": "function", "function": {...}}` envelope or the bare form
(`middleware.py:205` through `middleware.py:207`). An entry with no name is skipped rather than guessed
at (`middleware.py:209`, filtered at `middleware.py:221`), and a schema that cannot be rendered yields an
empty object schema — which reads as "takes no arguments" and therefore **exempts** the tool from the
guard rather than cancelling a legitimate call (`middleware.py:183`, docstring `middleware.py:170`).

## 5. The cycle counter and the TTL — derived, not stored

This is the single biggest structural difference from the Strands binding, which reads
`agent.event_loop_metrics.cycle_count`.

**The cycle counter is not stored at all.** `_cycle` (`middleware.py:253`) is the number of `AIMessage`
objects in the call's messages (`middleware.py:266`). Each model call appends exactly one, so counting
them *is* the counter — which makes it a function of the state rather than a second thing to keep in step
with it, and nothing to lose on a checkpoint restore (module docstring `middleware.py:25`, function
docstring `middleware.py:256`).

That derivation forces an off-by-one discipline that is worth stating explicitly, because the two seams
sit on opposite sides of an `AIMessage`:

- At rewrite time, `cycle = _cycle(messages)` is the cycle the call **about to go out** will run on
  (`middleware.py:668`) — its `AIMessage` does not exist yet.
- At guard time and inside `_load`, the `AIMessage` carrying the call is **already** in state, so the
  cycle the model chose on is one behind: `max(_cycle(messages) - 1, 0)` (`middleware.py:776`, comment
  `middleware.py:774`; `middleware.py:913`, comment `middleware.py:912`).

**Renewal is read out of the history, not written when a tool runs.** `_last_used`
(`middleware.py:269`) walks the `AIMessage` objects in order, counting one cycle each
(`middleware.py:304`), and a tool called by the `AIMessage` of cycle *k* counts as used at *k*
(`middleware.py:303`). A tool used across a stretch of cycles therefore keeps renewing itself with **no
state write at all** — which is why there is no post-call hook in this binding at all, where the Strands
reference needs an `AfterToolCallEvent`.

### 5.1 The two renewal-bug fixes

The docstring names them as two restrictions, "each one exists because of a case that breaks without it"
(`middleware.py:276`). Both are one line of code and both are load-bearing.

**Fix 1 — a call is not a load** (`middleware.py:302`, `if name in last`). Only a tool that
`loaded_tools` already holds is renewed. Without the test, renewing on *any* call would let a name the
model guessed at install itself in the live set, which is exactly the shortcut the guard exists to refuse
(`middleware.py:278`). The map `last` is seeded from `loaded` (`middleware.py:293`) precisely so that
membership in it means "was loaded", and `loaded` itself is never mutated (`middleware.py:291`).

**Fix 2 — a use on the cycle being decided for does not count** (`middleware.py:298`,
`if cycle >= before: break`). At model-call time no message is that recent, so the break is a no-op; at
guard time the call being judged is itself an `AIMessage` on the current cycle, and counting it would have
the call **vouch for its own tool** (`middleware.py:281`).

Their companion is the inequality in `_active_names`: a load or a use counts only when it happened on an
**earlier** cycle than the one being decided, `used < cycle` (`middleware.py:342`). That does two jobs at
once — it is trivially true on the normal path, since a tool loaded on cycle *k* is callable from *k+1*,
and it is what makes the guard correct, because a sibling `get_tool_details` that ran in the same batch as
a guessed call recorded the *current* cycle and so cannot make the guess look sanctioned (docstring
`middleware.py:317`). The test suite pins that case by name:
`test_a_load_from_the_same_batch_does_not_sanction_a_guess` in the binding's `tests/test_middleware.py`.

The TTL boundary belongs to the live side: `cycle - used <= ttl_cycles` is kept (`middleware.py:342`,
docstring `middleware.py:323`), matching the reference implementation.

**The load map is never pruned.** Expiry is a read-time decision in `_active_names`, not a deletion, and
the map is bounded by the number of distinct tools ever loaded on the thread (`_merge_loads` docstring
`middleware.py:132`). Keeping the *maximum* cycle is what makes the reducer order-independent: the same
set of loads produces the same map whichever order the writes arrive in (`middleware.py:128`, code
`middleware.py:144`).

## 6. The message fold and the adapter round trip

A `toolUse` block carries the tool's name **and its arguments**. Sitting next to a successful result it
reads as a template, and the model repeats it — which, once that tool has left the tool list, is a call to
a tool it cannot see. So the fold removes the call shape and keeps the evidence
(`fold_closed_exchanges` docstring `catalog.py:516`).

The decision is entirely the core's. What the binding adds is the round trip, and the one thing the round
trip must not lose is **object identity** (`_fold_messages` docstring `middleware.py:404`).

What is folded, and what is not:

- **Only closed turns.** `_current_turn_start` (`catalog.py:444`) returns the index of the last user
  message carrying no `toolResult` (`catalog.py:460`); everything from there on is the turn in flight,
  tool loop and latest assistant message included, and it passes through as **the very same objects**
  (`catalog.py:592`) — the single exception being the user message that opened the turn, which takes the
  folded span's trailing content when there is any (`catalog.py:595`). `messages[0]` is likewise
  untouched (`catalog.py:558`), and a boundary of `0` or `1` means nothing is folded at all
  (`catalog.py:541`).
- **Only closed pairs of non-callable tools.** A `toolUse` is enrolled when its tool is a disclosure tool
  or is not in this call's active set (`catalog.py:550`), and only if a `toolResult` answering it exists
  in the span (`catalog.py:554`). Nothing enrolled returns the caller's list unchanged
  (`catalog.py:556`).
- **The result becomes a sentence.** `fold_note` (`catalog.py:423`) renders `The tool X was called and
  the result was: Y`, or `... failed with: Y` when the result's status is an error (`catalog.py:440`);
  non-text, non-JSON parts of the result are kept as they are (`catalog.py:571`).
- **The disclosure tools' own exchanges go entirely**, with no sentence: they matter on the call right
  after them and are dead weight past that (`catalog.py:569`, `drop_names` default `catalog.py:512`).

The rewrite then has to leave a shape a provider accepts. Emptied messages are dropped
(`catalog.py:579`), same-role neighbours are merged (`catalog.py:584`), and `_tidy` (`catalog.py:483`)
fixes each message for its role: an assistant message that was rewritten loses its `reasoningContent`,
because a modified message can no longer carry a valid signature (`_without_reasoning`
`catalog.py:465`, applied `catalog.py:578`), and a user message puts its `toolResult` blocks first
(`_results_first` `catalog.py:473`) — a provider rejects text placed ahead of the `toolResult` that
answers the previous assistant message (`catalog.py:476`).

The safety net is the fold's own invariant, checked on its output. `pairs_intact` (`catalog.py:489`)
verifies that every `toolUse` is still answered in the very next message; when the fold broke an
adjacency that was intact on the way in, the original messages are returned with one warning
(`catalog.py:599`, `catalog.py:600`). A case the fold did not foresee therefore costs the saving on that
call, never the call itself. The last message is excluded from the check, since an assistant `toolUse`
still waiting for its result is a legitimate tail (`catalog.py:598`).

### 6.1 The round trip, and why it is asymmetric

```mermaid
sequenceDiagram
    participant RW as _rewrite [middleware.py:648]
    participant FM as _fold_messages [middleware.py:400]
    participant AD as _adapter
    participant Core as fold_closed_exchanges [catalog.py:508]

    Note over RW,FM: Band 1 · interface
    RW->>FM: (request.messages, active) [middleware.py:676]
    Note over AD: Band 2 · adapter
    FM->>AD: to_neutral_list(originals) [_adapter.py:131]
    AD-->>FM: neutral dicts · each carrying tracking_id when the message had an id [_adapter.py:126]
    Note over Core: Band 3 · core
    FM->>Core: fold_closed_exchanges(neutral, active) [middleware.py:419]
    alt nothing folded
        Core-->>FM: the SAME list object back [catalog.py:542,556]
        FM-->>RW: originals, untouched [middleware.py:421]
    else something folded
        Core-->>FM: a new list · unrewritten items are the SAME dict objects [catalog.py:575,592]
        FM->>FM: by_identity = id(neutral item) to original BaseMessage [middleware.py:423]
        loop each item of the folded list
            alt the item is one the core did not rewrite
                FM->>FM: append the ORIGINAL BaseMessage · byte-identical [middleware.py:428]
            else the item is a rewritten dict
                FM->>AD: to_langchain(item) [_adapter.py:153]
                AD-->>FM: a LIST · surviving ToolMessages then a HumanMessage for the sentence [_adapter.py:216]
            end
        end
        FM-->>RW: the folded messages [middleware.py:431]
    end
```

Three things this makes precise.

**The identity check is `folded is neutral`, not a content comparison** (`middleware.py:420`). The core's
contract is to return the caller's own sequence when it folded nothing (`catalog.py:537`), so the cheap
test is exact.

**`to_langchain` returns a list, not a message** (`_adapter.py:153`). The mapping is not one to one in
that direction: a neutral `user` message is the carrier for tool results, and the fold replaces a released
tool's `toolResult` with a text block on that same message. LangChain has no message type holding both,
so one neutral message renders as the surviving `ToolMessage` objects (`_adapter.py:204`) followed by a
`HumanMessage` carrying the text (`_adapter.py:215`). The tool result therefore stays immediately behind
the tool call, which is the ordering a provider requires, and the fold sentence lands behind it
(`_adapter.py:159`).

**Identity survives the trip in two different ways.** For a message the core did not touch, the original
`BaseMessage` object itself is reused (`middleware.py:428`) — which is what keeps a reasoning model's
latest assistant message byte-intact, since a rebuild would invalidate its signature
(`middleware.py:407`). For a message the core *did* rewrite, the LangChain `id` rides through as the
neutral `tracking_id` key and is restored on the way back (`_identified` `_adapter.py:120`, restored
`_adapter.py:175`). When a neutral message splits, the identity goes to the first message out — the one
that existed before the fold — and the synthetic carrier of the folded text gets none, because it is
per-call content rather than a persisted message (`_adapter.py:209`, `_adapter.py:215`, docstring
`_adapter.py:163`).

The neutral shape itself is defined in `message.py`: a message is `{"role": ..., "content": [block, ...]}`
with `NeutralBlock` and `NeutralMessage` aliased to plain dicts (`message.py:40`, `message.py:41`) so an
adapter can hand over a dict without a copy. `_ROLE_BY_TYPE` (`_adapter.py:46`) is the whole role
mapping, and a `ToolMessage` is carried as a `toolResult` block on a **user**-role neutral message
(`_adapter.py:101`, `tool_message_to_result_block` `_adapter.py:66`).

## 7. The catalog block and the summary cache

`build_catalog` (`catalog.py:376`) is the one entry point the binding needs, called with the specs, the
limit, the active names and the instance's cache (`middleware.py:678`). It derives the line of every tool
that is **not** callable on this call — from an explicit `summaries` mapping, else from the cache, else
from `summary_line` — and renders the block (`catalog.py:412`, `catalog.py:413`, `catalog.py:415`).
`catalog_chars=None` suppresses the catalog entirely and returns `""` (`catalog.py:403`), and
`catalog_prompt_block` returns `""` when there is nothing to list, so an empty catalog adds nothing rather
than a header promising a list (`catalog.py:370`, docstring `catalog.py:359`).

**The default summarizer is truncation, not a model call.** This is the other notable difference from the
Strands binding, whose default is one plain call to the agent's own model. Here `SummaryCache`
(`catalog.py:242`) takes an *optional* summarizer (`catalog.py:253`) and the middleware passes whatever
the caller configured, which is `None` by default (`middleware.py:604`, ctor default
`middleware.py:559`). With no summarizer, `SummaryCache.line` (`catalog.py:290`) skips the summarizer
branch outright (`catalog.py:307`) and falls to `summary_line` (`catalog.py:316`), which uses a
description that already fits verbatim and truncates a longer one at a sentence or word boundary
(`summary_line` `catalog.py:222`, `catalog.py:229`, `truncate_description` `catalog.py:127`). No model
call, no network, no token cost, and therefore no usage counter to bill — the core states the constraint
outright: no framework import, no I/O, no model call anywhere in the module (`catalog.py:23`).

A configured summarizer is tried only when the description does not fit (`catalog.py:307`), its answer is
clamped (`clamp_summary` `catalog.py:202`, called `catalog.py:309`), and one that raises is logged and
falls back to the same truncation (`catalog.py:311`, `catalog.py:316`) — so no tool is ever left without a
line and no summarizer failure escapes the rewrite.

Lines are cached keyed by `(name, description)` (`SummaryCache.key` `catalog.py:266`, store
`catalog.py:263`), so a tool is summarized once however many calls read it, and a re-registration with a
changed description gets a new line rather than a stale one behind a matching name (docstring
`catalog.py:245`). `prime` (`catalog.py:270`) is how a framework layer with a model injects an expensive
line, which keeps the catalog byte-stable across calls whichever way the line was produced
(`catalog.py:13`).

The search index is built on the same fingerprint. `_ensure_index` (`middleware.py:691`) keeps the
`(name, description)` pairs as a frozenset and compares them on every call (`middleware.py:702`,
`middleware.py:703`), so a tool arriving at runtime — and equally one re-registered with a new
description — triggers exactly one rebuild (`middleware.py:705`). The fingerprint is written **only after
the build returns** (`middleware.py:706`), so a build that raises is retried next call. Nothing is indexed
at construction time, because what the index covers are the specifications of a call and the first rewrite
is what has them (`middleware.py:598`, docstring `middleware.py:694`).

```mermaid
sequenceDiagram
    participant Agent as create_agent model node
    participant MW as wrap_model_call [middleware.py:612]
    participant RW as _rewrite [middleware.py:648]
    participant Idx as LexicalToolIndex [tool_index.py:177]
    participant Cat as build_catalog [catalog.py:376]
    participant Sum as SummaryCache.line [catalog.py:290]
    participant Provider as the model call

    Agent->>MW: request · tools = every bound tool · system_message = the caller's
    MW->>RW: _rewrite(request) [middleware.py:631]
    RW->>RW: _specs_of(bound) · BaseTool or provider dict alike [middleware.py:660,186]
    alt the disclosure tools are not both bound
        RW-->>MW: the request itself, unchanged [middleware.py:663]
    else both are bound
        RW->>Idx: _ensure_index · fingerprint changed, so this call builds [middleware.py:665,705]
        RW->>RW: cycle = number of AIMessages so far [middleware.py:668,266]
        RW->>RW: loaded = request.state loaded_tools [middleware.py:669]
        RW->>RW: _last_used then _active_names [middleware.py:670,671]
        RW->>RW: tools = _keep_active · caller objects, arrival order [middleware.py:675]
        RW->>RW: messages = _fold_messages · see section 6 [middleware.py:676]
        RW->>Cat: build_catalog(specs, catalog_chars, active, cache) [middleware.py:678]
        Cat->>Sum: line for each name NOT active [catalog.py:413]
        Sum-->>Cat: verbatim when it fits, else a boundary cut · no model call [catalog.py:316,222]
        Cat-->>RW: the header plus one line per hidden tool, or an empty string [catalog.py:373,370]
        alt the block is non-empty
            RW->>RW: system_message = _with_catalog(...) [middleware.py:680]
        end
        RW-->>MW: request.override(tools, messages, system_message) [middleware.py:689]
    end
    MW->>Provider: handler(request) [middleware.py:634]
    Note over Provider: tools = find_tools + get_tool_details + always_available + live loads
    Note over Provider: system = the caller's message plus the catalog block as one more text block
    Note over Provider: messages = the folded copy · no call shape for a tool absent from tools
```

Any failure on the rewrite path is swallowed and logged, and the request goes on **as received** — which
is the behaviour without the middleware (`middleware.py:632`, `middleware.py:633`, docstring
`middleware.py:619`). Nothing is remembered about the failure, so the very next call attempts the rewrite
again (`middleware.py:621`).

## 8. The system message — no setter, so a replacement is built

`_with_catalog` (`middleware.py:352`) does not append to the existing message; it constructs a new one.
The reason is a hard constraint of `langchain-core`: **`SystemMessage.content_blocks` is a read-only
property**, so the block list is read off it and a replacement message is built from it
(`middleware.py:355`, construction `middleware.py:374`).

Three shapes, each preserved:

- `block` empty → `system_message` back unchanged, **by identity** (`middleware.py:370`, docstring
  `middleware.py:365`).
- `system_message is None` → a new `SystemMessage` carrying only the catalog (`middleware.py:372`,
  `middleware.py:373`).
- otherwise → a new `SystemMessage` whose content is the existing blocks followed by one
  `{"type": "text", "text": block}` (`middleware.py:374`).

The existing blocks are kept as **separate** blocks rather than flattened into one string, because a
caller using the list form is usually placing cache checkpoints between them and collapsing it would move
them (`middleware.py:356`). Appending rather than prepending is deliberate: the caller's own prompt keeps
the opening position, so on a provider that caches by prefix the operator's text stays at a stable offset
(`middleware.py:360`).

## 9. The common path — catalog, load, call

Two model calls and no guessing: the names live in the system message, the schemas arrive through
`get_tool_details`, and use renews the TTL without any write at all.

`_load` (`middleware.py:893`) tolerates the two shapes a model actually sends: a bare string instead of a
list (`middleware.py:907`) and duplicates, which are de-duplicated with order kept and each name stripped
(`middleware.py:908`). A call that named nothing usable answers with `_DETAILS_EMPTY_GUIDANCE`
(`middleware.py:927`). A name the agent does not have — or either disclosure tool, which cannot be loaded
— is collected and reported rather than silently dropped (`middleware.py:920`, `middleware.py:931`). The
specification itself is **not** in the result text: it travels in `tools` on the next call, because a tool
result is resident in the history while a tool list is per call and forgettable (docstring
`middleware.py:896`).

```mermaid
sequenceDiagram
    participant Model
    participant MW as the middleware
    participant Tool as get_tool_details [middleware.py:837]
    participant State as loaded_tools channel [middleware.py:156]

    Note over Model: Call N · cycle = k · reads the catalog in the system message
    MW-->>Model: tools = callable only · system += catalog · messages folded [middleware.py:689]
    Model->>Tool: get_tool_details(["list_investment_transactions"])
    Tool->>MW: _load(names, runtime) [middleware.py:853,893]
    MW->>MW: bare string tolerated · duplicates dropped · names stripped [middleware.py:907,908]
    MW->>MW: cycle = number of AIMessages in state MINUS one [middleware.py:913]
    loop each requested name
        MW->>MW: spec present and not a disclosure tool, so loaded[name] = cycle [middleware.py:920,923]
        MW->>MW: line = name plus its cached catalog summary [middleware.py:924]
    end
    MW->>State: Command(update = messages + loaded_tools) [middleware.py:940,941,943]
    State->>State: _merge_loads keeps the later cycle per name [middleware.py:124,144]
    MW-->>Model: "Loaded. These tools are callable with their full parameters on your next call..." [middleware.py:81,935]
    Note over Model: Call N+1 · cycle = k+1 · used k is strictly earlier, so the tool is active [middleware.py:342]
    MW-->>Model: the loaded tool carried as the caller's own object [middleware.py:675]
    Model->>MW: list_investment_transactions(real args) · wrap_tool_call [middleware.py:712]
    MW->>MW: the name is in the active set for the cycle it chose on, so the call runs [middleware.py:781]
    MW->>MW: no write · the call itself renews through _last_used next time [middleware.py:303]
```

`get_tool_details` takes a **list**, which is what keeps a step needing three tools to one cycle rather
than three (docstring `middleware.py:841`). The answer `ToolMessage` is built by the middleware itself
(`middleware.py:935`) rather than by the framework wrapping a returned string, because the tool's return
type is `Command` — that is the only way a LangGraph tool writes a state channel, and `loaded_tools` is
added to the update **only** when something was actually loaded (`middleware.py:941`).

## 10. The fallback — `find_tools` searches, it does not load

`find_tools` exists for a need the model cannot map to any listed name. It ranks specifications and
reports names plus summaries, and then stops: the model still goes through `get_tool_details`, so there is
one road to a schema instead of two (docstring `middleware.py:860`).

A blank need is not searched at all and returns guidance (`middleware.py:871`, `middleware.py:872`). A
search that raises returns guidance too, worded the same way as a no-match, because from where the model
stands the two cases are one and the failure is the middleware's to log rather than the model's to reason
about (`middleware.py:879` through `middleware.py:881`, docstring `middleware.py:105`). A match the bound
set does not have, and either disclosure tool, are skipped (`middleware.py:886`).

```mermaid
sequenceDiagram
    participant Model
    participant Tool as find_tools [middleware.py:815]
    participant MW as _search [middleware.py:857]
    participant Idx as LexicalToolIndex [tool_index.py:177]

    Note over Model: no catalog name fits the need
    Model->>Tool: find_tools("list investment transactions")
    Tool->>MW: _search(need, runtime) [middleware.py:835]
    alt need is blank
        MW-->>Model: _EMPTY_NEED_GUIDANCE [middleware.py:872,98]
    else need is usable
        MW->>MW: specs from runtime.tools · _ensure_index [middleware.py:874,875]
        MW->>Idx: search(need, top_k) [middleware.py:878 · tool_index.py:214]
        alt search raised
            MW-->>Model: _SEARCH_FAILED_GUIDANCE [middleware.py:881,104]
        else matches returned
            Idx-->>MW: at most top_k non-zero scores · ties by indexing order [tool_index.py:241,239]
            MW->>MW: one line per match · bound and not a disclosure tool [middleware.py:884,886]
            alt nothing usable
                MW-->>Model: _NO_MATCH_GUIDANCE [middleware.py:890,101]
            else names to report
                MW-->>Model: _MATCHES_HEADER plus one line per match · NOTHING loaded [middleware.py:891,76]
                Model->>Tool: get_tool_details([names]) · then exactly as section 9
            end
        end
    end
```

The index is term-frequency and local, so a search needs no network (`LexicalToolIndex` `tool_index.py:177`,
`build` `tool_index.py:197`, `search` `tool_index.py:214`, constraint `tool_index.py:180`). It scores over
the **distinct** terms of the need, so a need repeating a word does not let that word own the ranking
(`tool_index.py:228`, `tool_index.py:234`, docstring `tool_index.py:183`), and a zero score is not
returned at all — a need sharing no term with any specification yields no matches, which the tool reports
as such instead of offering something arbitrary (`tool_index.py:235`, docstring `tool_index.py:188`).
The indexed text is the specification's **full** text: name, description and the parameter descriptions of
the `inputSchema`, walked to a bounded depth (`_spec_text` `tool_index.py:135`, `_schema_parts`
`tool_index.py:103`, `_MAX_SCHEMA_DEPTH` `tool_index.py:96`) — the parameters are what tell apart two
tools whose one-line summaries read alike. Tokenization treats punctuation and underscores as separators,
so `list_accounts` indexes as `list` plus `accounts` and a need phrased as "list the accounts" reaches it
(`_TOKEN_PATTERN` `tool_index.py:160`, `_tokenize` `tool_index.py:165`).

`ToolIndex` is a structural `Protocol` (`tool_index.py:53`), so any object exposing a callable `build` and
`search` is a valid implementation — which is exactly what `_validate_index` checks, by member rather than
by `isinstance` (`middleware.py:489`, loop `middleware.py:501`). The protocol allows either operation to
return an awaitable (`tool_index.py:66`, `tool_index.py:82`); this binding calls both synchronously
(`middleware.py:705`, `middleware.py:878`), so a network-backed index would have to resolve its own
awaitables.

## 11. The guard — a guessed call does not run

A guessed call is a name the model read in the catalog and called without loading it. The guard does not
consult a stored "what was projected" set; it **recomputes** the active set for the cycle the model chose
on (`middleware.py:776` through `middleware.py:780`). That single choice is what makes two cases come out
right at once: a sibling `get_tool_details` in the same batch recorded the current cycle and so cannot
sanction the guess, and a tool that expired *since* the call was issued is not mistaken for a guess
either (docstring `middleware.py:716`).

The exemptions are dealt with in order. The two disclosure tools and everything in `always_available`
return first (`middleware.py:763`). A name nothing bound has is not this middleware's to judge — the agent
answers it already (`middleware.py:770`, comment `middleware.py:769`). A name in the recomputed active set
returns (`middleware.py:781`). Only then is the parameter test applied (`middleware.py:786`).

**The guard loads nothing on the model's behalf.** It cancels with `_PREMATURE_CALL_MESSAGE`
(`middleware.py:791`), which names the tool and points at `get_tool_details`. A recovery that loaded the
tool and invited an immediate retry would teach the model that calling a catalog name directly works,
which is the very shortcut the catalog rule forbids (docstring `middleware.py:114`).

```mermaid
sequenceDiagram
    participant Model
    participant WTC as wrap_tool_call [middleware.py:712]
    participant Cancel as _cancellation [middleware.py:749]
    participant Handler as the tool

    Note over Model: the model calls a catalog name without loading it first
    Model->>WTC: ToolCallRequest · tool_call.name = X
    WTC->>Cancel: _cancellation(request) [middleware.py:737]
    alt X is a disclosure tool or in always_available
        Cancel-->>WTC: None · nothing written [middleware.py:763,764]
    else neither
        Cancel->>Cancel: specs from request.runtime.tools [middleware.py:766,767]
        alt X is bound to nothing
            Cancel-->>WTC: None · the agent answers it already [middleware.py:770,771]
        else X is bound
            Cancel->>Cancel: chose_on = AIMessages in state MINUS one [middleware.py:776]
            Cancel->>Cancel: _last_used then _active_names for chose_on [middleware.py:778,779]
            alt X is active on chose_on
                Cancel-->>WTC: None · the schema was visible, so the call is informed [middleware.py:781,782]
            else X was never visible
                alt X declares no required parameter
                    Cancel-->>WTC: None · an empty call to it is legitimate [middleware.py:786,787]
                else X requires parameters
                    Cancel->>Cancel: log the cancellation [middleware.py:789]
                    Cancel-->>WTC: ToolMessage · status error · _PREMATURE_CALL_MESSAGE [middleware.py:790,791,794]
                    WTC-->>Model: "'X' did not run: it is not loaded... call get_tool_details with [X] first"
                    Note over Model: X is NOT loaded by the guard · the model must call get_tool_details itself
                    Model->>WTC: get_tool_details(["X"]) · then exactly as section 9
                end
            end
        end
    end
    WTC->>Handler: handler(request) on any None above [middleware.py:737]
```

Only the `required` list decides — `_requires_parameters` (`middleware.py:224`) answers `True` only when
that list is non-empty (`middleware.py:245`): a tool whose parameters are all optional is callable with no
arguments, so an empty call to it is not the symptom of a missing schema (docstring `middleware.py:228`).
Any schema shape the function cannot read is treated as requiring nothing (`middleware.py:240`,
`middleware.py:243`), which errs towards letting the call run — and that is the same direction
`_schema_of` errs in when a schema cannot be rendered at all (`middleware.py:183`).

The guard does **not** require an empty input. Arguments or not, the model could not have known the
parameters, so either way the call is a guess (comment `middleware.py:784`).

Any failure inside `_cancellation` returns `None` and lets the call through, because cancelling a
legitimate call is the worse outcome of the two (`middleware.py:796` through `middleware.py:798`,
docstring `middleware.py:752`).

## 12. One tool's lifecycle

```mermaid
stateDiagram-v2
    [*] --> Catalog: bound · one line in the system-message catalog [catalog.py:368]
    Catalog --> Loaded: get_tool_details([name]) · Command writes loaded_tools[name] = cycle [middleware.py:923,943]
    Catalog --> Cancelled: guessed call · cancelled, and NOTHING is loaded [middleware.py:790]
    Cancelled --> Catalog: the model must call get_tool_details itself
    Loaded --> Live: the next call sees used strictly earlier than cycle [middleware.py:342]
    Live --> Live: called again · _last_used reads the use out of the AIMessage, no write [middleware.py:303]
    Live --> Idle: not called · cycle minus used is still at most ttl_cycles, so it is kept [middleware.py:342]
    Idle --> Catalog: cycle minus used exceeds ttl_cycles · absent from the active set [middleware.py:339]
    note right of Catalog
        ttl_cycles default = 3 (DEFAULT_TTL_CYCLES, middleware.py:64)
        Release withdraws the schema from the next call's tool list.
        The tool stays bound and callable (class docstring, middleware.py:525).
        Release also puts the name BACK into the catalog, since the block is
        built by skipping exactly the active names (catalog.py:365).
        Nothing is deleted from loaded_tools: expiry is a read-time decision
        (_merge_loads docstring, middleware.py:132).
    end note
```

`find_tools` appears nowhere in this diagram on purpose: a search writes no state, so a tool's lifecycle
is driven by `get_tool_details`, by use, and by inactivity.

## 13. Verbatim text the model sees

**Write targets.** `tools` and `messages` always, `system_message` when the catalog block is non-empty,
all through one `request.override` (`middleware.py:674` through `middleware.py:689`). The two tool
answers are **messages**: `find_tools` returns a plain string the framework wraps (`middleware.py:891`),
and `get_tool_details` builds its own `ToolMessage` inside a `Command` (`middleware.py:935`,
`middleware.py:943`) — which is why the schema itself is deliberately kept out of it (docstring
`middleware.py:82`), and why the fold drops those two exchanges once they are spent (`catalog.py:569`).

### 13.1 The system-message catalog header — `CATALOG_PROMPT_HEADER` (`catalog.py:83`)

Quoted literally, `{get_tool_details}` and `{find_tools}` being substituted with the two tool names at
`catalog.py:372`:

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

Immediately followed by one line per non-callable tool, built at `catalog.py:368`:

```python
lines.append(f"- {name}: {summary}" if summary else f"- {name}")
```

So the model sees, for example:

```text
- get_transactions: Transactions of an account over a date range, newest first.
- open_position: Opens a position on an instrument for an account and returns its id.
```

No sigil and no marker: the names are not in `tools`, so nothing in the call claims they are callable and
nothing has to be walked back. Step 3 is what makes release intelligible from the model's side — a
rejected call is a documented state with a documented recovery, not a surprise.

### 13.2 `get_tool_details` docstring — exactly as the model receives it (`middleware.py:839`)

```text
Load the full parameters of one or more tools from the catalog, so you can call them.

Pass every tool you are about to need in one call. They arrive complete in your tool list on
your next call. A tool left unused for a few calls is unloaded; to call it after that, call
this again with its name.

Args:
    names: Exact tool names, as written in the catalog or in a `find_tools` result.
    runtime: Injected by the framework. Not user-facing.

Returns:
    A state update recording what was loaded, and a message naming it plus any requested
    name that is not a tool.
```

Its input schema is not a source literal: it is derived by `@tool` (`middleware.py:837`) from the
signature `get_tool_details(names: list[str], runtime: ToolRuntime)` (`middleware.py:838`). The only
model-facing parameter is `names`, a list of strings — `ToolRuntime` is injected. The tool is a **closure,
not a method**, precisely so that no `self` surfaces as a parameter the model is asked to fill (docstring
`middleware.py:807`, `middleware = self` at `middleware.py:813`).

### 13.3 `find_tools` docstring — exactly as the model receives it (`middleware.py:817`)

```text
Search for tools that can do what you need, when no name in the tool catalog fits.

This only finds tools; it does not load them. It answers with matching tool names and one
line about each. To use any of them, call `get_tool_details` with their names, then call them.

If a name in the catalog already fits what you need, skip this and call `get_tool_details`
directly.

Args:
    need: What you are trying to do, described in your own words. A capability, not a tool
        name — "list the transactions of an investment account" works better than a guess at
        what the tool might be called.
    runtime: Injected by the framework. Not user-facing.

Returns:
    The matching tool names with a one-line summary of each, or guidance to describe the need
    or to reword it when there is nothing to list.
```

Same derivation, from `find_tools(need: str, runtime: ToolRuntime)` (`middleware.py:816`): the only
model-facing parameter is `need`, a string. Both names come from the core rather than from a constructor
knob — `FIND_TOOLS_NAME` (`catalog.py:57`) and `GET_TOOL_DETAILS_NAME` (`catalog.py:60`), collected into
`PLUGIN_TOOL_NAMES` (`catalog.py:66`) — and the second is deliberately not `get_details`: a verb-noun that
generic is one a domain tool can already hold, and a collision would silently shadow one of the two
(`catalog.py:63`).

### 13.4 `get_tool_details` results

Header literal `_DETAILS_LOADED_HEADER` (`middleware.py:81`), which states the release rule in the same
breath as the load:

```python
_DETAILS_LOADED_HEADER = (
    "Loaded. These tools are callable with their full parameters on your next call. A tool left unused for "
    f"a few calls is unloaded; to call it after that, call `{GET_TOOL_DETAILS_NAME}` again:"
)
```

Assembled with one line per loaded tool at `middleware.py:929`, each line built at `middleware.py:924`,
and followed when needed by `_DETAILS_UNKNOWN` (`middleware.py:88`, formatted at `middleware.py:931`):

```python
_DETAILS_UNKNOWN = "Not a tool, ignored: {names}. Use names from the catalog or from `" + FIND_TOOLS_NAME + "`."
```

A call that named nothing usable gets `_DETAILS_EMPTY_GUIDANCE` (`middleware.py:91`, returned at
`middleware.py:927` and again as the fallback at `middleware.py:932`):

```python
_DETAILS_EMPTY_GUIDANCE = (
    "Pass the names of the tools you want loaded, as a list. Take them from the catalog, or call `"
    + FIND_TOOLS_NAME
    + "` first."
)
```

### 13.5 `find_tools` results

Header literal `_MATCHES_HEADER` (`middleware.py:76`), which states in the same breath that nothing was
loaded:

```python
_MATCHES_HEADER = (
    f"Tools that match. Nothing is loaded yet: call `{GET_TOOL_DETAILS_NAME}` with the names you want, then call them."
)
```

The full result is assembled at `middleware.py:891` as `"\n".join([_MATCHES_HEADER, *lines])`, each line
built at `middleware.py:884`:

```python
f"- {match.name}: {self._short_description(specs[match.name])}"
```

`_short_description` (`middleware.py:945`) returns the cached catalog line, falling back to
`DEFAULT_CATALOG_CHARS` as the limit when the catalog is suppressed (`middleware.py:957`) —
`catalog_chars=None` drops the catalog from the prompt, it does not mean a search result should carry a
full description (docstring `middleware.py:948`).

### 13.6 `_PREMATURE_CALL_MESSAGE` (`middleware.py:109`)

```python
_PREMATURE_CALL_MESSAGE = (
    "'{name}' did not run: it is not loaded, so its parameters are unknown to you. Call `"
    + GET_TOOL_DETAILS_NAME
    + '` with ["{name}"] first, then call \'{name}\' with its real parameters.'
)
```

Formatted with the tool name at `middleware.py:791` and carried on a `ToolMessage` with
`status="error"` (`middleware.py:794`), answering the guessed call's own id (`middleware.py:792`). It names
the recovery instead of performing it: the tool is **not** loaded by the guard.

### 13.7 The fold sentence — `fold_note` (`catalog.py:423`)

Not a constant: the sentence is assembled from the tool name and the rendered result, with the verb chosen
by the result's status (`catalog.py:440`, `catalog.py:441`):

```python
outcome = "failed with" if result.get("status") == "error" else "the result was"
return f"The tool {name} was called and {outcome}: {' '.join(parts)}"
```

Text parts go in verbatim and JSON parts are serialized with `ensure_ascii=False` (`catalog.py:437`,
`catalog.py:439`). So a closed exchange of a tool that is no longer callable reads, in the messages of the
next call:

```text
The tool get_transactions was called and the result was: {"items": [...], "next": null}
```

It reaches the model as a text block on the neutral message (`catalog.py:570`) and, once
`to_langchain` has rendered it, as a `HumanMessage` following the surviving `ToolMessage` objects
(`_adapter.py:215`).

### 13.8 The remaining guidance strings

- `_EMPTY_NEED_GUIDANCE` (`middleware.py:98`), returned on a blank `need` (`middleware.py:872`):

  ```python
  _EMPTY_NEED_GUIDANCE = "Describe what you are trying to do, in your own words, then call this tool again."
  ```

- `_NO_MATCH_GUIDANCE` (`middleware.py:101`) — returned when nothing usable was found
  (`middleware.py:890`):

  ```python
  _NO_MATCH_GUIDANCE = "No tool matches that description. Try different wording, or answer directly."
  ```

- `_SEARCH_FAILED_GUIDANCE` (`middleware.py:104`) — returned when the search itself raised
  (`middleware.py:881`):

  ```python
  _SEARCH_FAILED_GUIDANCE = "Tool search is unavailable right now. Try a different description, or answer directly."
  ```

With the `CATALOG_PROMPT_HEADER` block and its lines, the two tool docstrings, `_MATCHES_HEADER`,
`_DETAILS_LOADED_HEADER`, `_DETAILS_UNKNOWN`, `_DETAILS_EMPTY_GUIDANCE`, `_PREMATURE_CALL_MESSAGE` and the
`fold_note` sentence, that is the complete set of model-facing text this binding can produce. The
`_ELLIPSIS = "..."` literal (`catalog.py:77`) can appear inside a catalog line that was cut mid-sentence
(`truncate_description` `catalog.py:127`, appended `catalog.py:158`).

There is **no summarizer system prompt** in this binding, because there is no default model summarizer to
send one — see section 7.

## 14. Configuration — every constructor parameter

Constructor: `ProgressiveToolDisclosureMiddleware.__init__` (`middleware.py:555`). All keyword-only (`*`
at `middleware.py:557`). Every check runs before any state is set, so a construction that fails registers
nothing on any agent (`middleware.py:586` through `middleware.py:591`, docstring `middleware.py:583`).

| Parameter | Default (`file.py:LINE`) | Accepts `None`? | Meaning |
|-----------|--------------------------|-----------------|---------|
| `catalog_chars` | `DEFAULT_CATALOG_CHARS = 80` (`catalog.py:70`; ctor `middleware.py:558`) | **Yes** | Character limit of one catalog line's summary. `None` adds no catalog at all (`catalog.py:403`), leaving the two disclosure tools' descriptions as the only hint that other tools exist — the cheapest configuration and the one with the least to go on (docstring `middleware.py:568`). Validated by `_validate_catalog_chars` (`middleware.py:454`) — `None` or int ≥ 1, `0` rejected because a zero-character line fits nothing (`middleware.py:465`). |
| `summarizer` | `None` (`middleware.py:559`) | **Yes** | What writes a line when a description does not fit. Receives `(spec, max_chars)` (`ToolSummarizer` `middleware.py:70`). `None` means **truncation at a sentence or word boundary**, not a model call (`catalog.py:307`, `catalog.py:316`), and a summarizer that fails falls back to that same truncation (`catalog.py:311`). `_validate_summarizer` (`middleware.py:506`) — `None` or callable. |
| `ttl_cycles` | `DEFAULT_TTL_CYCLES = 3` (`middleware.py:64`; ctor `middleware.py:560`) | No | Cycles a loaded tool survives without a call; a call renews it with no write (`middleware.py:303`). `_validate_positive_int` (`middleware.py:439`) — int ≥ 1, `bool` and `float` rejected (`middleware.py:451`, docstring `middleware.py:442`). |
| `always_available` | `()` empty tuple (`middleware.py:561`) | No | Names callable on every call, skipping the discovery cycle and exempt from the guard (`middleware.py:338`, `middleware.py:763`). `_validate_always_available` (`middleware.py:469`) — a sequence of non-empty strings; a bare string is rejected, since it would configure one name per character (`middleware.py:480`, docstring `middleware.py:473`). Stored as a tuple so the caller's sequence cannot change the configuration after the fact (`middleware.py:596`, comment `middleware.py:595`). |
| `index` | `None` → `LexicalToolIndex()` (`middleware.py:562`; instantiated `middleware.py:600`) | **Yes** | Search implementation behind `find_tools`. `None` selects the default term-frequency index (`tool_index.py:177`), which needs no network. `_validate_index` (`middleware.py:489`) — `None`, or an object with a callable `build` and `search`, checked by member because `ToolIndex` is a structural protocol (docstring `middleware.py:494`). |
| `top_k` | `DEFAULT_TOP_K = 3` (`middleware.py:67`; ctor `middleware.py:563`) | No | How many tools one search lists, passed at `middleware.py:878`. `_validate_positive_int` (`middleware.py:439`) — int ≥ 1. |

Six parameters, none positional, and **no placement flag, no `referenced_source` bridge and no tool-name
knob**. The instance keeps exactly six things after validation: the limit, the TTL, the always-available
tuple, `top_k`, the index and the summary cache, plus a `None` fingerprint the first rewrite fills
(`middleware.py:593` through `middleware.py:604`). Everything per-thread lives in graph state
(`middleware.py:156`), which is what lets a checkpoint restore resume with the same set of tools loaded.
