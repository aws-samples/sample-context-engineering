# OpenCode context-engineering plugin

Relevance-filters oversized tool results in [OpenCode](https://opencode.ai) (verified against the public
plugin API of **1.18.34**). The plugin under `plugins/context-engineering.ts` is a thin interface: it
reacts to the public `tool.execute.after` hook and the session lifecycle `event` hook, and talks
newline-delimited JSON (JSONL) to a persistent Python bridge. All decision logic — admission,
capability, chunking, scoring, verbatim preview, baseline preservation and the closed refusal — lives in
`context-core` (`context_core.opencode_bridge` over `context_core.opencode`). The plugin also registers the
public `retrieve_context` tool (partial reads of a filtered result), clears closed retrievals in the messages sent
to the model and keeps each session's retention lease alive with a heartbeat. No private OpenCode API is used.

It can also run practice D, the context graph, on top of that. D is **experimental and off by default**
(`CONTEXT_GRAPH=on`; see "Practice D" below for what it does, what it costs and why it stays off).

## Install

From a clone of the repository, install the core so the bridge module is importable:

```bash
uv pip install -e context-core
```

The plugin's own dependencies (`@opencode-ai/plugin` and `@opencode-ai/sdk`, pinned to the verified 1.18.34) are listed in
`.opencode/package.json`; OpenCode installs them at startup, and the runner and `bun test` need them too
(`cd .opencode && bun install`).
The plugin file is loaded automatically from `.opencode/plugins/` at OpenCode startup. The bridge command
is resolved in this order: `CONTEXT_BRIDGE_COMMAND`, then `PYTHON`, then the `python3` executable. Its
arguments remain configurable through `CONTEXT_BRIDGE_ARGS`; the default is
`-m context_core.opencode_bridge`. The plugin does not read `.env`.

## Configuration

Everything is environment-driven — there is no profile, region, model id or endpoint hardcoded anywhere.
The bridge **requires** `AWS_PROFILE` and never defaults a profile.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AWS_PROFILE` | yes | — | AWS profile used to build the boto3 session for Bedrock |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | no | profile default | AWS region |
| `CONTEXT_EXECUTION_MODEL_ID` | live only | unset | OpenCode agent model id used by live admission; no fallback to the reranker |
| `CONTEXT_EXECUTION_ALLOWED_MODELS` | live only | empty | comma-separated allowlist that must contain the agent model |
| `CONTEXT_RERANK_MODEL_ID` | no | `amazon.rerank-v1:0` | Bedrock reranking model id or ARN; independent from the agent model |
| `CONTEXT_RELEVANCE_THRESHOLD` | no | `0.5` | minimum chunk score, in `[0.0, 1.0]` |
| `CONTEXT_CHUNK_TOKENS` | no | `2500` | scoring granularity (tokens per chunk) |
| `CONTEXT_PREVIEW_TOKENS` | no | `1000` | preview budget (tokens) |
| `CONTEXT_MAX_RESULT_TOKENS` | no | `8000` | gate: results at/below this are left unchanged |
| `CONTEXT_PREVIEW_STYLE` | no | `scoped` | notice that travels with an excerpt: `scoped` says how much of the original it covers, which questions it cannot answer and the narrowest call that reaches the rest; `legacy` is the old one-line "read the full result" hint |
| `CONTEXT_RETRIEVAL_CLEANUP` | no | on | clear the results of closed `retrieve_context` calls in the messages sent to the model (see "Reading back a filtered result"); `0`, `false`, `off` or `no` turns it off |
| `CONTEXT_OPENCODE_TOOLS` | no | built-in read tools | comma-separated base tool classes explicitly eligible for filtering; incoming `ce60-fixture_<tool>` and `mcp_ce60-fixture_<tool>` names normalize to `<tool>` only when that base class is listed |
| `CONTEXT_METRICS_FILE` | no | unset | per-arm JSONL file containing only normalized class names, operations, byte counts and flags; no payload is written |

The bridge uses fail-closed name matching. It accepts the observed MCP forms `ce60-fixture_<tool>` and
`mcp_ce60-fixture_<tool>`, plus pure built-in names such as `read`; it strips only those exact prefixes.
The normalized class must be present in the explicit coverage set. An unknown or uncovered name is recorded
as `skipped` with `tool_not_covered`, never silently filtered. The default coverage contains only the six
read-shaped built-ins; a deployment must list each MCP base class (for example `list_accounts` or
`read_aws_documentation`) in `CONTEXT_OPENCODE_TOOLS` before either prefixed form can be filtered.

`CONTEXT_METRICS_FILE` is one JSONL file per arm/run. Each transform record contains only
`tool_class`, `operation`, `input_bytes`, `output_bytes`, `filtered`, `source_already_truncated`,
`fallback`, rerank attempt/result and retrieval counters. Retrieval records contain counters and success
flags only. The CE-60 driver reads the file after the arm and copies a scalar summary into
`host_evidence` in the manifest and sanitized result. A write failure is surfaced as
`metrics_status=collection_failed`; it never causes a payload to be logged.

## Run

Start OpenCode from the project root as usual; the plugin loads itself:

```bash
opencode
```

The first oversized result from a covered tool spawns the bridge, which builds the boto3 session and the
reranker lazily. A result that is under threshold, from an uncovered tool, or produced while the bridge
is unavailable is left exactly as the tool returned it. The plugin substitutes a result only when the
bridge answers `previewed` (a filtered preview) or `baseline` (the integral original, preserved on a
recoverable scoring failure); a refusal or partial payload is never injected.

> ⚠️ **Bedrock inference is billable.** When a covered result exceeds `CONTEXT_MAX_RESULT_TOKENS`, the
> bridge calls the Bedrock `rerank` operation under `AWS_PROFILE`, which incurs AWS charges. Small
> results and uncovered tools make no inference call and cost nothing.

## Reading back a filtered result

A filtered result carries a notice with an opaque `reference_id`, how much of the original the excerpt covers,
which questions an excerpt cannot answer (a maximum, total, count, ranking or comparison across every row) and the
narrowest call that reaches the rest. The public tool `retrieve_context` takes the `reference_id` plus at most one
way of narrowing the read:

| Argument | Returns |
|---|---|
| `pattern` (regex) | only the matching lines, numbered, with `context_lines` around each (default 5); the cheapest way to aggregate over specific rows |
| `line_range` `{start, end}` | exactly that 1-indexed inclusive span, using the line numbers of the `[... N lines omitted ...]` markers |
| `max_chunks` | the N most relevant chunks in document order, from the ranking already computed for the excerpt (no second rerank) |
| `max_tokens` | approximate size limit of the response; alone it returns the whole content cut to that budget |
| none of them | the whole original |

Precedence is `line_range`, then `pattern`, then `max_chunks`; `max_tokens` bounds every mode and defaults to
`CONTEXT_MAX_RESULT_TOKENS`, so a read never costs more than keeping the original would have. A malformed option is
answered with a correctable message (it never quotes content) and counts as a failed retrieval.

**Closed retrievals are cleared, not deleted.** With `CONTEXT_RETRIEVAL_CLEANUP` on, the experimental
`experimental.chat.messages.transform` hook marks the result of each completed `retrieve_context` call of an
earlier turn as cleared in the messages sent to the model, so OpenCode sends `[Old tool result content cleared]`
instead of the retrieved text. The call stays visible: deleting it made the model doubt its earlier answers and re-run
the same retrieval every turn. The stored session is untouched, and any failure of the hook leaves the messages as
they were.

**Retention.** Each session has a memory-only lease in the bridge. Every request for the session renews it and the
plugin sends a `heartbeat` every 120 s for the sessions it has seen, so a reference can be read back after a long
silence. Without any sign of life the lease lapses after 300 s and the reference expires after another 300 s.

The excerpt budgets above default to the reference filter's package defaults (8,000 / 2,500 / 1,000 tokens). The
reference also scores chunks against the tool-call arguments as well as the question; this port scores against the
question only.

## Practice D: context graph (experimental, opt-in)

With `CONTEXT_GRAPH=on` the plugin also runs practice D. Before each model call, the same experimental
`experimental.chat.messages.transform` hook asks the bridge what to change in **that one request**, and applies the answer to
the array it was handed. The model gets two tools to ask for anything back: `expand_card` (an earlier turn, by its title) and
`find_context` (search the collapsed turns by description). `expand_artifact` of the other hosts does not exist here: OpenCode
has no store of offloaded results for it to read.

**It is request-only.** The array the hook receives is a per-request copy that OpenCode rebuilds from storage on the next call,
so the stored session is never written, and no session-editing SDK call is made. This is shown, not asserted: every request
records a digest of what OpenCode handed the hook, taken before anything edited it, and after the run the runner reads the
stored session back and requires the same digest for every request (`<arm>-graph-durability.json`).

### Two policies (`CONTEXT_GRAPH_MODE`)

The same Cards, the same tools and the same request-only guarantee; they differ in what they decide to show.

| | `reference` (default) | `inplace` |
|---|---|---|
| decision | every user turn, from the question: each earlier turn is shown whole, as a description or as a title (embedding model `cohere.embed-multilingual-v3`) | a pure function of the transcript: no score, no embedding call, no state, the same answer after a restart |
| edit | leaves messages out and appends one trailing synthetic user message with the summary of what was left out | replaces the **tool-calling assistant messages** of each closed turn by one synthetic assistant message where the first of them was; appends nothing |
| `expand_card` | the named turns come back in full for the rest of the turn (the next request puts them back where they were) | the turn's content comes back as the **tool result**; the request does not change |
| measured on CE-60 (Sonnet 5.5, A+D against the baseline of the same repetition, 3 repetitions) | tokens -48.9 %, **dollars +215.7 %** (campaign v7) | tokens -65.2 %, **dollars -31.1 %**, 30/30 correct turns in all three repetitions (campaign v9); with the answers kept one more turn (`KEEP_DIALOGUE=1`) dollars **+16.6 %** (campaign v8) |

Why two, and why the in-place edit is shaped the way it is. OpenCode puts a cache checkpoint on the system prompt and on the last
two messages of every request, and the provider reads a request only up to a checkpoint an earlier request wrote whose prefix is
identical (not up to any common prefix); a cache write costs 25 times a read at the price list in use ($0.10 / $2.50 per million
tokens). Appending is free; editing an earlier message makes everything after the edit a write unless the edit sits right after a
message that ended an earlier request. The reference policy re-decides every turn, so turns flip between whole and collapsed and
the cache is rewritten: it cuts the tokens the model reads by about half and the bill rises more than threefold. The `inplace`
policy replaces the assistant side of a turn (its tool calls and its text answer) by one summary, the turn's Card (title, tools
used, the numeric lines copied literally), **directly after the user's question and only once, when the turn closes**. The first
call of that turn ended at the question, so the question is a checkpoint and the cache stays warm right up to it; only the
summary and the next question are written. Changing the same turn a second time one turn later (`KEEP_DIALOGUE=1` drops the
answers then) lands where no request ended and the whole compact history is written again every turn: that was campaign v8.
What `inplace` gives up is the relevance decision: an old turn is not shown again unless the model asks for it. It is a variant
of D, closer to the "context curator" idea the design folded into D, and is reported as such. The measurements, the cache
finding and the cost model are in `validation/opencode_gate_a/results/ce60-v7-v9-REPORT.md`.

How it maps onto the hook. OpenCode calls the hook once per **model call**, so a tool loop calls it several times per user
turn. The reference policy scores once per turn, freezes the choice and delivers it on every call of the turn, so the context
cannot shift mid-reasoning and an `expand_card` lasts exactly the rest of the turn; computing the choice on every call would
undo `expand_card`. The `inplace` policy derives the Cards once per turn and recomputes the plan on every call (OpenCode
rebuilds the request from storage each time and the edit has to be made again), which is cheap because it needs nothing from
outside. A turn is left as it is whenever anything about it is in doubt: no Card, a user-role message inside it, a compaction
summary, repeated or empty message ids.

Scoring (the reference policy, and `find_context` under either) uses the embedding model `cohere.embed-multilingual-v3` (the
thresholds are calibrated to it; another model needs its own calibration) and, like the relevance filter, only under an admitted
`live` admission. Without one the reference policy still derives Cards and never scores, which is a full pass.

| Variable | Default | Purpose |
|---|---|---|
| `CONTEXT_GRAPH` | off | `on` / `1` / `true` / `yes` switches practice D on; anything else is off |
| `CONTEXT_GRAPH_MODE` | `reference` | `reference` or `inplace` |
| `CONTEXT_GRAPH_KEEP_RECENT` / `_KEEP_DIALOGUE` | `0` / `0` | `inplace`: closed turns left untouched; closed turns after those that keep their question and text answers. The defaults are the configuration that paid (v9); anything above 0 changes a turn twice and was measured worse (`KEEP_DIALOGUE=1`: +16.6 % dollars) |
| `CONTEXT_GRAPH_RECALL_MAX_CHARS` | `40000` | `inplace`: most characters one `expand_card` returns (OpenCode saves a tool result above 50 KB or 2,000 lines to a file) |
| `CONTEXT_GRAPH_UNIQUE_TITLES` | off | `reference`: give every turn a Card of its own when two turns open with the same words (`inplace` always does) |
| `CONTEXT_GRAPH_EVIDENCE_FILE` | unset | per-request JSONL record (digests, counts and ids; never text) |
| `CONTEXT_GRAPH_TIMEOUT_MS` | `30000` | how long the plugin waits for the bridge on a projection (scoring calls an embedding model) |
| `CONTEXT_GRAPH_EXPAND_THRESHOLD` / `_COLLAPSE_FLOOR` / `_LINK_THRESHOLD` | `0.62` / `0.45` / `0.50` | the benchmark `GRAPH_TUNING` |
| `CONTEXT_GRAPH_DESCRIPTION_TOKENS` / `_BODY_BUDGET` | `100` / `40000` | description size; token ceiling across the turns kept in full (`none` removes it) |
| `CONTEXT_GRAPH_MAX_RETRIEVAL_CYCLES` / `_REUSE_TTL` | `8` / `5` | retrieval calls per user turn; model calls a recalled turn stays raised (`reference`) |
| `CONTEXT_GRAPH_TAGS_PER_CARD` / `_RARITY_WEIGHT` / `_MIN_CARDS` / `_NEIGHBORS` | `5` / `0.70` / `3` / `0` | as in the other hosts |
| `CONTEXT_GRAPH_EMBED_MODEL_ID` | `cohere.embed-multilingual-v3` | Bedrock embedding model |

With the flag off the plugin registers the same hooks and the same single tool as before, the bridge answers every `graph_*`
request `skipped`, and the startup response and the runner's `plugin_options` are byte-for-byte what they were.

The runner treats D as something arm A can carry, not as an arm: with `CONTEXT_GRAPH=on` it records `graph:on` in
`plugin_options` (plus `graph_mode:inplace,graph_keep:K,graph_dialogue:D` for the in-place policy), writes
`A-graph-evidence.jsonl` and `A-graph-durability.json`, and writes its paired comparison to `d-slice-comparison.json`
**instead of** `gate-a-decision.json`, because Gate A judges A alone. Gate D has its own evidence and evaluator in
`validation/opencode_gate_d/`. D's status here is `experimental` under an owner waiver of the Gate A precondition, never
`supported`, and no D result changes Gate A, which stays `no_go`.

## CE-60 host driver (`runner/ce60.ts`)

`runner/ce60.ts` is the real CE-60 host driver. It drives an actual OpenCode server over the **public**
SDK (`@opencode-ai/sdk`, pinned to `1.18.34`) — no private API — and produces the payload-free evidence
the offline Gate A layer (`validation/opencode_gate_a/`) consumes. It never scores or sanitises anything
itself: the deterministic oracle, the terminal classification, the sanitiser and the evaluator all live
in the Python `opencode_gate_a` package, which the driver shells out to.

What it does per arm:

1. starts OpenCode as a persistent process (`opencode serve`) bound to `127.0.0.1` on an **ephemeral
   port** with the arm's environment — `baseline` starts in `--pure` mode (no plugins), `A` starts
   normally so `plugins/context-engineering.ts` loads;
2. waits for `/global/health`;
3. creates one session via `client.session.create`;
4. sends the 60 CE-60 prompts **strictly sequentially** via `client.session.prompt` (never in
   parallel — CE-60 is one evolving conversation), keeping each raw answer only in memory;
5. hands the in-memory, payload-free per-turn observations to `python -m opencode_gate_a.host_driver
   finalize`, which writes only the manifest, the sanitized result, the evaluation and an evidence
   summary;
6. **always** stops the OpenCode process in a `finally`.

The 60 prompts come from the CE-60 scenario (`validation/community-plugin-A-B-D/src/scenario.py`) via
`python -m opencode_gate_a.host_driver emit-turns`. If that import fails, the driver fails closed and
asks for `--turns-json` with exactly 60 turns — it never truncates, pads or fabricates turns. Only arms
`baseline` and `A` exist; `A_plus_D` and `native_B` are refused by design.

The driver invokes Python as `python3` by default; set `PYTHON` to point at a specific interpreter. That
interpreter must be able to import the scenario, which depends on `strands-agents`
(`pip install -r validation/community-plugin-A-B-D/requirements.txt`). If the scenario is not importable
in your environment, pass `--turns-json` with exactly 60 `{turn_id, prompt}` entries instead.

### Safety: dry run is the default

Nothing — no OpenCode process, no model call — happens without `--execute-live`. The default is a dry
run that resolves the 60 turns and prints the plan (with secrets redacted). A raw prompt, answer, tool
payload, credential or log is never written to a file: the raw answer text is sent to the Python oracle
over a pipe, used to score correctness, and discarded.

`baseline` and `A` run in separate processes and sessions on the **same** host, host-version and model,
which is what makes their Gate A comparison a valid isolated pair. OpenCode itself exposes no CE-60 retrieval
count, so the driver derives the per-turn count of `retrieve_context` calls from the whole persisted session
history (`attributeSessionRetrieval`); the campaign acceptance check compares that total with the bridge's own
counters (`host_evidence.retrieval_attempts`). The baseline, which loads no plugin, records `host_not_exposed`
(never a fabricated zero). Arm A's evidence also records
`plugin_options` (for example `preview_style:scoped,retrieval_cleanup:on,plugin_rev:2`), so a result always says
which retrieval behaviour produced it.

### Commands

```bash
# Dry run (DEFAULT): resolve the 60 turns and print the plan. No process, no model call, no cost.
bun .opencode/runner/ce60.ts --dry-run

# Offline/fixture diagnostic: validates MCP normalization and sanitized metrics only; no OpenCode, AWS or model.
bun .opencode/runner/ce60.ts \
  --diagnostic-fixture \
  --metrics-file ./ce60-out/diagnostic-{arm}-{run_id}.jsonl

# Help.
bun .opencode/runner/ce60.ts --help

# Live run of both arms (BILLABLE — see the warning below).
AWS_PROFILE=my-profile AWS_REGION=us-west-2 \
  bun .opencode/runner/ce60.ts \
    --execute-live \
    --arm both \
    --model anthropic/claude-sonnet-4 \
    --approval-id APPR-1234 \
    --budget-ceiling 25 --budget-unit USD \
    --stat-plan ./stat-plan.json \
    --run-id ce60-2026-06 \
    --output-dir ./ce60-out

# A single arm, or an explicit 60-turn file instead of the scenario.
# A dry-run calibration plan for the real OpenCode A arm. These values are applied only to A.
bun .opencode/runner/ce60.ts \
  --dry-run --arm A \
  --max-result-tokens 4000 \
  --preview-tokens 800 \
  --chunk-tokens 500 \
  --relevance-threshold 0.02
```

The calibration flags default to `8000`, `1000`, `2500` and `0.5` respectively when omitted. The plan
prints the resolved values, and a live A payload records the same numeric values in host provenance without
including prompts or tool-result payloads. Baseline strips these four calibration variables and never
receives them.

Three flags change the conditions of BOTH arms identically and are recorded in the evidence:

- `--disable-tools question,webfetch` removes those tools through the per-prompt `tools` map. `question` blocks a
  headless session waiting for a human (the F002 stall); `webfetch` is a determinism change.
- `--system-prompt "<text>"` sends a per-prompt `system` field (1-500 characters), for example a response-language
  pin; only its sha256 is recorded.
- `--diagnostic-stop-after <turn_id>` stops after that turn and writes a diagnostic artifact that is never Gate A
  evidence.

The correctness oracle version is chosen with `CE60_ORACLE_VERSION` (`v3`, the community oracle, is the default;
`v4` only widens the accepted English wording of the C1 retry count). The campaign scripts under
`validation/opencode_gate_a/campaigns/` run the full billable campaigns; see
`validation/opencode_gate_a/results/ce60-v3-v6-REPORT.md` for what they found.

### AWS profile and environment

A live run **requires** `AWS_PROFILE` (never a default) and a region (`AWS_REGION` or
`AWS_DEFAULT_REGION`), plus `--model`, `--approval-id`, `--budget-ceiling`, `--budget-unit`,
`--stat-plan` and `--run-id`. For arm `A` the driver sets the context-engineering plugin's live
settings in that process only: `CONTEXT_EXECUTION_MODE=live`, `CONTEXT_APPROVAL_ID`,
`CONTEXT_BUDGET_CEILING`, `CONTEXT_BUDGET_UNIT`, `CONTEXT_EXECUTION_MODEL_ID` (the chosen agent model),
`CONTEXT_EXECUTION_ALLOWED_MODELS` (the same agent model) and `CONTEXT_RERANK_MODEL_ID` (defaults to
`amazon.rerank-v1:0`). The bridge uses the execution model for admission and the reranker only for
relevance scoring. No credential is minted by the driver; AWS access comes entirely from `AWS_PROFILE`.

> ⚠️ **A live run is billable.** `--execute-live` starts a real OpenCode server and sends 60 prompts to
> the chosen model under `AWS_PROFILE`; arm `A` additionally calls the Bedrock `rerank` operation on
> oversized tool results. Both incur AWS charges against the admitted budget. The default dry run starts
> nothing and costs nothing.

### Outputs

Per arm, under `--output-dir`: `<arm>-catalog.json`, `<arm>-manifest.json` (the offline/replay manifest
the evaluation is bound to), `<arm>-sanitized.json`, `<arm>-evaluation.json` and `<arm>-evidence.json`.
A live run also writes `<arm>-live-manifest.json`, the provenance manifest binding the admission,
budget, price and approved statistical plan (a live run is evaluated as a replay of itself, because the
offline evaluator never evaluates a live manifest directly). When both arms run, the driver also writes
`gate-a-decision.json`. The bridge writes one JSONL metrics file per arm/run when `CONTEXT_METRICS_FILE`
or `--metrics-file` is configured; the driver reads only its sanitized scalar summary and puts it in
`host_evidence` in both the manifest and sanitized result. Baseline remains `--pure`: it loads no plugin
and exposes no `retrieve_context`, so its retrieval and bridge metrics are `host_not_exposed`.

The CE-60 model-input metric remains exactly `tokens.input + cache.read + cache.write`. Schema/tool
overhead is separate: the driver reads `tool_spec_tokens` and `tool_spec_count` only if a future public
SDK response exposes them. SDK 1.18.34 does not, so both fields are recorded as `host_not_exposed`
rather than estimated from conversation text or tool-call counts.

### Tests and build

```bash
cd .opencode/runner
bun test                              # offline driver tests: plan, guards, 60-turn invariant, no-parallel
bun build ce60.ts --target=bun --outfile=/dev/null   # type transpilation check

# Python host bridge + offline Gate A layer:
cd ../../validation
python -m pytest opencode_gate_a -q
```

### Payload-free session diagnostics

After every `session.prompt`, the TypeScript driver reads the public `client.session.messages` endpoint and
keeps only bounded diagnostics: message count, tool-part count, MCP tool-part count, compaction observation,
timeout/error flags, request lifecycle flags and tool names. Message text, prompts, tool arguments and tool
outputs are discarded. The Python bridge validates the bounded `observability_summary` and persists it under
`host_evidence.observability` in the manifest, sanitized result and evidence file. The block contains
`message_count_max`, `message_count_median`, `tool_call_count_total`, `tool_call_count_max`,
`mcp_tool_call_count_total`, `timeout_count`, `timeout_turn_ids`, `tool_names` and the observed compaction
count. Unavailable session counts remain `null`; they are never converted to zero. A timeout observation
records its turn id and all counts available before the session abort, which supports diagnosis of F002
without retaining payload data.
