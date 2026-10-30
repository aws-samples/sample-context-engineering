# Gate D for the OpenCode port

Practice D (the context graph) on OpenCode is **request-only**. Before a model call it changes what that one request carries
and nothing the host keeps; it never writes to the session. It has two policies (`CONTEXT_GRAPH_MODE`): the `reference` one
leaves earlier messages out and appends one summary block, and the `inplace` one replaces the tool-calling messages of each old
turn by one short summary where they were (see `.opencode/README.md`). Gate D checks the request-only claim for both, request by
request, and turns the result into a decision. It is a separate gate from Gate A and never reclassifies it.

Gate A was decided on OpenCode and is `no_go` (1 of 3 repetitions in v6). The functional design keeps D `unsupported` until
Gate A is `go` (FR5.1 / BR5.1). The owner asked for D to be built and measured anyway and **waived that precondition**; the
evaluator records it as `precondition: waived_by_owner`, and a passing Gate D under a waiver makes the capability
`experimental`, never `supported`.

## What is checked

One evidence record per model request (`OpenCodeGateDEvidence`, `contracts.py`). It holds digests, opaque ids and booleans and
never a word of the conversation.

| Check | Rule | Reason code |
|---|---|---|
| request-only | the durable transcript before and after every request has the same digest, and `canonical_equality` holds | `TRANSCRIPT_DURABLY_CHANGED` |
| scenarios | `tool_loop`, `subagent`, `compact`, `resume` and `recovery` are all covered | `SCENARIO_MISSING:<name>` |
| Card sources | every request after the first closed turn names its Cards and the host message ids each one stands for | `CARD_SOURCE_MISSING` |
| Card identity | within a session an id keeps its sources; no id is shared between sessions | `CARD_ID_UNSTABLE`, `CARD_ID_COLLISION` |
| full pass | with the projection switched off the model input equals the input without the feature | `FULL_PASS_DIVERGES`, `FULL_PASS_MISSING` |
| no durable effect | no graph message in the store, no write, no change of the stored transcript | `DURABLE_EFFECT_DETECTED` |
| engagement | the graph projected at least once, so the evidence is about D and not about doing nothing | `NEVER_ENGAGED` |
| scenario facts | whatever a scenario reports by itself (a Card left without a source after a compaction, a turn that did not come back) | `SCENARIO_FAILED:<name>` |

## Two kinds of evidence

**The contract suite** (`suite.py`, offline, no network, deterministic) runs the real binding
(`context_core.opencode_graph.OpenCodeGraph` over the real graph core) through the five scenarios on recorded OpenCode sessions
and a synthetic one, under either policy (`--mode reference|inplace`). Its similarity matcher is a deterministic bag-of-words
one, so two runs must agree to the byte (the resume scenario depends on that); what the embedding model decides is measured
live (and the `inplace` policy scores nothing while it projects, so for it the suite is the whole decision). The suite can
fail: the tests break the binding on purpose (state shared between sessions, a mutated transcript, an in-place answer that
removes a user message, a recall that moves the request) and require `no_go`.

Under `inplace` the scenarios read the answer differently. `tool_loop`, `subagent` and `resume` compare the whole shape of an
answer (removed ids, block, summaries); `recovery` asks a collapsed turn back with `expand_card`, requires its content in the
answer and requires the next request to be **exactly the same** (a recall that changed the request would rewrite the provider's
cache from there on); and every in-place answer is checked on its own: summaries replace assistant messages only, each summary
stands where a removed message was, and nothing is appended.

**Live evidence** (`python -m opencode_gate_d live`) turns one A+D campaign run into request records from the two payload-free
files the runner writes: `A-graph-evidence.jsonl` (per request: the digest of what OpenCode handed the hook, taken before
anything edited it, and the digest of what was sent) and `A-graph-durability.json` (the session read back after the run: the
digest of the stored slice each request was handed). Equal digests mean the stored session is exactly what every request was
given: nothing a request did reached the host.

## Run

From `validation/`:

```bash
export PYTHONPATH="$PWD/../context-core/src:$PWD"

# the contract suite over recorded sessions (read from a COPY of the OpenCode store) and a synthetic one
python3.13 -m opencode_gate_d suite --title ce60-A-ce60-v6-sonnet55-r1-A --out opencode_gate_d/results/suite.json

# the same under the in-place policy (--keep-recent N / --keep-dialogue N mirror CONTEXT_GRAPH_KEEP_RECENT / _DIALOGUE)
python3.13 -m opencode_gate_d suite --mode inplace --title ce60-A-ce60-v8-sonnet55-r1-A --out opencode_gate_d/results/suite-inplace.json

# a live A+D run -> request evidence
python3.13 -m opencode_gate_d live --run-dir opencode_gate_a/results/ce60-v7-sonnet55-r1 --out opencode_gate_d/results/live-r1.json

# the decision, with the waiver stated
python3.13 -m opencode_gate_d decide --evidence opencode_gate_d/results/suite.json --evidence opencode_gate_d/results/live-r1.json \
  --gate-a-outcome no_go --waiver-by owner --waiver-requirement FR5.1/BR5.1 --waiver-reason "..." \
  --out opencode_gate_d/results/decision.json
```

`replay.py` is a different tool: it replays a recorded session request by request through the real binding (any policy) and
reports how much of each request practice D would have removed and what that does to the prompt cache and the bill (chars/4 for
the conversation, the fixed prefix kept exact, calibrated to the real baseline of the same repetition; output tokens are the
baseline's). It sizes an experiment; it does not measure one, and it cannot see the extra model calls and recalls a different
context causes.

The cache model matters. The default models what OpenCode and the provider really do: a checkpoint on the system prompt and on
the last two messages of every request, and a read only up to a checkpoint an earlier request wrote whose prefix is identical.
`--ideal-cache` restores the older model (a hit at any common prefix with the previous request), which is exact for a history
that only appends and optimistic for any edit; it predicted -30 % for a configuration that really cost +16.6 %. The default
reproduces the six in-place sessions of campaigns v8 and v9 to within 0.2 - 6 points of the real dollars
(`results/ce60-v7-v9-REPORT.md`). Result files under `results/replay/` made before this change used the older model and are
optimistic for the in-place variants.

The suite evidence files in `results/` (`suite-*.json.gz`, 3 - 4 MB each uncompressed) are stored gzipped; `gunzip -k` one
before passing it to `decide --evidence`. The decisions already made are there too: `decision-faithful.json` (reference
policy, `no_go`: `CARD_ID_UNSTABLE`), `decision-unique-titles.json`, `decision-no-waiver.json` (`unsupported`),
`decision-v8-inplace.json` and `decision-v9-inplace.json` (in-place policy, suite plus live evidence, `go` / `experimental` /
`waived_by_owner`), and `decision-inplace-suite-only.json`.

## Tests

```bash
cd validation && python3.13 -m pytest opencode_gate_d -q
```

`test_recorded.py` also runs the TypeScript serializer (`.opencode/lib/graph.ts`) and the Python one over the same transcript and
requires identical output, when `bun` is installed.

## Results kept in the repository

`results/` holds the decisions (`decision-*.json`), the recorded-suite evidence (`suite-*.json.gz`) and the replay
sizing (`replay/*.json`). The live request evidence of the A+D campaigns (`live-v{7,8,9}-r{1,2,3}.json`, about 7 MB) is
not versioned; `python3.13 -m opencode_gate_d live --run-dir <run directory of a campaign>` regenerates it.
