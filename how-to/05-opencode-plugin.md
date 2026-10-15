# Running OpenCode with the context-engineering plugin

> **⚠️ Not for production use.** This is a minimal working sample for experimentation and learning.
> It has no guardrails, no error handling and no operational hardening.

How to install two of the practices on [OpenCode](https://opencode.ai), the terminal coding agent:
[relevance filtering](../docs/design/design-a-relevance-filtering.md) (A) and the
[context graph](../docs/design/design-d-context-graph.md) (D).

There is no agent to write here. OpenCode is the agent: you copy one plugin into a project and set
environment variables. The logic is the same [`context-core`](../context-core/) as in the other guides
([Strands](02-community-plugins-agent-sample.md), [LangGraph](03-langgraph-plugins-agent-sample.md),
[Hermes](04-hermes-plugins-agent-sample.md)). The plugin, `.opencode/plugins/context-engineering.ts`, is a thin
TypeScript interface over OpenCode's public plugin API, verified against **OpenCode 1.18.34**; a Python bridge
(`context_core.opencode_bridge`) makes every decision. No private API is used.

| | Practice | On OpenCode |
|---|---|---|
| **A** | Relevance filtering | Works once you open the admission ([below](#a--relevance-filtering)). A result of `read`, `grep`, `glob`, `list`, `webfetch` or `bash` above 8,000 tokens becomes a verbatim excerpt; the original stays reachable through `retrieve_context`. |
| **D** | Context graph | **Experimental, off by default.** Closed turns collapse into short Cards, in the request only. `expand_card` and `find_context` bring them back. |
| **B** | Progressive tool disclosure | Not ported. The port declares it unsupported on OpenCode instead of approximating it. |

**Read [the things that will bite you](#things-that-will-bite-you) before judging a result.**

## Prerequisites

| | Requirement | Why |
|---|---|---|
| OpenCode | **1.18.34** | the version the plugin API was verified against |
| Python | **3.10 or newer** | the bridge; `context_core` must import in the interpreter OpenCode runs |
| AWS CLI | **v2** | only to configure and verify credentials |
| Bun | optional | only for this repository's tests and runner; OpenCode installs the plugin's two dependencies itself |

Bedrock models that must be **enabled in your account**, in the region you use:

| Model id | Used by |
|---|---|
| `amazon.rerank-v1:0` | relevance filtering (`CONTEXT_RERANK_MODEL_ID` changes it) |
| `cohere.embed-multilingual-v3` | `find_context` of the context graph |
| the model OpenCode runs | the agent itself; its id goes into the admission below |

Credentials come from the standard `boto3` chain under `AWS_PROFILE`. The bridge **requires** it and never
defaults a profile.

## Install

From a clone of this repository:

```bash
# 1. The core, in the Python that runs the bridge: python3, unless you set PYTHON or CONTEXT_BRIDGE_COMMAND.
pip install -e context-core

# 2. The plugin, into your project. OpenCode loads .opencode/plugins/ at startup.
P=/path/to/your/project
mkdir -p $P/.opencode/plugins $P/.opencode/lib
cp .opencode/plugins/context-engineering.ts $P/.opencode/plugins/
cp .opencode/lib/{bridge-transport,excerpt,graph,history,revision,tool-output}.ts $P/.opencode/lib/
cp .opencode/package.json $P/.opencode/        # or merge its two dependencies into your own
```

Leave the `*.test.ts` files behind. On its first start OpenCode runs `bun install` for the two dependencies in
`.opencode/package.json` (`@opencode-ai/plugin` and `@opencode-ai/sdk`, pinned to 1.18.34), so that start is slower
and needs the npm registry. If you run OpenCode inside a clone of this repository there is nothing to copy:
`.opencode/` is already there. The same layout under `~/.config/opencode/` is how OpenCode loads global plugins; it
was not tried here.

## A — relevance filtering

The filter calls Bedrock, so it stays off until you open an explicit admission. Without these variables a
covered, oversized result is left exactly as the tool returned it, and nothing is billed:

```bash
export AWS_PROFILE=<your-profile>  AWS_REGION=<region>
export CONTEXT_EXECUTION_MODE=live
export CONTEXT_APPROVAL_ID=local-dev                          # any label; it is recorded in the admission
export CONTEXT_BUDGET_CEILING=10  CONTEXT_BUDGET_UNIT=USD    # declarative: the admission checks you stated one
export CONTEXT_EXECUTION_MODEL_ID=amazon-bedrock/anthropic.claude-sonnet-5-5   # the model OpenCode runs
export CONTEXT_EXECUTION_ALLOWED_MODELS=$CONTEXT_EXECUTION_MODEL_ID
opencode
```

The budget is not a spend cap. Each filtered result costs one Bedrock `rerank` call under `AWS_PROFILE`.

What it does. A result from a covered tool whose size estimate (`ceil(characters / 4)`) is above
`CONTEXT_MAX_RESULT_TOKENS` (8,000) is cut into chunks of about 2,500 tokens and scored against your latest message.
The agent receives about 1,000 tokens of verbatim chunks (`CONTEXT_PREVIEW_TOKENS`), with the gaps marked, and a
notice with a `reference_id`. `retrieve_context` reads the rest: `pattern` (the matching lines), `line_range`,
`max_chunks`, or the whole original. Everything else is untouched.

Check that it works. Point `CONTEXT_METRICS_FILE` at a file, ask the agent to read a file larger than about 32 KB, and
look for the filtered result:

```bash
export CONTEXT_METRICS_FILE=/tmp/ce-metrics.jsonl
grep previewed /tmp/ce-metrics.jsonl      # input_bytes, output_bytes and counters; never the text
```

Verified on an empty project holding only the files above and a 130 KB text file: the `read` of 53,595 bytes reached the
agent as 3,996, and the agent found the line it needed with one `retrieve_context` call using a `pattern`.

## D — context graph (experimental)

```bash
export CONTEXT_GRAPH=on
export CONTEXT_GRAPH_MODE=inplace
```

When a user turn closes, the assistant side of that turn (its tool calls and its text) is replaced, in the request
only, by one synthetic message: a Card with the turn's title, the tools it used and the numeric lines copied
literally. The question stays as written. The agent gets `expand_card` (an earlier turn, by its title) and
`find_context` (a search over the collapsed turns). `find_context` scores with `cohere.embed-multilingual-v3` and needs the
admission above; `expand_card` does not.

The stored session is never written. Set `CONTEXT_GRAPH_EVIDENCE_FILE` to record, per model request, a digest of what
OpenCode handed the hook and how many messages were left out (counts and ids, never text). After a second turn the file
shows `projected` requests.

Verified in the same project: the second turn (`opencode run -c`) collapsed three messages on every request, and the
agent recovered the answer from the first turn with `expand_card` and `find_context`.

## Both

Set both groups of variables. They act at different moments: A on a result as it arrives, D on turns that have
closed.

## Things that will bite you

1. **A is a silent no-op until the admission is open.** The plugin only logs `live admission not requested; results will
   not be filtered` at debug level, the bridge refuses every oversized result and the agent sees it unchanged. Check the
   metrics file for `previewed`.
2. **A only acts on big results.** The gate is 8,000 tokens, about 32 KB, and OpenCode's `read` already returns at most
   about 50 KB. In a 16-turn coding session no result of a covered tool reached it (the largest was 31.7 KB). Lower
   `CONTEXT_MAX_RESULT_TOKENS` to make it act sooner; the benchmark harnesses of the other hosts use 4,000, with
   500-token chunks.
3. **`CONTEXT_OPENCODE_TOOLS` replaces the default list** (`read`, `grep`, `glob`, `list`, `webfetch`, `bash`). Setting it
   to your MCP tools turns the built-ins off for A, so list them too.
4. **The relevance threshold is a position in the scorer's distribution, not a number.** `0.5` was run with
   `amazon.rerank-v1:0`. Another reranker needs its own value; the other hosts' benchmarks use `0.02` with
   `cohere.rerank-v3-5`.
5. **Always set `CONTEXT_GRAPH_MODE=inplace`, and leave `CONTEXT_GRAPH_KEEP_RECENT` and `CONTEXT_GRAPH_KEEP_DIALOGUE` at
   0.** OpenCode's prompt cache reads a request only up to a checkpoint an earlier request wrote, and a cache write
   costs 25 times a read. The default `reference` policy re-decides every turn and raised the bill by 215 % while it cut
   the tokens by half. Keeping the answers one turn longer (`KEEP_DIALOGUE=1`) changes a turn twice and raised it by 17 %.
6. **`expand_card` wants the turn's title.** The model sometimes passes the whole question; the call then misses and it
   retries (23 of 84 calls in the CE-60 runs).
7. **A real OpenCode compaction with D on was not run.** The tests model one, and the plugin leaves a turn alone whenever
   anything about it is in doubt (a compaction summary included).
8. **Both practices cost something.** A pays a `rerank` call per filtered result, `find_context` an embedding call.

## What was measured

All on Sonnet 5.5, a few runs each. Treat them as hints, not as a benchmark.

| Setup | Result |
|---|---|
| A alone, the 60-turn CE-60 scenario, 3 repetitions | tokens −14 %, dollars −12 %. The port's own bar (median input −20 %, correctness within 3 points) was met in 1 of 3 repetitions, so Gate A stays `no_go`. |
| A + D (`inplace`), the same scenario, 3 repetitions | tokens −65 %, dollars −31 %, 30 of 30 correct turns every time. D is `experimental`. |
| A + D, a scripted 16-turn coding session, 2 plugin sessions against 3 baselines | tokens −71 %, dollars −32 %, same deliverables. A never fired there, so the gain is D. |

The reports: [`ce60-v3-v6-REPORT.md`](../validation/opencode_gate_a/results/ce60-v3-v6-REPORT.md) (A alone),
[`ce60-v7-v9-REPORT.md`](../validation/opencode_gate_a/results/ce60-v7-v9-REPORT.md) (D and the cache finding) and
[`validation/opencode_code_session/`](../validation/opencode_code_session/README.md) (the coding session). The plugin's
full reference, with every variable, is [`.opencode/README.md`](../.opencode/README.md).
