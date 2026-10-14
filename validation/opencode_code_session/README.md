# Real coding session on OpenCode

A scripted developer conversation, run once without the plugin and once with it, in a throwaway copy of this
repository. It is a measurement, not Gate A and not a campaign: it reports and decides nothing.

The task the agent is given is to write Claude Code plugins for this repository (`prompts.json`, 16 prompts: explore,
plan, build four packages, run the tests, validate, two recall questions, edit the README, summarise). The files it
writes are **throwaway**. They live in a sandbox that is deleted after scoring and are not part of this repository's
implementation; only counts, sizes and file names are kept.

| Why | How |
|---|---|
| CE-60 only uses MCP fixture tools. Practice A (relevance filter) covers OpenCode's built-in `read`, `grep`, `glob`, `list`, `bash` by default and practice D (context graph) works on every tool, so a real session is the first time either meets real file reads, edits and test runs. | The agent really reads, searches, writes and runs commands with OpenCode's own tools. |
| D only collapses a turn once it has closed. | The task is split into 16 user turns, as a developer would, not one long prompt. |

## Run

```bash
bash validation/opencode_code_session/run.sh                  # baseline then plugin, Sonnet 5.5, prompts.json
CODE_RUN_ID=code-v2 CODE_PLUGIN_MODE=d-only bash validation/opencode_code_session/run.sh
CODE_ARMS=plugin CODE_MODEL=amazon-bedrock/anthropic.claude-haiku-5-5 \
  CODE_TURNS=validation/opencode_code_session/smoke-turns.json bash validation/opencode_code_session/run.sh   # cheap smoke
```

Billable (Bedrock, `AWS_PROFILE=claude-code`, `us-west-2`). `.env` is never read. Environment: `CODE_RUN_ID`,
`CODE_MODEL`, `CODE_TURNS`, `CODE_ARMS`, `CODE_PLUGIN_MODE` (`a+d` default, `d-only`, `a-only`), `CODE_MAX_TURNS`,
`KEEP_SANDBOX=1`. Results go to `results/<run>/`.

## Arms

* `baseline`: `opencode serve --pure`, no plugin.
* `plugin`: the plugin as it ships, in the v9 configuration. Practice A runs with live admission over the built-in tools
  (above 8000 tokens a result becomes an excerpt plus `retrieve_context`); practice D runs in place with
  `KEEP_RECENT=0`, `KEEP_DIALOGUE=0`. `d-only` keeps the admission but gives A a coverage no tool matches; `a-only`
  turns the graph off.

Both arms get the same model, prompts and tool policy (`question`, `webfetch` and `task` disabled).

## Guardrails (the agent has a shell and write access)

* **Sandbox, not the repository.** `prepare_sandbox.py` copies the tree without `.git`, `.env`, local agent and AI-DLC
  configuration, results and virtual environments (and without this harness: `prompts.json` holds the probe answers),
  scans for credential-looking files and strings, writes a neutral `AGENTS.md`, and makes it a fresh git repository so
  `git status` is exactly what the agent changed. The Bedrock provider gets `chunkTimeout: 180000` in both arms.
* **Permission policy** (`.opencode/opencode.json` inside the sandbox): paths outside the sandbox, `curl`, `ssh`, `aws`
  and other credential helpers, package installs, `git push`, `env`, process killing and the interactive `question` tool are denied. A denied
  call fails fast; an `ask` would block a headless session.
* **Server.** Loopback only, with OpenCode's own HTTP basic auth (`OPENCODE_SERVER_PASSWORD`, random per run). The runner
  checks the listener with `lsof` and stops if it is anything else.
* **Environment.** The OpenCode process gets `PATH`, `HOME`, locale, `AWS_PROFILE` and `AWS_REGION`, not the caller's
  environment.

## Files

* `.opencode/runner/code-session.ts`: the driver (one arm). Uses the same SDK and process helpers as `ce60.ts`, which it
  imports but does not change.
* `prepare_sandbox.py`, `workcheck.py` (what the session left behind, static: nothing the agent wrote is executed),
  `report.py` (one run, both arms), `run.sh` (orchestration and the reading written before the first run).
* `prompts.json` (scenario, probe rules, expected files), `smoke-turns.json`.

## Reading it

The agent's path differs from run to run (which files it opens, how often it re-reads, which test it writes), so a gap
between two sessions is a hint, not a sample. Repeat the run before quoting a percentage. The probes check whether a fact
read early can still be recovered after practice D collapsed that turn; the deliverable check is deliberately shallow
(files exist, JSON parses, Python compiles, the agent's own tests passed in its transcript).

## First results (Sonnet 5.5, `results/code-v1` to `code-v3`)

Three baseline sessions and two plugin sessions (A+D, v9 configuration), 16 turns each, all turns completed in every
counted session. `code-v2` ran plugin then baseline, `code-v3` baseline then plugin.

| | baseline (3) | plugin A+D (2) | difference of the means |
|---|---|---|---|
| model input tokens (read + write + uncached) | 6.4M to 8.9M, mean 7.5M | 2.14M, 2.17M | -71 % |
| dollars (OpenCode cost counter) | 1.59 to 2.07, mean 1.82 | 1.24, 1.24 | -32 % |
| model calls | 59 to 75 | 75, 77 | +15 % |
| output tokens | 55k to 73k | 53k, 51k | -20 % |
| cache write tokens | 165k to 185k | 209k, 217k | +21 % |
| input per call, median / peak | 116-123k / 165-184k | 25-26k / 45-47k | -78 % / -74 % |
| wall time | 439 to 641 s | 415 to 420 s | -22 % |

Paired in the same run: `code-v2` -71 % tokens and -31 % dollars, `code-v3` -66 % and -22 %.

* Same deliverable in every counted session: all 19 expected files present, every JSON parses, every Python file
  compiles, the marketplace lists the four packages, the README gained the section, no existing package touched, the
  agent's own test runs ended green, both recall probes answered correctly (in the plugin arm the Hermes one through
  `expand_card`).
* Practice A never fired here. It saw 188 tool results; the 138 from tools it covers (`read`, `grep`, `glob`, `list`,
  `webfetch`, `bash`) were all under the gate of 8000 tokens (the largest, a `read`, was 31.7 KB of the 32 KB it takes),
  so it made no reranker call and produced 0 previews. The other 50 are tools it does not cover, among them two
  `expand_card` recalls of 36 and 42 KB, which are practice D's. The whole difference is therefore practice D. A does
  fire on a real large read (the Haiku smoke previewed a 56 KB document). What-if, not run: a gate of 4000 tokens would
  have caught 7 of the 138 covered results, 2000 tokens 21, 1000 tokens 52.
* Practice D stayed request-only: in all 152 plugin requests the stored session still matched what OpenCode handed the
  hook (`A-graph-durability.json`), none rejected.
* The first plugin attempt stalled on a Bedrock stream (a `write` call pending for ten minutes) and is not counted
  (`results/code-v1/attempt0-plugin-stalled/`); `code-v1` therefore holds a baseline only.
* Caveats: three and two sessions; the baseline's cost varied by 30 % between runs; one scenario and one model; sessions
  of 59 to 77 calls (CE-60 baseline: 98, with A+D: 118). The dollar gap is smaller than the token gap because the plugin
  makes more calls and writes more cache, as in CE-60 v9.
