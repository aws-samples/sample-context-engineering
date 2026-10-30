# F002 Diagnostic — Sonnet 5.5 / OpenCode

## Result

The isolated F002 prompt completed successfully in a fresh OpenCode session with:

- OpenCode `1.18.34`
- Model `amazon-bedrock/anthropic.claude-sonnet-5-5`
- MCP fixture `ce60-fixture`
- `ExecutionAdmission` admitted
- Bridge `python3.13` started successfully
- No API key loaded from `.env`

The model performed real MCP calls (`list_accounts`, `get_connector_status`) and returned a terminal answer.

## Contrast with CE-60

The same F002 turn timed out in the accumulated 60-turn Sonnet 5.5 baseline and A sessions, including with a 600-second per-turn timeout. Therefore the failure is not a basic provider, MCP startup, or bridge startup failure.

The failure is associated with the accumulated conversation state and/or the model/tool loop after the preceding CE-60 turns. The isolated prompt did not reproduce it.

## Observations

- Bridge startup: passed.
- Live admission: passed.
- MCP connection: passed.
- Model invocation: passed.
- Isolated F002 terminal response: passed.
- CE-60 accumulated F002: timed out in baseline and A.
- A plugin filtering was not needed for the isolated result because the model used small account/status outputs; this diagnostic was about lifecycle/timeout isolation.

## Decision

Do not treat the 5.5 CE-60 timeout as a provider-unavailable result. Before another full matrix, add an instrumented per-turn/session diagnostic for:

1. accumulated message count and estimated context size;
2. tool-call loop count per turn;
3. MCP tool latency and error state;
4. compaction/session summary events;
5. model request start/end and timeout boundary;
6. whether the failing turn follows a context compaction or an unresolved tool call.

A future benchmark must preserve the CE-60 conversation semantics, so silently resetting the session before F002 would invalidate the comparison. If a controlled compaction policy is introduced, it must be applied identically to baseline and A and recorded in the manifest.
## Follow-up — clean CE-60 Sonnet 5.5 R1

After the plugin, bridge, model-binding, MCP-prefix and telemetry fixes, a clean paired CE-60 R1 was executed with the real Bedrock model and the shared deterministic MCP fixture. The integration path passed, but `F002-filler` timed out again in both accumulated sessions with the 600-second per-turn deadline:

- baseline: 59/60 completed; 30/30 correctness-eligible turns correct;
- arm A: 59/60 completed; 30/30 correctness-eligible turns correct;
- both arms: one timeout, exactly `F002-filler`;
- no compaction event was observed;
- baseline telemetry: 2,258 total tool calls, 1,885 MCP calls, maximum 56 tool calls, maximum 159 messages, median 92.5 messages;
- arm A telemetry: 2,232 total tool calls, 1,814 MCP calls, maximum 55 tool calls, maximum 156 messages, median 90 messages;
- arm A filtering path: 54 transform events, one previewed/filtered result, one successful Bedrock reranker call;
- arm A model-input collection was valid for completed turns but invalid at run scope because the timed-out turn prevented a complete run metric.

The new evidence strengthens the original conclusion: the failure is reproducible only after accumulated CE-60 context/tool activity, not during isolated provider, MCP, bridge or reranker startup. Gate A therefore remains `not_evaluable`, and additional full repetitions are paused until the controlled compaction/tool-loop diagnostic is completed. Full evidence is in `ce60-5-5-REPORT.md` and the `ce60-mcp-5-5-sonnet-clean-r1` result directory.

## Root cause — `question` tool blocking a headless session

The follow-up section above over-read the telemetry: its tool-call totals were sums of cumulative snapshots (real whole-session count ~55-56) and "no compaction" should read `unavailable`. See the erratum in `ce60-5-5-REPORT.md`.

The F002 stall is the built-in OpenCode `question` tool. In the R1 baseline session, the timed-out turn (F002-filler, 19th turn) was an assistant message whose only tool part was `question`, with no `step-finish`, lasting 600.47 s, exactly the prompt timeout. `question` creates a pending request that only ends with a reply or reject from a human (SDK `question.reply` / `question.reject`); the CE-60 runner never answers. The isolated F002 run succeeded because, without the accumulated history, the model had no reason to ask.

Haiku 5.5 probe (`probe-question-haiku55.json`, `.opencode/runner/probe-question.ts`):

- control, no tools map: the model called `question`, the turn blocked for the full 120 s window, classification `blocked_question`; abort returned the session to idle;
- variant, `tools: { question: false }`: completed in ~2 s with no `question` part;
- MCP check with `{ question: false, webfetch: false }`: `ce60-fixture_list_accounts` ran normally;
- cleanup: zero residual processes.

Mitigation, applied identically to both arms and recorded in the manifests: disable `question` through the public per-prompt `tools` map (`--disable-tools question`). Campaign v2 additionally disables `webfetch` as a separate, declared determinism change, because the model was fetching live internet pages outside the deterministic MCP fixture.
