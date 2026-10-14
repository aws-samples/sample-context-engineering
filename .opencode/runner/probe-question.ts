/**
 * Payload-free probe for the F002 hypothesis, over the PUBLIC OpenCode SDK only.
 *
 * Questions answered:
 *   1. control  — does the built-in `question` tool block a headless `opencode serve` session?
 *   2. variant  — does the public per-prompt `tools: { question: false }` map remove the tool?
 *   3. mcp      — do the CE-60 MCP fixture tools still work with that tools map in place?
 *   4. status   — what does `session.status` report for an idle session after a completed prompt?
 *
 * Only tool names, statuses, booleans and classifications are printed. Prompts and answers are never
 * printed or written. Usage (billable, a few cents):
 *   AWS_PROFILE=... AWS_REGION=... bun .opencode/runner/probe-question.ts amazon-bedrock/anthropic.claude-haiku-5-5
 */

import { createOpencodeClient } from "@opencode-ai/sdk";

import {
  type AbortClient,
  classifyTimeoutState,
  collectSessionTelemetry,
  inheritedOpencodeEnvKeys,
  readSessionStatus,
  reapProcessTree,
  startOpencode,
  verifyAbort,
  waitForHealth,
} from "./ce60.ts";

const MODEL = process.argv[2] ?? "amazon-bedrock/anthropic.claude-haiku-5-5";
const TIMEOUT_MS = Number(process.env.PROBE_TIMEOUT_MS ?? 120000);
const [providerID, ...rest] = MODEL.split("/");
const modelID = rest.join("/");

const ASK_PROMPT =
  "Before doing anything else you must call the question tool to ask me which of my accounts I mean. " +
  "If the question tool is not available, reply with the single word UNAVAILABLE.";
const MCP_PROMPT = "List my linked financial accounts using the available tools, then summarize them in one line.";

type Client = ReturnType<typeof createOpencodeClient>;

async function promptOnce(
  client: Client,
  sessionId: string,
  text: string,
  tools?: Record<string, boolean>,
): Promise<Record<string, unknown>> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<"TIMEOUT">((resolve) => {
    timer = setTimeout(() => resolve("TIMEOUT"), TIMEOUT_MS);
  });
  const started = Date.now();
  const result = await Promise.race([
    client.session.prompt({
      path: { id: sessionId },
      body: { model: { providerID, modelID }, ...(tools ? { tools } : {}), parts: [{ type: "text", text }] },
    }),
    timeout,
  ]);
  if (timer) clearTimeout(timer);
  const telemetry = await collectSessionTelemetry(client, sessionId);
  const summary: Record<string, unknown> = {
    timed_out: result === "TIMEOUT",
    elapsed_ms: Date.now() - started,
    tool_names: telemetry.tool_names,
    last_tool: telemetry.assistant?.last_tool ?? null,
    assistant_completed: telemetry.assistant?.completed ?? null,
  };
  if (result === "TIMEOUT") {
    summary.classification = classifyTimeoutState(telemetry);
    const abort = await verifyAbort(client as unknown as AbortClient, sessionId);
    summary.abort_status = abort.abort_status;
    summary.idle_status = abort.idle_status;
  } else {
    summary.status_after = await readSessionStatus(client as unknown as AbortClient, sessionId);
  }
  return summary;
}

async function main(): Promise<number> {
  if (!process.env.AWS_PROFILE || !(process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION)) {
    console.error("AWS_PROFILE and AWS_REGION are required (no default profile is assumed)");
    return 2;
  }
  const env = { ...process.env };
  for (const key of inheritedOpencodeEnvKeys(env)) delete env[key];
  // `baseline` starts `opencode serve --pure`: no plugin, but the project MCP fixture still loads.
  const started = await startOpencode("baseline", env, 60000);
  try {
    await waitForHealth(started.url);
    const client = createOpencodeClient({ baseUrl: started.url });
    const newSession = async (title: string): Promise<string> => {
      const created = await client.session.create({ body: { title } });
      const id = (created.data as { id?: string } | undefined)?.id;
      if (!id) throw new Error("could not create a probe session");
      return id;
    };
    const control = await promptOnce(client, await newSession("probe-question-control"), ASK_PROMPT);
    const variantSession = await newSession("probe-question-variant");
    const variant = await promptOnce(client, variantSession, ASK_PROMPT, { question: false });
    const mcp = await promptOnce(client, variantSession, MCP_PROMPT, { question: false, webfetch: false });
    const report = {
      model: MODEL,
      timeout_ms: TIMEOUT_MS,
      control,
      variant,
      mcp,
      verdict: {
        question_blocks_headless_session: control.timed_out === true && control.classification === "blocked_question",
        tools_map_removes_question:
          variant.timed_out === false && !(variant.tool_names as string[]).includes("question"),
        mcp_tools_work_with_tools_map:
          mcp.timed_out === false && (mcp.tool_names as string[]).some((name) => name.includes("list_accounts")),
      },
    };
    console.log(JSON.stringify(report, null, 2));
    return 0;
  } finally {
    const cleanup = await reapProcessTree(started.proc);
    console.error(`[probe] cleanup residual=${cleanup.residual_process_count} sigkill=${cleanup.sigkill_needed}`);
  }
}

if (import.meta.main) {
  main()
    .then((code) => process.exit(code))
    .catch((error) => {
      console.error(String(error instanceof Error ? error.message : error));
      process.exit(1);
    });
}
