/**
 * End-to-end probe: does the plugin's replacement of an oversized MCP result actually reach the model?
 *
 * Starts the real OpenCode (arm A: plugin enabled, live admission, real Bedrock reranker), makes the
 * model call the one CE-60 tool whose result is above the gate, then reads the persisted tool part
 * through the PUBLIC session API and prints only SIZES and booleans: never any tool payload.
 *
 * Pass criteria (all three must hold):
 *   - the bridge metrics report the result as `previewed` (input_bytes > output_bytes);
 *   - the persisted tool output is smaller than the original (it carries the preview, not the result);
 *   - the persisted tool output carries the retrieve_context hint marker.
 *
 *   AWS_PROFILE=... AWS_REGION=... bun .opencode/runner/probe-writeback.ts amazon-bedrock/anthropic.claude-haiku-5-5
 */

import { createOpencodeClient } from "@opencode-ai/sdk";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildArmEnv, inheritedOpencodeEnvKeys, parseArgs, reapProcessTree, startOpencode, waitForHealth } from "./ce60.ts";

const MODEL = process.argv[2] ?? "amazon-bedrock/anthropic.claude-haiku-5-5";
const [providerID, ...rest] = MODEL.split("/");
const modelID = rest.join("/");
const MARKER = "[context-engineering]";
const BIG_TOOL = "list_investment_transactions";
const PROMPT =
  "Call the list_investment_transactions tool for account ACC-001 covering the last 120 days, " +
  "then tell me only how many transactions came back.";

async function main(): Promise<number> {
  if (!process.env.AWS_PROFILE || !(process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION)) {
    console.error("AWS_PROFILE and AWS_REGION are required (no default profile is assumed)");
    return 2;
  }
  const dir = mkdtempSync(join(tmpdir(), "probe-writeback-"));
  const metricsFile = join(dir, "metrics.jsonl");
  const opts = parseArgs([
    "--execute-live",
    "--model",
    MODEL,
    "--approval-id",
    "probe-writeback",
    "--budget-ceiling",
    "5",
    "--budget-unit",
    "USD",
    "--disable-tools",
    "question,webfetch",
  ]);
  const base = { ...process.env } as Record<string, string | undefined>;
  for (const key of inheritedOpencodeEnvKeys(base)) delete base[key];
  base.CONTEXT_OPENCODE_TOOLS = process.env.CONTEXT_OPENCODE_TOOLS ?? BIG_TOOL;
  const env = buildArmEnv("A", base, opts, metricsFile);
  const started = await startOpencode("A", env, 60000);
  try {
    await waitForHealth(started.url);
    const client = createOpencodeClient({ baseUrl: started.url });
    const created = await client.session.create({ body: { title: "probe-writeback" } });
    const sessionId = (created.data as { id?: string } | undefined)?.id as string;
    await client.session.prompt({
      path: { id: sessionId },
      body: {
        model: { providerID, modelID },
        tools: { question: false, webfetch: false },
        parts: [{ type: "text", text: PROMPT }],
      },
    });
    const messages = (await client.session.messages({ path: { id: sessionId } })).data as unknown[];
    const persisted: { tool: string; bytes: number; has_marker: boolean }[] = [];
    for (const message of messages) {
      const parts = ((message as { parts?: unknown[] }).parts ?? []) as Array<Record<string, unknown>>;
      for (const part of parts) {
        if (part.type !== "tool") continue;
        const state = (part.state ?? {}) as { output?: unknown };
        if (typeof state.output !== "string") continue;
        persisted.push({
          tool: String(part.tool),
          bytes: new TextEncoder().encode(state.output).byteLength,
          has_marker: state.output.includes(MARKER),
        });
      }
    }
    const metrics = existsSync(metricsFile)
      ? readFileSync(metricsFile, "utf8")
          .split("\n")
          .filter((line) => line.trim() !== "")
          .map((line) => JSON.parse(line) as Record<string, unknown>)
          .filter((row) => row.event === "transform")
          .map((row) => ({
            tool_class: row.tool_class,
            operation: row.operation,
            input_bytes: row.input_bytes,
            output_bytes: row.output_bytes,
            rerank_result: row.rerank_result,
          }))
      : [];
    const big = persisted.filter((entry) => entry.tool.includes(BIG_TOOL));
    const previewed = metrics.filter((row) => row.operation === "previewed");
    const report = {
      model: MODEL,
      bridge_transforms: metrics,
      persisted_tool_outputs: persisted,
      verdict: {
        bridge_previewed: previewed.length > 0,
        persisted_is_smaller_than_original:
          previewed.length > 0 && big.some((entry) => entry.bytes < Number(previewed[0].input_bytes) / 2),
        persisted_has_preview_marker: big.some((entry) => entry.has_marker),
      },
    };
    console.log(JSON.stringify(report, null, 2));
    const v = report.verdict;
    return v.bridge_previewed && v.persisted_is_smaller_than_original && v.persisted_has_preview_marker ? 0 : 3;
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
