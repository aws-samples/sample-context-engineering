import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import * as contextEngineeringModule from "./context-engineering.ts";
import {
  classifyOutputShape,
  describeUnreadableToolOutput,
  readOutputText,
  writeOutputText,
} from "../lib/tool-output.ts";
import { BridgeClient, type BridgeSpawner, resolveBridgeCommand } from "../lib/bridge-transport.ts";

describe("OpenCode plugin entrypoint", () => {
  test("exports only the default plugin factory", () => {
    expect(Object.keys(contextEngineeringModule)).toEqual(["default"]);
  });
});

describe("tool output extraction", () => {
  test("reads the public output.output string", () => {
    expect(
      readOutputText({
        title: "tool result",
        output: "public tool text",
        metadata: {},
      }),
    ).toBe("public tool text");
  });

  test("reads text-only MCP content blocks", () => {
    expect(
      readOutputText({
        title: "MCP result",
        content: [
          { type: "text", text: "first block" },
          { type: "text", text: "second block" },
        ],
        metadata: { isError: false },
      }),
    ).toBe("first block\nsecond block");
  });

  test("rejects an invalid shape without content and reports only payload-free diagnostics", () => {
    const secretArgument = "SECRET-ARGUMENT-CANARY";
    const secretContent = "SECRET-CONTENT-CANARY";
    const input = {
      tool: "mcp_fixture",
      sessionID: "session-1",
      callID: "call-1",
      args: { secret: secretArgument },
    };
    const output = {
      title: "invalid MCP result",
      content: [{ type: "image", data: secretContent }],
      metadata: {},
    };

    expect(readOutputText(output)).toBeNull();
    expect(readOutputText({ title: "no content", metadata: {} })).toBeNull();

    const diagnostics = describeUnreadableToolOutput(input, output);
    expect(diagnostics).toEqual({
      tool: "mcp_fixture",
      input_keys: ["tool", "sessionID", "callID", "args"],
      output_keys: ["title", "content", "metadata"],
      output_value_types: {
        title: "string",
        content: "array",
        metadata: "object",
      },
      has_output_string: false,
    });
    expect(JSON.stringify(diagnostics)).not.toContain(secretArgument);
    expect(JSON.stringify(diagnostics)).not.toContain(secretContent);
  });

  test("accepts a direct string defensively", () => {
    expect(readOutputText("direct tool text")).toBe("direct tool text");
  });
});

describe("resolveBridgeCommand", () => {
  test("prefers CONTEXT_BRIDGE_COMMAND over PYTHON", () => {
    expect(
      resolveBridgeCommand({
        CONTEXT_BRIDGE_COMMAND: "custom-python",
        PYTHON: "python3.13",
      }),
    ).toBe("custom-python");
  });

  test("falls back to PYTHON when CONTEXT_BRIDGE_COMMAND is unset", () => {
    expect(resolveBridgeCommand({ PYTHON: "python3.13" })).toBe("python3.13");
  });

  test("falls back to python3 when both overrides are unset", () => {
    expect(resolveBridgeCommand({})).toBe("python3");
  });
});

describe("BridgeClient transport", () => {
  test("writes through a FileSink-like stdin and correlates out-of-order JSONL responses", async () => {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const stdout = new ReadableStream<Uint8Array>({
      start(nextController) {
        controller = nextController;
      },
    });
    const requests: Array<{ id: string; type: string }> = [];
    const writes: string[] = [];
    let flushCount = 0;
    const stdin = {
      write(data: Uint8Array) {
        const line = decoder.decode(data);
        writes.push(line);
        requests.push(JSON.parse(line) as { id: string; type: string });
        return data.byteLength;
      },
      flush() {
        flushCount++;
      },
    };
    const spawn: BridgeSpawner = (_command, _options) => ({ stdin, stdout });
    const client = new BridgeClient(() => {}, { spawn });

    const firstRequest = client.request({ type: "first" });
    const secondRequest = client.request({ type: "second" });
    await Promise.resolve();

    expect(requests).toHaveLength(2);
    controller.enqueue(
      encoder.encode(
        [
          { id: requests[1].id, operation: "ok", result: "second" },
          { id: requests[0].id, operation: "ok", result: "first" },
        ]
          .map((response) => JSON.stringify(response))
          .join("\n") + "\n",
      ),
    );
    controller.close();

    await expect(firstRequest).resolves.toMatchObject({ operation: "ok", result: "first" });
    await expect(secondRequest).resolves.toMatchObject({ operation: "ok", result: "second" });
    expect(writes).toEqual([
      JSON.stringify({ type: "first", id: "req-1", idempotency_key: "req-1" }) + "\n",
      JSON.stringify({ type: "second", id: "req-2", idempotency_key: "req-2" }) + "\n",
    ]);
    expect(flushCount).toBe(2);
  });

  test("falls back to a WritableStream stdin when FileSink methods are absent", async () => {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const stdout = new ReadableStream<Uint8Array>({
      start(nextController) {
        controller = nextController;
      },
    });
    const stdin = new WritableStream<Uint8Array>({
      write(data) {
        const request = JSON.parse(decoder.decode(data)) as { id: string };
        controller.enqueue(encoder.encode(JSON.stringify({ id: request.id, operation: "ok" }) + "\n"));
        controller.close();
      },
    });
    const spawn: BridgeSpawner = (_command, _options) => ({ stdin, stdout });
    const client = new BridgeClient(() => {}, { spawn });

    await expect(client.request({ type: "stream-fallback" })).resolves.toMatchObject({ operation: "ok" });
  });
});

describe("tool output write-back (the model-facing text field)", () => {
  // OpenCode 1.18.34 builds an MCP tool's model-facing text from `content[].text` AFTER the
  // `tool.execute.after` hook, so a replacement must land in `content`, not in `output`.
  test("classifies built-in results as output_string and raw MCP results as mcp_content", () => {
    expect(classifyOutputShape({ title: "t", output: "text", metadata: {} })).toBe("output_string");
    expect(classifyOutputShape({ content: [{ type: "text", text: "x" }], isError: false })).toBe("mcp_content");
    expect(classifyOutputShape({ content: [{ type: "text", text: "x" }], output: "" })).toBe("mcp_content");
    expect(classifyOutputShape({ content: [], output: "" })).toBe("none");
    expect(classifyOutputShape(null)).toBe("none");
    expect(classifyOutputShape("direct string")).toBe("none");
  });

  test("writes an MCP replacement into content[] and never into a phantom output field", () => {
    const result: Record<string, unknown> = {
      content: [{ type: "text", text: "ORIGINAL-LARGE-RESULT" }],
      isError: false,
    };
    expect(writeOutputText(result, "PREVIEW")).toBe(true);
    expect(result.content).toEqual([{ type: "text", text: "PREVIEW" }]);
    expect("output" in result).toBe(false);
    expect(readOutputText(result)).toBe("PREVIEW");
  });

  test("collapses several MCP text blocks into one and keeps non-text blocks (attachments) intact", () => {
    const image = { type: "image", mimeType: "image/png", data: "AAAA" };
    const result: Record<string, unknown> = {
      content: [
        { type: "text", text: "part one" },
        image,
        { type: "text", text: "part two" },
      ],
    };
    expect(writeOutputText(result, "PREVIEW")).toBe(true);
    expect(result.content).toEqual([{ type: "text", text: "PREVIEW" }, image]);
  });

  test("keeps writing built-in results to output.output", () => {
    const result: Record<string, unknown> = { title: "read", output: "ORIGINAL", metadata: { n: 1 } };
    expect(writeOutputText(result, "PREVIEW")).toBe(true);
    expect(result.output).toBe("PREVIEW");
    expect(result.metadata).toEqual({ n: 1 });
  });

  test("refuses to write when there is nothing readable to replace", () => {
    expect(writeOutputText({ content: [] }, "PREVIEW")).toBe(false);
    expect(writeOutputText({ title: "t" }, "PREVIEW")).toBe(false);
    expect(writeOutputText("direct string", "PREVIEW")).toBe(false);
    expect(writeOutputText(null, "PREVIEW")).toBe(false);
  });

  test("the plugin writes through writeOutputText instead of assigning output.output", () => {
    const source = readFileSync(join(import.meta.dir, "context-engineering.ts"), "utf8");
    expect(source).toContain("writeOutputText(output, substituted)");
    expect(source).not.toMatch(/\(output as JsonObject\)\.output\s*=/);
  });
});

describe("fidelity with the reference relevance filter", () => {
  const source = readFileSync(join(import.meta.dir, "context-engineering.ts"), "utf8");

  test("closed retrieval results are cleared in the model-visible history through the experimental hook", () => {
    expect(source).toContain('"experimental.chat.messages.transform"');
    expect(source).toContain("clearClosedRetrievalResults(output.messages, RETRIEVE_CONTEXT_TOOL)");
    // The hook honours the switch and can never break the agent loop.
    expect(source).toContain("if (RETRIEVAL_CLEANUP && Array.isArray(output?.messages)) {");
    const hook = source.slice(source.indexOf('"experimental.chat.messages.transform"'));
    expect(hook.slice(0, hook.indexOf("tool: {"))).toContain("catch");
  });

  test("the notice is composed by the excerpt module; the legacy hint stays selectable but is not the default", () => {
    expect(source).toContain("composeFilteredResult({");
    expect(source).toContain("parsePreviewStyle(process.env.CONTEXT_PREVIEW_STYLE)");
    expect(source).toContain("parseRetrievalCleanup(process.env.CONTEXT_RETRIEVAL_CLEANUP)");
    expect(source).not.toContain("to read the full result verbatim");
  });

  test("the retrieval tool accepts the narrowing options and forwards each one to the bridge", () => {
    for (const option of ["pattern", "line_range", "context_lines", "max_chunks", "max_tokens"]) {
      expect(source).toContain(`${option}: tool.schema`);
      expect(source).toContain(`"${option}"`);
    }
    expect(source).toContain("request[option] = args[option]");
  });

  test("the tool description says when NOT to use it and names the narrow options", () => {
    const description = source.slice(source.indexOf("function retrievalToolDescription"), source.indexOf("const ContextEngineeringPlugin"));
    expect(description).toContain("ONLY when the excerpt cannot answer");
    expect(description).toContain("Do NOT use it to re-read");
    expect(description).toContain("`pattern`");
    expect(description).toContain("`line_range`");
    // The promise of removal is made only when the cleanup is on.
    expect(description).toContain("cleanup\n      ? ");
  });

  test("a malformed read option reaches the model as a correctable message that quotes no content", () => {
    expect(source).toContain('response.operation === "invalid_request"');
    expect(source).toContain("throw new Error(`retrieve_context: ${response.message}`);");
  });
});

describe("session liveness", () => {
  const source = readFileSync(join(import.meta.dir, "context-engineering.ts"), "utf8");

  test("sends a heartbeat for every session it has seen, well inside the 300 s lease window, without keeping the process alive", () => {
    expect(source).toContain("const HEARTBEAT_INTERVAL_MS = 120_000;");
    expect(source).toContain('void bridge.request({ type: "heartbeat", session_id: id });');
    expect(source).toContain(".unref?.()");
  });

  test("tracks a session when a tool runs or a retrieval is made, and forgets it when the session is deleted", () => {
    expect(source).toContain("if (sessionID) liveSessions.add(sessionID);");
    expect(source).toContain("liveSessions.add(sessionID);\n          const request");
    expect(source).toContain("liveSessions.delete(sessionID);");
  });
});

describe("plugin revision", () => {
  const source = readFileSync(join(import.meta.dir, "context-engineering.ts"), "utf8");

  test("is logged at startup and shared with the runner that records it in the evidence", async () => {
    const { PLUGIN_REVISION } = await import("../lib/revision.ts");
    expect(PLUGIN_REVISION).toBe(2);
    expect(source).toContain("revision: PLUGIN_REVISION,");
    expect(readFileSync(join(import.meta.dir, "..", "runner", "ce60.ts"), "utf8")).toContain("plugin_rev:${PLUGIN_REVISION}");
  });
});

describe("practice D is opt-in and adds nothing when it is off", () => {
  const source = readFileSync(join(import.meta.dir, "context-engineering.ts"), "utf8");

  test("the switch is CONTEXT_GRAPH and anything but an explicit on leaves the graph off", () => {
    expect(source).toContain("const GRAPH_ENABLED = parseGraphFlag(process.env.CONTEXT_GRAPH);");
  });

  test("the graph hook and the two tools exist only behind the switch", () => {
    expect(source).toContain("const graphHook = GRAPH_ENABLED\n    ? new GraphHook({");
    expect(source).toContain("if (!GRAPH_ENABLED) return {};");
    expect(source).toContain("const graphRun = graphHook && Array.isArray(output?.messages) ? graphHook.begin(output.messages) : null;");
  });

  test("the graph records what arrived BEFORE the retrieval cleanup edits the array, and acts after it", () => {
    const hook = source.slice(source.indexOf('"experimental.chat.messages.transform"'));
    const begin = hook.indexOf("graphHook.begin(output.messages)");
    const cleanup = hook.indexOf("clearClosedRetrievalResults(output.messages, RETRIEVE_CONTEXT_TOOL)");
    const finish = hook.indexOf("await graphRun.finish()");
    expect(begin).toBeGreaterThan(-1);
    expect(begin).toBeLessThan(cleanup);
    expect(cleanup).toBeLessThan(finish);
  });

  test("the plugin never writes to the session: no SDK call that edits messages is made for the graph", () => {
    const graph = readFileSync(join(import.meta.dir, "..", "lib", "graph.ts"), "utf8");
    for (const forbidden of ["session.update", "session.revert", "session.summarize", "session.fork", "session.delete", "message.update", "client.session"]) {
      expect(graph).not.toContain(forbidden);
    }
  });

  test("the registered tool names are the ones the bridge names in its guidance", () => {
    const graph = readFileSync(join(import.meta.dir, "..", "lib", "graph.ts"), "utf8");
    expect(graph).toContain('export const EXPAND_CARD_TOOL = "expand_card";');
    expect(graph).toContain('export const FIND_CONTEXT_TOOL = "find_context";');
  });

  test("the policy only chooses what the model is told expand_card does, and it is read from the same variable as the bridge", () => {
    expect(source).toContain("const GRAPH_MODE = parseGraphMode(process.env.CONTEXT_GRAPH_MODE);");
    expect(source).toContain('description: GRAPH_MODE === "inplace" ? EXPAND_CARD_RESULT_DESCRIPTION : EXPAND_CARD_DESCRIPTION,');
    // Nothing else in the plugin branches on the policy: the bridge decides, the graph module applies whatever comes back.
    expect(source.match(/GRAPH_MODE/g)?.length).toBe(4);
  });
});
