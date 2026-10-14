import { describe, expect, test } from "bun:test";

import { clearClosedRetrievalResults, type HistoryMessage } from "./history.ts";

const TOOL = "retrieve_context";
const NOW = 1_700_000_000_000;

const user = (text = "question"): HistoryMessage => ({ info: { role: "user" }, parts: [{ type: "text", text }] });
const assistant = (...parts: NonNullable<HistoryMessage["parts"]>): HistoryMessage => ({ info: { role: "assistant" }, parts });
const call = (tool: string, status = "completed", time: Record<string, unknown> | null = { start: 1, end: 2 }) => ({
  type: "tool",
  tool,
  state: { status, output: "x".repeat(2000), time },
});
const retrieval = (status = "completed", time: Record<string, unknown> | null = { start: 1, end: 2 }) =>
  call(TOOL, status, time);

describe("clearClosedRetrievalResults", () => {
  test("marks a retrieval of a closed turn as cleared and changes nothing else about the message", () => {
    const kept = call("ce60-fixture_list_accounts");
    const messages = [user("turn 1"), assistant({ type: "step-start" }, kept, retrieval()), user("turn 2")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(1);
    expect(messages).toHaveLength(3);
    expect(messages[1].parts).toHaveLength(3); // nothing removed, nothing reordered
    const cleared = messages[1].parts![2];
    expect(cleared.state?.time?.compacted).toBe(NOW);
    expect(cleared.state?.status).toBe("completed");
    expect(kept.state.time).toEqual({ start: 1, end: 2 }); // another tool is untouched
  });

  test("keeps the retrieval of the turn that is still in flight", () => {
    const closed = retrieval();
    const inFlight = retrieval();
    const messages = [user("turn 1"), assistant(closed), user("turn 2"), assistant({ type: "step-start" }, inFlight)];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(1);
    expect((closed.state.time as { compacted?: number }).compacted).toBe(NOW);
    expect((inFlight.state.time as { compacted?: number }).compacted).toBeUndefined();
  });

  test("clears every retrieval of every closed turn", () => {
    const messages = [user("a"), assistant(retrieval(), retrieval()), user("b"), assistant(retrieval()), user("c")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(3);
  });

  test("never drops a message and never rewrites reasoning (the stored shape stays valid for the provider)", () => {
    const reasoning = { type: "reasoning", text: "thinking", metadata: { anthropic: { signature: "sig" } } };
    const messages = [user("a"), assistant(reasoning, retrieval()), user("b")];
    clearClosedRetrievalResults(messages, TOOL, NOW);
    expect(messages).toHaveLength(3);
    expect(messages[1].parts).toContain(reasoning);
    expect(messages[1].parts).toHaveLength(2);
  });

  test("leaves calls that are pending, running or failed alone", () => {
    const pending = retrieval("pending");
    const running = retrieval("running");
    const failed = retrieval("error");
    const messages = [user("a"), assistant(pending, running, failed), user("b")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(0);
    for (const part of [pending, running, failed]) {
      expect((part.state.time as { compacted?: number }).compacted).toBeUndefined();
    }
  });

  test("never touches another tool, user messages or aborted assistant messages' tool names", () => {
    const other = call("ce60-fixture_list_accounts");
    const messages = [user("a"), assistant(other), user("b")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(0);
    expect((other.state.time as { compacted?: number }).compacted).toBeUndefined();
  });

  test("a history with no newer user message has no closed turn", () => {
    expect(clearClosedRetrievalResults([user("a"), assistant(retrieval())], TOOL, NOW)).toBe(0);
    expect(clearClosedRetrievalResults([], TOOL, NOW)).toBe(0);
  });

  test("a result already cleared by OpenCode itself is not counted again", () => {
    const already = retrieval("completed", { start: 1, end: 2, compacted: 5 });
    const messages = [user("a"), assistant(already), user("b")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(0);
    expect((already.state.time as { compacted?: number }).compacted).toBe(5);
  });

  test("a completed part without a time object is skipped rather than crashing the converter", () => {
    const noTime = retrieval("completed", null);
    const messages = [user("a"), assistant(noTime), user("b")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(0);
  });

  test("tolerates malformed messages without throwing", () => {
    const messages = [
      user("a"),
      {} as HistoryMessage,
      { info: { role: "assistant" }, parts: null } as HistoryMessage,
      { info: null, parts: [retrieval()] } as HistoryMessage,
      user("b"),
    ];
    expect(() => clearClosedRetrievalResults(messages, TOOL, NOW)).not.toThrow();
  });

  test("is idempotent: a second pass over the cleared history changes nothing", () => {
    const messages = [user("a"), assistant(retrieval()), user("b")];
    expect(clearClosedRetrievalResults(messages, TOOL, NOW)).toBe(1);
    expect(clearClosedRetrievalResults(messages, TOOL, NOW + 1)).toBe(0);
    expect((messages[1].parts![0].state!.time as { compacted?: number }).compacted).toBe(NOW);
  });
});
