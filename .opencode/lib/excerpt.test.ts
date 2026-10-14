import { describe, expect, test } from "bun:test";

import { approximateTokens, composeFilteredResult, parsePreviewStyle, parseRetrievalCleanup } from "./excerpt.ts";

const ORIGINAL = Array.from({ length: 480 }, (_, i) => `row ${i}`).join("\n");
const PREVIEW = "row 0\nrow 1\n[... 478 lines omitted ...]";
const BASE = {
  preview: PREVIEW,
  referenceId: "REF-1",
  originalText: ORIGINAL,
  excerpt: { total_lines: 480, total_chunks: 4, shown_chunks: 1, shown_spans: [[1, 120]] },
  retrievalTool: "retrieve_context",
  style: "scoped" as const,
  cleanup: true,
};

describe("composeFilteredResult (scoped)", () => {
  const text = composeFilteredResult(BASE);

  test("starts with the marker the campaign analysis recognises and says it is an excerpt", () => {
    expect(text.startsWith("[context-engineering] EXCERPT, not the whole result")).toBe(true);
    expect(text).toContain("original: 480 lines");
    expect(text).toContain("shown: lines 1-120 (1 of 4 chunks)");
  });

  test("says which questions the excerpt cannot answer and does not invite a blind full read", () => {
    expect(text).toContain("needs every row");
    expect(text).toContain("maximum, minimum, total, count");
    expect(text).not.toContain("read the full result verbatim");
  });

  test("names the narrowest options and the reference, and keeps the preview verbatim at the end", () => {
    expect(text).toContain('reference_id="REF-1"');
    expect(text).toContain("`pattern`");
    expect(text).toContain("`line_range`");
    expect(text).toContain("`max_chunks`");
    expect(text.endsWith(PREVIEW)).toBe(true);
  });

  test("promises that retrieved content is cleared only when the cleanup is on, and says not to re-check", () => {
    expect(text).toContain("cleared from the conversation after the turn that uses it");
    expect(text).toContain("[Old tool result content cleared]");
    expect(text).toContain("do not retrieve again just to re-check an earlier answer");
    const off = composeFilteredResult({ ...BASE, cleanup: false });
    expect(off).not.toContain("cleared from the conversation");
    expect(off).not.toContain("re-check");
  });

  test("falls back to its own line count when the bridge reported no statistics", () => {
    const bare = composeFilteredResult({ ...BASE, excerpt: null });
    expect(bare).toContain("original: 480 lines");
    expect(bare).toContain("shown: ~");
  });

  test("without a stored reference it tells the model to say the result was filtered, not to retrieve", () => {
    const noRef = composeFilteredResult({ ...BASE, referenceId: null });
    expect(noRef).toContain("Say that the result was filtered");
    expect(noRef).not.toContain("reference_id=");
  });

  test("never contains a content hash, and mentions no row of the original", () => {
    expect(text).not.toContain("sha256");
    expect(text).not.toContain("row 300");
  });
});

describe("composeFilteredResult (legacy)", () => {
  test("restores the old one-line hint exactly, after the preview", () => {
    const text = composeFilteredResult({ ...BASE, style: "legacy" });
    expect(text).toBe(
      `${PREVIEW}\n\n[context-engineering] This result was relevance-filtered to a preview. ` +
        'Call the retrieve_context tool with reference_id="REF-1" to read the full result verbatim.',
    );
  });

  test("returns the bare preview when no reference exists", () => {
    expect(composeFilteredResult({ ...BASE, style: "legacy", referenceId: null })).toBe(PREVIEW);
  });
});

describe("option parsing", () => {
  test("preview style defaults to scoped and only 'legacy' selects the old one", () => {
    expect(parsePreviewStyle(undefined)).toBe("scoped");
    expect(parsePreviewStyle("")).toBe("scoped");
    expect(parsePreviewStyle("nonsense")).toBe("scoped");
    expect(parsePreviewStyle(" Legacy ")).toBe("legacy");
  });

  test("retrieval cleanup is on unless explicitly switched off", () => {
    expect(parseRetrievalCleanup(undefined)).toBe(true);
    expect(parseRetrievalCleanup("1")).toBe(true);
    for (const off of ["0", "false", "OFF", " no "]) expect(parseRetrievalCleanup(off)).toBe(false);
  });

  test("approximateTokens is ceil(chars / 4)", () => {
    expect(approximateTokens("abcd")).toBe(1);
    expect(approximateTokens("abcde")).toBe(2);
    expect(approximateTokens("")).toBe(0);
  });
});
