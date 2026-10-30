/**
 * The notice that travels with a relevance preview.
 *
 * A preview answers "where does the result mention X". It cannot answer anything that needs every row (a maximum,
 * a total, a count), and a model that is not told computes the aggregate over the excerpt and is confidently
 * wrong, or distrusts the sample and fetches everything. The reference Strands filter therefore puts a
 * disclaimer in front of the excerpt: it says this is an EXCERPT, how much of the original it covers, which
 * questions it cannot answer, and the narrowest call that reaches the rest. The port used to append one line
 * that invited the model to read "the full result verbatim", and every preview was fetched back whole.
 *
 * `legacy` keeps that old line, only so the change can be measured against it.
 */

export type PreviewStyle = "scoped" | "legacy";

/** Payload-free numbers the bridge reports about what its selection did. */
export interface ExcerptStats {
  total_lines?: number;
  total_chunks?: number;
  shown_chunks?: number;
  shown_spans?: number[][];
}

const CHARS_PER_TOKEN = 4;

export function approximateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

function describeSpans(spans: number[][] | undefined): string {
  if (!spans || spans.length === 0) return "none";
  return spans
    .filter((span) => span.length === 2)
    .map(([start, end]) => (start === end ? `${start}` : `${start}-${end}`))
    .join(", ");
}

function countLines(text: string): number {
  return text === "" ? 0 : text.split("\n").length;
}

export interface NoticeParams {
  /** The preview the bridge built (a verbatim excerpt with gap markers). */
  preview: string;
  /** Opaque reference to the preserved original, or null when none could be stored. */
  referenceId: string | null;
  /** The original result; only its size is used. */
  originalText: string;
  excerpt?: ExcerptStats | null;
  retrievalTool: string;
  style: PreviewStyle;
  /** Whether retrieved content is cleared from later turns (so the notice may say so). */
  cleanup: boolean;
}

/**
 * Compose the text that replaces an oversized tool result.
 *
 * Always starts with the `[context-engineering]` marker, which the campaign analysis uses to recognise a
 * persisted preview.
 */
export function composeFilteredResult(params: NoticeParams): string {
  const { preview, referenceId, originalText, excerpt, retrievalTool, style, cleanup } = params;

  if (style === "legacy") {
    if (!referenceId) return preview;
    return (
      `${preview}\n\n[context-engineering] This result was relevance-filtered to a preview. ` +
      `Call the ${retrievalTool} tool with reference_id="${referenceId}" to read the full result verbatim.`
    );
  }

  const totalLines = excerpt?.total_lines ?? countLines(originalText);
  const totalTokens = approximateTokens(originalText);
  const shownChunks = excerpt?.shown_chunks;
  const totalChunks = excerpt?.total_chunks;
  const coverage =
    shownChunks !== undefined && totalChunks !== undefined
      ? `shown: lines ${describeSpans(excerpt?.shown_spans)} (${shownChunks} of ${totalChunks} chunks)`
      : `shown: ~${approximateTokens(preview)} tokens`;
  const head =
    `[context-engineering] EXCERPT, not the whole result | original: ${totalLines.toLocaleString("en-US")} lines, ` +
    `~${totalTokens.toLocaleString("en-US")} tokens | ${coverage}\n` +
    "Use the excerpt for anything it answers. It cannot answer a question that needs every row: a maximum, " +
    "minimum, total, count, average, ranking or any comparison across the whole result. Do not estimate those " +
    "from the excerpt.";

  if (!referenceId) {
    return `${head} Say that the result was filtered instead of computing it from the excerpt.\n\n${preview}`;
  }
  const removal = cleanup
    ? " What you retrieve is cleared from the conversation after the turn that uses it (it then shows as " +
      "[Old tool result content cleared]), so state the figures you relied on in your answer, and do not retrieve " +
      "again just to re-check an earlier answer."
    : "";
  const how =
    `\nTo read more, call \`${retrievalTool}\` with reference_id="${referenceId}" and the narrowest option that ` +
    "works: `pattern` (regex for only the rows you need), `line_range` ({start, end} using the same line numbers as " +
    "the \"[... N lines omitted ...]\" markers) or `max_chunks`. With no option it returns the whole result " +
    `(~${totalTokens.toLocaleString("en-US")} tokens).${removal}`;
  return `${head}${how}\n\n${preview}`;
}

/** Parse the preview style from an env value; anything unknown selects the default. */
export function parsePreviewStyle(value: string | undefined): PreviewStyle {
  return (value ?? "").trim().toLowerCase() === "legacy" ? "legacy" : "scoped";
}

/** Whether closed retrieval exchanges are removed from later turns. On unless explicitly disabled. */
export function parseRetrievalCleanup(value: string | undefined): boolean {
  return !["0", "false", "off", "no"].includes((value ?? "").trim().toLowerCase());
}
