type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function valueType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function readTextBlocks(content: unknown): string | null {
  if (!Array.isArray(content) || content.length === 0) return null;

  // Only text blocks carry model-facing text. Non-text blocks (image, resource) are ignored here and
  // preserved on write-back, so a result that mixes text with an attachment is still filterable.
  const textBlocks: string[] = [];
  for (const block of content) {
    if (!isRecord(block)) return null;
    if (block.type !== "text") continue;
    if (typeof block.text !== "string") return null;
    textBlocks.push(block.text);
  }

  const text = textBlocks.join("\n");
  return text.length > 0 ? text : null;
}

/** Read only text forms exposed by the public tool hook or an MCP result. */
export function readOutputText(output: unknown): string | null {
  if (typeof output === "string") return output.length > 0 ? output : null;
  if (!isRecord(output)) return null;

  if (typeof output.output === "string" && output.output.length > 0) return output.output;
  return readTextBlocks(output.content);
}

/** Where the text of a tool result lives, which decides where a replacement must be written. */
export type OutputShape = "output_string" | "mcp_content" | "none";

/**
 * Classify a tool result by the field OpenCode 1.18.34 actually reads the model-facing text from.
 *
 * Built-in tools return `{ title, output: string }` and the text is read from `output`. A direct MCP
 * tool returns the raw MCP result `{ content: [{ type: "text", text }, ...] }`; OpenCode builds the
 * model-facing text from `content[].text` AFTER the `tool.execute.after` hook, so for MCP results a
 * replacement written to `output` is silently discarded and `content` must be rewritten instead.
 */
export function classifyOutputShape(output: unknown): OutputShape {
  if (!isRecord(output)) return "none";
  // MCP results carry `content` blocks and no usable string `output` at hook time.
  if (readTextBlocks(output.content) !== null && !(typeof output.output === "string" && output.output.length > 0)) {
    return "mcp_content";
  }
  if (typeof output.output === "string" && output.output.length > 0) return "output_string";
  return "none";
}

/**
 * Write a replacement text into the field OpenCode reads, in place.
 *
 * - `output_string`: sets `output.output`.
 * - `mcp_content`: replaces the text blocks with ONE text block carrying the replacement and keeps any
 *   non-text block (image, resource) untouched so attachments are never lost.
 *
 * @returns Whether a replacement was written. A result with no readable text is left untouched.
 */
export function writeOutputText(output: unknown, replacement: string): boolean {
  if (!isRecord(output)) return false;
  const shape = classifyOutputShape(output);
  if (shape === "output_string") {
    output.output = replacement;
    return true;
  }
  if (shape === "mcp_content" && Array.isArray(output.content)) {
    const nonText = output.content.filter((block) => !(isRecord(block) && block.type === "text"));
    output.content = [{ type: "text", text: replacement }, ...nonText];
    return true;
  }
  return false;
}

export type UnreadableToolOutputDiagnostics = {
  tool: string | null;
  input_keys: string[];
  output_keys: string[];
  output_value_types: Record<string, string>;
  has_output_string: boolean;
};

/** Describe only object shape and primitive value types; never include tool payloads. */
export function describeUnreadableToolOutput(
  input: unknown,
  output: unknown,
): UnreadableToolOutputDiagnostics {
  const inputRecord = isRecord(input) ? input : null;
  const outputRecord = isRecord(output) ? output : null;

  return {
    tool: typeof inputRecord?.tool === "string" ? inputRecord.tool : null,
    input_keys: inputRecord ? Object.keys(inputRecord) : [],
    output_keys: outputRecord ? Object.keys(outputRecord) : [],
    output_value_types: outputRecord
      ? Object.fromEntries(
          Object.entries(outputRecord).map(([key, value]) => [key, valueType(value)]),
        )
      : {},
    has_output_string: typeof outputRecord?.output === "string",
  };
}
