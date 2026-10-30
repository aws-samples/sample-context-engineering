Plugin arm, first attempt (code-v1), stopped by the operator at turn 8 (T08-build-relevance).

At 10:50:06Z the model had started a third `write` tool call in one assistant message; the part stayed `pending`
(no input, no output tokens, message not finished) for more than ten minutes with an ESTABLISHED connection to Bedrock
and no CPU. OpenCode's default `chunkTimeout` is 15 minutes, so it would have waited until then. Provider-side stream
stall, not a plugin failure as far as the logs show (the bridge and graph requests were answering until 10:50).
The attempt is not counted. The harness now sets chunkTimeout (180 s) for both arms, stops an arm at the first turn that
does not succeed, and retries that arm once in a fresh sandbox.
