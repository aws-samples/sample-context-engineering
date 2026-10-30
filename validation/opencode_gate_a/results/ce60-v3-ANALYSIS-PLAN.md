# CE-60 v3 — analysis plan (written before R2/R3 results were read)

Written after v3 Sonnet R1 and before R2/R3 so that the interpretation is not fitted to the numbers.

## What R1 established (measured, not assumed)

1. **The plugin now works on MCP tools.** The persisted tool output of the oversized call is a ~4.1 KB preview with the retrieval hint instead of the original 33,677 B (v2 persisted the original).
2. **Where the plugin acts, the effect is large and real.** Matching the identical call in both sessions, the next assistant step read **20,612 fewer input tokens** in arm A (33,677 B → 4,138 B).
3. **The model used the retrieval path once** (`retrieve_context`, turn 6, ~8.4k tokens returned) and the content was recovered. The preview-then-retrieve contract works; its cost shows up as one extra step.
4. **Gate A's statistic is dominated by something the plugin does not control.** The Gate A decision thresholds the *median per-turn model input* across the run. In R1 the arm-A median was +30% above baseline. The step-level data show why: for the same turn, the agent chose different tool arguments in the two sessions (a connector-log read of 7,935 B in baseline vs 15,822 B in A; both below the 32 KB gate, so the plugin did not touch either). That one divergence lifts every later turn of that session by ~7k tokens. Across 39 matched calls the plugin never touched, the next-step input already differs by +145 to +13,851 tokens between sessions.

## Consequence

The CE-60 Gate A statistic cannot separate "practice A works" from "the two sessions happened to make different tool calls", because baseline and A are independent stochastic sessions. The inert-plugin control (v2) showed the same statistic ranging from −1.7% to +8.3% with a true effect of zero. The plugin's real, measured benefit (~20k tokens, once, on the single result above the gate) is smaller than that noise on a ~120k-token median.

## Rules fixed in advance

- Gate A thresholds and the evaluator are **not** changed. A `no_go` stays a `no_go`.
- A repetition counts only if both arms are 60/60 with zero timeouts and the persisted-preview check passes.
- Report, for every repetition, three numbers side by side: (a) the Gate A median delta, (b) the matched-call plugin effect (`analyze_matched_calls.py`), (c) the inert-control noise range from v2. No single number is promoted over the others.
- If all three v3 repetitions are `no_go` on model input while the matched-call effect is consistently negative (a saving), the honest conclusion is: *practice A works as designed and is correctly rejected by Gate A on this workload, because Gate A measures a quantity that agent variance dominates*. That is a finding about the benchmark, and it must not be reported as "A failed" or as "A passed".
- Any proposal to change the gate or the measurement (for example replaying identical recorded tool results in both arms) is a **new experiment definition** to be agreed separately, never applied retroactively to these runs.
