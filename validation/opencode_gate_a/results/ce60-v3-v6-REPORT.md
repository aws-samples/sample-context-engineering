# CE-60 OpenCode campaigns v3 to v6: does practice A pass Gate A?

## Verdict
- **Practice A on OpenCode (Sonnet 5.5, plugin revision 2) passes the unchanged Gate A in 1 of 3 repetitions.** v6 R1 `no_go` (median model input -14.6 %), R2 `go` (-24.3 %), R3 `no_go` (-14.7 %, plus a one-turn correctness drop that is an oracle false positive). The rule written before v6 ran (at least 2 of 3 repetitions `go`) is **not met**, so the hypothesis is **not supported**.
- What the data do support, in every v6 repetition: retrieval is valid (bridge counters equal the evaluated counters, zero failures, no `GATE_RETRIEVAL_INVALID`); every preview was read back partially and none whole; no re-retrieval loop; both arms 60/60 with zero timeouts; correctness at parity except the false positive below; a saving of 26k to 37k input tokens per turn in turns 13 to 26, once the previews exist (before any later re-run erodes it).
- Mean change of the per-turn median over the three v6 repetitions: **-17.9 %** (range -24.3 % to -14.6 %). With three repetitions that interval contains the 20 % the gate asks for, so this is a near-miss, not a clear failure and not a pass. BENCHMARK.md calls anything under +/-20 % of tokens noise for the same reason.
- Which side of 20 % a repetition lands on is decided by one model behaviour, not by the plugin alone: whether the model re-runs `list_investment_transactions` in the filler turns of arm A only (see "Why the median lands where it does").
- Gate A thresholds and the scenario were not changed at any point, and oracle v4 (introduced in v4, before any v4 result, identical in both arms) was left as it was for v5 and v6. Opus 5.5 was not run on the final plugin.

## Results (Sonnet 5.5, 60-turn paired runs, Gate A unchanged: median input -20 % or better, correctness drop at most 3.0 pp, retrieval valid)
| Campaign | Rep | Correct (base / A) | Median input (base to A) | Change | Gate | Fails on | Previews (full / partial / none reads) |
|---|---|---|---|---:|---|---|---|
| v3 | R1 | 30/30 / 29/30 | 121,394 to 158,110 | +30.2 % | no_go | correctness, input | 2 |
| v3 | R2 | 30/30 / 29/30 | 129,282 to 129,720 | +0.3 % | no_go | correctness, input | 1 |
| v3 | R3 | 29/30 / 29/30 | 124,796 to 134,110 | +7.5 % | no_go | input | 1 |
| v4 | R1 | 30/30 / 30/30 | 139,514 to 159,690 | +14.5 % | no_go | input | 2 (2 / 0 / 0) |
| v4 | R2 | 30/30 / 29/30 | 146,874 to 176,814 | +20.4 % | no_go | correctness, input | 3 (3 / 0 / 0) |
| v4 | R3 | 30/30 / 30/30 | 151,902 to 171,941 | +13.2 % | no_go | input | 3 (3 / 0 / 0) |
| v5 (rev 1) | R1 | 30/30 / 30/30 | 143,230 to 115,865 | -19.1 % | no_go | input, retrieval | 2 (0 / 2 / 0) |
| **v6 (rev 2)** | R1 | 30/30 / 30/30 | 146,477 to 125,022 | **-14.6 %** | no_go | input | 4 (0 / 4 / 0) |
| **v6 (rev 2)** | R2 | 30/30 / 30/30 | 155,944 to 118,019 | **-24.3 %** | **go** | - | 4 (0 / 4 / 0) |
| **v6 (rev 2)** | R3 | 30/30 / 29/30 | 149,700 to 127,656 | **-14.7 %** | no_go | correctness, input | 2 (0 / 2 / 0) |

Opus 5.5 completed three repetitions only in v3 (R1 +25.3 %, R2 +15.3 %, R3 +36.0 %, all `no_go`, two of them hit by response-language drift); its v2 and v4 attempts were stopped (a provider 503, then the decision to move to the plugin fixes). v2 is omitted: arm A did not filter anything there (MCP write-back defect, see `ce60-v2-REPORT.md`). v5 R2 was aborted because the plugin files were edited while it ran; it is not evidence.

v6 totals over all model steps (a second view of the same runs): input summed over every step changed -9.0 %, -19.6 % and -13.1 %. Previews and retrievals in v6: 4 / 4, 4 / 4, 2 / 2 (previews / retrievals; bridge counters equal the evaluated per-turn counters in all three).

## The one correctness failure (v6 R3, arm A, 29/30)
It is the turn `B1-status` (a connector status question, no tool result was previewed in it). The answer says "The other four accounts may be affected too", which is correct for a user with five accounts. The oracle's universal `no-fabricated-account-count` check rejects the substring "four accounts" anywhere in an answer. The same false positive cost v4 R2 one turn. Under the rules the oracle is not changed after the fact, so R3 stays `no_go`; the input shortfall alone (-14.7 %) would have made it `no_go` anyway.

## Why the median lands where it does
- The plugin does what it was built to do. In R1, at A5-statement, B2-logs and C1-lambda-docs (twice), it replaces 33 to 79 KB results with 4.9 KB excerpts; the model then reads back only the rows it needs (`pattern`, 1.9 to 19.7 KB, never the whole). In turns 13 to 26, arm A carries 26k to 37k fewer input tokens per turn than the baseline (R1 29k to 31k, R2 30k to 37k, R3 26k), about 20 % to 26 % of the context at that point.
- That lead erodes in later turns when arm A re-runs a tool the baseline does not. The filler turns F013, F017, F025 and F029 ask for "the transactions of account X over the last 30 days". The fixture returns the same generic ledger for every account. Where the baseline still holds the earlier full export, it answers without a call ("I already pulled this export"). Where arm A holds only an excerpt and a cleared retrieval, it calls `list_investment_transactions` again, an 11 KB result (about 8k tokens) that then stays in the context.
- Repetitions by who re-ran the tool in those turns: **R1** only arm A (4 calls), lead falls from -30k to +10k tokens by turn 60, median -14.6 %. **R3** only arm A (2 calls, plus a larger log read), lead falls from -26k to -10k, median -14.7 %. **R2** both arms (the baseline also re-ran it in turns 31, 36 and 45), lead stays at -36k to -38k from turn 21 to 45 and ends at -23k, median -24.3 %, `go`. v5 R1 sits in between (-19.1 %).
- The median of 60 per-turn values sits around turn 30 to 36, exactly where these first re-runs happen, which is why a single repetition flips between -14 % and -24 %. Baseline medians across v4 to v6 span 139k to 156k tokens while arm A with the revision 1 and 2 plugin (v5 R1, v6) spans 116k to 128k.
- This is the normal price of relevance filtering (a filtered result is no longer in the context when a later turn needs it), not a defect of the port. It is also why the reference benchmark reports total tokens and treats anything under +/-20 % as noise.

## What each campaign fixed
| Campaign | Defect found and fixed | Result |
|---|---|---|
| v2 | F002 timeout was the built-in `question` tool waiting for a human; disabled in both arms. | Gate A evaluable. Arm A void (MCP write-back wrote `output.output`, which MCP results ignore). |
| v3 | Write-back through `content[]` (`writeOutputText`), `isRetryable` parse, Python allow-lists for the new error fields. | 6 of 6 `no_go`. Fixture docs cache empty and a lexical oracle made C1 wording noise; per-turn retrieval counters blind; Portuguese drift in either arm. |
| v4 | Real AWS pages pinned (`docs-v4/MANIFEST.json`), oracle v4 (C1 retry-count wording only), retrieval counted from the whole session history, English pinned per prompt. | Verdicts trustworthy; Sonnet 3 of 3 `no_go` at +13 % to +20 % (worse): every preview was read back whole and kept in the session. |
| v5 R1 (plugin rev 1) | Fidelity with the reference Strands filter: partial reads (`pattern`, `line_range`, `max_chunks`, `max_tokens`), an excerpt notice that says what an excerpt cannot answer, closed retrieval exchanges removed from the model-visible history. | -19.1 %. Two defects of rev 1 exposed: the retention lease was never renewed (every reference died 300 s after the first preview, so `GATE_RETRIEVAL_INVALID`), and deleting the exchange made the model re-run the same retrieval every turn (23 retrievals for 2 previews). |
| v6 (plugin rev 2) | Lease renewed on every request plus a plugin heartbeat every 120 s; closed retrieval results are cleared (OpenCode's own `[Old tool result content cleared]`) instead of deleted. | 1 of 3 `go`, mean -17.9 %, retrieval valid in all three. |

## Lease check
No campaign run read an old reference back, so the lease fix was checked on its own, live, with the real plugin and the real bridge (no model call, one Bedrock rerank per session; `results/ce60-v6-lease-soak/`).
- Two sessions each store the same 42.5 KB result (4.9 KB excerpt). The treated session runs inside the real plugin, whose heartbeat fires every 120 s. The control uses a second bridge with no heartbeat and no other request.
- At +660 s (300 s heartbeat window plus 300 s grace) the control's read-back is **refused**: its lease expired. At +665 s the treated session's read-back **returns the requested rows** (7.8 KB). So the heartbeat is what keeps a reference alive through a long silence, and the check discriminates.
- An earlier run with read-backs at +330 s and +420 s succeeded in both sessions, because any request inside the grace window renews the lease (revision 2 renews on every request). Revision 1 renewed nothing and refused exactly at +300 s. That run is not a discriminating test and its log was not kept.
- Not covered: a late read-back inside a full 60-turn model run, which no repetition produced.

## Limits
- n = 3 for the final configuration, one model (Sonnet 5.5), one scenario with a fixture that returns the same ledger for every account. The 95 % interval of the v6 mean is wide (about 4 % to 32 %).
- The pass/fail depends on a behaviour of the baseline arm (skipping a re-run) that the plugin does not control; the same plugin gives -14.6 % and -24.3 % on identical settings.
- Plugin revisions 1 and 2 were designed after reading v4 and v5 data, so v6 is a confirmation run of a changed design, not a blind one; the gate, scenario and oracle did not move.
- Oracle v4 still has one known false positive ("four accounts"), which can flip a repetition on its own (3.33 pp per turn against a 3.0 pp limit).
- The history clearing uses OpenCode's `experimental.chat.messages.transform` hook, which may change between releases (verified on 1.18.34). The stored session is untouched; the effect shows as lower per-turn input after a retrieval turn, not in the database.
- Every retrieval in v6 happened in the same turn as its preview, 3 to 6 seconds later. No campaign run read an old reference back in a later turn, so the lease renewal was not exercised by a late read-back (see the lease check); the model chose to re-run the original tool instead.
- One turn of v6 R2 (baseline arm) waited more than three minutes on a slow provider response and completed inside the 600 s prompt timeout; it did not change that repetition's acceptance.

## What would change the answer (needs your decision)
1. **Accept the near-miss.** Report practice A on OpenCode as a median reduction of 15 % to 24 % (mean 18 %) with correctness parity, below the 20 % gate in 2 of 3 repetitions. No further spend.
2. **Opus 5.5 on the final plugin** (`bash validation/opencode_gate_a/campaigns/ce60-v6.sh opus`, about 1.5 h). It would show whether the effect holds on the other model, with the same rule.
3. **Close one more documented gap with the reference**: the reference scores chunks against the question plus the tool-call arguments; the port uses only the question. Small expected effect, and a new campaign (v7) with its own rule written first.
4. **Revisit the gate itself** (for example a threshold on total input or a larger n). That is a change of the success criterion and is yours to make; I did not touch it.

## Reproduction
- Offline suites: `cd .opencode && bun test`; `cd validation && python3.13 -m pytest opencode_gate_a test_opencode_mcp_server.py -q`; `PYTHONPATH="$PWD/context-core/src" context-core/.venv/bin/python -m pytest context-core/tests -q`.
- Campaign (billable, AWS profile `claude-code`, us-west-2, keep the Mac on AC power): `caffeinate -i -s bash validation/opencode_gate_a/campaigns/ce60-v6.sh sonnet`. Preflight only: `... ce60-v6.sh none`.
- Table of all campaigns: `python3.13 validation/opencode_gate_a/campaigns/summarize_campaigns.py`.
- Per-turn pairing of one run (where the median comes from, which tool one arm re-ran): `python3.13 validation/opencode_gate_a/campaigns/paired_series.py ce60-v6-sonnet55-r1 --tools 31,36`.
- Live lease check: `caffeinate -i -s bash validation/opencode_gate_a/results/ce60-v6-lease-soak/lease_soak.sh` (about 11 minutes, one rerank call per session).
- Per repetition (written by the campaign script): `results/ce60-v6-sonnet55-rN/{gate-a-decision,A-evidence,baseline-evidence,preview-outcomes,matched-calls}.json`.
- Plan and approval: `stat-plan-v6-5-5-3-reps.json`, `user-approved-ce60-v6-5-5-2026`. Probe of revision 2: `results/ce60-v6-probe-a1/probe-report.txt`.
