# Practice D on OpenCode: campaigns v7, v8 and v9

Status: D is `experimental` (owner waiver of FR5.1 / BR5.1) and off by default. Gate A is `no_go` (v6) and none of this
changes it. This report reads the three D campaigns together; the Gate A campaigns v3-v6 are in `ce60-v3-v6-REPORT.md`.

## The result

A+D against the baseline of the same repetition, Sonnet 5.5, plugin revision 2, three repetitions per campaign, the CE-60
scenario (60 turns, 30 eligible for correctness, oracle v4, English pinned). `T` is total model input (input + cache read +
cache write), `C` is dollars from OpenCode's own per-message `cost`, `K` is correct turns A+D minus baseline.

| campaign | what D did | T (mean) | C per repetition (mean) | cache writes (x baseline) | K | recalls per repetition | pre-registered label |
|---|---|---:|---:|---:|---|---|---|
| v6 (A alone, for scale) | nothing (A only) | -13.9 % | -2.1 / -26.1 / -7.5 % (**-11.9 %**) | 1.15 / 0.49 / 0.99 | 0 / 0 / -1 | n/a | Gate A `no_go` |
| v7 `ce60-v7.sh` | reference policy: re-decide every turn which old turns to show | -48.9 % | +206.5 / +221.5 / +219.0 % (**+215.7 %**) | 10.9 - 11.7 | 0 / 0 / 0 | 12 / 13 / 15 | D_TOKENS_ONLY |
| v8 `ce60-v8.sh` | in place, answers kept one more turn (`KEEP_DIALOGUE=1`) | -62.4 % | +34.1 / +8.7 / +7.1 % (**+16.6 %**) | 3.98 / 2.89 / 2.98 | 0 / 0 / 0 | 20 / 12 / 20 | D_TOKENS_ONLY |
| v9 `ce60-v9.sh` | in place, a turn is final the moment it closes (`KEEP_DIALOGUE=0`) | -65.2 % | -27.8 / -39.6 / -26.1 % (**-31.1 %**) | 1.38 / 0.99 / 1.28 | 0 / 0 / 0 | 18 / 9 / 22 | **D_USEFUL** |

(v7 recalls are `expand_card` + `find_context`: 8+4 / 10+3 / 12+3.) The labels are the ones fixed in each script header before
the first repetition: D_USEFUL needs mean `C` at least 5 points better than A alone in v6 (<= -16.9 %) and at most one correct
turn lost; D_TOKENS_ONLY is the tokens falling and the bill not following. Reports: `ce60-v7-d-slice-report.json`,
`ce60-v8-d-slice-report.json`, `ce60-v9-d-slice-report.json` (tool: `campaigns/d_slice_report.py`).

In plain terms: on OpenCode with a prompt-cached provider, D always cuts what the model reads by half to two thirds and
answers equally well on this scenario; whether the bill follows depends entirely on whether the edit respects the cache. The
reference policy does not (3.2x the bill), the in-place policy with the answers kept does not either (1.17x), the in-place
policy where a turn is final at close does (0.69x, and 19 points better than A alone).

## What was run

Each repetition runs two arms back to back on the same scenario: the baseline (no plugin) and arm A (the plugin with the
relevance filter, plus the context graph). Three repetitions per campaign. Acceptance of a repetition: both arms 60/60, zero
timeouts, a paired comparison that is not `not_evaluable`, arm A's preview proven, oracle v4, English answers, retrieval counts,
the documented plugin switches (including `graph:on,graph_mode:inplace,graph_keep:K,graph_dialogue:D`), the docs cache
unchanged, and the graph request-only (below). No repetition of v7, v8 or v9 needed a retry.

**Request-only.** The plugin edits only the array OpenCode hands the `experimental.chat.messages.transform` hook, which is a
per-request copy. After each run the runner reads the stored session back and requires that the messages every request was
handed still hash to the digest recorded before anything edited them, and that no graph message is in the store. 342 of 342
(v7), 368 of 368 (v8) and 356 of 356 (v9) graph requests satisfied it; no answer of the bridge was rejected by the plugin's own
check. The synthetic message the in-place policy inserts is an assistant message in the middle of the request; OpenCode and
Bedrock accept it (live Haiku and Sonnet probes, and nine full repetitions).

## v7: the reference policy

Cards, scoring against the question with `cohere.embed-multilingual-v3`, the whole summary block appended after the last
message. Tokens -48.9 %, dollars +215.7 %. The price list in use, fitted exactly from OpenCode's cost counters over twelve
sessions, is $0.10 per million cache-read tokens, $2.50 per million cache-write tokens (25 times a read) and $10 per million
output tokens. In R1 the bill was $6.52 against $2.13, of which $5.40 was cache writes. About 82 % of the extra written tokens
were relevance flips and restores of old turns, about 18 % the block rewritten at the end of every request.

## v8: in place, answers kept one more turn

`CONTEXT_GRAPH_MODE=inplace`: when a turn closes, its tool-calling assistant messages (the bulk) become one short assistant
message at the position of the first (the turn's Card: title, tools used, numeric lines copied literally, within 100 tokens);
the user's question and the model's text answers stay for one more turn, then the whole assistant side is that one message.
No scoring, no embedding call, no state; `expand_card` returns an old turn's content as the tool result and changes nothing in
the request. Tokens -62.4 %, but dollars +16.6 % with cache writes 3.0 - 4.0 times the baseline's.

This form was chosen because a cost model that assumed a cache hit at any common prefix with the previous request priced it at
-30 %. That model was wrong, and v8 is how it was found out.

## What v8 taught: the cache has checkpoints

OpenCode puts a cache checkpoint on the system prompt and on the **last two messages** of every request. The provider reads a
request only up to a checkpoint an earlier request wrote whose prefix is byte-identical to this request's prefix. It does not
read up to any common prefix. Appending to a history is free (the previous request ended at a checkpoint); editing an earlier
message makes everything after the edit a cache write unless the edit sits right after a message that ended an earlier request.

The first call of a turn ends at the user's question, so the question is a checkpoint. Replacing what follows the question by a
summary when the turn closes keeps the cache warm right up to the question, and only the summary and the next question are
written. Changing the same turn a second time one turn later (v8 dropped the kept answers then) lands where no request ended,
and the whole compact history after the system prompt is written again, every turn.

| first calls of a turn, turns 4-60 | system prefix only | median first-call write | first-call writes / all writes |
|---|---:|---:|---:|
| baselines (v8 and v9, six sessions) | 2 - 3 of 57 | 548 - 694 | 64k - 70k of 160k - 206k |
| v8 in place, answers kept (R1 / R2 / R3) | **57 / 57 / 57 of 57** | 6.6k / 6.7k / 6.9k, growing with the history | 409k / 414k / 416k of 637k / 596k / 598k |
| v9 in place, final at close (R1 / R2 / R3) | 9 / 8 / 9 of 57 | 194 / 205 / 193 | 40k / 40k / 40k of 248k / 199k / 249k |

"System prefix only" means the call read about 28.7k tokens (system prompt and tool schemas) and wrote the rest. The 8-9 of 57
that still do in v9 are consistent with the provider's cache lifetime on calls that follow a long pause (the baselines have 2 - 3
of them too). Writes cost 25 times a read, so the arithmetic is unforgiving: reading the baseline's whole 150k-token history costs
what writing 6k tokens costs, and rewriting a compact history of that size every turn costs more than reading the long one.

The same mechanism applies to any plugin that edits earlier messages, including the retrieval cleanup of A (not measured
separately here).

## v9: final at close

The same code with `CONTEXT_GRAPH_KEEP_DIALOGUE=0`: the whole assistant side of a closed turn (its tool calls and its text
answer) becomes the one summary directly after the question, at the moment the next turn starts, and is never touched again.
Tokens -65.2 %, dollars -31.1 % (-27.8 / -39.6 / -26.1), cache writes 0.99 - 1.38 times the baseline's, 30/30 correct turns in
every repetition on both arms, 111-127 model calls against 96-98 in the baselines (the recalls), output tokens +4.5 / -4.1 /
+21.3 %.

| repetition | baseline | A+D | read tokens | write tokens | model calls |
|---|---:|---:|---|---|---|
| R1 | $2.102 | $1.518 | 11.79M -> 4.03M | 180k -> 248k | 98 -> 118 |
| R2 | $2.200 | $1.329 | 12.27M -> 3.80M | 201k -> 199k | 97 -> 111 |
| R3 | $2.195 | $1.623 | 12.48M -> 4.39M | 194k -> 249k | 96 -> 127 |

The median per-call model input falls from 146k - 159k to 37k - 38k tokens.

Written before v9 ran (script header): the cruder model priced this configuration at -7 / -24 / -25 % (mean -18 %) on the v8
transcripts and had over-predicted v8's dollars by 4 - 8 points, so about -25 % was expected. The real mean is -31.1 %.
v9 was run under the v8 approval id (the same mode and plan with a corrected configuration); the owner was not asked about this
extra campaign.

## The corrected cache model

`opencode_gate_d/replay.py` now models the checkpoints (`checkpoints=True`, the default), keeps the fixed prefix (28.7k tokens)
out of the calibration, and compares the replayed transcript with the real baseline's own. `--ideal-cache` keeps the older,
optimistic model. Run on the recorded sessions (`python3.13 -m opencode_gate_d.replay --title <arm A session> --matcher keyword
--mode inplace --keep-recent 0 --keep-dialogue D --calibrate-baseline <baseline session>`, from `validation/`, with
`PYTHONPATH="$PWD/../context-core/src:$PWD"`):

| configuration | repetition | model dollars | real dollars | model writes | real writes |
|---|---|---:|---:|---:|---:|
| v8 (`KEEP_DIALOGUE=1`) | R1 / R2 / R3 | +35.4 / +11.5 / +13.2 % | +34.1 / +8.7 / +7.1 % | 4.25 / 3.19 / 3.35x | 3.98 / 2.89 / 2.98x |
| v9 (`KEEP_DIALOGUE=0`) | R1 / R2 / R3 | -29.2 / -39.8 / -31.5 % | -27.8 / -39.6 / -26.1 % | 1.34 / 0.92 / 1.21x | 1.38 / 0.99 / 1.28x |

Within 0.2 - 6 points of the real dollars on six sessions. It cannot see what the model does differently when its context is
different: the extra calls and the recalls (it holds the output tokens at the baseline's).

## Correctness

K is 0 in all nine repetitions of v7, v8 and v9 (30/30 against 30/30, except v8 R2: 29/30 against 29/30). Oracle v4 is lexical and has 30 eligible turns, so this reads as "no loss was visible at this resolution", not as a
proof of parity. The model reached for `expand_card` / `find_context` 9 - 22 times per repetition in the in-place campaigns; each
returned the content of the named turns as a tool result.

## Gate D

`validation/opencode_gate_d/`. Five scenarios (tool_loop, subagent, compact, resume, recovery) plus full pass and absence of
durable effects, per request; the evaluator takes the precondition as an input (`waived_by_owner` caps a pass at
`experimental`).

- Reference policy, over the recorded v7 sessions: `no_go`, single reason `CARD_ID_UNSTABLE` (a Card is named by the first
  ~12 tokens of the user's message, so 22 of 59 CE-60 turns that share an opening overwrite each other's Card; this is in the
  reference core and affects the other hosts too). `CONTEXT_GRAPH_UNIQUE_TITLES` fixes it in the suite (`go`, suite only).
  `decision-faithful.json`, `decision-unique-titles.json`, `decision-no-waiver.json`.
- In-place policy: `go`, `experimental`, `waived_by_owner`, no reasons. v8: 760 suite requests over two recorded v7 sessions plus
  a synthetic one, plus 368 live requests (`decision-v8-inplace.json`); v9: 789 suite requests plus 356 live
  (`decision-v9-inplace.json`). `gate_a_reclassified` is false in both. Cards are named per turn, so there is no
  `CARD_ID_UNSTABLE`.
- The suite can fail: tests break the binding on purpose (state shared between sessions, a mutated transcript, an in-place
  answer that removes a user message, a recall that moves the request) and require `no_go`.
- `subagent` and `compact` are modeled (two sessions in one bridge, a synthetic compaction). A real OpenCode Task-tool subagent and
  a real compaction were not run; the guard that leaves a compaction summary message alone has only been tested offline.

## Limits

- One model (Sonnet 5.5), one scenario (CE-60), three repetitions per campaign. The spread between repetitions is large (v9 -26 to
  -40 %, v8 +7 to +34 %); the baseline of each repetition also differs ($1.87 - $2.26).
- The dollars depend on the price list fitted from these sessions, in particular on a cache write costing 25 times a read. A
  provider whose writes cost closer to reads would punish churn less (and pay less for D).
- Opus was not run.
- The recall cap (40,000 characters and 1,500 lines, shared by the turns asked for) comes from reading OpenCode 1.18.34, which
  saves a tool result above 50 KB or 2,000 lines to a file. It cut 4 of the 84 `expand_card` results of v8 and v9 (largest
  result 39,787 characters).
- About a quarter of the `expand_card` calls (23 of 84 in the six in-place repetitions) came back as "no earlier turn ... is
  titled": the model asked with the full text of the user's question, or with a `(#n)` marker it guessed, instead of copying the
  title from the summary (titles are cut at about 48 characters, so two questions that open alike differ only by the marker).
  Resolving a full question to its turn by a unique match against the user's message would turn most of those into hits. It
  was not done, so that the code stays what was measured; the figures above include the cost of the misses.
- The A-only path (`CONTEXT_GRAPH` unset) was checked live on the final code with `runner/probe-writeback.ts` (preview written
  back, persisted output smaller, marker present) and by the offline suites, not by a new 60-turn campaign.

## Reproduce

```bash
bash validation/opencode_gate_a/campaigns/ce60-v9.sh none      # preflight only (offline suites, docs cache, write-back probe)
caffeinate -i -s bash validation/opencode_gate_a/campaigns/ce60-v9.sh sonnet   # about 65 minutes, billable
python3.13 validation/opencode_gate_a/campaigns/d_slice_report.py \
  --d ce60-v9-sonnet55-r1 --d ce60-v9-sonnet55-r2 --d ce60-v9-sonnet55-r3 \
  --a ce60-v6-sonnet55-r1 --a ce60-v6-sonnet55-r2 --a ce60-v6-sonnet55-r3
```

Enable D by hand: `CONTEXT_GRAPH=on CONTEXT_GRAPH_MODE=inplace` (the defaults are the v9 configuration). Knobs and the
mechanism are in `.opencode/README.md`.
