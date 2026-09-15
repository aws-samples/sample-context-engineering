"""The conversation script every configuration replays.

The shape is taken from the measured session, because that session is
what motivated all three strategies: a main subject, a long detour, and a return.

**Topic lines are five turns or more.** This is the harness's second important calibration,
alongside the schema budget, and the line length is what decides whether it measures anything.
Compacting a two-turn subject removes two turns of history — too little to register, so the
history strategy reads as doing nothing when the truth is there was nothing worth compacting.
The case the design was built for had *four* turns of connector debugging accumulating
3,080,381 input tokens, 44.7% of the session, still being resent on every subsequent turn. A
topic line has to carry real mass before removing it can save anything, so each line here
accumulates five or more turns of tool results before the conversation moves on.

Four phases:

- ``line-a`` — investments, five turns. Establishes the subject the run returns to.
- ``line-b`` — a connector failure, five turns. The detour. The history strategy can act on it.
- ``line-c`` — AWS infrastructure, five turns. A second detour, so by the time the run
  returns to line A there are *two* stale subjects resident, not one. This is what
  separates a strategy that tracks the active topic from one that can compact several.
- ``return`` — three turns back on line A, each depending on facts established in the
  opening five. If a stale subject was compacted and not restored, these degrade visibly.

Prompts are phrased as a user would phrase them, with no hint about which tool to call, so
tool selection stays the model's job and the disclosure strategy is tested rather than
bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One user turn in the script."""

    label: str
    phase: str
    prompt: str
    rationale: str


SCENARIO: tuple[Turn, ...] = (
    # --- Line A: investments (5 turns) ---------------------------------------------
    Turn(
        label="A1-accounts",
        phase="line-a",
        prompt=(
            "Good morning. I want a full review of my net worth. "
            "Start by listing all my accounts, with institution and balance."
        ),
        rationale="Opens line A. Establishes the account set the return turns depend on.",
    ),
    Turn(
        label="A2-positions",
        phase="line-a",
        prompt=(
            "In the FinBank investment account, what positions do I hold? "
            "I want the instrument, quantity and value of each one."
        ),
        rationale="The core facts of line A. The return turns ask about these values specifically.",
    ),
    Turn(
        label="A3-allocation",
        phase="line-a",
        prompt="How is that portfolio distributed by asset class?",
        rationale="Adds mass to line A with a third tool result.",
    ),
    Turn(
        label="A4-projection",
        phase="line-a",
        prompt=(
            "If I leave that portfolio untouched, how much should it yield over 12 months "
            "in the base scenario? And in the hawkish scenario?"
        ),
        rationale="Two tool calls in one turn, growing line A further.",
    ),
    Turn(
        label="A5-statement",
        phase="line-a",
        prompt=(
            "Export the last 90 days of statements for that account and tell me what the largest "
            "CDB redemption in the period was and on what date."
        ),
        rationale=(
            "Closes line A with its heaviest payload: a ~34k character CSV where the answer "
            "is one row. Protected content, so numbers must survive verbatim."
        ),
    ),
    # --- Line B: the connector detour (5 turns) ------------------------------------
    Turn(
        label="B1-status",
        phase="line-b",
        prompt=(
            "Hold on, changing the subject: the app has been showing a stale FinBank balance since "
            "the day before yesterday. Is the connector having trouble?"
        ),
        rationale="Hard topic switch. The explicit 'changing the subject' is the signal the classifier can use.",
    ),
    Turn(
        label="B2-logs",
        phase="line-b",
        prompt="Pull the connector's error logs so we can understand the root cause.",
        rationale="Large repetitive log tail — the worst case for a positional prefix preview.",
    ),
    Turn(
        label="B3-force-sync",
        phase="line-b",
        prompt=(
            "MFA_CHALLENGE_TIMEOUT on almost every attempt. Force a full sync "
            "and tell me whether it was accepted."
        ),
        rationale="Extends line B.",
    ),
    Turn(
        label="B4-rotate",
        phase="line-b",
        prompt=(
            "That did not fix it. Rotate the connector's credentials and open a high severity "
            "case for the platform team."
        ),
        rationale="Two more tool calls on line B.",
    ),
    Turn(
        label="B5-verify",
        phase="line-b",
        prompt=(
            "Confirm the connector status after the rotation, in verbose mode, and list the "
            "WARN level logs to see whether the pattern changed."
        ),
        rationale=(
            "Closes line B at five turns with another heavy payload. Line B is now the "
            "connector-debugging mass measured at 44.7% of that session."
        ),
    ),
    # --- Line C: AWS infrastructure (5 turns) --------------------------------------
    Turn(
        label="C1-lambda-docs",
        phase="line-c",
        prompt=(
            "Different subject now. Our connector runs on Lambda. I need to know exactly "
            "how asynchronous invocation retry works and for how long the execution "
            "environment stays reusable — read the AWS documentation and answer with what "
            "the docs say."
        ),
        rationale=(
            "Second topic switch. Pulls a 60k+ character document whose answer sits "
            "mid-page, which is the turn that discriminates the two preview strategies."
        ),
    ),
    Turn(
        label="C2-s3-naming",
        phase="line-c",
        prompt=(
            "We keep the statements in S3. Fetch the official bucket naming rules page "
            "and tell me whether 'octank.extratos.2026' is a valid name and why."
        ),
        rationale="Rendered HTML via Playwright: the opening characters are navigation chrome.",
    ),
    Turn(
        label="C3-metrics",
        phase="line-c",
        prompt=(
            "What is the average duration of the Lambda function over the last 24 hours? "
            "And how much would the connector cost per month at that volume?"
        ),
        rationale="Needs two tools not yet used, so disclosure must spend a find_tools cycle.",
    ),
    Turn(
        label="C4-iam",
        phase="line-c",
        prompt=(
            "Describe the 'connector-worker' IAM role with its inline policies and tell me "
            "whether anything violates least privilege according to AWS best practices."
        ),
        rationale="Another tool plus a documentation read, growing line C.",
    ),
    Turn(
        label="C5-dynamo",
        phase="line-c",
        prompt=(
            "Last one: read the DynamoDB capacity docs and explain when "
            "on-demand comes out cheaper than provisioned for our case."
        ),
        rationale=(
            "Closes line C at five turns. Three stale-capable subjects now exist, and lines B "
            "and C together dominate the history."
        ),
    ),
    # --- Return to line A (3 turns) ------------------------------------------------
    Turn(
        label="R1-largest-asset",
        phase="return",
        prompt=(
            "Ok, set the connector and the infrastructure aside, that is resolved. "
            "Back to my portfolio: considering the positions you pulled at the start of the "
            "conversation, what is my largest asset and what share of the total does it represent?"
        ),
        rationale=(
            "The turn the graph is judged on. It explicitly refers back to A2 and drops two "
            "subjects at once. If either was compacted without restoring line A, this degrades."
        ),
    ),
    Turn(
        label="R2-cross-reference",
        phase="return",
        prompt=(
            "And the CDB redemption you found in the statement: was it larger or smaller than the "
            "Fundo FinBank Absoluto position? Give me both values."
        ),
        rationale=(
            "Requires two facts from two different line-A turns, A5 and A2, ten turns back. "
            "The hardest recall in the script."
        ),
    ),
    Turn(
        label="R3-consolidate",
        phase="return",
        prompt=(
            "Wrap it up for me: a short summary of my net worth and one line on the "
            "connector's final status."
        ),
        rationale=(
            "Needs line A and line B at once — the case where over-aggressive curation shows, "
            "since the connector subject was just declared closed."
        ),
    ),
)

SYSTEM_PROMPT = (
    "You are an Octank financial assistant who also operates the platform's AWS infrastructure. "
    "Answer in English, directly.\n\n"
    "Rules:\n"
    "- Use the available tools to obtain real data before answering. Never invent "
    "values, balances, dates or passages of documentation.\n"
    "- When citing monetary values, reproduce the source's spelling exactly.\n"
    "- If a tool result comes back truncated with storage references, use the "
    "retrieval tool with a pattern or a line range to fetch the missing passage, "
    "instead of retrieving the whole content.\n"
    "- If you do not have the tool you need loaded, search for it by describing what you need "
    "to do before concluding that the capability does not exist.\n"
    "- Be concise: answer what was asked, without repeating the history."
)

PHASES = ("line-a", "line-b", "line-c", "return")

# --- the long script ---------------------------------------------------------------------
#
# Selection and persistence both answer to conversation *length*, and eighteen turns cannot
# show either. The final block costs one entry per Card in every call, so its price is linear
# in the number of turns and invisible at eighteen; and the rebuild scan measured 30ms over an
# 18-turn conversation against 2.9s over a 200-turn one.
#
# The eighteen scored turns are not touched: they stay the accuracy spine, in the same order,
# with the same expectations, so accuracy remains comparable across script lengths. What the
# filler adds is mass — real tool calls producing real payloads, and therefore real Cards —
# and it is deliberately *unscored*: `accuracy.score_turn` returns `scored=False` for a label
# it has no checks for, so filler cannot move the accuracy figure in either direction.
#
# The return turns stay last, which is the whole point of lengthening the script: they now ask
# about facts established eighty turns earlier rather than thirteen.

FILLER_PROMPTS = (
    "What is the current balance of account {index} and how did it change over the last week?",
    "Show me the transactions of account {index} over the last 30 days.",
    "What does the asset class allocation look like on account {index}?",
    "Project the yield of account {index} over 12 months in the base scenario.",
    "Is the TestBank connector syncing account {index} normally?",
    "Pull the error logs of the NeoBank connector for account {index}.",
    "What are the duration metrics of the Lambda that processes account {index}?",
    "Describe the IAM role that account {index} uses to write to S3.",
    "How much would it cost to run the processing of account {index} in another region?",
    "What objects exist in the statements bucket of account {index}?",
)
"""Prompts that add mass without adding expectations.

Ten shapes cycled with a changing index, so each one is a distinct subject with its own Card and
its own tool result rather than a repeat the offloader would serve from one reference. Phrased
like the scored turns — no hint about which tool to call — so tool selection stays the model's job.
"""


def long_script(total: int = 100) -> tuple[Turn, ...]:
    """The scored script padded to ``total`` turns, with the return turns kept last.

    Args:
        total: How many turns the script should carry. Values at or below the scored count return
            the scored script unchanged, so a short run is exactly the script it always was.

    Returns:
        The turns, in order: the opening scored lines, then the filler, then the return.
    """
    scored = SCENARIO
    opening = tuple(turn for turn in scored if turn.phase != "return")
    closing = tuple(turn for turn in scored if turn.phase == "return")

    needed = total - len(scored)
    if needed <= 0:
        return scored

    filler = tuple(
        Turn(
            label=f"F{index:03d}-filler",
            phase="filler",
            prompt=FILLER_PROMPTS[index % len(FILLER_PROMPTS)].format(index=index + 1),
            rationale="Unscored mass: a real tool result, a real Card, and no expectation of its own.",
        )
        for index in range(needed)
    )
    return (*opening, *filler, *closing)


def turns(
    limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    total: int | None = None,
) -> tuple[Turn, ...]:
    """Return the script, optionally lengthened, truncated or filtered by phase.

    ``limit`` is for smoke runs. Filtering by phase is for isolating one strategy, but note
    what each removal costs: dropping ``line-b`` or ``line-c`` removes the stale mass the
    graph exists to compact, dropping ``return`` removes the only check that compacting was
    reversible, and dropping ``line-a`` leaves the return turns asking about facts that were
    never established.

    ``total`` pads the script with unscored filler up to that many turns, keeping the return
    turns last. It is for the two mechanisms that answer to length rather than to subject
    structure: the per-Card cost of the final block, and the cost of deriving the graph.
    """
    selected = SCENARIO if total is None else long_script(total)
    if phases:
        selected = tuple(turn for turn in selected if turn.phase in phases)
    if limit is not None:
        selected = selected[:limit]
    return selected


def line_mass() -> dict[str, int]:
    """Turn count per phase, so a run can assert the lines are long enough to matter."""
    counts: dict[str, int] = {}
    for turn in SCENARIO:
        counts[turn.phase] = counts.get(turn.phase, 0) + 1
    return counts
