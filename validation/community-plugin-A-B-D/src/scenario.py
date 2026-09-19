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
# The eighteen hand-written turns above are the spine: the same order, the same expectations, so
# accuracy stays comparable across script lengths. What ``--total-turns`` adds is conversation, and
# there are two kinds of it.
#
# **Scored filler** carries expectations of its own, keyed by the *kind* in its label rather than by
# the label itself (see ``accuracy.FILLER_CHECKS``). Its prompts are the ones whose answer the mocked
# tools return identically for every account, so one static expectation is correct for all of them.
# The script targets HALF of the requested turns being scored: 60 turns means 30 scored, of which 18
# are the spine and 12 are filler.
#
# **Unscored filler** is the rest. It is where the heavy payloads live -- a 30-day statement export is
# the biggest single result in the suite -- and it carries no expectation because its answer depends
# on the account, on a window the agent chooses, or on both.
#
# Both kinds now ask about accounts that EXIST. They used to interpolate a running index, so "account
# 11" matched nothing in the fixture: measured over 60 turns, 36 of 42 filler turns made no tool call
# at all, and the full stack answered 36 of them by claiming six accounts where there are five,
# duplicating one identifier and mislabelling two institutions. Unscored turns cannot fail, so that
# drift was invisible -- which is exactly why half the script is scored now.

SCORED_FILLER_SPECS: tuple[tuple[str, str], ...] = (
    (
        "allocation",
        "What does the asset class allocation look like on account {account}?",
    ),
    (
        "projection",
        "Project the yield of account {account} over 12 months in the base scenario.",
    ),
    (
        "connector",
        "Is the FinBank Invest connector syncing account {account} normally?",
    ),
    (
        "logs",
        "Pull the error logs of the NeoBank connector while it syncs account {account}.",
    ),
    (
        "metrics",
        "What are the duration metrics of the Lambda that processes account {account}?",
    ),
    (
        "iam",
        "Describe the IAM role that account {account} uses to write to S3.",
    ),
    (
        "objects",
        "What objects exist in the statements bucket for account {account}?",
    ),
)
"""``(kind, prompt)`` pairs whose answers the mocked tools return identically for every account.

That independence is the whole selection criterion: it is what lets one static expectation in
``accuracy.FILLER_CHECKS`` be correct for every turn built from the pair, without the harness having
to derive a per-account expectation at runtime. The ``kind`` is what the label carries and what the
expectation is looked up by.

Phrased like the spine's turns -- no hint about which tool to call -- so tool selection stays the
model's job and the disclosure strategy is tested rather than bypassed.
"""

UNSCORED_FILLER_PROMPTS: tuple[str, ...] = (
    "What is the current balance of account {account} and how did it change over the last week?",
    "Show me the transactions of account {account} over the last 30 days.",
    "How much would it cost to run the processing of account {account} in another region?",
    "Which Lambda function configuration is behind the export for account {account}?",
)
"""Prompts that add mass without adding expectations.

Each one's answer depends on the account, on a window the agent picks, or on an argument it chooses,
so there is no single string a static expectation could require. The statement export in particular
is the heaviest result in the suite, which is what this filler is here to contribute.
"""


def _scored_filler_positions(filler_count: int, scored_count: int) -> frozenset[int]:
    """Return the filler positions that carry expectations, spread evenly across the run.

    Spread rather than grouped, because a scored turn's value is the depth it sits at: the point of
    lengthening the script is to ask a checkable question after the history has grown, and clustering
    the checks at one end would measure one depth many times instead of many depths once.

    Args:
        filler_count: How many filler turns the script has room for.
        scored_count: How many of them must carry expectations.

    Returns:
        The positions, as offsets into the filler sequence.
    """
    if scored_count <= 0 or filler_count <= 0:
        return frozenset()
    if scored_count >= filler_count:
        return frozenset(range(filler_count))
    return frozenset(round(k * filler_count / scored_count) for k in range(scored_count))


def long_script(total: int = 100) -> tuple[Turn, ...]:
    """The scored spine padded to ``total`` turns, with the return turns kept last.

    Half of ``total`` is scored, the spine included: at 60 turns that is 30 scored turns, 18 of them
    the hand-written spine and 12 built from :data:`SCORED_FILLER_SPECS`. The spine is the floor, so a
    ``total`` below twice its length simply yields the spine's own count rather than dropping any of
    it -- an expectation is never removed to hit a ratio.

    Args:
        total: How many turns the script should carry. Values at or below the spine's length return
            the spine unchanged, so a short run is exactly the script it always was.

    Returns:
        The turns, in order: the opening scored lines, then the filler, then the return.
    """
    from .tools import account_ids

    spine = SCENARIO
    opening = tuple(turn for turn in spine if turn.phase != "return")
    closing = tuple(turn for turn in spine if turn.phase == "return")

    filler_count = total - len(spine)
    if filler_count <= 0:
        return spine

    accounts = account_ids()
    # The spine is the floor: never negative, so a short run scores the spine and nothing more.
    scored_needed = max(0, total // 2 - len(spine))
    scored_positions = _scored_filler_positions(filler_count, scored_needed)

    filler: list[Turn] = []
    scored_seen = 0
    for position in range(filler_count):
        account = accounts[position % len(accounts)]
        if position in scored_positions:
            kind, template = SCORED_FILLER_SPECS[scored_seen % len(SCORED_FILLER_SPECS)]
            # The kind travels in the label, which is how accuracy.score_turn finds the expectation
            # without a dictionary entry per generated turn.
            filler.append(
                Turn(
                    label=f"S{scored_seen:03d}-{kind}",
                    phase="filler-scored",
                    prompt=template.format(account=account),
                    rationale=(
                        "Scored mass: a real tool result and a checkable answer, asked at this depth "
                        "of the conversation."
                    ),
                )
            )
            scored_seen += 1
        else:
            template = UNSCORED_FILLER_PROMPTS[position % len(UNSCORED_FILLER_PROMPTS)]
            filler.append(
                Turn(
                    label=f"F{position:03d}-filler",
                    phase="filler",
                    prompt=template.format(account=account),
                    rationale="Unscored mass: a real tool result, and no expectation of its own.",
                )
            )

    return (*opening, *tuple(filler), *closing)


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
