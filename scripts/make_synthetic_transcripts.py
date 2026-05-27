"""Generate the named synthetic benchmark transcripts in `transcripts/`.

Every transcript shipped in this repository is **synthetic**: procedurally
assembled from per-domain speaker rosters, topic banks, and utterance
templates with a fixed RNG seed. No real meeting recordings, personal data,
or third-party corpus text is used. Re-running this script reproduces the
exact same files byte-for-byte.

The six files keep domain-flavoured names so the benchmark sweep and the
figures/summary in `results/` stay meaningful:

    small_ami.txt          product-design team kickoff      (~4.0k words)
    small_swe.txt          engineering sprint standup       (~4.6k words)
    medium_welsh.txt       parliamentary committee session  (~10k words)
    medium_finance.txt     quarterly finance review         (~17k words)
    large_finance.txt      annual budget board meeting      (~46k words)
    large_meetingbank.txt  city council public meeting      (~50k words)

Usage:
    python scripts/make_synthetic_transcripts.py            # all six
    python scripts/make_synthetic_transcripts.py --only small_ami small_swe
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS_DIR = os.path.join(ROOT, "transcripts")
SEED = 20240601


@dataclass(frozen=True)
class Domain:
    """A meeting domain: who is in the room and what they talk about."""
    filename: str
    title: str
    speakers: list[str]
    topics: list[str]
    decisions: list[str]
    actions: list[str]
    target_words: int
    # domain-specific statement fragments speakers riff on
    statements: list[str] = field(default_factory=list)


# Shared conversational connective tissue (domain-agnostic meeting flow).
GENERIC = [
    "Let me make sure I understand the dependency before we commit to a date.",
    "Can someone own that and report back at the next session?",
    "I'd rather we held off until the numbers are in front of us.",
    "That's out of scope for today, let's park it and circle back.",
    "I want to flag a risk the rest of the group may not have visibility into.",
    "We agreed on this direction last time, so let's not relitigate it now.",
    "Honestly, I'm not convinced the timeline is realistic given the scope.",
    "Quick point of order before we move on to the next item.",
    "Let's capture that as an action item with a named owner.",
    "I'll take the first pass and share a draft by the end of the week.",
    "Could we get a second opinion from the team that owns that surface?",
    "For the record, I want my reservations on this noted.",
    "If there are no objections, I'd like to move that we proceed.",
    "Let's timebox this discussion to five minutes and then decide.",
    "I think the data supports the more conservative of the two options.",
    "There's a hard dependency here we need to resolve before we ship.",
    "Can we revisit the assumptions behind that estimate?",
    "I'd like to hear the dissenting view before we lock this in.",
    "Let's make the decision reversible so we can adjust if it goes sideways.",
    "Noted. I'll fold that feedback into the revised proposal.",
]


DOMAINS: dict[str, Domain] = {
    "small_ami": Domain(
        filename="small_ami.txt",
        title="Remote-control product design kickoff",
        target_words=4000,
        speakers=["Project Manager", "Industrial Designer",
                  "User Interface", "Marketing"],
        topics=["the project scope and budget", "the target user persona",
                "the casing material and ergonomics",
                "the button layout and on-screen menu"],
        decisions=["ship a limited beta of the voice-control variant first",
                   "cap the unit cost at twelve euros fifty",
                   "drop the LCD in favour of a simpler LED indicator"],
        actions=["update the requirements doc and recirculate it",
                 "mock up two casing options for the next review",
                 "run a quick usability pass on the menu flow"],
        statements=[
            "The trend research says younger users expect a companion app.",
            "If we keep the curved casing we add tooling cost on the mould.",
            "A single multifunction button tests badly in our usability data.",
            "Marketing wants the brand colour on the bezel, not the buttons.",
            "Battery life is the feature users rank first in the survey.",
            "We can reuse the chipset from the previous generation to save cost.",
            "The menu should never be more than two levels deep.",
            "An LED ring is cheaper than an LCD and still feels premium.",
        ],
    ),
    "small_swe": Domain(
        filename="small_swe.txt",
        title="Platform engineering sprint standup",
        target_words=4600,
        speakers=["Eng Lead", "Backend", "Frontend", "SRE", "QA"],
        topics=["yesterday's production incident", "the sprint burndown",
                "the database migration rollout", "the flaky test suite"],
        decisions=["roll the migration forward behind a feature flag",
                   "freeze non-critical deploys until the incident is closed",
                   "split the monolith test job into three parallel shards"],
        actions=["write the postmortem and share it by Thursday",
                 "add a dashboard alert for queue depth",
                 "quarantine the three flaky integration tests"],
        statements=[
            "The incident was a cache stampede after the config push at nine.",
            "p99 latency doubled for about eleven minutes before we rolled back.",
            "The migration is backwards-compatible if we keep the old column.",
            "CI is red again on the same async test; it's a real race condition.",
            "We're carrying two points of tech debt into the next sprint.",
            "The alert threshold was set too high to catch this early.",
            "I can have the backfill script ready behind a flag by tomorrow.",
            "Let's not deploy on a Friday afternoon again.",
        ],
    ),
    "medium_welsh": Domain(
        filename="medium_welsh.txt",
        title="Parliamentary committee scrutiny session",
        target_words=10000,
        speakers=["Chair", "Member for the North", "Member for the South",
                  "Minister", "Clerk", "Witness"],
        topics=["the draft transport bill", "the rural broadband programme",
                "the health board funding settlement",
                "the consultation responses from local authorities"],
        decisions=["write to the Minister requesting the impact assessment",
                   "schedule a second evidence session with the regulator",
                   "recommend a six-month review clause in the bill"],
        actions=["circulate the written evidence to all members",
                 "publish the committee's interim findings",
                 "invite the auditor general to the next session"],
        statements=[
            "The consultation drew over four hundred responses, mostly critical.",
            "Local authorities say the funding formula disadvantages rural areas.",
            "Minister, can you confirm when the impact assessment will be laid?",
            "The witness's evidence contradicts the department's own figures.",
            "We have a duty to scrutinise the value-for-money case here.",
            "The regulator has not yet provided the data we requested in March.",
            "I'd remind members that this session is being broadcast.",
            "The bill as drafted gives ministers very broad delegated powers.",
        ],
    ),
    "medium_finance": Domain(
        filename="medium_finance.txt",
        title="Quarterly finance review",
        target_words=17000,
        speakers=["CFO", "Controller", "FP&A Lead", "Treasurer",
                  "Audit Partner", "Board Observer"],
        topics=["the Q3 revenue variance", "the cash runway and burn",
                "the headcount and hiring freeze",
                "the audit findings and remediation"],
        decisions=["reforecast full-year revenue down by four percent",
                   "extend the hiring freeze through the next quarter",
                   "draw ten million on the revolving credit facility"],
        actions=["update the board deck with the revised forecast",
                 "close the two open audit findings before year end",
                 "renegotiate the payment terms with the top three vendors"],
        statements=[
            "Revenue came in three point two percent under plan this quarter.",
            "Gross margin held, but operating expense crept up on cloud spend.",
            "Our runway is fourteen months at the current burn rate.",
            "Days sales outstanding have stretched to fifty-one days.",
            "The auditors flagged a control gap in revenue recognition.",
            "We should hedge the euro exposure before the next earnings call.",
            "The deferred revenue balance is healthier than the headline number.",
            "If we draw on the facility now we lock in a better rate.",
        ],
    ),
    "large_finance": Domain(
        filename="large_finance.txt",
        title="Annual budget and strategy board meeting",
        target_words=46000,
        speakers=["CFO", "CEO", "Controller", "FP&A Lead", "Treasurer",
                  "Audit Partner", "Head of Sales", "Head of Engineering"],
        topics=["the three-year financial plan", "the capital allocation policy",
                "the pricing and packaging change",
                "the new market expansion business case",
                "the cost base and efficiency programme"],
        decisions=["approve the capital expenditure envelope for next year",
                   "raise list prices by seven percent on the enterprise tier",
                   "green-light the expansion into the new region in H2",
                   "set the efficiency target at one hundred basis points"],
        actions=["finalise the board-approved budget and distribute it",
                 "model three pricing scenarios for the next meeting",
                 "stand up the expansion programme office",
                 "report monthly on the efficiency programme savings"],
        statements=[
            "The three-year plan assumes a gradual recovery in net retention.",
            "Capital allocation should favour the highest-return product lines.",
            "A seven percent list increase nets to about four after discounting.",
            "The expansion case breaks even in the eighth quarter on base case.",
            "We carry meaningful concentration risk in our top ten accounts.",
            "The efficiency programme targets duplicated tooling and real estate.",
            "Sales capacity is the binding constraint on the upside scenario.",
            "Engineering cloud commitments give us a volume discount next year.",
            "The treasury policy caps counterparty exposure per institution.",
            "We should stress-test the plan against a flat-revenue downside.",
        ],
    ),
    "large_meetingbank": Domain(
        filename="large_meetingbank.txt",
        title="City council regular public meeting",
        target_words=50000,
        speakers=["Mayor", "Council Member Alvarez", "Council Member Brooks",
                  "City Clerk", "Public Works Director", "City Attorney",
                  "Budget Officer", "Resident"],
        topics=["the proposed zoning ordinance", "the annual capital budget",
                "the public transit fare adjustment",
                "the parks and recreation bond measure",
                "the public comment period on the housing plan"],
        decisions=["adopt the zoning ordinance on second reading",
                   "approve the capital budget as amended",
                   "defer the fare adjustment pending a fare-equity study",
                   "place the parks bond measure on the November ballot"],
        actions=["publish the adopted ordinance in the city record",
                 "direct staff to prepare the fare-equity study",
                 "schedule the bond measure public hearing",
                 "respond in writing to the residents' petition"],
        statements=[
            "The ordinance rezones the corridor for mixed-use development.",
            "Residents have raised concerns about parking and traffic impact.",
            "The capital budget funds the bridge repair deferred last year.",
            "A fare increase falls hardest on our lowest-income riders.",
            "The bond measure would fund three new neighbourhood parks.",
            "Staff recommend approval subject to the planning conditions.",
            "I move that we open the floor for public comment on this item.",
            "The city attorney advises that the measure meets statutory notice.",
            "We received over two hundred written comments on the housing plan.",
            "Let the record reflect the motion carried five to two.",
        ],
    ),
}


def _opening(d: Domain, rng: random.Random) -> list[str]:
    chair = d.speakers[0]
    return [
        f"{chair}: Thank you all for joining. On today's agenda we have "
        + ", ".join(d.topics) + ".",
        f"{chair}: Let's work through the items in order and keep to time.",
    ]


def _closing(d: Domain) -> list[str]:
    chair = d.speakers[0]
    workers = d.speakers[1:] or [chair]
    lines = [f"{chair}: Before we close, let me summarise what we decided."]
    for dec in d.decisions:
        lines.append(f"{chair}: We agreed to {dec}.")
    for i, act in enumerate(d.actions):
        owner = workers[i % len(workers)]
        lines.append(f"{chair}: Action item — {owner} will {act}.")
    lines.append(f"{chair}: Thanks everyone. Meeting adjourned.")
    return lines


def _segment(d: Domain, topic: str, rng: random.Random) -> list[str]:
    """One discussion segment about a single topic."""
    chair = d.speakers[0]
    lines = [f"{chair}: Moving on to {topic}. Who'd like to open?"]
    pool = d.statements + GENERIC
    n_turns = rng.randint(6, 12)
    for _ in range(n_turns):
        spk = rng.choice(d.speakers)
        # occasionally chain two sentences for a more natural cadence
        parts = [rng.choice(pool)]
        if rng.random() < 0.35:
            parts.append(rng.choice(pool))
        lines.append(f"{spk}: " + " ".join(parts))
    # land the segment on a mini-decision or action sometimes
    if rng.random() < 0.5:
        lines.append(f"{chair}: Let's note that and move on.")
    return lines


def build_transcript(d: Domain, rng: random.Random) -> str:
    lines = _opening(d, rng)
    word_target = d.target_words
    closing = _closing(d)
    closing_words = sum(len(line.split()) for line in closing)

    def word_count(ls: list[str]) -> int:
        return sum(len(line.split()) for line in ls)

    # Fill with topic segments until we approach the target (leaving room
    # for the closing summary), cycling through the topic list.
    i = 0
    while word_count(lines) + closing_words < word_target:
        topic = d.topics[i % len(d.topics)]
        lines.extend(_segment(d, topic, rng))
        i += 1

    lines.extend(closing)
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=list(DOMAINS),
                   help="Generate only the named transcripts (default: all).")
    args = p.parse_args()

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    names = args.only or list(DOMAINS)
    for name in names:
        d = DOMAINS[name]
        # deterministic per-file seed → reproducible bytes
        rng = random.Random(SEED + sum(ord(c) for c in name))
        text = build_transcript(d, rng)
        out = os.path.join(TRANSCRIPTS_DIR, d.filename)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        words = len(text.split())
        print(f"[synth] {d.filename}: ~{words} words (~{int(words/0.75)} tok) "
              f"-> transcripts/{d.filename}")


if __name__ == "__main__":
    main()
