"""
COSC726 Lab 7 — multi-agent systems (support module)
====================================================
Real model, no mocks, no other lab required.

    pip install openai pydantic
    ollama pull qwen2.5:7b && ollama serve
    python layla_crew_solution.py

What this week actually asks
----------------------------
Not "build a crew" -- that is thirty lines. The question is whether the crew
was worth building, and the 2026 evidence says the burden of proof is on
you. Reported figures: multi-agent implementations typically use 3-10x the
tokens of a single agent for equivalent tasks, and at EQUAL token budgets a
single agent matches or beats multi-agent on reasoning work.

So this kit is built to measure, not to impress. Everything is instrumented
per agent: tokens, wall-clock, and what each role actually contributed.

Public API
----------
    BRIEFS              four research briefs with gold claim sets
    Role, RoleSpec      the three workers plus a critic
    Message             one inter-agent message, logged
    CrewTrace           per-agent tokens, latency, handoffs, verdicts
    single_agent()      the baseline you must beat
    score_output()      grounded claims, unsupported claims, coverage
    compare()           the ledger: crew vs baseline
    CORPUS, search()    a small closed corpus, so failures are yours

The failure taxonomy this lab exercises
---------------------------------------
A 2026 study of five multi-agent frameworks over 150+ tasks catalogued 14
failure modes in three families, and concluded that many are STRUCTURAL --
not fixable by better prompts. The three families:

    1. SPECIFICATION      the roles or the task were badly defined
                          (reported as the largest single category)
    2. INTER-AGENT        misalignment: context lost at a handoff, one agent
                          contradicting another, sycophantic agreement
    3. VERIFICATION       nobody checked, or the crew could not stop

Exercises 2-4 produce one from each family. You are asked to classify what
you observe, not merely to report it.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CORPUS", "search", "BRIEFS", "Brief", "Role", "RoleSpec", "ROLES",
    "Message", "AgentRun", "CrewTrace", "Finding", "Analysis", "Memo",
    "Verdict", "score_output", "compare", "make_client", "MODEL", "PROVIDER",
    "SYSTEMS",
]

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = (os.getenv("OLLAMA_MODEL", "qwen2.5:7b") if PROVIDER == "ollama"
         else os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"))


def make_client():
    """The Week 2 seam. One function; nothing else knows the provider."""
    from openai import OpenAI
    if PROVIDER == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OpenAI(base_url=f"{base}/v1", api_key="ollama")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (or use LLM_PROVIDER=ollama)")
    return OpenAI()


# ---------------------------------------------------------------------------
# 1. The corpus — small and closed, so every failure is one you caused
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "SUP-2401": (
        "Supplier review, Northwind Retail, Q1. Delta Logistics delivered "
        "94% of consignments on time in Q1, down from 97% in Q4. The decline "
        "is concentrated in the northern depot. Delta's contract renews in "
        "September and includes a 2% liquidated-damages clause per week of "
        "delay, capped at 10%."),
    "SUP-2402": (
        "Supplier review, Northwind Retail, Q2. Delta Logistics delivered "
        "89% on time in Q2. Northwind raised two formal claims under the "
        "damages clause, totalling 4% of consignment value. Delta attributes "
        "the decline to a depot relocation completed in June."),
    "FIN-1180": (
        "Cost note. Late-delivery credits paid to customers rose from "
        "£4,100 in Q1 to £9,700 in Q2. The credit policy threshold is three "
        "working days. Approximately 71% of Q2 credits related to "
        "consignments routed through the northern depot."),
    "OPS-0907": (
        "Operations note. The northern depot relocation ran from April to "
        "June. Throughput during the move was roughly 60% of normal. "
        "Contingency routing through the southern depot was available but "
        "was used for only 12% of affected consignments."),
    "LEG-0455": (
        "Legal note. The liquidated-damages clause in the Delta contract is "
        "the sole remedy for late supply; Northwind cannot additionally "
        "claim consequential losses such as customer credits. Terminating "
        "for convenience requires ninety days' notice."),
}


def search(query: str, k: int = 3) -> list[dict[str, str]]:
    """Lexical retrieval over the corpus. Deliberately simple: retrieval is
    not this week's subject, and a fancy retriever would hide the coordination
    failures we are here to see."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for doc_id, text in CORPUS.items():
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(q & words) / (len(q) or 1)
        scored.append((overlap, doc_id, text))
    scored.sort(reverse=True)
    return [{"doc_id": d, "text": t} for s, d, t in scored[:k] if s > 0]


# ---------------------------------------------------------------------------
# 2. The briefs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Brief:
    brief_id: str
    question: str
    gold_docs: set[str]
    gold_claims: list[str]      # substrings that must appear, grounded
    trap: str


BRIEFS: list[Brief] = [
    Brief("B1",
          "Why did customer late-delivery credits rise in Q2, and what can "
          "we recover from the supplier?",
          {"FIN-1180", "SUP-2402", "OPS-0907", "LEG-0455"},
          ["northern depot", "relocation", "sole remedy"],
          "The answer spans four documents and the LEGAL constraint reverses "
          "the obvious conclusion: you cannot recover the credits. A crew "
          "that splits research from analysis often loses LEG-0455 at the "
          "handoff \u2014 inter-agent misalignment, and the commonest MAST "
          "family after specification."),

    Brief("B2",
          "Summarise Delta Logistics' on-time performance across Q1 and Q2.",
          {"SUP-2401", "SUP-2402"},
          ["94", "89"],
          "Two documents, two numbers, no reasoning. A single agent should "
          "match or beat the crew here, and the token ratio is the point: "
          "this is the case that shows coordination costing more than it "
          "buys."),

    Brief("B3",
          "Should we terminate the Delta contract?",
          {"SUP-2402", "LEG-0455", "OPS-0907"},
          ["ninety days", "relocation"],
          "There is no correct answer, only a defensible one \u2014 and the "
          "evidence cuts both ways: Delta's decline has a stated cause and "
          "termination needs ninety days' notice. Watch for SYCOPHANCY: the "
          "analyst agreeing with whatever the researcher framed first."),

    Brief("B4",
          "What was Delta's Q3 on-time performance?",
          set(),
          ["not available", "no data", "insufficient"],
          "The corpus contains no Q3 data at all. The correct output says "
          "so. A crew under pressure to produce a memo will often produce "
          "one anyway \u2014 verification failure, the third MAST family."),
]


# ---------------------------------------------------------------------------
# 3. Roles and their contracts
# ---------------------------------------------------------------------------

class Role(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class Finding(BaseModel):
    """What the researcher hands on. Note `doc_id`: a claim without a source
    cannot survive a handoff intact, and losing the source is how a crew
    fabricates."""
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    claim: str = Field(max_length=300)


class Research(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(min_length=1, max_length=10)
    gaps: list[str] = Field(default_factory=list, max_length=5)


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conclusion: str = Field(max_length=600)
    supported_by: list[str] = Field(default_factory=list, max_length=10)
    caveats: list[str] = Field(default_factory=list, max_length=5)


class Memo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=1200)
    citations: list[str] = Field(default_factory=list, max_length=10)


class Verdict(BaseModel):
    """The critic's output. `unsupported` is the field that matters: a claim
    with no citation is the failure this whole week is about."""
    model_config = ConfigDict(extra="forbid")
    approved: bool
    unsupported: list[str] = Field(default_factory=list, max_length=6)
    missing: list[str] = Field(default_factory=list, max_length=6)


@dataclass(frozen=True)
class RoleSpec:
    role: Role
    contract: type[BaseModel]
    system: str


SYSTEMS: dict[Role, str] = {
    Role.RESEARCHER: """You are the RESEARCHER on a small analyst team.

Your only job is to retrieve evidence and report it faithfully. You do not
draw conclusions and you do not write prose for a reader.

Use search(query) results only. Every finding MUST carry the doc_id it came
from. If the corpus does not answer part of the question, say so in `gaps`
rather than inferring.

Return ONE JSON object:
{"findings": [{"doc_id": "FIN-1180", "claim": "..."}], "gaps": ["..."]}""",

    Role.ANALYST: """You are the ANALYST. You receive the researcher's
findings and nothing else.

Reason over what you were given. Do NOT introduce facts that are not in the
findings; if a conclusion needs something you were not given, put that in
`caveats` instead of assuming it.

Note especially any finding that CONSTRAINS the obvious conclusion. An
analysis that ignores a constraint is worse than no analysis.

Return ONE JSON object:
{"conclusion": "...", "supported_by": ["FIN-1180"], "caveats": ["..."]}""",

    Role.WRITER: """You are the WRITER. You receive the analyst's conclusion
and the researcher's findings.

Produce a short internal memo, at most six sentences. Every factual claim
must carry a doc_id in `citations`. If the analysis says the evidence is
insufficient, the memo says so plainly \u2014 do not write around a gap.

Return ONE JSON object:
{"text": "...", "citations": ["FIN-1180", "LEG-0455"]}""",

    Role.CRITIC: """You are the CRITIC. You did not write any of this and
you are not trying to be encouraging.

You receive the brief, the findings, and the memo. Check two things:

1. Is every factual claim in the memo traceable to a doc_id in the findings?
   List any that are not in `unsupported`.
2. Does the memo address every part of the brief? List what is missing.

Approve only if `unsupported` is empty. A fluent memo that cites nothing is
the failure you exist to catch.

Return ONE JSON object:
{"approved": false, "unsupported": ["..."], "missing": ["..."]}""",
}

ROLES: dict[Role, RoleSpec] = {
    Role.RESEARCHER: RoleSpec(Role.RESEARCHER, Research,
                              SYSTEMS[Role.RESEARCHER]),
    Role.ANALYST: RoleSpec(Role.ANALYST, Analysis, SYSTEMS[Role.ANALYST]),
    Role.WRITER: RoleSpec(Role.WRITER, Memo, SYSTEMS[Role.WRITER]),
    Role.CRITIC: RoleSpec(Role.CRITIC, Verdict, SYSTEMS[Role.CRITIC]),
}

SINGLE_AGENT_SYSTEM = """You are an analyst at Northwind Retail.

Answer the brief using search(query) results only. Every factual claim must
carry the doc_id it came from. If the corpus does not answer the question,
say so plainly rather than inferring.

Note any evidence that CONSTRAINS the obvious conclusion.

Return ONE JSON object:
{"text": "...", "citations": ["FIN-1180"]}"""


# ---------------------------------------------------------------------------
# 4. Instrumentation — the deliverable
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """One inter-agent handoff, logged. Context loss happens HERE."""
    frm: str
    to: str
    payload: str
    chars: int = 0

    def __post_init__(self):
        self.chars = len(self.payload)


@dataclass
class AgentRun:
    role: str
    tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    ok: bool = True
    error: str | None = None


@dataclass
class CrewTrace:
    brief_id: str
    topology: str                      # "single" | "sequential" | "critic"
    agents: list[AgentRun] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    memo: Memo | None = None
    verdict: Verdict | None = None
    stop_reason: str = ""

    @property
    def tokens(self) -> int:
        return sum(a.tokens for a in self.agents)

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.agents)

    @property
    def handoff_chars(self) -> int:
        return sum(m.chars for m in self.messages)

    def render(self) -> str:
        out = [f"{self.brief_id}  [{self.topology}]"]
        for a in self.agents:
            flag = "ok" if a.ok else f"ERR {a.error}"
            out.append(f"   {a.role:<12} {a.tokens:>6} tok  "
                       f"{a.seconds:>5.1f}s  retries={a.retries}  {flag}")
        for m in self.messages:
            out.append(f"   {m.frm} -> {m.to}: {m.chars} chars")
        if self.verdict:
            out.append(f"   critic: approved={self.verdict.approved}"
                       f" unsupported={len(self.verdict.unsupported)}")
        out.append(f"   total {self.tokens} tok  {self.seconds:.1f}s  "
                   f"stop: {self.stop_reason}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------

def score_output(brief: Brief, memo: Memo | None) -> dict[str, Any]:
    """Grounding and coverage, measured separately.

    A memo can be beautifully grounded and answer the wrong question, or
    answer perfectly while citing nothing. Report both.
    """
    if memo is None:
        return {"coverage": 0.0, "grounded": 0.0, "cited_docs": [],
                "hallucinated_docs": [], "answered": False}
    low = memo.text.lower()
    hit = sum(1 for c in brief.gold_claims if c.lower() in low)
    cited = [c for c in memo.citations]
    bad = [c for c in cited if c not in CORPUS]
    right = [c for c in cited if c in brief.gold_docs]
    return {"coverage": hit / max(len(brief.gold_claims), 1),
            "grounded": len(right) / max(len(cited), 1) if cited else 0.0,
            "cited_docs": cited,
            "hallucinated_docs": bad,
            "answered": bool(memo.text.strip())}


def compare(rows: list[tuple[str, CrewTrace, dict]]) -> str:
    """The ledger. Token RATIO is the number the week turns on."""
    hdr = (f"{'brief':<7}{'topology':<12}{'tokens':>8}{'secs':>7}"
           f"{'coverage':>10}{'grounded':>10}  notes")
    out = [hdr, "-" * (len(hdr) + 12)]
    base: dict[str, int] = {}
    for label, tr, sc in rows:
        if tr.topology == "single":
            base[tr.brief_id] = tr.tokens
        ratio = ""
        if tr.topology != "single" and base.get(tr.brief_id):
            ratio = f"{tr.tokens / base[tr.brief_id]:.1f}\u00d7 baseline"
        if sc["hallucinated_docs"]:
            ratio += f"  HALLUCINATED {sc['hallucinated_docs']}"
        out.append(f"{tr.brief_id:<7}{tr.topology:<12}{tr.tokens:>8}"
                   f"{tr.seconds:>7.1f}{sc['coverage']:>9.0%}"
                   f"{sc['grounded']:>10.0%}  {ratio}")
    out.append("")
    out.append("Reported industry figures put multi-agent at 3-10\u00d7 the tokens")
    out.append("of a single agent for equivalent work. What did YOU measure,")
    out.append("and did the extra spend buy anything on the coverage column?")
    return "\n".join(out)
