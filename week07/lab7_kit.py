"""
COSC726 Lab 6 — planning, reflection and re-planning (support module)
=====================================================================
Real model, no mocks, no other lab required.

    pip install openai pydantic
    ollama pull qwen2.5:7b && ollama serve
    python layla_planner_solution.py

What changes this week
----------------------
Weeks 4-6 built an agent that decides ONE step at a time. That is ReAct, and
its defining property is that Thought and Action fire in the same turn --
there is no point at which a plan exists to be inspected before anything
runs.

This week the agent commits to a plan first. That buys you a checkpoint
before irreversible actions, and it costs you the ability to adapt freely.
The lab is about measuring that trade, not asserting it.

    Plan  ->  Execute  ->  Critique  ->  Re-plan  ->  (stop)

Public API
----------
    GOALS                 four multi-step requests, with gold step sets
    Step, Plan            the plan contract (Pydantic, validated)
    TOOLS                 the Week 4 tool set, with tiers and gates
    PlanTrace             every plan version, every step, every critique
    Critique              the critic's verdict, as a type
    detect_oscillation()  the loop detector
    goal_drift()          did the plan stop serving the goal?
    score_plan()          plan quality against the gold step set
    PLANNER_SYSTEM / CRITIC_SYSTEM     prompt scaffolds

The finding this lab exists to produce
--------------------------------------
Reflexion improves things a lot when the failure is *diagnosable from the
output* -- reported gains on coding benchmarks run to roughly twenty points
over a single attempt. It does nothing at all when the failure is
STRUCTURAL: a missing permission, a tool that does not exist, a policy
threshold not met. Reflecting harder on "permission denied" produces a more
eloquent way of being denied.

Exercise 3 makes that concrete. Watch the critic loop three times on a
problem no amount of reflection can solve, then decide what the agent should
have done instead.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "ORDERS", "KNOWN_IDS", "THRESHOLD_DAYS", "Tier", "TOOLS", "ToolSpec",
    "Step", "Plan", "Critique", "PlanTrace", "PlanVersion", "StepResult",
    "GOALS", "Goal", "detect_oscillation", "goal_drift", "score_plan",
    "PLANNER_SYSTEM", "CRITIC_SYSTEM", "make_client", "MODEL",
]

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = (os.getenv("OLLAMA_MODEL", "qwen2.5:7b") if PROVIDER == "ollama"
         else os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"))


def make_client():
    """The Week 2 seam. Ollama and OpenAI speak the same dialect."""
    from openai import OpenAI
    if PROVIDER == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OpenAI(base_url=f"{base}/v1", api_key="ollama")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (or use LLM_PROVIDER=ollama)")
    return OpenAI()


# ---------------------------------------------------------------------------
# 1. The world — the same Northwind, so nothing here is new to learn
# ---------------------------------------------------------------------------

ORDERS: dict[str, dict[str, Any]] = {
    "A1032": {"promised": "Tue", "eta": "Fri", "days_late": 3,
              "status": "delayed_at_depot", "value": 84.00},
    "A1044": {"promised": "Mon", "eta": "Mon", "days_late": 0,
              "status": "out_for_delivery", "value": 31.50},
    "A1080": {"promised": "Thu", "eta": "Fri", "days_late": 1,
              "status": "delayed_in_transit", "value": 126.00},
    "A1091": {"promised": "Mon", "eta": "Fri", "days_late": 14,
              "status": "delayed_at_depot", "value": 59.99},
}
KNOWN_IDS = set(ORDERS)
THRESHOLD_DAYS = 3
APPROVALS: dict[str, dict] = {}


def ok(**f):
    return {"ok": True, **f}


def err(code, **f):
    return {"ok": False, "error": code, **f}


class Tier(str, Enum):
    READ = "read"
    WRITE = "write"
    CONSEQUENTIAL = "consequential"


def track_order(order_id: str) -> dict:
    row = ORDERS.get(order_id)
    if row is None:
        return err("order_not_found", order_id=order_id)
    return ok(order_id=order_id, **row)


def get_policy() -> dict:
    return ok(threshold_days=THRESHOLD_DAYS, credit_percent=10,
              text=f"Orders {THRESHOLD_DAYS}+ working days late qualify for "
                   "a 10% credit, which requires supervisor approval. "
                   "Billing disputes are handled by the billing team, never "
                   "by support.")


def check_address_changeable(order_id: str) -> dict:
    row = ORDERS.get(order_id)
    if row is None:
        return err("order_not_found", order_id=order_id)
    changeable = row["status"] != "out_for_delivery"
    return ok(order_id=order_id, changeable=changeable,
              reason=("still at depot" if changeable
                      else "already with the courier; customer must arrange "
                           "redelivery"))


def request_approval(order_id: str, amount_percent: int) -> dict:
    """CONSEQUENTIAL. Creates a PENDING record. Applies nothing."""
    if order_id not in ORDERS:
        return err("order_not_found", order_id=order_id)
    ref = f"APR-{2048 + len(APPROVALS)}"
    APPROVALS[ref] = {"order_id": order_id, "state": "pending"}
    return ok(approval_ref=ref, state="pending", account_changed=False)


def escalate_to_billing(order_id: str, description: str) -> dict:
    return ok(escalated=True, team="billing", order_id=order_id,
              description=description)


def escalate_to_human(reason: str) -> dict:
    return ok(escalated=True, reason=reason)


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., dict]
    tier: Tier
    description: str
    args: list[str]


TOOLS: dict[str, ToolSpec] = {
    "track_order": ToolSpec(
        track_order, Tier.READ,
        "Look up ONE order: status, days_late, value. Read-only.",
        ["order_id"]),
    "get_policy": ToolSpec(
        get_policy, Tier.READ,
        "Return the late-delivery policy and its threshold. Read-only.",
        []),
    "check_address_changeable": ToolSpec(
        check_address_changeable, Tier.READ,
        "Can this order's delivery address still be changed? Read-only.",
        ["order_id"]),
    "request_approval": ToolSpec(
        request_approval, Tier.CONSEQUENTIAL,
        "Create a PENDING credit approval. Applies nothing.",
        ["order_id", "amount_percent"]),
    "escalate_to_billing": ToolSpec(
        escalate_to_billing, Tier.WRITE,
        "Hand a payment dispute to the billing team.",
        ["order_id", "description"]),
    "escalate_to_human": ToolSpec(
        escalate_to_human, Tier.WRITE,
        "Hand the whole case to a person, with context.",
        ["reason"]),
}


# ---------------------------------------------------------------------------
# 2. The plan contract
# ---------------------------------------------------------------------------

class Step(BaseModel):
    """One step of a plan. Note it is a PROPOSAL: nothing has run yet."""
    model_config = ConfigDict(extra="forbid")
    n: int = Field(ge=1, le=12)
    tool: str
    args: dict = Field(default_factory=dict)
    why: str = Field(max_length=200,
                     description="What this step establishes, in one line.")


class Plan(BaseModel):
    """A whole plan, produced before anything executes.

    THIS is what Plan-and-Execute buys you over ReAct: an artefact that
    exists before any action, and can therefore be inspected, validated,
    priced or shown to a human. ReAct has no such moment -- its Thought and
    Action fire in the same turn.
    """
    model_config = ConfigDict(extra="forbid")
    goal_restated: str = Field(max_length=300)
    steps: list[Step] = Field(min_length=1, max_length=12)

    def signature(self) -> str:
        """Identity of the plan's shape, for oscillation detection."""
        return "|".join(f"{s.tool}({json.dumps(s.args, sort_keys=True)})"
                        for s in self.steps)


class Critique(BaseModel):
    """The critic's verdict. A type, not a paragraph.

    `structural` is the field that matters. A structural failure is one that
    re-planning CANNOT fix: a missing permission, a tool that does not
    exist, a policy threshold not met. Reflecting harder on those produces a
    more eloquent way of being stuck.
    """
    model_config = ConfigDict(extra="forbid")
    goal_met: bool
    problems: list[str] = Field(default_factory=list, max_length=5)
    structural: bool = Field(
        default=False,
        description="True when no re-plan can fix this and a human is needed.")
    revise: bool = False


# ---------------------------------------------------------------------------
# 3. The trace — instrumentation is the deliverable
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    n: int
    tool: str
    args: dict
    tier: str | None
    ok: bool
    error: str | None = None
    observation: dict = field(default_factory=dict)


@dataclass
class PlanVersion:
    version: int
    plan: Plan | None
    results: list[StepResult] = field(default_factory=list)
    critique: Critique | None = None
    tokens: int = 0
    invalid_reason: str | None = None

    @property
    def signature(self) -> str:
        return self.plan.signature() if self.plan else "(invalid)"


@dataclass
class PlanTrace:
    goal_id: str
    versions: list[PlanVersion] = field(default_factory=list)
    stop_reason: str = ""
    answer: str | None = None

    @property
    def total_tokens(self) -> int:
        return sum(v.tokens for v in self.versions)

    @property
    def replans(self) -> int:
        return max(len(self.versions) - 1, 0)

    def render(self) -> str:
        out = [f"goal {self.goal_id}"]
        for v in self.versions:
            out.append(f"  --- plan v{v.version} ---")
            if v.plan is None:
                out.append(f"      INVALID: {v.invalid_reason}")
                continue
            for s in v.plan.steps:
                res = next((r for r in v.results if r.n == s.n), None)
                mark = ("      " if res is None
                        else ("  ok  " if res.ok else f" ERR  "))
                detail = "" if res is None or res.ok else f"({res.error})"
                out.append(f"   {mark}{s.n}. {s.tool}"
                           f"({json.dumps(s.args)}) {detail}")
            if v.critique:
                flag = " STRUCTURAL" if v.critique.structural else ""
                out.append(f"      critic: goal_met={v.critique.goal_met}"
                           f" revise={v.critique.revise}{flag}")
                for p in v.critique.problems:
                    out.append(f"        - {p}")
        out.append(f"  stop: {self.stop_reason}")
        out.append(f"  plans: {len(self.versions)}  "
                   f"re-plans: {self.replans}  tokens: {self.total_tokens}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. Failure detectors
# ---------------------------------------------------------------------------

def detect_oscillation(trace: PlanTrace) -> str | None:
    """Has the planner produced a plan it already tried?

    Oscillation is the planning-era version of the Week 4 no-progress
    detector, and it is why that detector had to be built at the loop level
    rather than inside any single step.
    """
    seen: dict[str, int] = {}
    for v in trace.versions:
        sig = v.signature
        if sig in seen:
            return (f"plan v{v.version} repeats v{seen[sig]} exactly "
                    f"({len(v.plan.steps) if v.plan else 0} steps)")
        seen[sig] = v.version
    return None


def goal_drift(goal: "Goal", plan: Plan) -> list[str]:
    """Which parts of the goal does this plan no longer address?

    Goal drift is quiet: each re-plan looks locally reasonable while the
    plan as a whole stops serving the request. You cannot see it from one
    version; you have to compare against the ORIGINAL goal every time.
    """
    covered = {s.tool for s in plan.steps}
    return [need for need, tools in goal.requires.items()
            if not (set(tools) & covered)]


def score_plan(goal: "Goal", plan: Plan) -> dict[str, Any]:
    """Plan quality against the gold step set, measured before execution."""
    proposed = [s.tool for s in plan.steps]
    unknown = [t for t in proposed if t not in TOOLS]
    gold = goal.gold_tools
    hit = len(gold & set(proposed))
    return {"steps": len(proposed),
            "gold_covered": hit / max(len(gold), 1),
            "hallucinated_tools": unknown,
            "missing": sorted(gold - set(proposed)),
            "extra": sorted(set(proposed) - gold - set(unknown))}


# ---------------------------------------------------------------------------
# 5. The goals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    goal_id: str
    text: str
    gold_tools: set[str]
    requires: dict[str, list[str]]      # need -> tools that satisfy it
    note: str


GOALS: list[Goal] = [
    Goal("G1",
         "My order A1091 is very late. I want to know where it is, whether "
         "I'm owed anything, and I'd like it sent to my work address instead.",
         {"track_order", "get_policy", "request_approval",
          "check_address_changeable"},
         {"locate": ["track_order"],
          "remedy": ["get_policy", "request_approval"],
          "address": ["check_address_changeable"]},
         "Three needs in one message. Tests decomposition and coverage. "
         "A1091 is 14 days late, so the credit qualifies."),

    Goal("G2",
         "Order A1080 arrived a day late and I'd like compensation.",
         {"track_order", "get_policy"},
         {"locate": ["track_order"], "policy": ["get_policy"]},
         "One day late, below the 3-day threshold. The correct plan gathers "
         "evidence and then does NOT propose a credit. Tests whether the "
         "planner can plan its way to 'no'."),

    Goal("G3",
         "My order A1032 is late AND I think I've been charged twice for it.",
         {"track_order", "get_policy", "request_approval",
          "escalate_to_billing"},
         {"delivery": ["track_order", "get_policy"],
          "billing": ["escalate_to_billing"]},
         "Two issues, ONE of which is out of remit. Tests whether the plan "
         "splits them rather than trying to resolve both."),

    Goal("G4",
         "Order A9999 hasn't turned up and I want a refund today.",
         {"track_order", "escalate_to_human"},
         {"locate": ["track_order"], "handoff": ["escalate_to_human"]},
         "The order does not exist and there is no refund tool at all. Every "
         "plan will fail at step 1. STRUCTURAL: no amount of re-planning "
         "fixes a missing tool. Tests whether the critic says so."),
]


# ---------------------------------------------------------------------------
# 6. Prompt scaffolds
# ---------------------------------------------------------------------------

def _tool_catalogue() -> str:
    lines = []
    for name, spec in TOOLS.items():
        sig = ", ".join(spec.args) or ""
        lines.append(f"  {name}({sig})  [{spec.tier.value}]")
        lines.append(f"      {spec.description}")
    return "\n".join(lines)


PLANNER_SYSTEM = f"""You are the planner for Layla, a support agent at
Northwind Retail.

Given ONE customer message, produce a complete plan BEFORE anything runs.

TOOLS — you may use only these. There are no others.
{_tool_catalogue()}

RULES
- Gather evidence before proposing any remedy.
- request_approval is consequential: it may only appear after track_order
  and get_policy have both appeared earlier in the plan.
- Billing disputes are out of remit: plan to escalate them, never to
  resolve them.
- If the request needs a tool that does not exist, plan to escalate to a
  human instead of inventing one.
- Keep the plan as short as the goal allows.

Return ONE JSON object and nothing else:
{{"goal_restated": "...",
  "steps": [{{"n": 1, "tool": "track_order",
              "args": {{"order_id": "A1032"}},
              "why": "establish the delay"}}]}}"""


CRITIC_SYSTEM = """You are the critic. You did not write the plan and you
are not trying to be encouraging.

You receive the customer's goal, the plan that was tried, and what each step
actually returned. Decide three things.

1. goal_met — did the executed plan actually serve every part of the
   customer's request? Partial is not met.

2. structural — is the reason for failure something a NEW PLAN COULD NOT
   FIX? A missing tool, a permission that will always be denied, an order
   that does not exist, a policy threshold that is not met. If so, say so:
   re-planning cannot help and a human must take over.

3. revise — should we try a different plan? Only true when the failure is
   NOT structural and you can name what would change.

Return ONE JSON object and nothing else:
{"goal_met": false, "problems": ["..."], "structural": false,
 "revise": true}"""
