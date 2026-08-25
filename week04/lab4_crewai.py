"""
COSC726 · Lab 4 — Rebuild Layla as a Crew
=========================================
Real CrewAI, real model, runnable from VS Code.

The notebook is for reading and for the lecture. This is for working: it has
a CLI, it is importable, and it is the shape your project repository should
take.

Setup
-----
    python -m venv .venv
    source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
    pip install crewai openai

    # local, free, no account:
    ollama pull qwen2.5:7b
    ollama serve

Run
---
    python lab4_crewai.py --help
    python lab4_crewai.py gates          # no model needed; start here
    python lab4_crewai.py baseline       # the hand-rolled agent
    python lab4_crewai.py crew           # the same job, as a crew
    python lab4_crewai.py prompt         # the prompt you never wrote
    python lab4_crewai.py compare        # both, on three fixtures
    python lab4_crewai.py runaway        # max_iter=15, and what it costs

Switch provider:
    LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python lab4_crewai.py crew

In VS Code
----------
Set the interpreter to .venv, then use the Run button. `gates` needs no
model at all, so put a breakpoint in `request_approval` and step through it
before you spend a token on anything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# PROVIDER — CrewAI routes on the model-name prefix (it uses LiteLLM)
# ---------------------------------------------------------------------------

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")  # PINNED


def get_llm():
    """The seam. One function; nothing else in this file knows the provider."""
    from crewai import LLM
    if PROVIDER == "ollama":
        return LLM(model=f"ollama/{OLLAMA_MODEL}", base_url=OLLAMA_URL)
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Either export it, or use the "
                 "default LLM_PROVIDER=ollama.")
    return LLM(model=OPENAI_MODEL)


# ---------------------------------------------------------------------------
# THE WORLD
# ---------------------------------------------------------------------------

ORDERS: dict[str, dict] = {
    "A1032": {"promised": "Tue", "eta": "Fri", "days_late": 3,
              "status": "delayed_at_depot"},
    "A1044": {"promised": "Mon", "eta": "Mon", "days_late": 0,
              "status": "out_for_delivery"},
    "A1080": {"promised": "Thu", "eta": "Fri", "days_late": 1,
              "status": "delayed_in_transit"},
    "A1091": {"promised": "Mon", "eta": "Fri", "days_late": 12,
              "status": "delayed_at_depot"},
}
KNOWN_IDS = set(ORDERS)

# Deliberately 10, not the obvious 3. A 7B model has seen thousands of
# "3 days late -> credit" examples and may answer correctly WITHOUT reading
# the policy. An unusual threshold forces a real tool call, and lets you tell
# reasoning from recall. Only A1091 qualifies.
THRESHOLD_DAYS = 10

APPROVALS: dict[str, dict] = {}


def _track(order_id: str) -> dict:
    row = ORDERS.get(order_id)
    if row is None:
        return {"ok": False, "error": "order_not_found", "order_id": order_id}
    return {"ok": True, "order_id": order_id, **row}


def _policy() -> dict:
    return {"ok": True, "threshold_days": THRESHOLD_DAYS, "credit_percent": 10,
            "text": f"Orders {THRESHOLD_DAYS}+ days late qualify for a 10% "
                    "credit, which requires human approval."}


def _approve(order_id: str, amount_percent: int) -> dict:
    ref = f"APR-{2048 + len(APPROVALS)}"
    APPROVALS[ref] = {"order_id": order_id, "state": "pending"}
    return {"ok": True, "approval_ref": ref, "state": "pending",
            "account_changed": False}


# ---------------------------------------------------------------------------
# RUN STATE — note WHERE this has to live
# ---------------------------------------------------------------------------
# In Lab 3 the dispatcher held this. CrewAI gives you nowhere to put it, so
# it becomes module state. Two concurrent requests would share it. That is a
# real production problem and it belongs in your memo.

SEEN: set[str] = set()
OBSERVED: dict[str, int | None] = {"days_late": None}


def reset_run_state() -> None:
    SEEN.clear()
    OBSERVED["days_late"] = None


# ---------------------------------------------------------------------------
# THE TOOLS — with the gates restored (Task 4)
# ---------------------------------------------------------------------------
# @tool derives the schema from the type hints and the description from the
# docstring. What it CANNOT derive: the ^A[0-9]{4}$ pattern, that the order
# must exist, that an approval needs prior evidence, or that request_approval
# is consequential while track_order is not.
#
# Those are judgements, and judgements do not live in type hints. So they end
# up here, in the tool bodies -- scattered, with no single place to audit them
# and nothing forcing the next tool anyone adds to have any.

from crewai.tools import tool  # noqa: E402  (after the module docstring)


@tool("track_order")
def track_order(order_id: str) -> str:
    """Look up the delivery status of ONE order by its ID. Read-only: this
    never changes anything. Returns status, promised date, eta and days_late.
    Use this before making any claim about where an order is."""
    if not re.fullmatch(r"A[0-9]{4}", str(order_id)):            # gate 2
        return json.dumps({"ok": False, "error": "bad_pattern",
                           "hint": "Order IDs look like A1032."})
    if order_id not in KNOWN_IDS:                                # gate 3
        return json.dumps({"ok": False, "error": "unknown_order",
                           "hint": "Ask the customer to confirm the ID."})
    result = _track(order_id)
    if result["ok"]:
        SEEN.add("track_order")
        OBSERVED["days_late"] = result["days_late"]
    return json.dumps(result)


@tool("get_policy")
def get_policy() -> str:
    """Return the late-delivery policy and its numeric threshold. Read-only.
    Use this before offering any remedy. Do not assume the threshold."""
    SEEN.add("get_policy")
    return json.dumps(_policy())


@tool("request_approval")
def request_approval(order_id: str, amount_percent: int) -> str:
    """Create a PENDING approval request for a credit. This does NOT apply
    anything. After calling it you may say the request is pending, never that
    it is done."""
    if not re.fullmatch(r"A[0-9]{4}", str(order_id)):            # gate 2
        return json.dumps({"ok": False, "error": "bad_pattern"})
    if not (1 <= int(amount_percent) <= 100):                    # gate 2
        return json.dumps({"ok": False, "error": "amount_out_of_range"})
    if order_id not in KNOWN_IDS:                                # gate 3
        return json.dumps({"ok": False, "error": "unknown_order"})
    if "track_order" not in SEEN or "get_policy" not in SEEN:    # gate 4
        return json.dumps({"ok": False, "error": "no_evidence",
                           "hint": "Check the order and the policy first."})
    if (OBSERVED["days_late"] or 0) < THRESHOLD_DAYS:            # gate 4
        return json.dumps({"ok": False, "error": "below_threshold",
                           "hint": f"Policy requires {THRESHOLD_DAYS}+ days."})
    return json.dumps(_approve(order_id, amount_percent))        # only now


# ---------------------------------------------------------------------------
# THE CREW (Task 2)
# ---------------------------------------------------------------------------

def build_crew(verbose: bool = False, max_iter: int = 6):
    from crewai import Agent, Crew, Process, Task

    support = Agent(
        role="Support triage agent for Northwind Retail",
        goal="Resolve one customer order query using only tool results",
        backstory=("You never claim an action happened without a tool result "
                   "confirming it. You check the policy before offering any "
                   "remedy, and you never assume the threshold."),
        tools=[track_order, get_policy, request_approval],
        allow_delegation=False,      # stops agents handing work back and forth
        max_iter=max_iter,           # the Week 4 turn cap, renamed. Default 15.
        llm=get_llm(),
        verbose=verbose)

    task = Task(
        description="Handle this customer message: {message}",
        expected_output=("One short reply stating the order status and "
                         "whether any approval is pending."),
        agent=support)

    return Crew(agents=[support], tasks=[task],
                process=Process.sequential, verbose=verbose)


# ---------------------------------------------------------------------------
# THE HAND-ROLLED BASELINE — what the crew is compared against
# ---------------------------------------------------------------------------

BASE_SYSTEM = ("You are Layla, a support agent for Northwind Retail. Resolve "
               "ONE order query using the tools. Never state a fact a tool "
               "has not returned. Check the policy before offering a remedy.")

BASE_TOOLS = [
    {"type": "function", "function": {
        "name": "track_order",
        "description": "Look up ONE order by ID. Read-only.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_policy",
        "description": "Return the late-delivery policy and its threshold.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {}, "required": []}}},
]


def base_client():
    from openai import OpenAI
    if PROVIDER == "ollama":
        return OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"), OLLAMA_MODEL
    return OpenAI(), OPENAI_MODEL


def run_baseline(message: str, max_steps: int = 6) -> dict:
    """The Week 4 shape: gates before execution, a cap on the loop.

    Every gate is visible in this function. Keep that in mind when you look
    for them in the crew version.
    """
    client, model = base_client()
    msgs = [{"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": message}]
    trace, tokens = [], 0

    for step in range(1, max_steps + 1):                          # TURN CAP
        resp = client.chat.completions.create(
            model=model, messages=msgs, tools=BASE_TOOLS, temperature=0)
        tokens += resp.usage.total_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:                                    # COMPLETE
            return {"answer": msg.content, "trace": trace,
                    "tokens": tokens, "stop": "complete", "steps": step}

        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments or "{}")
        oid = args.get("order_id")

        if oid is not None and not re.fullmatch(r"A[0-9]{4}", str(oid)):
            obs = {"ok": False, "error": "bad_pattern"}           # gate 2
        elif oid is not None and oid not in KNOWN_IDS:
            obs = {"ok": False, "error": "unknown_order"}         # gate 3
        elif call.function.name == "track_order":
            obs = _track(oid)
        else:
            obs = _policy()

        trace.append({"tool": call.function.name, "ok": obs.get("ok")})
        msgs += [msg, {"role": "tool", "tool_call_id": call.id,
                       "content": json.dumps(obs)}]

    return {"answer": None, "trace": trace, "tokens": tokens,
            "stop": "capped_turns", "steps": max_steps}           # NEVER silent


# ---------------------------------------------------------------------------
# COMMANDS
# ---------------------------------------------------------------------------

MSG = "My order A1032 was due Tuesday and it still hasn't arrived."

FIXTURES = [
    ("A1032 late", "My order A1032 was due Tuesday and it hasn't arrived."),
    ("A1080 below", "Order A1080 is one day late. Can I get compensation?"),
    ("A9999 unreal", "Where is my order A9999?"),
]


def cmd_gates(_):
    """Prove the gates fire. Needs no model, costs nothing.

    Put a breakpoint in request_approval and step through this first.
    """
    reset_run_state()
    print("Gates, with no model involved:\n")
    checks = [
        ("gate 3  fabricated id",
         lambda: track_order.run(order_id="A9999")),
        ("gate 2  bad pattern",
         lambda: track_order.run(order_id="1102")),
        ("gate 4  no evidence yet",
         lambda: request_approval.run(order_id="A1032", amount_percent=10)),
    ]
    for label, fn in checks:
        print(f"  {label:<26} {fn()}")

    track_order.run(order_id="A1032")
    get_policy.run()
    print(f"  {'gate 4  below threshold':<26} "
          f"{request_approval.run(order_id='A1032', amount_percent=10)}")

    reset_run_state()
    track_order.run(order_id="A1091")           # 12 days late: qualifies
    get_policy.run()
    print(f"  {'all gates pass':<26} "
          f"{request_approval.run(order_id='A1091', amount_percent=10)}")
    print("\nNote the last one: account_changed=false. It proposed; a human "
          "disposes.")


def cmd_baseline(args):
    """The hand-rolled agent."""
    r = run_baseline(args.message)
    for s in r["trace"]:
        print(f"  {s['tool']:<20} ok={s['ok']}")
    print(f"  stop: {r['stop']}  steps: {r['steps']}  tokens: {r['tokens']}")
    print(f"\n  {r['answer']}")


def cmd_crew(args):
    """The same job, as a crew."""
    reset_run_state()
    APPROVALS.clear()
    crew = build_crew(verbose=args.verbose, max_iter=args.max_iter)
    t0 = time.time()
    result = crew.kickoff(inputs={"message": args.message})
    print(f"\n({time.time() - t0:.1f}s)\n")
    print(result.raw)
    print(f"\ntokens: {result.token_usage.total_tokens}")
    print(f"approvals created: {APPROVALS}")


def cmd_prompt(_):
    """Recover the system prompt CrewAI assembled from role/goal/backstory."""
    from crewai.utilities.i18n import get_i18n
    crew = build_crew()
    agent = crew.agents[0]

    print("=== the three inputs you wrote ===")
    print("role     :", agent.role)
    print("goal     :", agent.goal)
    print("backstory:", agent.backstory)

    print("\n=== the template CrewAI wraps them in ===")
    print(get_i18n().slice("role_playing"))

    print("\n=== tool descriptions, sent on EVERY call ===")
    for t in agent.tools:
        print(f"  {t.name}: {t.description[:100]}")
        print(f"      schema: {t.args_schema.model_json_schema().get('properties')}")

    approx = (len(agent.role) + len(agent.goal) + len(agent.backstory)
              + sum(len(t.description) for t in agent.tools)) // 4
    print(f"\napprox {approx} prompt tokens on every single call")
    print("\nLook at the schemas above. Where is the ^A[0-9]{4}$ pattern?")


def cmd_compare(args):
    """Both implementations, same fixtures, same model."""
    rows = []
    for name, msg in FIXTURES:
        b = run_baseline(msg)
        reset_run_state()
        APPROVALS.clear()
        crew = build_crew(max_iter=args.max_iter)
        t0 = time.time()
        k = crew.kickoff(inputs={"message": msg})
        rows.append((name, b["tokens"], b["steps"],
                     k.token_usage.total_tokens, round(time.time() - t0, 1)))

    head = (f"{'case':<14}{'base tok':>10}{'base steps':>12}"
            f"{'crew tok':>10}{'crew s':>9}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>10}{r[2]:>12}{r[3]:>10}{r[4]:>9}")

    bt = sum(r[1] for r in rows)
    kt = sum(r[3] for r in rows)
    print(f"\ntotals: baseline {bt}, crew {kt}  ->  {(kt - bt) / max(bt, 1):+.0%}")
    print("\nWhere did the difference go? Run `prompt` and count what rides "
          "on every call.")


def cmd_runaway(args):
    """max_iter at the framework default, on a task it cannot finish."""
    reset_run_state()
    crew = build_crew(verbose=args.verbose, max_iter=15)
    t0 = time.time()
    result = crew.kickoff(inputs={"message": "Where is my stuff?"})
    print(f"\n({time.time() - t0:.1f}s)  "
          f"tokens: {result.token_usage.total_tokens}")
    print(result.raw[:400])
    print("\nYour Lab 3 agent also had a no-progress detector. Where would "
          "that live here?")


COMMANDS = {
    "gates": cmd_gates, "baseline": cmd_baseline, "crew": cmd_crew,
    "prompt": cmd_prompt, "compare": cmd_compare, "runaway": cmd_runaway,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="COSC726 Lab 4 — Layla as a CrewAI crew.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with `gates`: it needs no model and costs nothing.")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--message", default=MSG)
    ap.add_argument("--max-iter", type=int, default=6,
                    dest="max_iter", help="turn cap (CrewAI default is 15)")
    ap.add_argument("--verbose", action="store_true",
                    help="CrewAI's own step-by-step output")
    args = ap.parse_args()

    print(f"provider: {PROVIDER}  "
          f"model: {OLLAMA_MODEL if PROVIDER == 'ollama' else OPENAI_MODEL}\n")
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
