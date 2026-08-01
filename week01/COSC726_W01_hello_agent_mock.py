#!/usr/bin/env python3
"""
COSC726 - Agentic Artificial Intelligence
Week 1 / Lab 0 - Deterministic support-agent trace (STUDENT VERSION)

This program is a trace SIMULATOR, not a live AI agent. Every event is
pre-recorded and deterministic, so you can study control flow without a model,
an API key, an internet connection, or a paid account.

The DECIDE entries are concise authored rationale summaries written for teaching
and audit. They are NOT a claim that a model's private reasoning was exposed.

Running example: the customer-support agent helping Layla with order #A1032.
It follows the module all term, gaining one capability each week.

Usage (local terminal):
    python COSC726_W01_hello_agent_mock.py --self-test
    python COSC726_W01_hello_agent_mock.py --mock
    python COSC726_W01_hello_agent_mock.py --mock --fast --upto 4
    python COSC726_W01_hello_agent_mock.py --check
    python COSC726_W01_hello_agent_mock.py --json

Usage (Google Colab - prefix with an exclamation mark):
    !python COSC726_W01_hello_agent_mock.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TraceEvent:
    step: int
    kind: str
    text: str


_RAW_TRACE: Final[tuple[tuple[int, str, str], ...]] = (
    (1, "USER",
     "Hi, I'm Layla. Where is my order #A1032? It was meant to arrive Tuesday."),
    (2, "DECIDE",
     "I need the current order status. Use the read-only lookup_order tool."),
    (3, "ACT",
     'lookup_order(order_id="A1032")'),
    (4, "OBSERVE",
     '{"order_id":"A1032","item":"desk lamp","status":"delayed at depot",'
     '"promised_day":"Tuesday","new_eta":"Friday"}'),
    (5, "DECIDE",
     "The delay is three calendar days. Check the policy before mentioning "
     "compensation."),
    (6, "ACT",
     'get_late_delivery_policy(topic="goodwill credit")'),
    (7, "OBSERVE",
     '{"eligibility":"delay of 3 or more calendar days",'
     '"remedy":"10% goodwill credit","approval_required":true}'),
    (8, "DECIDE",
     "The order appears eligible. Applying a credit changes the account, so "
     "request human approval rather than claim it happened."),
    (9, "ACT",
     'request_approval(action="apply_goodwill_credit", order_id="A1032", '
     'amount_percent=10)'),
    (10, "OBSERVE",
     '{"approval_id":"APR-2048","status":"pending","account_changed":false}'),
    (11, "DECIDE",
     "I have verified the facts and opened the required approval request. Stop "
     "and report the status, evidence, and next step."),
    (12, "RESPOND",
     "Hi Layla - order #A1032 is delayed at the depot and is now due Friday. "
     "Because that is three calendar days late, it appears eligible for a 10% "
     "goodwill credit. I submitted approval request APR-2048; no account change "
     "has been made yet."),
)

MOCK_TRACE: Final[tuple[TraceEvent, ...]] = tuple(
    TraceEvent(step, kind, text) for step, kind, text in _RAW_TRACE
)

VALID_PHASES: Final[set[str]] = {"sense", "reason", "act", "observe"}

# ─────────────────────────────────────────────────────────────────────────────
# YOUR TASK (Lab 0, Part C)
#
# Replace every "..." with one of: "sense", "reason", "act", "observe".
# Then run:  python COSC726_W01_hello_agent_mock.py --check      (target 12/12)
#
# Steps 4, 7 and 10 accept EITHER "observe" or "sense". Pick one and be ready to
# argue for it - that argument is the point, not the mark.
# ─────────────────────────────────────────────────────────────────────────────
ANNOTATIONS: dict[int, str] = {
    1: "sense",
    2: "reason",
    3: "act",
    4: "observe",
    5: "reason",
    6: "act",
    7: "observe",
    8: "reason",
    9: "act",
    10: "observe",
    11: "reason",
    12: "act",
}

_ANSWER_KEY_RAW: Final[dict[int, list[str]]] = {
    1: ["sense"],
    2: ["reason"],
    3: ["act"],
    4: ["observe", "sense"],
    5: ["reason"],
    6: ["act"],
    7: ["observe", "sense"],
    8: ["reason"],
    9: ["act"],
    10: ["observe", "sense"],
    11: ["reason"],
    12: ["act"],
}
ANSWER_KEY: Final[dict[int, set[str]]] = {
    step: set(values) for step, values in _ANSWER_KEY_RAW.items()
}


def validate_trace() -> list[str]:
    """Return a list of trace-integrity problems; an empty list means PASS.

    This is the module's first example of a control PROVEN BY TEST rather than
    asserted in prose - a habit that becomes mandatory from Lab 3 onward.
    """
    problems: list[str] = []

    steps = [event.step for event in MOCK_TRACE]
    if steps != list(range(1, len(MOCK_TRACE) + 1)):
        problems.append("steps must be consecutive and start at 1")

    expected_kinds = [
        "USER", "DECIDE", "ACT", "OBSERVE", "DECIDE", "ACT",
        "OBSERVE", "DECIDE", "ACT", "OBSERVE", "DECIDE", "RESPOND",
    ]
    kinds = [event.kind for event in MOCK_TRACE]
    if kinds != expected_kinds:
        problems.append(f"unexpected event sequence: {kinds!r}")

    if set(ANSWER_KEY) != set(steps):
        problems.append("answer key must cover every trace step")
    if any(not allowed or not allowed.issubset(VALID_PHASES)
           for allowed in ANSWER_KEY.values()):
        problems.append("answer key contains an invalid phase")

    joined = " ".join(event.text.lower() for event in MOCK_TRACE)
    if "3 or more calendar days" not in joined:
        problems.append("policy threshold must be internally consistent")
    if '"approval_required":true' not in joined:
        problems.append("policy observation must require approval")
    if '"account_changed":false' not in joined:
        problems.append("approval observation must record no account change")
    if "no account change has been made" not in MOCK_TRACE[-1].text.lower():
        problems.append("final response must not claim an unexecuted change")
    return problems


def play_trace(*, fast: bool = False, upto: int | None = None) -> None:
    """Print the trace event by event."""
    events = MOCK_TRACE if upto is None else MOCK_TRACE[:upto]
    width = max(len(event.kind) for event in MOCK_TRACE)
    print("=" * 100)
    print(" COSC726 Week 1 - DETERMINISTIC MOCK TRACE - no model / no API")
    print("=" * 100)
    for event in events:
        print(f"[{event.kind:<{width}}] step {event.step:>2}: {event.text}")
        print("-" * 100)
        if not fast:
            time.sleep(0.30)
    print(f"Displayed {len(events)} of {len(MOCK_TRACE)} events.")


def print_json() -> None:
    """Print the trace as structured JSON (observability stretch task)."""
    print(json.dumps([asdict(e) for e in MOCK_TRACE], indent=2, ensure_ascii=False))


def check_annotations() -> int:
    """Score ANNOTATIONS and return the number correct."""
    correct = 0
    for step, expected in ANSWER_KEY.items():
        got = str(ANNOTATIONS.get(step, "")).strip().lower()
        ok = got in expected
        correct += int(ok)
        if ok:
            verdict = "OK"
        elif got in {"", "..."}:
            verdict = "UNANSWERED"
        else:
            verdict = "EXPECTED " + " or ".join(sorted(expected))
        print(f"step {step:>2}: {got or '(blank)':<10} {verdict}")
    total = len(ANSWER_KEY)
    print(f"\nScore: {correct}/{total}")
    if correct == total:
        print("PASS - you can read the evidence-action loop in any agent trace.")
    else:
        print("Complete or revise the annotations, then run --check again.")
    return correct


def run_self_test() -> bool:
    problems = validate_trace()
    if problems:
        print("TRACE SELF-TEST FAILED")
        for problem in problems:
            print(f"- {problem}")
        return False
    print("TRACE SELF-TEST PASSED")
    print(f"{len(MOCK_TRACE)} events · {len(ANSWER_KEY)} annotation targets "
          "· approval gate verified")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COSC726 Week 1 deterministic agent-trace simulator")
    parser.add_argument("--mock", action="store_true",
                        help="play the trace (default action)")
    parser.add_argument("--fast", action="store_true", help="skip pauses")
    parser.add_argument("--upto", type=int, metavar="N",
                        help="show only the first N events")
    parser.add_argument("--check", action="store_true", help="score ANNOTATIONS")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the trace as JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the trace and answer key")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.upto is not None and not 1 <= args.upto <= len(MOCK_TRACE):
        print(f"--upto must be between 1 and {len(MOCK_TRACE)}", file=sys.stderr)
        return 2
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.check:
        return 0 if check_annotations() == len(ANSWER_KEY) else 1
    if args.as_json:
        print_json()
        return 0
    play_trace(fast=args.fast, upto=args.upto)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
