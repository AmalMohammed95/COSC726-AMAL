"""COSC726 Week 7 Lab 6: planning, validation, execution, and measurement."""

import os
import re
import json
from pydantic import ValidationError

os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

import lab7_kit as K
from lab7_kit import (
    CRITIC_SYSTEM, PLANNER_SYSTEM, Critique, Plan,
    PlanTrace, PlanVersion, Step, StepResult, Tier
)

client = K.make_client()
JSON_OBJ = re.compile(r"\\{.*\\}", re.S)
REPAIRS = {"unfenced": 0, "retries": 0, "gave_up": 0}

def _ask(system, user, max_tokens=700):
    r = client.chat.completions.create(
        model=K.MODEL, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or "", r.usage.total_tokens


def _parse(raw, model_cls):
    obj = None
    try:
        obj = json.loads(raw)                 # unrepaired, and counted
    except json.JSONDecodeError:
        m = JSON_OBJ.search(raw)
        if m:
            REPAIRS["unfenced"] += 1
            try: obj = json.loads(m.group(0))
            except json.JSONDecodeError: obj = None
    if obj is None:
        return None, "not valid JSON"
    try:
        return model_cls.model_validate(obj), None
    except ValidationError as exc:
        e = exc.errors()[0]
        return None, f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"


def propose(system, user, model_cls, tries=3):
    """Ask, validate, hand the error back. Bounded."""
    prompt, total = user, 0
    for _ in range(tries):
        raw, tok = _ask(system, prompt)
        total += tok
        obj, why = _parse(raw, model_cls)
        if obj is not None:
            return obj, None, total
        REPAIRS["retries"] += 1
        prompt = (f"{user}\n\nYour previous reply was rejected: {why}. "
                  "Return ONLY the corrected JSON object.")
    REPAIRS["gave_up"] += 1
    return None, why, total


def validate_plan(plan: Plan) -> list[str]:
    problems = []
    completed_tools = set()

    for step in plan.steps:
        spec = K.TOOLS.get(step.tool)

        # Gate 1: the named tool must exist.
        if spec is None:
            problems.append(
                f"Step {step.n}: unknown tool '{step.tool}'"
            )
            continue

        # Gate 2: all required arguments must be present.
        for arg in spec.args:
            if arg not in step.args:
                problems.append(
                    f"Step {step.n}: '{step.tool}' is missing '{arg}'"
                )

        # Gate 2: validate any supplied order ID.
        if "order_id" in step.args:
            order_id = step.args["order_id"]
            if not isinstance(order_id, str) or not re.fullmatch(
                r"A[0-9]{4}", order_id
            ):
                problems.append(
                    f"Step {step.n}: invalid order_id {order_id!r}"
                )

        # Gate 4: evidence-gathering steps must precede approval.
        if step.tool == "request_approval":
            missing_evidence = {
                "track_order", "get_policy"
            } - completed_tools
            if missing_evidence:
                problems.append(
                    f"Step {step.n}: request_approval comes before "
                    f"{', '.join(sorted(missing_evidence))}"
                )

        completed_tools.add(step.tool)

    return problems


def execute(plan: Plan, allow_consequential: bool = True) -> list[StepResult]:
    results = []
    tracked_orders = {}
    policy = None

    for step in plan.steps:
        spec = K.TOOLS.get(step.tool)
        tier = spec.tier.value if spec else None
        error = None
        observation = {}

        if spec is None:
            error = "unknown_tool"
        elif any(arg not in step.args for arg in spec.args):
            error = "missing_required_argument"
        elif "order_id" in step.args and step.args["order_id"] not in K.KNOWN_IDS:
            error = "order_not_found"
        elif spec.tier == Tier.CONSEQUENTIAL and not allow_consequential:
            error = "consequential_action_not_allowed"
        elif step.tool == "request_approval":
            order_id = step.args["order_id"]
            if order_id not in tracked_orders or policy is None:
                error = "evidence_missing"
            elif tracked_orders[order_id]["days_late"] < policy["threshold_days"]:
                error = "below_credit_threshold"

        if error is None:
            try:
                observation = spec.fn(**step.args)
                if not observation.get("ok", False):
                    error = observation.get("error", "tool_failed")
            except (TypeError, ValueError) as exc:
                error = f"invalid_arguments: {exc}"

        result = StepResult(
            n=step.n,
            tool=step.tool,
            args=step.args,
            tier=tier,
            ok=error is None,
            error=error,
            observation=observation,
        )
        results.append(result)

        # Only successful observations count as evidence for later steps.
        if result.ok and step.tool == "track_order":
            tracked_orders[step.args["order_id"]] = observation
        elif result.ok and step.tool == "get_policy":
            policy = observation

    return results


def run(goal, max_versions: int = 3, allow_consequential: bool = True):
    """Plan, validate, execute, critique, and re-plan within a fixed limit."""
    trace = PlanTrace(goal_id=goal.goal_id)
    feedback = ""

    for version in range(1, max_versions + 1):
        planner_input = f"CUSTOMER MESSAGE:\n{goal.text}"
        if feedback:
            planner_input += (
                "\n\nPREVIOUS ATTEMPT AND FEEDBACK:\n"
                + feedback
                + "\nProduce a revised complete plan for the ORIGINAL message."
            )

        plan, why, plan_tokens = propose(
            PLANNER_SYSTEM, planner_input, Plan
        )
        current = PlanVersion(
            version=version,
            plan=plan,
            tokens=plan_tokens,
        )
        trace.versions.append(current)

        if plan is None:
            current.invalid_reason = why
            trace.stop_reason = "planner could not produce a valid plan"
            break

        # Check the proposal before any tool runs.
        problems = validate_plan(plan)
        if problems:
            current.invalid_reason = "; ".join(problems)
            trace.stop_reason = "plan failed validation before execution"
            break

        # Detect a repeated plan before executing it again.
        repeated = K.detect_oscillation(trace)
        if repeated:
            trace.stop_reason = f"oscillation: {repeated}"
            break

        # Always compare with the customer's ORIGINAL goal.
        drift = K.goal_drift(goal, plan)
        if drift:
            current.invalid_reason = "goal drift: " + ", ".join(drift)
            trace.stop_reason = current.invalid_reason
            break

        current.results = execute(
            plan,
            allow_consequential=allow_consequential,
        )

        critique_input = (
            f"ORIGINAL CUSTOMER GOAL:\n{goal.text}\n\n"
            f"PLAN:\n{plan.model_dump_json(indent=2)}\n\n"
            f"EXECUTION RESULTS:\n"
            f"{json.dumps([r.__dict__ for r in current.results], indent=2)}"
        )

        critique, critic_error, critic_tokens = propose(
            CRITIC_SYSTEM, critique_input, Critique
        )
        current.tokens += critic_tokens

        if critique is None:
            trace.stop_reason = (
                f"critic could not produce a valid verdict: {critic_error}"
            )
            break

        current.critique = critique

        if critique.structural:
            trace.stop_reason = "structural failure — escalate to a human"
            break

        if critique.goal_met:
            trace.stop_reason = "goal met"
            trace.answer = "Goal completed according to the critic."
            break

        if not critique.revise:
            trace.stop_reason = "critic declined further revision"
            break

        feedback = (
            "Plan: " + plan.model_dump_json()
            + "\nResults: "
            + json.dumps([r.__dict__ for r in current.results])
            + "\nCritic problems: "
            + "; ".join(critique.problems)
        )

    if not trace.stop_reason:
        trace.stop_reason = "maximum plan versions reached"

    return trace


def measure(max_versions: int = 3):
    header = (
        f"{'Goal':<6} {'Gold coverage':<15} "
        f"{'Re-plans':<10} {'Tokens':<9} Stop reason"
    )
    print(header)
    print("-" * len(header))

    traces = []

    for goal in K.GOALS:
        trace = run(goal, max_versions=max_versions)
        traces.append(trace)

        first_plan = next(
            (v.plan for v in trace.versions if v.plan is not None),
            None,
        )

        if first_plan is None:
            coverage = "N/A"
        else:
            quality = K.score_plan(goal, first_plan)
            coverage = f"{quality['gold_covered']:.0%}"

        print(
            f"{goal.goal_id:<6} "
            f"{coverage:<15} "
            f"{trace.replans:<10} "
            f"{trace.total_tokens:<9} "
            f"{trace.stop_reason}"
        )

    return traces
