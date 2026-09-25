"""COSC726 Week 8 Lab 7: single-agent and crew comparison."""

import os
import re
import json
import time
from pydantic import ValidationError

os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

import lab8_kit as K
from lab8_kit import (
    AgentRun, Analysis, Brief, CrewTrace, Memo, Message,
    Research, Role, ROLES, Verdict
)

client = K.make_client()
JSON_OBJ = re.compile(r"\\{.*\\}", re.S)
REPAIRS = {"unfenced": 0, "retries": 0, "gave_up": 0}

def call_agent(role_system, user, contract, label, tries=3):
    """Returns (parsed | None, AgentRun carrying that agent's cost)."""
    run = AgentRun(role=label)
    t0, prompt = time.time(), user
    for _ in range(tries):
        r = client.chat.completions.create(
            model=K.MODEL, temperature=0, max_tokens=800,
            messages=[{"role": "system", "content": role_system},
                      {"role": "user", "content": prompt}])
        run.tokens += r.usage.total_tokens
        raw = r.choices[0].message.content or ""
        obj = None
        try:
            obj = json.loads(raw)                      # unrepaired, counted
        except json.JSONDecodeError:
            m = JSON_OBJ.search(raw)
            if m:
                REPAIRS["unfenced"] += 1
                try: obj = json.loads(m.group(0))
                except json.JSONDecodeError: obj = None
        if obj is not None:
            try:
                parsed = contract.model_validate(obj)
                run.seconds = time.time() - t0
                return parsed, run
            except ValidationError as exc:
                why = exc.errors()[0]["msg"]
        else:
            why = "not valid JSON"
        run.retries += 1
        REPAIRS["retries"] += 1
        prompt = (f"{user}\n\nYour previous reply was rejected: {why}. "
                  "Return ONLY the corrected JSON object.")
    REPAIRS["gave_up"] += 1
    run.ok, run.error = False, "no valid output"
    run.seconds = time.time() - t0
    return None, run


def evidence(brief, k=4):
    hits = K.search(brief.question, k=k)
    return ("\n".join(f"[{h['doc_id']}] {h['text']}" for h in hits)
            or "(the search returned nothing)")


def single_agent(brief: Brief) -> CrewTrace:
    """One model call with all retrieved evidence."""
    trace = CrewTrace(brief_id=brief.brief_id, topology="single")

    prompt = (
        f"BRIEF:\n{brief.question}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence(brief)}"
    )

    memo, agent_run = call_agent(
        K.SINGLE_AGENT_SYSTEM,
        prompt,
        Memo,
        label="single",
    )

    trace.agents.append(agent_run)
    trace.memo = memo
    trace.stop_reason = (
        "memo produced" if memo is not None
        else "single agent produced no valid memo"
    )
    return trace


def crew(brief: Brief, with_critic: bool = True,
         max_revisions: int = 1) -> CrewTrace:
    trace = CrewTrace(
        brief_id=brief.brief_id,
        topology="critic" if with_critic else "sequential",
    )

    # 1. Researcher sees the retrieved documents.
    research_prompt = (
        f"BRIEF:\n{brief.question}\n\n"
        f"SEARCH RESULTS:\n{evidence(brief)}"
    )
    research, run = call_agent(
        K.SYSTEMS[Role.RESEARCHER],
        research_prompt,
        Research,
        label="researcher",
    )
    trace.agents.append(run)

    if research is None:
        trace.stop_reason = "researcher produced no valid findings"
        return trace

    findings_payload = research.model_dump_json()
    trace.messages.append(
        Message(frm="researcher", to="analyst", payload=findings_payload)
    )

    # 2. Analyst sees findings only, never the original corpus.
    analysis, run = call_agent(
        K.SYSTEMS[Role.ANALYST],
        f"BRIEF:\n{brief.question}\n\nFINDINGS:\n{findings_payload}",
        Analysis,
        label="analyst",
    )
    trace.agents.append(run)

    if analysis is None:
        trace.stop_reason = "analyst produced no valid analysis"
        return trace

    analysis_payload = analysis.model_dump_json()
    trace.messages.append(
        Message(frm="analyst", to="writer", payload=analysis_payload)
    )
    trace.messages.append(
        Message(frm="researcher", to="writer", payload=findings_payload)
    )

    # 3. Writer creates the first memo.
    writer_prompt = (
        f"BRIEF:\n{brief.question}\n\n"
        f"ANALYSIS:\n{analysis_payload}\n\n"
        f"FINDINGS:\n{findings_payload}"
    )

    for revision in range(max_revisions + 1):
        memo, run = call_agent(
            K.SYSTEMS[Role.WRITER],
            writer_prompt,
            Memo,
            label="writer" if revision == 0 else f"writer_revision_{revision}",
        )
        trace.agents.append(run)
        trace.memo = memo

        if memo is None:
            trace.stop_reason = "writer produced no valid memo"
            return trace

        if not with_critic:
            trace.stop_reason = "memo produced without critic"
            return trace

        memo_payload = memo.model_dump_json()
        trace.messages.append(
            Message(frm="writer", to="critic", payload=memo_payload)
        )
        trace.messages.append(
            Message(frm="researcher", to="critic", payload=findings_payload)
        )

        # 4. Critic checks the memo against the findings it received.
        verdict, run = call_agent(
            K.SYSTEMS[Role.CRITIC],
            f"BRIEF:\n{brief.question}\n\n"
            f"FINDINGS:\n{findings_payload}\n\n"
            f"MEMO:\n{memo_payload}",
            Verdict,
            label="critic",
        )
        trace.agents.append(run)
        trace.verdict = verdict

        if verdict is None:
            trace.stop_reason = "critic produced no valid verdict"
            return trace

        if verdict.approved:
            trace.stop_reason = "critic approved"
            return trace

        if revision == max_revisions:
            trace.stop_reason = "revision limit reached"
            return trace

        feedback = verdict.model_dump_json()
        trace.messages.append(
            Message(frm="critic", to="writer", payload=feedback)
        )
        writer_prompt = (
            f"BRIEF:\n{brief.question}\n\n"
            f"ANALYSIS:\n{analysis_payload}\n\n"
            f"FINDINGS:\n{findings_payload}\n\n"
            f"PREVIOUS MEMO:\n{memo_payload}\n\n"
            f"CRITIC FEEDBACK:\n{feedback}\n\n"
            "Revise the memo and return the required JSON."
        )

    trace.stop_reason = "revision limit reached"
    return trace


def measure(briefs=None, with_critic: bool = True):
    briefs = K.BRIEFS if briefs is None else briefs
    rows = []

    for brief in briefs:
        baseline = single_agent(brief)
        baseline_score = K.score_output(brief, baseline.memo)
        rows.append(("baseline", baseline, baseline_score))

        team = crew(brief, with_critic=with_critic)
        team_score = K.score_output(brief, team.memo)
        rows.append(("crew", team, team_score))

        print(f"Completed {brief.brief_id}", flush=True)

    print()
    print(K.compare(rows))
    return rows
