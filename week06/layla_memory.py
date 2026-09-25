import json
import lab6_kit as K
from lab6_kit import (
    MemoryScope, EpisodicStore, SemanticStore,
    ProceduralStore, ScopeError, AblationRow
)

def make_search_tool(retriever, parents=None):
    """Create a scoped search result from retrieved policy passages."""

    def search_policy(query, k=3):
        if not isinstance(query, str) or len(query.strip()) < 3:
            return {"ok": False, "error": "Query is too short."}

        if not isinstance(k, int) or k < 1:
            return {"ok": False, "error": "k must be a positive integer."}

        chunks = retriever.retrieve(query.strip(), k=k)

        if not chunks:
            return {"ok": False, "error": "No matching policy found."}

        passages = []
        for chunk in chunks:
            passages.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": (
                    parents[chunk.parent_id]
                    if parents is not None and chunk.parent_id in parents
                    else chunk.text
                ),
            })

        return {
            "ok": True,
            "passages": passages,
            "evidence_ids": [chunk.chunk_id for chunk in chunks],
        }

    return search_policy


def seed_memory():
    ep = EpisodicStore()
    sem = SemanticStore()
    proc = ProceduralStore()

    layla = MemoryScope(user_id="cust-4417")
    other = MemoryScope(user_id="cust-9001")

    # Episodic memory: what happened to each customer?
    first = ep.write(
        layla,
        what="late delivery of order 4417",
        action="customer contacted support",
        outcome="case escalated for review",
        when="2026-07-10",
        provenance=["support-case-101"],
    )
    second = ep.write(
        layla,
        what="late delivery of order 4417",
        action="customer contacted support again",
        outcome="case escalated for review",
        when="2026-07-25",
        provenance=["support-case-102"],
    )
    ep.write(
        other,
        what="address change for order 9001",
        action="customer contacted support",
        outcome="address updated at depot",
        when="2026-07-20",
        provenance=["support-case-201"],
    )

    # Semantic memory: the newer preference supersedes the older one.
    sem.write(
        layla, "contact_preference", "email",
        asserted="2026-07-01"
    )
    sem.write(
        layla, "contact_preference", "phone",
        asserted="2026-07-26"
    )
    sem.write(
        other, "contact_preference", "email",
        asserted="2026-07-20"
    )

    # Procedural memory: propose a rule with traceable evidence,
    # then record its review.
    proc.propose(
        rule_id="repeat-late-delivery",
        when="the same customer reports late delivery more than once in 60 days",
        then="escalate to a human with the context from both contacts",
        provenance=[first.episode_id, second.episode_id],
    )
    proc.review("repeat-late-delivery", approve=True)

    return ep, sem, proc


def isolation_test(ep, sem):
    alice = MemoryScope(user_id="cust-4417")
    bob = MemoryScope(user_id="cust-9001")

    alice_episodes = ep.recall(alice, k=99)
    bob_episodes = ep.recall(bob, k=99)

    assert alice_episodes and bob_episodes
    assert all(e.scope.user_id == alice.user_id for e in alice_episodes)
    assert all(e.scope.user_id == bob.user_id for e in bob_episodes)

    alice_ids = {e.episode_id for e in alice_episodes}
    bob_ids = {e.episode_id for e in bob_episodes}
    assert alice_ids.isdisjoint(bob_ids)

    assert sem.get(alice, "contact_preference").value == "phone"
    assert sem.get(bob, "contact_preference").value == "email"

    try:
        ep.recall(None)
    except ScopeError:
        pass
    else:
        raise AssertionError("A scopeless episode read was allowed")

    try:
        sem.get(None, "contact_preference")
    except ScopeError:
        pass
    else:
        raise AssertionError("A scopeless fact read was allowed")


def answer(q, retriever, parents, ep, sem, proc, scope,
           use_episodic, use_semantic, use_procedural):
    """Return the evidence context and an approximate token count."""
    if retriever is None:
        return "insufficient evidence", 0

    chunks = retriever.retrieve(q.question, k=3)
    found_authority = bool({c.doc_id for c in chunks} & q.gold_docs)

    lines = []
    for chunk in chunks:
        passage = (
            parents[chunk.parent_id]
            if parents and chunk.parent_id in parents
            else chunk.text
        )
        lines.append(f"[POLICY {chunk.doc_id}/{chunk.chunk_id}] {passage}")

    repeat_confirmed = False
    if use_episodic:
        episodes = ep.recall(scope, k=5)
        for item in episodes:
            lines.append(
                f"[EPISODE {item.episode_id}] "
                f"{item.when}: {item.what}; {item.action}; {item.outcome}"
            )
        repeat_confirmed = (
            ep.count_since(scope, "late delivery", "2026-05-29") >= 2
        )

    if use_semantic:
        fact = sem.get(scope, "contact_preference")
        if fact:
            lines.append(f"[FACT] contact_preference: {fact.value}")

    if use_procedural:
        for rule in proc.active():
            lines.append(
                f"[PROCEDURE {rule.rule_id}] "
                f"when {rule.when}; then {rule.then}"
            )

    context = "\n".join(lines)
    approx_tokens = round(len(context.split()) * 1.33)

    has_answer_term = any(
        term.lower() in context.lower()
        for term in q.gold_answer_contains
    )

    # Q6 needs evidence of this customer's repeated contacts,
    # in addition to the authoritative escalation policy.
    can_answer = found_authority and has_answer_term
    if q.qid == "Q6":
        can_answer = can_answer and repeat_confirmed

    return (
        context if can_answer else "insufficient evidence",
        approx_tokens,
    )


def run_ablation(embedder):
    fixed = K.chunk_fixed(K.POLICY_DOCS)
    recursive = K.chunk_recursive(K.POLICY_DOCS)
    children, parents = K.chunk_parent_child(K.POLICY_DOCS)

    configurations = [
        ("no memory at all", None, None, False, False, False),
        ("fixed + keyword",
         K.KeywordRetriever(fixed), None, False, False, False),
        ("recursive + keyword",
         K.KeywordRetriever(recursive), None, False, False, False),
        ("recursive + embeddings",
         K.ChromaRetriever(
             recursive, embedder, collection="ablation-recursive"
         ), None, False, False, False),
    ]

    pc_retriever = K.ChromaRetriever(
        children, embedder, parents=parents,
        collection="ablation-parent-child"
    )

    configurations.extend([
        ("parent-child + embeddings",
         pc_retriever, parents, False, False, False),
        ("+ episodic",
         pc_retriever, parents, True, False, False),
        ("+ episodic + semantic",
         pc_retriever, parents, True, True, False),
        ("everything",
         pc_retriever, parents, True, True, True),
    ])

    rows = []
    for name, retriever, parent_docs, use_ep, use_sem, use_proc in configurations:
        if retriever is None:
            recall = precision = 0.0
        else:
            scores = K.score_retrieval(retriever, K.QUESTIONS)
            recall, precision = scores["recall"], scores["precision"]

        correct = 0
        total_tokens = 0

        for q in K.QUESTIONS:
            result, tokens = answer(
                q, retriever, parent_docs, ep, sem, proc, scope,
                use_ep, use_sem, use_proc
            )
            correct += result != "insufficient evidence"
            total_tokens += tokens

        rows.append(AblationRow(
            name=name,
            recall=recall,
            precision=precision,
            answered=correct / len(K.QUESTIONS),
            tokens=total_tokens,
        ))

    return rows

if __name__ == "__main__":
    ep, sem, proc = seed_memory()
    scope = MemoryScope(user_id="cust-4417")
    isolation_test(ep, sem)
    embedder = K.OllamaEmbedder()
    print(K.ablate(run_ablation(embedder)))
