# COSC726 Week 7 Lab 6 — Decision Memo

Model used: qwen2.5:3b through Ollama.
Measurement limit: one plan version per goal.

## Measurement table

| Goal | First-plan gold coverage | Re-plans | Tokens | Stop reason |
|---|---:|---:|---:|---|
| G1 | 100% | 0 | 1913 | Critic labelled failure structural |
| G2 | 100% | 0 | 1569 | Critic labelled failure structural |
| G3 | 50% | 0 | 602 | Goal drift: billing |
| G4 | 50% | 0 | 615 | Goal drift: handoff |

Total measured tokens: 4699. Re-plans were zero because this
measurement allowed only one plan version per goal.

## 1. What did planning buy over ReAct?

The complete plan could be checked before tools ran. The validator
detected an invented refund tool, an invalid order ID, and approval
scheduled before evidence gathering. In G3 and G4, the goal-drift
detector stopped incomplete plans before any step executed. Without
this checkpoint, a step-by-step agent could attempt an invalid action
before discovering the problem.

## 2. What did it cost?

The one-version measurement used 4699 tokens across four goals.
Planning also committed the agent to steps before it saw tool results;
G2 included unnecessary address and escalation steps. A comparison
with three permitted versions did not complete within the available
runtime, so this run does not establish a measured cost multiplier.

## 3. Where did reflection help, and where did it not?

No successful re-plan was observed in this measurement because it
allowed one version per goal. G4 stopped for missing handoff coverage
before the order lookup ran, so this trace did not test whether the
critic would recognize the nonexistent order or missing refund tool.
A missing tool and nonexistent order cannot be repaired merely by
asking the same model to re-plan.

## 4. What did the detectors catch that the critic missed?

Goal drift detected that G3 omitted the billing handoff and G4 omitted
a human handoff before execution. G1 had no detected drift, but its
critic incorrectly called the outcome structural: the order was 14
days late, above the three-day threshold, and request_approval was
available and executed. The critic also misread the address result.
This shows that the critic's verdict needs independent checks.

## 5. Where does the agent still trust something it should not?

The loop trusts the model critic's structural verdict even when it
contradicts tool observations. The plan validator checks required
arguments and order ID format, but does not fully validate the
meaning of a requested amount: G4 proposed a 100% credit in response
to a refund request. Tool outputs and authorization must be checked
before treating a proposed action as justified.

## 6. What did this lab not tell you?

There were only four hand-written goals, one model, and one measured
run per goal, so there is no variance estimate or evidence of
generalization. The three-version comparison did not complete.
Gold tool coverage measures proposed tool names, not customer outcome.
The model critic is not an independent authority and made factual
errors in this run. Results may change with another model, prompt,
runtime, or sample of goals.
