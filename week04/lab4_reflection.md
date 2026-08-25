# Lab 4 Reflection — Layla as a CrewAI Crew

## 1. Where did the gates go?

In the CrewAI implementation, the safety gates were placed inside the tool bodies. The `track_order` and `request_approval` tools validate the order-ID pattern and verify that the order exists. The `request_approval` tool also checks whether the required evidence was collected and whether the delay meets the policy threshold.

A newly added tool does not automatically inherit these gates. Its developer must manually implement the necessary validation and authorization controls. This makes the controls scattered and harder to audit than the centralized dispatcher used in Lab 3.

## 2. What is `SEEN`?

`SEEN` is module-level state used to record which tools have been called. `OBSERVED` stores the observed number of late days.

During debugging, the first approval attempt showed an empty `SEEN` state and `days_late: None`, so the `no_evidence` gate rejected it. After calling `track_order` and `get_policy`, `SEEN` contained both tools and `days_late` was 3. The approval was still rejected because the policy requires at least 10 late days. For order `A1091`, `days_late` was 12, so all gates passed and a pending approval request was created.

Because this state is shared at module level, two concurrent requests could modify or read the same state. Evidence collected for one customer could therefore be mistakenly used for another customer. Production code should keep state isolated per request or session.

## 3. Where is the order-ID pattern?

The generated CrewAI schema contained only basic types such as `string` and `integer`. It did not contain the required pattern `^A[0-9]{4}$`.

Python type hints communicate data types, but they do not automatically communicate business judgments such as valid ID formats, order existence, evidence requirements, approval consequences, or policy thresholds. These rules therefore had to be implemented manually inside the tool bodies.

## 4. Baseline and CrewAI Comparison

Across the three test fixtures, the baseline used 1,826 tokens, while CrewAI used 4,122 tokens. CrewAI therefore consumed approximately 126% more tokens.

The additional cost came from framework instructions and the repeated inclusion of the agent’s role, goal, backstory, tool descriptions, and generated schemas. The prompt inspection estimated approximately 185 additional prompt tokens on every model call.

## 5. Runaway Behaviour

The runaway test received the incomplete message, “Where is my stuff?”, without an order ID. However, the agent produced a response about order `A1032`, even though the user had not supplied that ID. The run consumed 2,179 tokens in 26 seconds.

The `max_iter` setting limits the number of iterations, but it does not detect whether the agent is making meaningful progress. A no-progress detector would need to be implemented in a wrapper around crew execution or in a custom callback that records repeated actions and observations.

## 6. Remaining Safety Problem

The tool gates prevented invalid or unauthorized actions, but they did not guarantee that the final natural-language answer was truthful.

For example, the baseline incorrectly stated that a three-day delay might qualify for compensation, although the actual threshold was 10 days. The CrewAI response also referred to a potentially pending approval while `approvals created` was empty.

Therefore, the agent can still make unsupported claims in its final answer even when no consequential tool action occurs. I would add a final-response validator that checks every claimed status, policy threshold, and approval state against the actual tool results before returning the response to the customer.

## Conclusion

CrewAI reduced the amount of orchestration code required to create the agent, but it did not automatically provide the application-specific safety controls from Lab 3. Validation rules still had to be placed manually inside tools, shared module state created concurrency risks, token consumption increased, and the final answer could still contain unsupported claims.
