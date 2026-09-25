# COSC726 Week 8 Lab 7 — Decision Memo

Model: qwen2.5:3b via Ollama.
Evaluation: four briefs, one measured run per configuration.

## Measurement ledger

| Brief | Topology | Tokens | Seconds | Coverage | Grounded |
|---|---|---:|---:|---:|---:|
| B1 | Single | 569 | 2.4 | 100% | 75% |
| B1 | Crew + critic | 3313 | 9.1 | 33% | 67% |
| B2 | Single | 680 | 4.5 | 100% | 50% |
| B2 | Crew + critic | 3961 | 13.7 | 100% | 50% |
| B3 | Single | 593 | 3.3 | 100% | 67% |
| B3 | Crew + critic | 3866 | 11.8 | 50% | 67% |
| B4 | Single | 562 | 2.6 | 0% | 0% |
| B4 | Crew + critic | 2746 | 7.3 | 0% | 0% |

## 1. Did the crew beat the baseline?

No measured brief showed a coverage or grounding gain from the crew.
On B1, coverage fell from 100% to 33% and grounding from 75% to 67%.
On B2, both systems achieved 100% coverage and 50% grounding.
On B3, coverage fell from 100% to 50%, while grounding remained 67%.
Both systems explicitly said that Q3 performance data was unavailable
on B4. Its 0%/0% scores do not adequately represent this correct abstention.

## 2. What was the token ratio?

The crew used 5.8 times the baseline tokens on B1 and B2, 6.5 times
on B3, and 4.9 times on B4. These observed ratios fall within the
3–10 times comparison described in the lab. On B2, the extra cost
bought no measured improvement.

## 3. Where did context get lost?

The analyst received the researcher's findings rather than the source
documents. On B1, the final analysis and memo omitted the legal
sole-remedy constraint in LEG-0455. The trace confirms that the
analyst's conclusion did not carry that constraint to the writer.
The printed research handoff was truncated, so this run alone does
not establish whether the source was omitted by the researcher or
ignored by the analyst. Inspecting the complete researcher message
would distinguish these possibilities.

## 4. What did the critic buy, and what did it cost?

Across four briefs, the crew without a critic used 6,441 tokens and
28.1 seconds. The measured crew with a critic and up to one revision
used 13,886 tokens and 41.9 seconds: 7,445 more tokens and 13.8 more
seconds, about 2.16 times the tokens. The critic rejected unsupported
B1 and B3 memos, but the permitted revision did not restore their
missing coverage. These are separate model runs, so the difference
also contains ordinary output variation.

## 5. Classify the observed failures

- Specification: B2 used a multi-agent workflow for two simple
  performance numbers; its roles added cost without measured benefit.
  The evaluation metric also gave B4 zero coverage and grounding
  despite both systems correctly reporting that Q3 data was absent.
- Inter-agent misalignment: B1's legal constraint did not survive into
  the analyst's conclusion and final memo. The exact point of loss
  requires inspection of the complete researcher handoff.
- Verification: B3's final memo omitted the ninety-day termination
  notice and introduced an unsupported condition about stable southern
  depot throughput. The critic did not produce an approved correction
  within the revision limit. On B4, both systems avoided fabricating
  a Q3 performance percentage, so that anticipated verification
  failure was not observed.

## 6. What would I ship, and what did this lab not tell me?

For these short, sequential briefs I would start with the single-agent
baseline, require source citations, and add a deterministic check for
missing required sources when such a list is available. I would use a
crew only when a larger task demonstrates a quality gain that justifies
its extra cost. This lab did not test parallel work, where multiple
agents may have a stronger case. Four briefs and one run each provide
no variance estimate. The results use qwen2.5:3b; a 7B or frontier
model may fail differently. The corpus and scoring rules are small
and constructed for this exercise.
