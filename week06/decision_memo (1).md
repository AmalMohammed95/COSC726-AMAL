# Lab 5 — Memory Decision Memo

## 1. Which layer helped most, and on which questions?
Recursive embedding search raised the answered rate from 67% to 83%.
Episodic memory raised it to 100% by making Q6 answerable.

## 2. Which layer made things worse, and how do you know?
Parent-child retrieval increased estimated tokens from 492 to 1,221
without improving the answered rate. Semantic and procedural memory
added more tokens without improving it further.

## 3. Where did embeddings beat keywords?
For Q5, keyword search missed POL-BILLING because the question said
"charged twice" while the policy said "duplicate charges". Embedding
search found the correct policy. I would test both on more real questions
before choosing a production retrieval strategy.

## 4. What did episodic memory make possible?
It verified that cust-4417 had contacted support twice. A general policy
document cannot establish an individual customer's contact history.

## 5. Where does memory access sit in the authorisation model?
Every read and write requires a user scope enforced by the store.
The application must obtain the user identity from an authenticated
session, rather than trusting a user_id supplied in a request.

## 6. What did the measurement not tell you?
Five short documents and six exercise questions form only a smoke test.
The deterministic answer check does not assess generated answer quality.
Token counts are estimates, and only one embedding model was tested.
Changing that model would require re-indexing the vector collection.
