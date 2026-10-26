# Agentic RAG: an investigation, and what we measured

This note answers two parts of the brief: *investigate agentic RAG as a whole*
and *investigate the differences between traditional and agentic RAG*. It
explains the idea, the common patterns, the trade-offs and when not to use an
agent, then shows what this project measured on the same 40 questions in both
modes.

## 1. From RAG to agentic RAG

**Retrieval-augmented generation (RAG)** gives a language model facts it was not
trained on: retrieve relevant passages, put them in the prompt, and ask the
model to answer only from them, with citations. The pipeline is fixed:

```
question ─► retrieve top k ─► LLM writes the answer ─► done
```

It works well when one search with the question as typed finds everything
needed. It fails in predictable ways:
- **one search can't cover two topics:** a comparison of e-KYC with AML rules
  mostly retrieves one document;
- **it can't follow references:** e-KYC 8.10 says "as required under
  paragraph 8.4", and 8.4 is never fetched;
- **the question may not stand alone:** "what about takaful operators?" means
  nothing without the conversation;
- **it can't tell "not covered" from "covered"** before searching: it always
  retrieves *something*, and the model may stretch it into an answer;
- **it can't ask:** "what does para 8.2 say?" exists in all three documents.

**Agentic RAG** lets the model make decisions inside the pipeline: what to
search for, whether the results are enough, whether to search again, and when
to stop. Retrieval becomes a **tool** the model calls, inside a loop the system
controls.

## 2. Workflows vs agents

A useful distinction (made, for example, in Anthropic's *Building effective
agents*):
- a **workflow** runs LLM calls along code paths the developer wrote;
- an **agent** lets the LLM choose the path: which tool, with which input, and
  when it is done.

Most production systems sit between the two. This project's agentic mode is a
**bounded agent**: the LLM makes the decisions at fixed routing points (the
planner's route, the reviewer's "enough or not"), and code enforces every limit
(2 review rounds, 3 searches per plan, 5 LLM calls and 6 searches per
question, and no new round or regeneration after 60 s; the time limit stops
new work, not a call already running). The traditional mode is a pure workflow.

## 3. The patterns, and where each one is in this project

| Pattern | Idea | Here |
|---|---|---|
| **Routing** | Classify the request first and send it down the right path | The planner returns a `route`: search, lookup, search and lookup together, clarify, out of scope, small talk (`app/agent/planning.py`, `routing.py`). Out-of-scope questions are refused **before** any search. |
| **Query rewriting** | Turn a follow-up or vague question into a standalone query | `standalone_question`: "what about takaful operators?" → "Can takaful operators onboard customers without meeting them face to face?", and the sector switches |
| **Decomposition** | Split a multi-part question into sub-queries | Up to 3 searches per plan, each with its own document filter: a comparison becomes one search in e-KYC and one in AML |
| **Tool use / parallel tool calls** | The model chooses functions and their arguments; code runs them | The planner (and the reviewer) choose the tool and its arguments by filling a schema: searches with document filters, paragraph references. Code validates the choice and runs `search_policies` / `lookup_paragraphs` (`app/agent/tools.py`) in parallel with LangGraph `Send`. Tool use through structured output, not the provider's function-calling API |
| **Corrective RAG** (CRAG, Yan et al. 2024) | Grade the retrieved evidence; if it is weak, retrieve again or differently | The reviewer (`review.py`) marks relevant evidence, says what is missing, and asks for follow-up searches or cross-references; at most 2 rounds |
| **Self-RAG** (Asai et al. 2023) | The model critiques its own retrieval and output | Partly: the reviewer critiques the evidence, but the answer is checked by **code** (the verifier), not by the model judging itself |
| **Adaptive RAG** (Jeong et al. 2024) | Spend more effort only on harder questions | The route and the review loop adapt the work: a refusal costs 1 call, a lookup 3–4, a comparison with a follow-up more |
| **Reflection / verification** | Check the output, then fix it | The verifier checks every citation, quote, MUST vs guidance and cross-reference; one regeneration with the problems listed |
| **ReAct** (Yao et al. 2022) | Interleave reasoning and tool calls in a free loop until the model stops | **Not used**: a free loop is hard to bound and audit. Listed in the backlog as a comparison mode. |

## 4. Trade-offs

| | Traditional RAG | Agentic RAG |
|---|---|---|
| LLM calls | 1 (2 with a regeneration; 1.05 on average here) | several (2.85 on average here, at most 5) |
| Latency | low | higher, and it varies with the path |
| Predictability | same path every time | the path varies, so it must be bounded and traced |
| Failure modes | misses what one search can't find | wrong routing, over-filtered searches, loops (bounded here) |
| Testing | one path | every path needs a test (a scripted fake LLM here) |
| Best for | single-topic factual questions | multi-part, cross-referenced, follow-up, ambiguous and out-of-scope questions |

Agentic RAG does not make retrieval better by itself: it decides *what* to
retrieve. It still depends on a good search engine. Both modes here share one.

## 5. When not to use an agent

- **Simple, single-topic questions.** Both modes found the answer to every
  direct lookup in our set (4 questions); the agent only added time and cost.
- **Strict latency or cost budgets.** Here the agent took a median 25 s versus
  4 s, at about 2.7× the calls.
- **When a deterministic path is required,** for example a regulated decision
  that must follow the same steps every time and be explained step by step.
  Then prefer a workflow, or bound the agent tightly and log every decision, as
  done here.
- **When the routing decision is trivial for code.** If a keyword or a form
  field decides the path, use code: it is free, exact and testable.
- **When the corpus is small enough to put in the prompt.** Then there is
  nothing to search.

## 6. What we measured

The same 40-question golden set (8 categories), both modes, the same search
engine, generator and verifier. Full report:
`evals/results/20261026-0338-full/report.md` (and the app's Evaluation page).

| | Traditional | Agentic |
|---|---|---|
| Outcome accuracy (answer / refuse / clarify, as expected) | 95% | **100%** |
| Citation recall (gold paragraphs cited) | 80% | **91%** |
| Evidence recall (gold paragraphs retrieved)¹ | 82% | **92%** |
| Faithfulness (statements supported by their citation; LLM judge) | 98% | 99% |
| LLM calls per question | **1.05** | 2.85 |
| Latency p50 / p95 | **4.4 s / 11.0 s** | 25.1 s / 69.9 s |

¹ Agentic counts every passage it gathered over several searches; traditional
counts its top 5. p50 is the median; p95 means 95% of questions were faster.

By category (citation recall unless stated):

| Category | Traditional | Agentic | What made the difference |
|---|---|---|---|
| Cross-reference | 62% | **100%** | the reviewer followed "as required under paragraph 8.4" and fetched 8.4; the verifier's cross-reference check made the generator use it |
| Comparison | 56% | **83%** | decomposition: one filtered search per document |
| Follow-up | 75% | **100%** | query rewriting from the conversation, sector switched |
| Ambiguous (outcome) | 0% | **100%** | only the agent can ask which document |
| Out of scope (outcome) | 100% | 100% | traditional refused after searching; agentic before searching, in 1 call |
| Direct lookup | 100% | 100% | no difference: an agent isn't needed |
| Paraphrased | 87% | 92% | small difference |
| Sector-specific | **100%** | 75% | one agentic miss (below) |

The categories are small (2–10 questions; in a 4-question category one question
is 25 points), and the cross-reference questions name the reference ("…including
the rule it refers to"). Read the gaps as direction, not precise effect sizes.
The judge (`gemini-3.1-flash-lite`) is the same model as the reviewer, so the
family that gathers the evidence also scores the answers; a cross-check by a
second model was planned but stopped after 3 calls on 504 errors.

**Where the agent was worse.** For "How may a money changer verify a
customer's identity remotely?" the planner rewrote it as an e-KYC question
("…remotely using e-KYC?") and never retrieved AML 14C.16.12, the
money-services rule on video calls. The
answer was correct but incomplete. This is the agent's typical failure: a
planning decision that narrows the search too far. Traditional searched all
documents and found it. A known limit: the next prompt revision tells the
planner not to narrow a sector-rule question to one document.

**Conclusion.** On this corpus the agent is worth its cost for the question
types that need it (cross-references, comparisons, follow-ups, ambiguity and
out-of-scope), and it is not worth it for direct lookups and simple facts. A
production system would route cheap questions to the workflow and hard ones to
the agent, which is Adaptive RAG one level up.
