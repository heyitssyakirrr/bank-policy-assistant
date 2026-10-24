# Handoff: where the build stands and how to learn it

> Written 24 Oct 2026, after M6, by the Claude Code session that built
> M1–M6. **Part A is for Syakir. Part B is for the next Claude Code
> session.** It is a map: the full teaching documents
> (`docs/LEARNING_GUIDE.md`, `docs/BUILD_REPORT.md`) come in M8, after the
> evaluation.

---

# Part A: for Syakir

## A1. Where things stand

| Milestone | What | Commit | Status |
|---|---|---|---|
| M1 | LLM layer + traditional RAG | `6749bd2` | done |
| M2 | Agentic LangGraph graph | `224cbe5` | done |
| M3 | Postgres memory, audit log, PII masking, FastAPI SSE | `3c5ea97` | done |
| M4 | Streamlit UI (Chat, Compare, Evaluation) | `fdc8d1e` | done |
| M5 | Golden set + evaluation harness | `b0cf396` | done |
| M6 | Docker, CI, LangSmith | `c613af2` | done |
| M7 | First full evaluation, then fixes | | **waiting for you to run it** |
| M8 | Docs: README, BUILD_REPORT, LEARNING_GUIDE, … | | after M7 |
| M9 | Final evaluation, tag `v1.0-assessment` | | after M8 |

- **Tests:** 287 passing (`pytest -q`), ruff clean.
- **Branch:** `feat/agentic-rag`. Nothing is pushed (Claude Code is not
  allowed to push; you can).
- **Waiting for you:** the full evaluation (A8), the golden-set spot-check
  (A8), your Cohere monthly balance, a look at `.env.example`.

## A2. What happened during the automation

Each milestone ran the same loop (CLAUDE.md "Every milestone runs this loop"):

```
plan files → write code + tests → ruff + pytest green → code-reviewer subagent
  → fix every finding → small real-API smoke check → update PROGRESS.md → commit
```

The **code-reviewer** is a separate Claude with a fresh context: it never
saw the conversation that wrote the code, so it reads it like a stranger.
It found real bugs in every milestone (about 70 findings in total, all
fixed). The ones worth knowing, because they make good interview stories:

| Milestone | What the reviewer caught | Why it mattered |
|---|---|---|
| M1 | Counters (LLM calls, evidence) leaked from one question to the next when memory was on | The request budget would have tripped on the 3rd question of a chat |
| M1 | A quote like "Board ... the ... shall" passed the quote check | Any citation would have "verified" |
| M2 | The generator read the raw follow-up ("what about takaful?") instead of the rewritten question | Follow-ups, a headline feature, answered the wrong thing |
| M3 | A client closing the tab mid-answer left no audit row | An audit log with holes is not an audit log |
| M3 | Alembic would have deleted LangGraph's memory tables on the next migration | Silent data loss |
| M4 | A worse regeneration could replace a correct first answer | Quality went down after "fixing" |
| M5 | An outage would have counted as a correct refusal | Refusal accuracy would lie |
| M5 | Some gold labels were wrong (FAR question labelled with para 13, which is what *BNM* may do) | The evaluation would blame the model for a labelling mistake |
| M6 | pytest could upload fake runs to your LangSmith | Tests must never call outside services |

**Deviations from the plan** (all written into PROJECT_PLAN.md §5.7 and
the PROGRESS.md decisions log, with reasons):

1. **The "must" check** flags a statement only if *all* its citations are
   guidance (G). The FAR rule (e-KYC Appendix 2) has no S/G marker but is an
   obligation, and the strict rule rejected the correct answer.
2. **Verifier check 7 (new):** if a cited passage depends on another
   passage that was retrieved ("as required under paragraph 8.4") but the
   answer leaves it out, regenerate once.
3. **Thinking level per role:** planner/reviewer "low" (default thinking
   made the planner take 11 s), generator "medium" (on "low" it ignored the
   8.4 passage it had been given).
4. **The full evaluation costs ~295 calls,** not the plan's ~575.
5. Smaller ones: one Docker image for three services, the runner lives in
   `app/api/`, Streamlit pages in `ui/views/`.

## A3. The architecture

```
 Browser ── Streamlit UI (ui/) ──HTTP + SSE──► FastAPI (app/api/)
                                                   │
                                     TurnRunner (app/api/runner.py)
                                     1. mask PII   2. run the graph   3. write audit rows
                                                   │
                       ┌───────────────────────────┴───────────────────────────┐
                       ▼                                                       ▼
         traditional graph (app/agent/traditional.py)          agentic graph (app/agent/graph.py)
         retrieve → generate → verify → finish                 plan → search/lookup (parallel)
                                                                → review (≤2 rounds) → generate
                                                                → verify → finish
                       │  both share: generator, verifier, search engine, LLM adapter │
                       └───────────────────────────┬───────────────────────────┘
          ┌──────────────────────┬─────────────────┼────────────────────┬──────────────────┐
          ▼                      ▼                 ▼                    ▼                  ▼
  SearchService           ChatModelClient     PostgresSaver        Our audit tables     Qdrant
  (app/services/)         (app/llm/models.py) (agent memory)       (conversations,      1,219 chunks
  embed → Qdrant hybrid   Gemini 3.1/3.5 Lite                       turns, turn_events,  dense + BM25
  → RRF → Cohere rerank   rate limit, cache,                        feedback)
  → sector selection      retries, fallback
```

**Where everything lives:**

| Folder | What | Start reading at |
|---|---|---|
| `app/ingestion/` | Offline: PDFs → 1,219 chunks → Qdrant (built before M1) | `pipeline.py`, `indexer.py` |
| `app/services/` | The search engine | `search_service.py` |
| `app/llm/` | The one adapter every LLM call goes through | `models.py` |
| `app/agent/` | Both graphs, their nodes, prompts, the verifier | `graph.py`, `traditional.py` |
| `app/api/` | FastAPI routes, the runner, what a turn looks like | `runner.py`, `views.py` |
| `app/db/` + `migrations/` | Postgres tables, Alembic | `models.py` |
| `ui/` | Streamlit (talks to the API over HTTP only) | `app.py`, `components.py` |
| `evals/` | Golden set, metrics, judge, report | `run.py`, `golden_set.yaml` |
| `tests/` | 287 tests; `fakes.py` is the scripted LLM | `test_agentic_graph.py` |

## A4. One question, end to end (real inputs and outputs)

The question: **"What does e-KYC paragraph 8.10 require?"** in agentic mode.
This is the cross-reference case from PROJECT_PLAN §2: 8.10 says "as
required under paragraph 8.4", so a good answer needs both.

**1. The UI sends** `POST /chat`:
```json
{"question": "What does e-KYC paragraph 8.10 require?", "mode": "agentic", "sector": "banking"}
```

**2. The runner** (`app/api/runner.py`) masks PII (nothing to mask here; a
MyKad number would become `[NRIC]`), creates the conversation in Postgres,
and streams events back. The first one:
```json
{"type": "start", "conversation_id": "9ec6…", "turn_id": "b81b…", "mode": "agentic", "pii_masked": []}
```

**3. The planner** (`understand_question`, `app/agent/planning.py`): one LLM
call returns a `QueryPlan` (schema in `app/agent/schemas.py`):
```json
{"route": "lookup", "standalone_question": "What does paragraph 8.10 of the e-KYC policy document require?",
 "searches": [], "paragraph_refs": [{"document": "ekyc", "paragraph_id": "8.10"}],
 "reasoning": "The user names a paragraph and its document."}
```
Code then **sanitises** it (max 3 searches, paragraph IDs must look like
`8.10`, the route must match the content).

**4. The tool** `lookup_paragraphs` (`app/agent/tools.py`) fetches e-KYC 8.10
by number. Evidence is labelled `E1`, `E2`, … by code.

**5. The reviewer** (`review_evidence`, `app/agent/review.py`): one LLM call
returns an `EvidenceReview`:
```json
{"relevant_evidence": ["E1"], "is_sufficient": false,
 "missing_information": "The text refers to requirements under paragraph 8.4…",
 "cross_references": [{"document": "ekyc", "paragraph_id": "8.4"}]}
```
Not sufficient, so the graph loops (round 2): `lookup_paragraphs` fetches
8.4, and the reviewer runs again: "2 relevant passages, enough to answer".

**6. The generator** (`generate_answer`, `app/agent/nodes.py`) sees the
passages as numbered blocks. Code marks the dependency it found:
```
[E1] e-KYC Policy Document, para 8.10 (G), p. 7
Cross-reference: E2 (para 8.4)
"""
G 8.10 In identifying and verifying an individual's identity through e-KYC as required under paragraph 8.4 …
"""
```
It returns a `GroundedAnswer`: statements, each with `evidence_ids`, an
exact `quote`, and `obligation` (must / should_or_may / information).

**7. The verifier** (`verify_answer` → `check_answer`,
`app/agent/verification.py`) is pure code. Every cited ID must exist, the
quote must really be in the cited passage, a "must" can't cite only
guidance, and a cross-reference can't be left out. One problem means one
regeneration with feedback.

**8. The answer** (real output, the Compare page, `docs/screenshots/compare.png`):

| | Traditional | Agentic |
|---|---|---|
| Answer | MAY: 8.10's measures only | **MUST** (8.4, S): measures must be secure and effective, proportionate to risk. **MAY** (8.10, G): document verification, biometrics, liveness |
| LLM calls | 1 | 5 (planner, 2 reviews, generator, 1 regeneration) |
| Latency | 7 s | 41 s |

**9. Every step was streamed** as an event and saved:
```json
{"type": "step", "node": "review_evidence", "lines": ["Reviewed evidence: missing …", "Following cross-reference e-KYC para 8.4"], "ms": 23600}
```
The final `answer` event carries the whole turn (summary, statements,
sources with PDF links, cost). Postgres gets one `turns` row (question,
plan, evidence, answer, every verification report, models, prompt and index
versions, tokens, latency) plus one `turn_events` row per step. Both tables
are **append-only**: database triggers reject UPDATE, DELETE and TRUNCATE.

**Other real runs worth knowing** (PROGRESS.md "Results so far"):
- **Follow-up:** "Can a bank onboard customer 900101-14-5678 without
  meeting face to face?" became `[NRIC]` before reaching Gemini, the logs
  or the database. Then "what about money changers?" in the same
  conversation was rewritten from memory, the sector switched to money
  services, and the answer cited the AML **14C** copies, not the 14A ones.
- **Refusal:** "What's the interest rate cap on credit cards?": the planner
  routed it to a refusal in 1 call, before any search.
- **Clarification:** "What does para 8.2 say?": 8.2 exists in all three
  documents, so agentic asks which one; traditional just answers.

## A5. The design decisions you must be able to defend

1. **LLM for judgment, code for facts.** The LLM decides what to search,
   whether the evidence is enough, and how to phrase the answer. Code
   checks citations, quotes, S/G labels, cross-references, limits and PII,
   because those have one correct answer (PROJECT_PLAN §3).
2. **A bounded agent, not a free loop.** Max 2 review rounds, max 3
   searches per plan, a request budget (5 LLM calls, 6 searches, 60 s) and
   a recursion limit. A bank needs predictable, auditable behaviour.
3. **Citations by ID.** The LLM cites `E3`; it never types a paragraph
   number, so it cannot invent a citation. Code maps `E3` back to "e-KYC
   para 8.4 (S), p. 6".
4. **Same engine, two modes.** Both graphs share the search, the generator
   and the verifier, so the comparison measures only the decision-making.
5. **Degrade, never crash.** Planner down → plain search; reviewer down →
   answer from all evidence; reranker down → fused order; embedding down →
   BM25 only; one failing search → the others still answer.

## A6. Tests (pytest): what they prove

```
evaluation (evals/, real models, run on demand)   "is it actually good?"
API tests (tests/test_api.py)                     "does the service contract hold?"
graph-path tests (scripted fake LLM)              "does each decision path work?"
unit tests (pure functions)                       "is each piece correct?"
```

- **The scripted fake LLM** (`tests/fakes.py` → `ScriptedLLM`) returns fixed
  plans, reviews and answers, so every graph path runs the same way every
  time: lookup, parallel search, loop stopping at 2 rounds, refusal,
  clarification, cross-reference, regeneration, planner/reviewer/generator
  outages, budget limits, follow-ups with memory.
- No test calls Gemini, Cohere, Qdrant or Postgres. Real quality is
  measured by the evaluation, never by pytest.
- Run: `pytest -q` (about 7 seconds).

## A7. Evaluation: what it measures

- **Golden set:** `evals/golden_set.yaml`, 40 questions in 8 categories
  (lookup, paraphrased, comparison, cross-reference, sector, follow-up, out
  of scope, ambiguous), each with its gold paragraphs and expected outcome.
- **Retrieval ablation:** the same question through each search stage
  (dense → BM25 → hybrid → + exact lookup → + rerank → + selection),
  scored Hit@5, Recall@5, MRR@5. This proves each optimisation helps.
- **Answers, both modes:** outcome accuracy (answer / refuse / clarify),
  citation recall, evidence recall, verifier pass rate, our LLM judge's
  faithfulness and relevancy, LLM calls, tokens, latency.
- **Smoke results so far** (5 answer items, `evals/results/20261024-1457-smoke/report.md`):

  | | Traditional | Agentic |
  |---|---|---|
  | Outcome accuracy | 80% | 100% |
  | Citation recall | 50% | 83% |
  | Faithfulness | 100% | 100% |
  | LLM calls per question | 1.0 | 2.6 |

  The full run (40 items) is M7.

## A8. Your hands-on plan (on the Claude Code desktop)

Do these in order. Each step says what it costs.

**Step 1. Get the stack running** (free)
```powershell
cd "C:\AI Projects\bank-policy-assistant"
.venv\Scripts\activate
docker compose up -d db qdrant
pytest -q                                   # expect 287 passed
```

**Step 2. Ask questions from the command line** (each new question ~1–5
Gemini calls; repeats are free from the cache)
```powershell
python -m scripts.ask "What must a bank do if FAR exceeds 5%?"
python -m scripts.ask "What does e-KYC paragraph 8.10 require?" --mode agentic --steps
python -m scripts.ask "Can banks onboard remotely?" --mode agentic --steps --follow-up "what about takaful operators?"
```
Watch the steps: planner → tools → reviewer → generator → verifier.

**Step 3. Run the app like the demo** (two terminals, venv active in both)
```powershell
uvicorn app.main:app --port 8000
streamlit run ui/app.py
```
Open http://localhost:8501: try Chat, Compare ("What does e-KYC paragraph
8.10 require?"), and a question with a fake MyKad number to see the
masking notice. http://localhost:8000/docs shows the API.

**Step 4. Look at the audit trail** (free)
```powershell
docker exec bank-policy-assistant-db-1 psql -U postgres -d bankpolicy -c "select mode, left(question,50), status, llm_calls, latency_ms from turns order by created_at desc limit 5;"
```

**Step 5. Spot-check the golden set** (free; the plan asks for ≥ 10 items)
```powershell
python -m evals.golden --show fact-6 xref-1 compare-6
```
Open the PDF at the paragraph and confirm the gold paragraphs answer the
question. Add each ID you checked under `syakir:` in
`evals/golden_review.yaml`.

**Step 6. Run the full evaluation (M7)**, best right after the 3 pm
Malaysia-time quota reset:
```powershell
python -m evals.run --full --estimate       # free: prints the expected calls
python -m evals.run --full                  # ~178 + ~102 + 15 Gemini calls, 30–60 min
```
Then open `evals/results/<time>-full/report.md` (or the Evaluation page)
and ask Claude Code: "analyse the full evaluation and fix the top failures
(M7)".

**Step 7. Try Docker the way the assessor would** (free to index)
```powershell
# stop the local uvicorn/streamlit first: Docker uses ports 8000 and 8501
docker compose up --build
```

## A9. Known issues and limits (be honest about these in the demo)

- Traditional mode ignores a document named in the question: "paragraph
  9.1 of the Fair Treatment … policy" ranks e-KYC 9.1 and AML 9.1 first.
- Agentic answers are slow on the free tier: p50 ~34 s on the smoke set
  (Gemini 504 retries, the generator's medium thinking, 2 review rounds).
- Cohere's reranker sometimes demotes the gold paragraph (compare-1, sector-1).
- The generator is non-deterministic: the same question can produce a
  slightly different answer; the verifier keeps it honest, the evaluation
  measures it.
- The golden set was written by Claude Code from the chunks; it needs your
  spot-check to be trustworthy.

## A10. Check yourself (explain it in your own words)

1. **Why does the agentic mode answer the 8.10 question better?**
   *Model answer:* the reviewer reads the evidence, sees "as required under
   paragraph 8.4", and asks for 8.4 before answering; traditional retrieves
   once and never looks again.
2. **Why can't the LLM invent a citation?** *Model answer:* it only sees
   labels like E3 and must cite those; code checks every cited label exists
   and the quote is really in that passage, and code turns E3 into the
   paragraph and page.
3. **Why a scripted fake LLM in tests?** *Model answer:* real output
   changes run to run; scripting it makes every decision path testable
   exactly. Quality is measured separately, by the evaluation.
4. **What stops the agent looping forever?** *Model answer:* a round
   counter (max 2 reviews), the request budget (5 LLM calls, 6 searches,
   60 s) checked by code, and LangGraph's recursion limit as a last resort.

---

# Part B: for the next Claude Code session

1. **Read first:** this file, then PROGRESS.md (decisions log, quota
   ledger, known issues), then PROJECT_PLAN.md §5.7 (every build-time
   change) and §10. CLAUDE.md's rules still apply in full.
2. **The next milestone is M7.** Syakir runs `python -m evals.run --full`
   himself (CLAUDE.md quota rules). When he says it's done: read the newest
   `evals/results/*-full/report.md` and `results.json`, analyse the failures
   by category, fix the top ones (the M7 candidates are listed in
   PROGRESS.md "Known issues"), re-run only the affected items with
   `python -m evals.run --only <ids>` (cached calls are free), and record
   everything. Then M8 and M9.
3. **Teaching mode.** Syakir wants to understand everything before the
   demo (Mon 26 Oct). When he types `/tour`, follow .claude/skills/tour: first the
   concept ladder (PROJECT_PLAN §11) from one LLM call upwards, then Part A one
   section at a time: show the real file and lines, run the free commands
   with him, ask the check-yourself questions, and wait for his answer
   before moving on. Use `/explain` style for any concept (CLAUDE.md
   "Explaining to Syakir"). Never run anything that spends quota without
   saying the cost first.
4. **Quota.** Check the PROGRESS.md ledger before any real call; ≤ 30
   Gemini calls per milestone of your own smoke checks.
