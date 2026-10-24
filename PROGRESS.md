# Progress: Bank Policy Compliance Assistant

> Update after every work session. To continue in a new chat, paste this file,
> PROJECT_PLAN.md, and the files you're working on.

**Last updated:** 2026-10-24
**Goal:** live Streamlit demo of the agentic RAG, **Mon 26 Oct 2026**
**Current step:** M5, golden set + evaluation harness (autonomous build by
Claude Code; see PROJECT_PLAN.md §10)
**Tests:** 271 passing
**Repo:** _(add your GitHub link)_

---

## Status at a glance

```
DONE ████████████████████  Ingestion: 3 PDFs → 1,219 structured chunks
DONE ████████████████████  Indexing: Qdrant dense + BM25 sparse, embedding cache
DONE ████████████████████  Hybrid retrieval: RRF + exact paragraph lookup
DONE ████████████████████  Rerank + selection: Cohere, sector collapse/demotion
DONE ████████████████████  M1  LLM layer + traditional RAG
DONE ████████████████████  M2  LangGraph agent
DONE ████████████████████  M3  Postgres memory + audit + FastAPI
DONE ████████████████████  M4  Streamlit UI
NEXT ░░░░░░░░░░░░░░░░░░░░  M5  Golden set + evaluation harness
     ░░░░░░░░░░░░░░░░░░░░  M6  Docker, CI, LangSmith
     ░░░░░░░░░░░░░░░░░░░░  M7–M9  Evaluation runs, fixes, docs, freeze
```

**What exists** is the *retrieval engine*: give it a question and it returns
the 5 best paragraphs. **What's missing** is everything that uses it: the LLM
answering, the agent deciding, memory, the API, the UI, evaluation, and
packaging.

---

## Checklist (from PROJECT_PLAN.md §10)

### Already built
- [x] Ingestion pipeline (`app/ingestion/`): Docling → paragraphs, S/G, tables,
  footnotes, callouts, sectors
- [x] Qdrant indexing with an embedding cache and quota-safe resume
  (`indexer.py`)
- [x] Hybrid retrieval: dense + BM25 in one round trip, RRF k=60, exact
  lookup (`retrieval_service.py`)
- [x] Cohere rerank + graceful fallback (`rerank_service.py`,
  `search_service.py`)
- [x] Sector duplicate collapse + other-sector demotion + exact-match
  guarantee (`selection.py`)
- [x] `scripts/try_search.py` debugging view
- [x] 85 unit tests

### Autonomous build milestones (PROJECT_PLAN.md §10)
- [x] Gemini limits checked (Flash-Lite 15 RPM / 500 RPD; Flash 5 RPM / 20 RPD)
- [x] M1 LLM layer + traditional RAG (24 Oct). `app/llm/` (one adapter
  `ChatModelClient`: per-model sliding-window limiter, SQLite response cache,
  live retries, 1 repair retry, fallback model), `app/agent/` (schemas,
  evidence E-labels + merge reducer, prompts `PROMPT_VERSION`, verifier,
  per-turn state, shared generator/verifier nodes, traditional graph),
  `scripts/ask.py`. Search engine: document filters, document-aware lookup,
  caller-supplied sector, BM25 fallback when embedding fails, no-reranker
  fallback when no Cohere key. Reviewed by code-reviewer: 5 major + 12 minor
  findings, all fixed except the one noted in the decisions log.
- [x] M2 Agentic graph + graph-path tests (24 Oct). `app/agent/`:
  `planning.py` (planner + plan sanitiser + fallback plan), `tools.py`
  (`search_policies`, `lookup_paragraphs`, parallel via `Send`, failures
  degrade), `review.py` (reviewer + sanitiser + budget trimming),
  `routing.py`, `responses.py` (refusal, greeting, clarification),
  `budget.py`, `graph.py`, `steps.py` (one line per step for CLI/API/UI),
  `checkpoint.py` (registered state types for the checkpointer).
  `scripts/ask.py --mode agentic --steps --follow-up`. 47 new tests: every
  §8.1 graph path. Reviewed by code-reviewer: 3 major + 8 minor findings,
  all fixed.
- [x] M3 Postgres checkpointer + audit tables + PII masking + FastAPI SSE
  (24 Oct). `app/db/` (models, session, audit repository), Alembic
  (`alembic.ini`, `migrations/` 0001 tables + append-only triggers, 0002
  no-TRUNCATE), `app/core/pii.py`, `app/api/` (runner, views, schemas,
  deps, routes: chat SSE, conversations, feedback, documents, health),
  `app/main.py` app factory with a PostgresSaver. Old echo app removed.
  Reviewed by code-reviewer: 2 major + 7 minor findings, all fixed.
- [x] M4 Streamlit: Chat, Compare, Evaluation (24 Oct). `ui/` (app.py
  navigation, views/chat, compare, evaluation; components; api_client),
  `scripts/screenshot_ui.py`, `docs/screenshots/`. While testing the UI:
  per-role thinking, code-marked cross-references, verifier check 7.
  Reviewed by code-reviewer: 4 major + 8 minor findings, all fixed.
- [ ] M5 Golden set + evaluation harness (smoke + full)
- [ ] **Syakir:** spot-check ≥ 10 golden-set items against the PDFs
- [ ] M6 Docker (index from committed cache), `.env.example`, LangSmith, CI
- [ ] M7 Evaluation #1 (**Syakir runs it** when Claude Code says ready) + fixes
- [ ] M8 Docs: README, architecture, investigation, testing, demo script,
      BUILD_REPORT, LEARNING_GUIDE
- [ ] M9 Final evaluation (**Syakir runs it**), tag `v1.0-assessment`
- [ ] **Syakir:** read LEARNING_GUIDE + BUILD_REPORT, make slides, rehearse
      twice, record a backup video

### Quota ledger (update after every real-API run)
| Date/time | Run | Gemini 3.1-lite | Gemini 3.5-lite | Cohere |
|---|---|---|---|---|
| 24 Oct ~02:10 | M1 smoke: FAR question + credit-card refusal (traditional), + 2 query embeddings | 0 | 2 | 2 |
| 24 Oct ~02:20–02:38 | M2 smoke: the five §2 cases in agentic mode (8.10 ×3, comparison ×2, follow-up pair, para 8.2, credit card), incl. 3 retried 504s, + 9 query embeddings | 21 | 7 | 9 |
| 24 Oct ~02:46 | M3 smoke: 2 API requests (agentic follow-up across requests, NRIC in turn 1), incl. 1 retried 504, + 3 query embeddings | 5 | 2 | 3 |
| 24 Oct ~03:00–03:40 | M4: UI chat question, 3 Compare runs (8.10), generator prompt/thinking trials via ask.py (cached planner/reviewer) | 15 | 11 | 5 |
| **24 Oct total** | | **41** | **22** | **19** |

---

## Results so far (real data)

| Query | Result |
|---|---|
| "What must a bank do if FAR exceeds 5%?" | para 13, the FAR table and para 11 in the final top 3. The reranker lifted para 11 from fused #8 to #3 |
| "14A.9.6" | [EXACT] #1. Dense alone missed it entirely, which is why hybrid exists |
| "can a customer be onboarded without meeting face to face" | Final: 14A.15.1, 14A.15.7, e-KYC 8.26, 14A.15.2, 14A.15.9. 12 sector copies collapsed; money-services-only rules demoted |
| "what does para 8.10 say about biometric matching" | [EXACT] 8.10 #1; 8.11 (money services only) demoted |
| M1 traditional: "What must a bank do if FAR exceeds 5%?" | ANSWERED, 5 statements (3 MUST, 2 MAY) citing e-KYC App. 2 paras 10 (table), 11, 13; verification passed first time; 1 LLM call, 1,705 in / 606 out tokens, 6.7 s |
| M1 traditional: "What is the interest rate cap on credit cards?" | NOT_COVERED, no statements; 1 LLM call, 5.6 s |
| Cached re-run of the FAR question | 0.1 s, 0 API requests |
| M2 agentic: "What does e-KYC paragraph 8.10 require?" | planner → lookup ekyc:8.10 → reviewer: "refers to 8.4" → lookup ekyc:8.4 → reviewer: enough → answered, verified. 4 LLM calls. The generator still cited only 8.10 (quality item for M7) |
| M2 agentic: "How do the e-KYC verification rules differ from AML's non-face-to-face rules?" | split into 2 filtered searches (e-KYC, AML/CFT) in parallel; reviewer followed 14A.15.7; answered with 5 statements across both documents (8.10, App. 4 para 2, 14A.15.9, 14A.15.2, 14A.15.7). 4 LLM calls, 29 s |
| M2 agentic follow-up: "Can a customer be onboarded without meeting face to face?" then "what about takaful operators?" | turn 2 rewritten from history, sector switched to insurance/takaful ("Applies to: insurers and takaful operators"), both answered and verified, 3 LLM calls each |
| M2 agentic: "What does para 8.2 say?" | CLARIFY: asks which of the 3 documents; 1 LLM call, no search |
| M2 agentic: "What's the interest rate cap on credit cards?" | OUT_OF_SCOPE refusal before any search; 1 LLM call |
| M3 API: "Can a bank onboard customer 900101-14-5678 without meeting face to face?" then "what about money changers?" (same conversation_id) | NRIC masked in prompts, logs and `turns`; turn 2 rewritten from the Postgres checkpoint, sector → money services, cites the AML 14C copies (14C.16.1, 14C.16.10-12); 31 s (one 504 retry) and 9 s; 2 `turns` + 13 `turn_events` rows; UPDATE and TRUNCATE on the audit tables rejected |
| M4 UI Compare: "What does e-KYC paragraph 8.10 require?" | traditional 7.2 s, 1 LLM call: only 8.10 (MAY). Agentic 40.6 s, 5 LLM calls: followed 8.10 → 8.4, regenerated once (check 7), answered MUST from 8.4 (S) + MAY from 8.10 (G). Screenshot: docs/screenshots/compare.png |
| Generator thinking trial (8.10) | low: ignored the fetched 8.4 even with an explicit prompt rule; medium: used it (4.3 s generator) |
| Gemini latency | planner 11 s with default thinking → 3 s with `thinking_level=low`. gemini-3.1-flash-lite returned 504 DEADLINE_EXCEEDED on ~1 in 8 calls (retried); a stalled call cost up to 30 s, now 20 s |

**Latency:** embed ~0.7–0.9 s · Qdrant search ~30–45 ms · rerank ~0.8–1.3 s

**Ingestion quality:** e-KYC 57/57 and FTFC 120/120 main-body paragraphs
correct. AML: 2 boundaries wrong (14A.14.1 and 14B.16.2 merged into a
neighbour; text kept). 182/182 split-off S/G markers matched by position.

---

## Current repo structure

```
bank-policy-assistant/
  app/
    core/        config.py (pydantic-settings), logging.py, retry.py (tenacity), rate_limit.py
    db/          vector_repository.py (only file that talks to Qdrant)
    ingestion/   OFFLINE: models, parser (Docling), structure (rules), chunker,
                 manifest, quality, debug, pipeline (CLI), indexer (CLI)
    services/    bm25_encoder, embedding_service, retrieval_service,
                 rerank_service, selection, search_service
    data/        raw_pdfs/, ingest_manifest.yaml, interim/ (caches), processed/chunks.jsonl, debug/
    api/, schemas/, static/, main.py   ← old Phase 0 echo app: replaced on Day 3
  scripts/       try_search.py
  tests/         test_ingestion, test_indexing, test_retrieval, test_rerank (85 tests)
  docker-compose.yml  (postgres + qdrant v1.19.1)
  requirements.txt / requirements-ingest.txt / requirements-dev.txt, pyproject.toml
```

## Environment notes
- Windows, Python 3.12, `.venv`, PowerShell. Extract zips with
  `Expand-Archive -Path <zip> -DestinationPath . -Force` (not "Extract All",
  which creates a subfolder).
- Docker Desktop must be running before `docker compose up -d` (enable "start
  at login").
- Postgres driver: `psycopg[binary]` v3, with `DATABASE_URL` starting
  `postgresql+psycopg://`.
- Keys in `.env` only (never committed): `GEMINI_API_KEY`, `COHERE_API_KEY`;
  later `LANGSMITH_API_KEY`.
- Free-tier limits hit so far: Gemini embeddings count **each text** as a
  request (100/min, 1,000/day), and the full index took 2 days. Cohere trial:
  10 calls/min, 1,000 calls/month.

---

## Known issues / open items
- AML 14A.14.1 and 14B.16.2 boundaries (Docling merges them). Accepted.
- Exact lookup ignores which document a paragraph is in ("8.2" exists in
  several). **Fixed on Day 2** by document-aware refs from the planner.
- `detect_sector()` keyword list is brittle. **Replaced on Day 1–2** by the
  planner's `sector` plus the user's institution type.
- Everything is synchronous. Fine for a demo; async is in the backlog.
- To compare in evaluation: Qdrant server-side RRF (k=2 default) vs ours
  (k=60); Gemini Embedding 2 vs 001.

---

## Decisions log (why things are the way they are)

**Autonomous build, M4 (2026-10-24)**
- See PROJECT_PLAN §5.7 (Gemini settings, cross-references, Streamlit UI,
  cache keys) and §5.4 check 7.
- The reviewer found that a regeneration could replace a correct first
  answer, and that the first cross-reference parser linked "paragraph 6.2
  of the AML/CFT policy document" to e-KYC 6.2 and missed lists; both fixed
  and tested on the real corpus (53 chunks carry a dependency).

**Autonomous build, M3 (2026-10-24)**
- See PROJECT_PLAN §5.7 "Persistence + API details". The reviewer found
  that a client disconnect (GeneratorExit) or a failed audit write lost the
  turn, and that `alembic check` wanted to drop LangGraph's checkpoint
  tables; both fixed and tested.
- `/health` is a pure `readiness()` function (Qdrant count, SELECT 1,
  settings): it can't spend quota.

**Autonomous build, M2 (2026-10-24)**
- See PROJECT_PLAN §5.7 "Agentic graph details". Key points: the generator
  reads the standalone question; sector promotion only when every search
  names one sector; explicitly requested sector copies are kept; the
  review node trims follow-ups to rounds/calls/time; lookups need 2 search
  slots; a lookup miss searches the question; tool failures degrade.
- Prompts (`PROMPT_VERSION` 2026-10-24.4): an injection rule in every prompt,
  a passages rule for the reviewer and generator only; an example in every
  prompt; the reviewer sees the queries already run and must follow an
  "as required under paragraph X" reference; comparisons are "answered"
  from each side's rules (the first comparison run returned 1 statement).
- Gemini `thinking_level=low` and a 20 s timeout (see the Results table).

**Autonomous build, M1 (2026-10-24)**
- Verifier check 4 = "a `must` statement must not cite only G". Unmarked
  appendix/table passages state obligations (e-KYC App. 2 para 13, the FAR
  rule has `provision: null`). See PROJECT_PLAN §5.7.
- Quote check: each `...` fragment ≥ 3 words, fragments in order (reviewer
  showed "Board ... the ... shall" matched almost anything).
- `finalize` rebuilds the summary from surviving statements; the model's
  summary is unverified and could repeat a removed must-from-guidance claim.
- Per-turn state reset via `turn_input()` + reset-capable reducers; only
  `messages` carries over in a thread (reviewer reproduced the leak with
  `InMemorySaver`).
- Not changed (reviewer finding 8): cached LLM results keep reporting their
  original call count, so the request budget and graph path are the same
  with and without the cache; `cached` and `ChatModelClient.requests` show
  what was actually sent.
- Deferred to M3 (reviewer finding 14): register our Pydantic modules with
  the checkpointer serializer when the Postgres saver is built.
- Generation lives in `app/agent/nodes.py` with the verifier node; the pure
  checks are `verification.check_answer`. `state.py` was needed in M1.

**Assessment refocus (2026-10-20)**
- Project refocused on the agentic RAG assessment (live demo 26 Oct,
  Streamlit). Postgres chat history and Docker packaging are kept; MCP is
  dropped.
- **Two modes, one engine:** the traditional and agentic graphs share
  retrieval, generation and verification, so the comparison isolates the
  decision-making.
- **Bounded agent (LangGraph StateGraph), not a free ReAct loop:** compliance
  needs auditable, predictable behaviour. LLM decisions happen at the routing
  points, and code enforces limits (≤2 review rounds, ≤3 sub-searches,
  recursion limit). A ReAct mode is in the backlog for comparison.
- **LLM for judgment, code for facts:** question understanding (sector,
  documents, paragraph refs, decomposition, scope) moves to the LLM planner.
  Ingestion rules, retrieval math, duplicate collapse and citation checks stay
  in code because they have one correct answer.
- **Citations by evidence ID** (`E3`), verified by code, plus the S/G
  consistency check. The LLM never types a citation.
- **LLM access changes to LangChain chat models** (`langchain-google-genai`)
  instead of the OpenAI-compatible endpoint, for native structured output,
  token metadata and automatic LangSmith tracing. This supersedes the earlier
  generation decision. Embeddings stay on the native `google-genai` SDK.
- **Models (revised 22 Oct after checking AI Studio limits):**
  - `gemini-3.1-flash-lite`: planner, reviewer and evaluation judge.
  - `gemini-3.5-flash-lite`: generator.
  - Each is the other's fallback when its quota runs out. Separate models
    have separate quotas: 2 × 500 calls a day.
  - The Flash models allow only 20 calls a day, so they're rejected for the
    agent (`gemini-3.8-flash` is used only to cross-check the judge).
  - The 2.5 models are closed to new projects.
  - 3.1-flash-lite shuts down in May 2027; changing it is a config edit.
- **Provider-agnostic LLM layer (22 Oct):** each role (planner, reviewer,
  generator, judge) is configured by one setting such as
  `PLANNER_MODEL=google_genai:gemini-3.1-flash-lite` or
  `deepseek:deepseek-flash`, built with LangChain's `init_chat_model`.
  - Per-role settings: the structured-output method and the rate limit.
  - Switching provider = change `.env` + re-run the evaluation.
  - Embeddings are NOT switchable this way: a new embedding model means
    re-embedding and re-indexing all chunks.
- **Memory:** a LangGraph PostgresSaver (agent state) plus our own
  `conversations`/`turns`/`feedback` tables (product record and audit),
  migrated with Alembic.
- **UI:** Streamlit, separate from the FastAPI backend (SSE streaming). The
  old static HTML UI is retired.

**Retrieval (2026-10-11 → 10-20)**
- Qdrant for chunks (dense + BM25 sparse), Postgres for chat. RRF written in
  Python first (k=60), to compare with Qdrant's server-side fusion later.
- Exact paragraph lookup puts named paragraphs first. RRF alone ranked
  14A.9.7 above 14A.9.6.
- Question words were added to BM25 stopwords (the FAR table's BM25 rank went
  from 4 to 2).
- Cohere `rerank-v4.0-pro` scores all 30 candidates. If it fails, the fused
  order is used (`reranked=False`).
- Sector copies (AML 14A/14B/14C/14D) are collapsed at Jaccard ≥ 0.85,
  ignoring tokens with digits; the user's sector copy is kept. Rules only for
  other sectors are demoted, not deleted.
- Each Qdrant point stores `embedding_model`, and a collection refuses vectors
  from another model.

**Ingestion (2026-10-11)**
- Parser: Docling structured items, OCR off. It won the trial against PyMuPDF
  on reading order, S/G markers and labelled furniture/footnotes. Paragraph
  boundaries and hierarchy come from rules on the text, because Docling's
  labels are inconsistent.
- One chunk per numbered paragraph. Tables are their own chunks, stitched
  across pages with their lead-in sentence. Footnotes are attached to the
  referencing chunk. Contextual header in `embed_text`. UUIDv5 stable IDs.
- AML layout tables are unpacked into paragraphs; S/G markers are matched by
  bounding-box line position.
- Appendices are kept. Blank forms are tagged `form_template` in the reviewed
  manifest and filtered at query time ("tag, don't delete").
- Printed page numbers come from page headers (they differ from the PDF
  index: e-KYC +2, FTFC +1, AML +3).
- Embeddings: `gemini-embedding-001`, 768 dimensions, task types
  RETRIEVAL_DOCUMENT/QUERY, via the native SDK, cached by content hash.

---

## Session log
- **2026-10-11:** ingestion built and verified on the real PDFs; Qdrant
  indexer built.
- **Between sessions:** indexing ran over two days (free-tier embedding
  quota).
- **2026-10-20:** index complete (1,219 points); Step 2 hybrid retrieval and
  Step 3 rerank + selection built and verified; plan refocused on the
  agentic RAG assessment.
