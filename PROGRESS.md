# Progress: Bank Policy Compliance Assistant

> Update after every work session. A new Claude Code session reads
> docs/HANDOFF.md first (where things stand, the system end to end, and how
> to guide Syakir with `/tour`).

**Last updated:** 2026-10-26
**Goal:** submit a deployed Streamlit link + the repo, **Mon 26 Oct 2026**
**Current step:** deployment (Syakir: accounts, index upload, push,
docs/DEPLOY.md) and M7, the first full evaluation (Syakir runs it after the
3 pm quota reset)
**Tests:** 335 passing
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
DONE ████████████████████  M5  Golden set + evaluation harness
DONE ████████████████████  M6  Docker, CI, LangSmith
NEXT ░░░░░░░░░░░░░░░░░░░░  M7–M9  Evaluation runs, fixes, docs, freeze
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
- [x] M5 Golden set + evaluation harness (smoke + full) (24 Oct).
  `evals/`: golden_set.yaml (40 items, §8.2 counts), golden.py (check/show),
  golden_review.yaml (who checked what), metrics.py, retrieval_eval.py
  (ablation + regressions), answer_eval.py, judge.py, report.py, run.py
  (`--smoke/--full/--only/--estimate`). Evaluation page lists full runs
  first. Reviewed by code-reviewer: 11 major-ish + 8 minor findings, all
  fixed (gold labels corrected: fact-6, compare-2/4/5/6, fact-3).
- [ ] **Syakir:** spot-check ≥ 10 golden-set items against the PDFs
- [x] M6 Docker (index from committed cache), `.env.example`, LangSmith, CI
  (24 Oct). Dockerfile (one image, non-root), docker-compose.yml (db,
  qdrant with healthcheck, index bootstrap, api, ui), `.dockerignore`,
  committed `chunks.jsonl.gz` + embedding cache `.gz`, `indexer
  --require-cached` + snapshot rewrite, `.env.example`, LangSmith settings,
  `.github/workflows/ci.yml` + `requirements-ci.txt`. Clean-clone check
  passed twice (0 embedding calls, API healthy, UI 200, one real question
  answered). Reviewed by code-reviewer: 4 major + 9 minor findings, all
  fixed.
- [x] Public deployment mode + UI redesign (26 Oct, added for the submission).
  `ui/backend.py` (one interface), `ui/local_backend.py` (the TurnRunner in
  the Streamlit process: alembic upgrade, PDFs into `ui/static/`),
  `connect_qdrant()` (Qdrant Cloud API key), daily question limit in
  `TurnRunner.start` (429 from the API), per-visitor limit and history,
  `LIVE_RESPONSE_CACHE`, structured `TurnView.steps` for a step timeline,
  redesigned pages without emojis (Ask, Compare modes, Evaluation, How it
  works) and a theme in `.streamlit/config.toml`. Page tests with
  Streamlit's AppTest over the scripted stack (`tests/test_ui_local.py`).
  `docs/DEPLOY.md`: Qdrant Cloud, Neon, Streamlit Cloud, secrets, checks.
  Visual check in the browser against local Qdrant/Postgres: the FAR
  question in traditional mode came from the cache (0 API calls).
- [ ] **Syakir:** create Qdrant Cloud, Neon and Streamlit Cloud accounts,
  upload the index, push, deploy (docs/DEPLOY.md)
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
| 24 Oct 14:09 | M5 smoke run #1 (8 retrieval + 4 answer items, judged) | 13 | 7 | 12 |
| 24 Oct 14:49 | M5 smoke run #2 after review fixes (+ followup-1; judge prompt changed) | 13 | 4 | 6 |
| 24 Oct 14:57 | M5 smoke report regenerated (all cached) | 0 | 0 | 0 |
| 24 Oct ~14:40 | M6 clean-clone Docker check: 1 API question (+ 0 embedding calls for indexing) | 2 | 1 | 2 |
| **24 Oct total (window ending 15:00)** | | **69** | **34** | **39** |
| 26 Oct ~02:20 | UI redesign visual check: 1 question, answered from the response cache | 0 | 0 | 0 |
| 26 Oct 16:15 | M7 re-run of the 8 paragraph-number items, traditional (`20261026-1615-only`) after the named-document fix: all answers cached | 0 | 0 | 1 |
| 26 Oct 03:38 (Syakir) | **M7 full evaluation** (`20261026-0338-full`): 40 items × 2 modes, judged; cross-check stopped after 3 (gemini-3.8-flash 504s) | 185 | 89 | 68 |
| 26 Oct (Syakir) | Live Streamlit Cloud check: 8.10, onboarding + takaful follow-up, NRIC, credit-card Compare, 8.10 again (cached); calls estimated from the answers' LLM-call counts | ~11 | ~5 | ~4 |

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
| **M7 full evaluation (20261026-0338-full)** | 40 items × 2 modes, 0 errors. Agentic vs traditional: outcome 100% vs 95% (traditional can't ask to clarify: 0/2), citation recall 91% vs 80%, evidence recall 92% vs 82%, faithfulness 99% vs 98%, relevancy 5.00 vs 4.84, 2.85 vs 1.05 LLM calls, p50 25.1 s vs 4.4 s (p95 69.9 vs 11.0). Agentic wins most on cross-reference (citation recall 100% vs 62%), comparison (83% vs 56%) and follow-up (100% vs 75%); traditional is better on sector items (100% vs 75%: sector-4, the planner filtered a money-changer question to e-KYC and missed AML 14C.16.12). Ablation (28 items): Hit@5 dense 93% / BM25 86% / hybrid 96%; MRR@5 dense 0.86, hybrid 0.85, + exact lookup 0.82 (lookup-3: the document named in the question is ignored), + rerank 0.86, Recall@5 81% → 84%. 4 answers judged by the generator's model during 503s (disclosed). |
| M5 smoke (20261024-1457-smoke) | agentic: outcome 100%, citation recall 83%, relevancy 5.0; traditional: outcome 80% (answered the ambiguous "para 8.2" instead of asking), citation recall 50% (missed 8.4 for xref-1 and 14B.17.1 for the follow-up). Faithfulness 100% both. Ablation (8 items): hybrid MRR@5 0.94 vs dense 0.75 / BM25 0.74; rerank lifts Recall@5 78% → 88% |
| M6 Docker (clean-clone copy, own ports and volumes) | index bootstrap: 0 Gemini calls, 1,219 points from the committed cache; API healthy; one agentic question answered and verified (3 LLM calls, 24.6 s); UI page 200 |
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
- Docker Desktop must be running first (enable "start at login").
  - Local development: `docker compose up -d db qdrant` (databases only),
    then the API and UI from the venv.
  - Whole app in Docker: `docker compose up --build` (needs Compose 2.24+);
    it uses ports 8000 and 8501, so stop a local uvicorn/streamlit first.
  - After re-running the indexer on the host, commit the rewritten
    `app/data/processed/chunks.jsonl.gz` and the embedding cache `.gz`.
- Postgres driver: `psycopg[binary]` v3, with `DATABASE_URL` starting
  `postgresql+psycopg://`.
- Keys in `.env` only (never committed): `GEMINI_API_KEY`, `COHERE_API_KEY`;
  later `LANGSMITH_API_KEY`.
- Free-tier limits hit so far: Gemini embeddings count **each text** as a
  request (100/min, 1,000/day), and the full index took 2 days. Cohere trial:
  10 calls/min, 1,000 calls/month.

---

## Known issues / open items
- **FTFC callout boxes carry the S marker (found by the M7 docs review):** the
  66 "Good practices" / "Poor practice" chunks inherit the provision of the
  paragraph they illustrate (50 S, 16 G), so an example can be presented as an
  obligation and verifier check 4 can't catch it (fact-10, faithfulness 0.43).
  Fix after submission: give callout chunks `provision: null` in the chunker,
  re-embed the 66 changed chunks (their header changes), re-index locally and
  on Qdrant Cloud, commit the new snapshots, re-run the FTFC items.
- **sector-4:** the planner narrows a money-changer question to e-KYC; prompt
  fix after submission.
- **For M7 (found by the M5 smoke run and review):**
  - traditional exact lookup ignores a document named in the question:
    "paragraph 9.1 of the Fair Treatment … policy" puts e-KYC 9.1 and AML
    9.1 above FTFC 9.1 (lookup-3, MRR@5 1.00 → 0.33);
  - agentic fact-6 never retrieved e-KYC Appendix 2:11 (the notification
    contents);
  - Cohere rerank demotes the gold paragraph in compare-1 and sector-1
    (MRR@5 1.00 → 0.50);
  - agentic latency p50 ~34 s on the smoke set (504 retries and the
    generator's medium thinking).
- Temporary Docker volumes `bpaclean_pgdata` and `bpaclean_qdrant_data`
  from the M6 clean-clone check can be deleted (`docker volume rm …`).
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

**DeepSeek as the main LLM (2026-10-26)**
- Syakir paid for DeepSeek after the Gemini free tier's repeated 503s.
  Planner, reviewer and generator use `deepseek-flash` (thinking off); Gemini
  stays as their fallback, the judge and the embeddings. See PROJECT_PLAN §5.7
  "DeepSeek". The M7 numbers are Gemini's; the evaluation is re-run on
  DeepSeek before the docs quote new ones.

**M7 fix: the document named in the question (2026-10-26)**
- lookup-3: "paragraph 9.1 of the Fair Treatment … policy" put e-KYC 9.1 and
  AML 9.1 above FTFC 9.1. The traditional mode now orders the exact matches
  so the named document's come first (PROJECT_PLAN §5.7 "Sector"). The first
  version filtered by the name; code-reviewer showed a name used as a topic
  would then lose the match, so it only orders. sector-4 (the planner
  filtering a money-changer question to e-KYC) is left as a known limit:
  a prompt change on submission day was judged too risky.

**Model cooldown (2026-10-26)**
- From Syakir's live logs: gemini-3.1-flash-lite answered 503 "high demand"
  on nearly every call, so each planner/reviewer step lost ~5–6 s retrying
  before the fallback (an agentic answer took 35–50 s). A failed model is
  now tried last for 120 s, and live answers reuse a cached answer from
  either model; the evaluation keeps the old, faithful behaviour (see
  PROJECT_PLAN §5.7). Reviewed by code-reviewer: 1 major (eval attribution)
  + 4 minor findings, fixed; the call-count gap after failed retries is
  older than this change and recorded as known.

**Deployment and UI redesign (2026-10-26)**
- The submission needs a public link. Syakir chose option A: Streamlit
  Community Cloud running the UI and the agent in one process
  (`UI_BACKEND=local`), Qdrant Cloud for the index, Neon for Postgres. The
  FastAPI + Docker stack stays the reference architecture. Rejected: a
  separate free API host (sleeps, ~1 min cold start) and a self-managed VM
  (too much setup for one day). See PROJECT_PLAN §5.7 "Public deployment".
- Quota protection for a public link: shared daily limit from the audit
  log, per-visitor limit, response cache for live answers.
- UI redesign without emojis (Syakir: "it looks AI-built"), see §5.7.
- Reviewed by code-reviewer: 5 major + 8 minor findings, all fixed:
  the checkpointer pool now checks connections (Neon drops idle ones);
  the Compare page keeps each side's error across its rerun; a failed
  start closes what it opened and shows a message; step times are per
  step, not cumulative; the Docker image copies `.streamlit/config.toml`;
  the quota day starts at midnight Pacific (`QUOTA_TIMEZONE`, daylight
  saving included; `tzdata` added); a client leaving right after "start"
  is now audited (with an empty state, not the previous turn's). Accepted
  and documented: the daily limit is soft (questions in flight aren't
  counted yet).

**Autonomous build, M6 (2026-10-24)**
- See PROJECT_PLAN §5.7 "Packaging". The reviewer found that pytest could
  upload fake runs to LangSmith when tracing is on in `.env`, that the
  index job raced Qdrant's start, that every `docker compose up` would roll
  Qdrant back to the committed snapshot, and that `ask.py` traced
  unmasked questions; all fixed.
- `.env.example` was regenerated from the settings without reading the old
  file (reading `.env*` is blocked); Syakir should check nothing he added is
  missing.

**Autonomous build, M5 (2026-10-24)**
- See PROJECT_PLAN §5.7 "Evaluation details". Corrected gold labels after
  the review re-read them: fact-6 is App. 2 paras 10 + 11 (para 13 is what
  the Bank may do); added e-KYC 8.27 to fact-3/compare-5, 8.3 to compare-2,
  AML 24.1 to compare-4, 14C.16.1/10/11 to compare-6. Sector items use the
  sector they ask about (followup-1 keeps banking: the follow-up switches).
- Full-run estimate ~160 + ~100 + 15 calls (not ~575).

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
