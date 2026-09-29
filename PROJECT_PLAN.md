# Bank Policy Compliance Assistant — Project Plan (v2)

## Context for whoever (or whichever Claude session) picks this up
Portfolio project for an AI/ML engineer job search, built to prove real
hands-on skill — not a tutorial follow-along. The user is a CS graduate
with backend/data engineering experience (FastAPI, Docker, SQL, OCR
pipelines) but no prior hands-on experience with RAG, agent frameworks,
MCP, or evals.

**"Raw" in this plan means:** the user writes the retrieval, fusion, and
reranking logic himself instead of hiding it behind a LangChain
retriever object — NOT "avoid real infrastructure." Real infra
(Postgres + pgvector, Docker, structured logging) is used from day one,
because that's how actual AI engineers build these systems. Frameworks
(LangGraph, LlamaIndex) are introduced later (Phase 4+) once the
mechanics are understood, specifically so the user can explain what the
framework automates.

**Product:** a chatbot web app where a bank employee asks plain-English
questions about Malaysian banking regulation and gets a grounded, cited
answer, instead of manually searching BNM policy PDFs.

**Corpus (in `app/data/raw_pdfs/`):**
- `ekyc_policy.pdf` — BNM e-KYC Policy (2024, 30p)
- `fair_treatment_policy.pdf` — BNM Fair Treatment of Financial Consumers (2024, 54p)
- `aml_cft_fi_policy.pdf` — BNM AML/CFT/CPF/TFS Policy for Financial Institutions (181p)

**Stack:** FastAPI backend, plain HTML/CSS/JS frontend (served as static
files by FastAPI — no Next.js, no build step), Postgres + pgvector for
both vector storage and chat history, Gemini API (free tier) for
embeddings + generation, Cohere free tier for reranking. No live
deployment — GitHub repo + Docker Compose + demo video is the
deliverable.

**Note on layout:** the repo is flat at the root (no `/backend` wrapper
folder) — since the frontend is just static files served by FastAPI
rather than a separate app, there's no need for a top-level backend/
frontend split. `app/` sits directly at the repo root.

---

## Repo structure (target — build incrementally per phase)
```
app/
  main.py                    # FastAPI app factory, mounts routers, static files
  core/
    config.py                 # pydantic-settings: env vars, model IDs, DB URL
    logging.py                 # logger setup, request-id middleware, timing helper
    db.py                       # Postgres/SQLAlchemy session management
    retry.py                     # tenacity-based retry wrapper for external API calls
  api/
    routes/
      chat.py                   # POST /chat
      sessions.py                # GET /sessions, GET /sessions/{id}
      health.py                   # GET /health
  schemas/
    chat.py                     # Pydantic request/response models (incl. structured
                                  # Gemini output schema: answer, sources, grounded)
  services/
    ingestion_service.py         # PDF -> chunks -> embeddings -> pgvector (Phase 1)
    retrieval_service.py          # vector search + full-text search + RRF fusion
    rerank_service.py              # Cohere rerank wrapper
    generation_service.py           # Gemini call w/ structured output
    chat_service.py                  # orchestrates retrieve->rerank->generate,
                                       # persists messages, builds prompt w/ history
  db/
    models.py                    # SQLAlchemy models: Chunk, ChatSession, Message
    vector_repository.py          # pgvector similarity + full-text queries
    chat_repository.py             # session/message CRUD
  static/                       # plain HTML/CSS/JS frontend
    index.html
    style.css
    chat.js
  data/
    raw_pdfs/                    # the 3 source PDFs go here
scripts/
  ingest_documents.py            # one-off script to populate pgvector from PDFs
evals/
  questions.json                 # hand-written eval question set (Phase 2)
  results/                       # RAGAS run outputs, one file per pipeline config
tests/                          # optional but good practice, add as you go
docs/
  architecture.md                  # simple diagram + explanation (Phase 8)
  demo.gif
requirements.txt
docker-compose.yml               # postgres (with pgvector image) + app
Dockerfile
.env.example                     # GEMINI_API_KEY=, COHERE_API_KEY=, DATABASE_URL=, ...
README.md
PROJECT_PLAN.md
PROGRESS.md
```

## Database schema (Postgres, single DB for vectors + chat)
```sql
-- vector storage
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  document TEXT NOT NULL,          -- e.g. 'ekyc_policy.pdf'
  section TEXT,                     -- e.g. '4.2 Identity Verification'
  page INT,
  content TEXT NOT NULL,
  embedding VECTOR(768),            -- dimension depends on the Gemini embedding model used
  content_tsv TSVECTOR              -- for full-text search
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (content_tsv);

-- chat history
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY,
  title TEXT,                       -- derived from first message
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
  id SERIAL PRIMARY KEY,
  session_id UUID REFERENCES chat_sessions(id),
  role TEXT NOT NULL,               -- 'user' | 'assistant'
  content TEXT NOT NULL,
  sources JSONB,                     -- [{document, section}], null for user messages
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## API contract (keep stable — extend, don't break)
`POST /chat`

Request:
```json
{ "session_id": "uuid (client-generated, omit to start a new session)", "message": "string" }
```

Response (structured — this is what Gemini's JSON output mode returns,
validated against a Pydantic schema, not parsed from free text):
```json
{
  "session_id": "uuid",
  "answer": "string",
  "sources": [{"document": "ekyc_policy.pdf", "section": "4.2 Identity Verification"}],
  "grounded": true,
  "trace_id": "string (added Phase 6, null until then)"
}
```

`GET /sessions` → list of `{id, title, created_at}` for the history sidebar
`GET /sessions/{id}` → full message list for that session

## Default technical parameters (starting points — justify any change in PROGRESS.md)
- Chunk size: ~500 tokens, 50-token overlap
- Retrieval: top-8 candidates from each of vector search and full-text
  search → RRF fusion → reranked to top-3 passed to the LLM
- Embedding model: check Gemini's current embedding model ID via the API/docs
  rather than hardcoding — note the actual one used in PROGRESS.md
- Generation model: Gemini's current flash-tier model, same rule — verify,
  don't assume
- Structured output: Gemini JSON mode constrained to the response schema above
- **Database driver: `psycopg` (v3), not `psycopg2-binary`** — SQLAlchemy's
  default Postgres dialect expects `psycopg` v3, and it has more reliable
  prebuilt wheels on Windows anyway. `DATABASE_URL` must use the
  `postgresql+psycopg://` prefix, not plain `postgresql://`.
- **LLM calls use the `openai` Python SDK pointed at Gemini's OpenAI-compatible
  endpoint** (`base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`,
  `api_key=<GEMINI_API_KEY>`), not the native `google-generativeai` SDK. This
  keeps every request in the standard `messages`/`role`/`content` shape —
  the same shape LangGraph/LangChain convert to internally in Phase 4 — so
  the raw code and the framework code stay conceptually consistent, and the
  code would need minimal changes to swap providers later.

## Costs & free tiers (verified — nothing here should cost anything for this project)
- Gemini API: free tier, rate-limited, no card required
- Cohere Rerank: free trial key, no card, 1,000 calls/month cap, 10 req/min
  on rerank — trial keys are labeled non-commercial/evaluation use only,
  which is fine for a portfolio repo but worth knowing
- Postgres + pgvector: free, open source, runs in your own Docker container
- LangSmith: free Developer plan, 5,000 traces/month, 14-day retention, 1 seat
- Docker, LangGraph, LangChain, LlamaIndex, RAGAS, MCP: all free/open source

---

## Phase 0 — Foundations: Infra, Config, Logging
**Goal:** every cross-cutting concern exists before any business logic does.

**Tasks:**
- `docker-compose.yml`: a Postgres service using a pgvector-enabled image
  (e.g. `pgvector/pgvector:pg16`), plus the backend service
- `core/config.py`: pydantic-settings loading `DATABASE_URL`,
  `GEMINI_API_KEY`, `COHERE_API_KEY`, model IDs, from `.env`
- `core/logging.py`: structured logger, a FastAPI middleware that
  generates a request ID per request and injects it into every log line
  for that request, plus a simple `@log_duration` decorator/context
  manager for timing pipeline steps
- `core/retry.py`: a `tenacity`-based retry decorator (exponential
  backoff) for external API calls, used later on Gemini/Cohere calls
- `app/main.py`: FastAPI app, mounts `static/` for the frontend, a
  `GET /health` route that checks DB connectivity
- Apply the DB schema above via a migration or a simple init script
- Minimal `static/index.html` + `chat.js`: a chat box that calls a stub
  `/chat` endpoint (echo response) — confirms the whole stack is wired

**Definition of done:** `docker compose up` starts Postgres + backend;
opening `index.html` and sending a message round-trips through FastAPI
with a request ID visible in the logs; `/health` returns OK.

---

## Phase 1 — RAG Core: Ingestion, Hybrid Search, Rerank
**Goal:** real, hybrid, reranked, cited answers — built the way it's
actually done, from the start. (This replaces the old separate Phase 1
"raw" and Phase 2 "hybrid+rerank" — no reason to build a throwaway
in-memory version first.)

**Tasks:**
- `scripts/ingest_documents.py`: extract text per PDF (PyMuPDF/pdfplumber)
  with page/section metadata, chunk it (write the chunking logic
  yourself), embed each chunk via Gemini, insert into the `chunks` table
  (embedding + tsvector both populated) — run this once to populate the DB
- `retrieval_service.py`: two hand-written SQL queries — cosine similarity
  via pgvector, and full-text rank via `ts_rank` — then Reciprocal Rank
  Fusion combining the two ranked lists, written by hand (no library)
- `rerank_service.py`: send the fused top-N to Cohere's rerank API
  (wrapped in the retry decorator from Phase 0), keep top-3
- `generation_service.py`: build the prompt from the top-3 chunks +
  recent chat history, call Gemini with structured JSON output matching
  the response schema
- `chat_service.py`: orchestrates the above, persists the user message
  and assistant response (with sources) to `messages`, creates a
  `chat_sessions` row on first message of a session
- `/chat` route wired to `chat_service`
- Log timing for each step (retrieval, rerank, generation) per request
- Frontend: real chat UI — message bubbles, loading state while waiting,
  citations shown under each answer, using the actual `/chat` response

**Definition of done:** a real e-KYC/AML/Fair Treatment question in the
UI returns a correct, cited answer; logs show per-step timing; the
conversation is saved to Postgres and visible via `GET /sessions/{id}`.

---

## Phase 2 — Evaluation Framework (RAGAS)
**Goal:** measured quality, not vibes.

**Tasks:**
- `evals/questions.json`: 12–15 hand-written Q&A pairs across all 3
  documents, including 2–3 cross-document ones (e.g. KYC red flag → AML
  process), each with expected answer and expected source
- Install `ragas`; `evals/run_eval.py` runs the question set through the
  live pipeline, scores faithfulness, answer relevancy, context
  precision/recall
- Run at least two configs (e.g. with vs. without reranking, or two
  chunk sizes) and save both to `evals/results/`
- Record the comparison and what changed in PROGRESS.md's decision log

**Definition of done:** `evals/results/` has at least two dated RAGAS
runs with a written explanation of what changed between them.

---

## Phase 3 — Using History Well + History UI
**Goal:** history isn't just stored (Phase 1 already does that) — it's
used correctly in the pipeline, and browsable in the UI.

**Tasks:**
- `chat_service.py`: include recent turns (last ~6) from `messages` in
  the prompt so follow-ups resolve correctly (e.g. "what about for
  corporate accounts?" after an e-KYC question)
- Frontend: a sidebar listing past sessions (`GET /sessions`), click to
  load a session's full history (`GET /sessions/{id}`) into the chat window
- Test explicitly with a vague follow-up that only makes sense with
  memory; confirm it resolves correctly; test reopening an old session

**Definition of done:** a multi-turn conversation resolves follow-ups
correctly; closing and reopening the browser still shows chat history
via the sidebar.

---

## Phase 4 — Framework Refactor: LangGraph Agent
**Goal:** rebuild as a proper agent graph now that the mechanics are understood.

**Tasks:**
- `agent_graph.py`: LangGraph graph with nodes `retrieve → rerank →
  generate → grounding_check`
- Conditional edge on `grounding_check`: if the answer isn't supported
  by retrieved context, route to a fixed "not covered in these policies"
  response — this becomes the real implementation of `grounded: false`
- Replace manual history-in-prompt logic with LangGraph's state/
  checkpointing where it genuinely simplifies things
- `chat_service.py` now invokes the graph; document in PROGRESS.md
  exactly what LangGraph replaced vs. what you kept hand-written

**Definition of done:** the app runs on a LangGraph graph; grounding/
refusal is a graph branch, not a prompt instruction.

---

## Phase 5 — MCP Tool Integration
**Goal:** real tool-calling via MCP, matching what's already on the resume.

**Tasks:**
- `mcp_server.py`: minimal MCP server exposing `search_policy_document(query)`,
  wrapping `retrieval_service`
- Connect the LangGraph agent to it as an MCP client, replacing the
  direct function call in the `retrieve` node with a real MCP tool call
- Confirm end-to-end: a request triggers a real MCP tool call, result
  flows back into the graph

**Definition of done:** the retrieval step goes through MCP, confirmed
visible in the LangSmith trace (Phase 6).

---

## Phase 6 — Observability: LangSmith Tracing
**Goal:** real visibility into what the agent is doing per request.

**Tasks:**
- Wire LangSmith into the LangGraph pipeline
- Confirm a single trace shows: retrieved chunks, rerank scores, the MCP
  tool call, and the final generation
- Re-run the Phase 2 eval set through the traced pipeline
- Populate `trace_id` in the `/chat` response per the API contract

**Definition of done:** a LangSmith trace screenshot/link in the README
showing one full request end-to-end.

---

## Phase 7 — Guardrails & Hardening
**Goal:** behavior a real bank would actually accept.

**Tasks:**
- Basic PII detection on user input (flag patterns resembling IC/account
  numbers), redact before logging
- Adversarial test set: out-of-scope requests ("give me investment
  advice", "how do I get around AML checks") — confirm graceful refusal
- Confirm the `retry.py` wrapper + logging from Phase 0 handle a Gemini/
  Cohere API failure gracefully (clean error to frontend, no crash,
  logged with full context)

**Definition of done:** a "safety testing" section in the README listing
adversarial cases tried and how the app handled each.

---

## Phase 8 — Polish & Portfolio Packaging
**Goal:** something you'd be proud to screen-share.

**Tasks:**
- UI polish: streaming responses if feasible, clean loading/refusal
  states, responsive layout
- Confirm `docker-compose.yml` + `Dockerfile` still bring the whole app
  up with one command (this has been true since Phase 0 — just verify
  nothing broke)
- `README.md`: architecture diagram, the raw-logic-on-real-infra story,
  eval results summary, MCP + LangSmith usage, GIF of it working
- Record a 2–3 minute demo video walking through the app and 1–2
  architectural decisions

**Definition of done:** a repo a stranger could clone, run via
`docker compose up`, and understand in 5 minutes of reading the README.

---

## How to resume work in a new chat
Share, in order: this `PROJECT_PLAN.md`, your current `PROGRESS.md`, and
your GitHub repo link (or the specific files you're working on). That's
enough for a fresh session to understand the architecture, current
state, and write code that fits what already exists.
