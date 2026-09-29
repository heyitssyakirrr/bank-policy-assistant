# Progress Tracker — Bank Policy Compliance Assistant

Update this after every work session. When starting a new chat, paste
this file + PROJECT_PLAN.md + your GitHub repo link (or the specific
files you're currently working on) so Claude can pick up exactly where
you left off, including writing code that fits what already exists.

**Last updated:** 2026-09-27
**Current phase:** Phase 1 (starting)
**Repo:** _(fill in your GitHub link once pushed)_

---

## Environment setup notes
- Windows, Python 3.12, venv at `.venv`
- Driver switch: use `psycopg[binary]` (v3), not `psycopg2-binary` — hit a
  `ModuleNotFoundError: No module named 'psycopg'` because SQLAlchemy's
  default Postgres dialect wants v3. `DATABASE_URL` uses the
  `postgresql+psycopg://` prefix.
- Docker Desktop required enabling virtualization in BIOS + WSL2 features
  before it would start (Windows-specific hurdle, resolved)
- Repo has no `/backend` wrapper folder — flat layout with `app/` at the
  root, since the frontend is static files served by FastAPI, not a
  separate app (see PROJECT_PLAN.md's "Note on layout")

## Repo structure so far
```
bank-policy-assistant/
  app/
    main.py
    core/
      config.py
      db.py
      logging.py
    api/routes/
      chat.py         # POST /chat — echo stub only, no RAG yet
      health.py        # GET /health — checks DB connectivity
    schemas/
      chat.py          # ChatRequest / ChatResponse Pydantic models
    static/
      index.html
      style.css
      chat.js
  docker-compose.yml    # postgres (pgvector/pgvector:pg16) — running
  requirements.txt
  .env / .env.example
  .gitignore

Not yet created (planned, not started):
  app/core/retry.py           # deferred — no external API calls yet to retry
  app/api/routes/sessions.py
  app/db/ (models.py, vector_repository.py, chat_repository.py)
  app/services/
  app/data/raw_pdfs/           # PDFs not yet placed here
  scripts/, evals/, tests/, docs/
  Dockerfile
```

## Current DB schema (actual, not planned)
Postgres + pgvector container is running and reachable (`/health` confirms
`"database": "connected"`), but the actual schema (chunks / chat_sessions /
messages tables from PROJECT_PLAN.md) has NOT been applied yet — no
migration/init script has been run. This is first up in Phase 1.

## Current API contract (actual, not planned)
`POST /chat`
Request: `{"session_id": "string | null", "message": "string"}`
Response: `{"session_id": "string", "answer": "string", "sources": [], "grounded": true}`
— currently just echoes the input back as `answer` (`"Echo: <message>"`),
no retrieval or LLM call involved yet.

`GET /health` → `{"status": "ok", "database": "connected"}`

---

## Phase 0 — Foundations: Infra, Config, Logging
Status: [x] Done
Notes:
- Postgres + pgvector running via Docker Compose, `/health` confirms connectivity
- Structured logging working: each request gets a short request ID that
  appears on every log line for that request, plus a total duration log
  (confirmed working: `req=d2f32588` tied "Received message" and
  "completed" log lines together)
- Config centralized in `core/config.py` via pydantic-settings — nothing
  reads env vars directly elsewhere
- `/chat` (echo stub) and `/health` working end-to-end through the real
  static frontend, not just curl/Postman
- Not yet done, carried into Phase 1: DB schema not applied, `retry.py`
  not created (nothing to retry yet), `app/data/raw_pdfs/` not created

## Phase 1 — RAG Core: Ingestion, Hybrid Search, Rerank
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 2 — Evaluation Framework (RAGAS)
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 3 — Using History Well + History UI
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 4 — Framework Refactor: LangGraph Agent
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 5 — MCP Tool Integration
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 6 — Observability: LangSmith Tracing
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 7 — Guardrails & Hardening
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

## Phase 8 — Polish & Portfolio Packaging
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

---

## Open questions / blockers
-

## Decisions made along the way
_(e.g. "chose chunk_size=500/overlap=50 because...", "chat history kept
in same Postgres DB as vectors, not split out")_
-
