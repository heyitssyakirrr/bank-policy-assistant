# Progress Tracker — Bank Policy Compliance Assistant

Update this after every work session. When starting a new chat, paste
this file + PROJECT_PLAN.md + your GitHub repo link (or the specific
files you're currently working on) so Claude can pick up exactly where
you left off, including writing code that fits what already exists.

**Last updated:** _(fill in date)_
**Current phase:** _(fill in)_
**Repo:** _(link)_

---

## Environment setup notes
_(package versions that mattered, Gemini/Cohere model IDs actually in
use, pgvector image version, anything you had to work around)_
-

## Repo structure so far
_(paste `tree backend -L 3` after each session, so a new session sees
the real current layout, not just the target one)_
```

```

## Current DB schema (actual, not planned)
_(paste `\d chunks`, `\d chat_sessions`, `\d messages` output, or your
current migration file, if it has diverged from the plan)_
```sql

```

## Current API contract (actual, not planned)
_(paste the real current request/response JSON shape of /chat)_
```json

```

---

## Phase 0 — Foundations: Infra, Config, Logging
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
-

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
