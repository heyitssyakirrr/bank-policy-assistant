# Progress Tracker — Bank Policy Compliance Assistant

Update this after every work session. When starting a new chat, paste
this file + PROJECT_PLAN.md + your GitHub repo link (or the specific
files you're currently working on) so Claude can pick up exactly where
you left off, including writing code that fits what already exists.

**Last updated:** 2026-10-11
**Current phase:** Phase 1 (in progress — structure-aware ingestion pipeline built; next: run it with Docling, then Qdrant + embeddings)
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
    core/            config.py, db.py, logging.py, retry.py (empty)
    api/routes/      chat.py (echo stub), health.py, sessions.py (empty)
    schemas/         chat.py
    services/        embedding_service.py (renamed from embeddings.py)
                     retrieval/rerank/generation/chat_service.py (empty placeholders)
    db/              models.py, chat_repository.py, vector_repository.py (empty placeholders)
    ingestion/       OFFLINE pipeline, never imported by the API
      models.py      Block, AnnotatedBlock, Chunk (+ enums): the contract between stages
      parser.py      Docling -> Blocks (only file that imports Docling; JSON cache by PDF sha256)
      structure.py   regex rules: paragraph ids, S/G, headings, section path, appendix scope
      chunker.py     paragraph chunks, table stitching + lead-in, footnotes, size splits, stable ids
      manifest.py    loads data/ingest_manifest.yaml
      quality.py     per-page word coverage vs PyMuPDF (flags dropped/duplicated text)
      debug.py       human-readable dumps per stage + report
      pipeline.py    CLI: python -m app.ingestion.pipeline [--doc ekyc] [--reparse]
    data/
      raw_pdfs/                 the 3 BNM PDFs
      ingest_manifest.yaml      reviewed per-document decisions (skip/tag pages)
      interim/docling/          cached Docling parses (gitignored)
      processed/chunks.jsonl    pipeline output = input of the embedding step (gitignored)
      debug/                    *.1_blocks / *.2_structure / *.3_chunks .txt + report.txt (gitignored)
    static/          index.html, style.css, chat.js
  scripts/           test_embeddings.py
  tests/             test_ingestion.py (27 tests, no models/PDFs needed)
  requirements.txt (app) / requirements-ingest.txt (+docling, pymupdf, pyyaml) / requirements-dev.txt (+pytest, ruff)
  pyproject.toml     pytest + ruff config
  docker-compose.yml, Dockerfile (empty), docs/, evals/ (empty placeholders)
```

## Document structure findings (from reading the PDFs, 2026-10-11)
- Paragraphs are numbered (e-KYC `8.1`; AML `14C.1`, `28.2.1`) and each
  carries a marker: **S** = standard/obligation (must comply;
  non-compliance may mean enforcement action), **G** = guidance.
  Answers must distinguish "must" (S) from "should/may" (G).
- Hierarchy: Part > numbered Section (grey bar heading) > sub-section
  (`28.1`) > paragraph (`28.1.1`) > items `(a)` / `(i)`.
- Printed page numbers differ from the PDF page index (e-KYC +2,
  FTFC +1, AML +3). The header carries the printed number ("Page 6 of
  28", "15 of 178") and our header stripping currently discards it.
  Citations should use the printed page and keep the PDF index too.
- Boxed callouts ("Good practices", example lists) are illustrative,
  not binding.
- Real tables: AML definitions glossary, AML risk-factor example tables
  (rows split mid-cell across pages), AML Appendix 5 (CTR info), e-KYC
  Table 1 FAR thresholds (header row NOT repeated on the continuation
  page), FTFC Appendix 3 checklist. The `find_tables()` counts (e-KYC
  4, FTFC 56, AML 67) overstate this; boxes and S/G layouts are
  probably being counted.
- Blank forms/templates (AML Appendix 2, 8a/8b, last pages): rotated
  headers and empty grids produce junk text. Exclude or tag as
  `form_template`.
- Footers vary ("Issued on" vs "Amendment date"); footnotes sit at the
  page bottom; one bilingual English/Malay customer notice (AML p.61).
- Design consequences: chunk by numbered paragraph (sub-items stay with
  their parent); metadata = document, printed_page, pdf_page, section
  path, paragraph number, S/G, block_type (paragraph / table / callout /
  form); tables become their own chunks and are stitched across pages;
  cite as "para 8.2 (S), p.6 of 28".
- Parser trial done (see decisions log): Docling structured items won.
- Paragraph id shapes across all 3 PDFs: `8.1`, `10.1.1`, `14A.9.6`.
  Numbering RESTARTS in appendices (AML Appendix 1 has its own 1.1), so
  chunk identity = document + scope (main / Appendix N) + paragraph id.
- Some numbered lines are titles, not provisions (`29.1 Definition and
  Interpretation`, `4.0 Relationship-based Risk Assessment`): short, no
  S/G, no sentence ending -> heading.
- Wrapped cross-references ("...under paragraphs / 8.10 and 8.13 of...")
  look like paragraph starts; real paragraphs start with a capital or "(".
- Blank forms confirmed: AML Appendix 2 (pp.148-151), 8a/8b (169-170),
  9/9a/9b (171-181). Appendix 5 (CTR table) is real content and stays.

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
Status: [ ] Not started / [x] In progress / [ ] Done
Notes:
- Real PDF filenames in `app/data/raw_pdfs/`: `pd-ekyc-apr2024.pdf`,
  `pd-ftfc-mar24.pdf`, `PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf`
- (SUPERSEDED 2026-10-11, see below) `app/services/ingestion_service.py` built: `extract_pages` (PyMuPDF,
  per-page text), `detect_header_length` + `clean_pages` (strips a
  repeated running header/footer per document — detected generically by
  growing a matched leading-word prefix across pages, digits normalized,
  not hardcoded per document), `chunk_pages` (fixed-size word chunking,
  stays within a page so citations keep an accurate page number)
- Header stripping verified correct by eye against each PDF (page 1
  cover pages correctly left untouched since they don't match the
  header pattern; inner pages correctly cleaned) — see decisions log
- (old) Default chunking: 350 words/chunk, 50-word overlap (~465-470 tokens,
  within the typical 200-500 token production range)
- (old word-count chunker) Final clean chunk counts: e-KYC 47, Fair Treatment 81, AML/CFT/CPF/TFS
  193 — total 321 chunks across the corpus
- `scripts/ingest_documents.py` (Step 1, extraction+chunking only, no
  API calls) confirmed working end-to-end with clean sample output
- Embeddings approach decided: `app/services/embeddings.py` using the
  **native `google-genai` SDK**. Reason (corrected 2026-10-10): Gemini's
  OpenAI-compatible endpoint DOES support a synchronous
  `embeddings.create`, but its docs only list `input` and `model` —
  no documented `dimensions` or task type. We need both:
  `output_dimensionality=768` (pgvector indexes cap at 2000 dims, and
  the default 3072 is too big) and `task_type` (RETRIEVAL_DOCUMENT for
  chunks vs RETRIEVAL_QUERY for user questions). Model
  `gemini-embedding-001`. Batches of 50 texts/call, paced to stay
  under the free-tier 30k TPM limit (321 chunks → ~7 batches, well
  under 100 RPM / 1,000 RPD)
- `scripts/test_embedding.py` sanity check PASSED: 2 embeddings,
  768 dimensions confirmed
- Still to do: apply the DB schema (nothing is stored in Postgres
  yet), embed all 321 chunks into pgvector, build hybrid search
  (vector + full-text) + RRF fusion + Cohere rerank, wire into
  `/chat`, persist chat history
- 2026-10-11: ingestion rebuilt as `app/ingestion/` (replaces
  `ingestion_service.py` + `scripts/ingest_documents.py`; old word-count
  chunker and header-stripping heuristic deleted). One chunk per
  numbered paragraph (sub-items, trailing text and page-break
  continuations included), tables as own chunks (stitched across pages,
  prefixed with their lead-in sentence), footnotes attached to the chunk
  that references them, oversized units split at block boundaries
  (max 350 words), contextual header prepended to `embed_text`,
  deterministic UUIDv5 chunk ids (idempotent upserts later), printed
  page numbers read from page headers for citations.
- Layout differences handled (checked on FTFC p.11-12, AML p.9/40/132):
  FTFC "Good practices"/"Poor practices" boxes -> separate chunks tagged
  `callout`, tied to the paragraph above, never leaking into the next
  paragraph's section path; AML's S/G sits in its own far-left column ->
  a lone "S"/"G" block is carried to the next paragraph number; AML's
  definitions glossary rows cut by a page break (empty first cell) are
  glued back onto their term; "APPENDICES" divider resets the path.
- 2026-10-11 first REAL Docling run (e-KYC): 341 blocks -> 112 chunks,
  all section headings correct (Parts A-C, sections 1-12, Appendices
  1-6), para 8.10 and the stitched FAR table exactly right. Fixed from
  that run: footnote marker after a paragraph number ("8.22 11 ."),
  footnotes spanning several blocks (fn 11 + its (i)/(ii)), footnote on a
  heading ("legal person 5" -> first chunk under it), Docling reading
  Appendix 2 para 11 (a numbered list) as a table -> converted back to a
  paragraph, coverage report ignores manifest-skipped pages. Replayed on
  the real blocks: 108 chunks, 0 unmatched footnotes, 0 duplicates, all
  57 main-body paragraphs present. Known, accepted: Docling scrambles
  the order of App. 3 para 7(g)/(h) on p.24 (text kept, order off).
- Verified: 27 unit tests built from real Docling dumps + PDF pages pass; full run
  on all 3 PDFs with a stand-in parser had no crashes, unique ids, no
  main-body duplicate paragraph ids. NOT yet run with real Docling on
  the full PDFs (the cloud workspace could not download Docling's
  models) -> first thing next session: run the pipeline locally and
  read the report + debug files.
- Next: `python -m app.ingestion.pipeline --doc ekyc`, review debug
  files, then all docs; then Qdrant in docker-compose, embed
  chunks.jsonl, upsert dense + BM25 sparse vectors.

## Phase 2 — Evaluation Framework (RAGAS)
Status: [ ] Not started / [ ] In progress / [ ] Done
Notes:
- Planned experiment (not started): compare the current fixed-size
  word chunking (350/50 overlap) against clause-number-aware chunking
  — splitting on the BNM documents' own numbered clauses (e.g. "1.4",
  "4.2") via a regex boundary like `\n\d+\.\d+\s` — since these policy
  PDFs are already segmented into self-contained numbered rules.
  Run both through the RAGAS question set and compare context
  precision/recall to see whether clause-aware splitting measurably
  improves retrieval over blind word-count splitting.
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
- Header/footer stripping must be conditional per-page (check each
  page's own leading words against the detected pattern before
  stripping), not applied unconditionally — an early version blindly
  stripped the first N words from every page, which would have eaten
  real content on pages (like cover pages) that don't actually start
  with the header. Verified by inspecting page 1 of each document
  after the fix.
- Embeddings use the native `google-genai` SDK, not the OpenAI-
  compatible endpoint used for generation. Correct reason: the
  compatible endpoint's embeddings call documents only `input` and
  `model`, so we can't rely on it for `output_dimensionality` and
  `task_type`, which retrieval quality and the pgvector index limit
  depend on. (An earlier version of this note wrongly blamed the
  async Batch API; synchronous embeddings do work on the compatible
  endpoint.) Generation calls still use the OpenAI-compatible endpoint
  as originally decided.
- Parser: **Docling, read via its structured items (not the markdown
  export), OCR off** (2026-10-11). Trial on e-KYC pp.8-10 and 18-25
  vs PyMuPDF. Docling put section headers in the correct reading
  order, kept S/G markers on the paragraph line, and labelled page
  headers as furniture and footnotes as `footnote`. The structured
  items kept the original list markers (`1.`..`10.`, `i.`, `(i)`) and
  had a page number on every item; the renumbering and `&amp;` seen in
  the markdown were export artifacts. Known gaps: tables split across
  pages (we stitch), item labels are inconsistent (S/G paragraphs are
  sometimes `list_item`, sometimes `text`; "A. General requirements"
  is read as a list item), and section headers come back flat with no
  levels. So paragraph boundaries and the section hierarchy come from
  regex on the text, not from Docling's labels. Printed page number is
  read from the furniture header ("Page 16 of 28") before it is dropped.
- Vector store: **Qdrant for chunks (dense + BM25 sparse), Postgres for
  chat sessions/messages** (2026-10-11). RRF written in Python first,
  then compared with Qdrant's server-side fusion in Phase 2.
- Appendices are kept; blank forms/templates are tagged
  `form_template` via a reviewed ingest manifest and filtered at query
  time, not deleted.
-
