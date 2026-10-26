# Architecture

How a question becomes a cited answer: the components, the data flow from PDF to
answer, the two graphs, the state and schemas, and the design decisions with
their reasons. Names follow [design_qa.md](design_qa.md): planner, reviewer,
generator, verifier, tool, evidence (E1…), round.

## 1. Components

```
 Browser ── Streamlit UI (ui/) ──┬── HTTP + SSE ──► FastAPI (app/api/)      local / Docker
                                 └── in-process ──►                         Streamlit Cloud
                                                   │
                                     TurnRunner (app/api/runner.py)
                       daily limit → mask PII → run the graph → write the audit rows
                                                   │
                     ┌─────────────────────────────┴─────────────────────────────┐
                     ▼                                                           ▼
   traditional graph (app/agent/traditional.py)            agentic graph (app/agent/graph.py)
                     │      both share: search engine, generator, verifier, LLM adapter
                     └─────────────────────────────┬─────────────────────────────┘
          ┌────────────────────────┬───────────────────┼───────────────────┐
          ▼                        ▼                   ▼                   ▼
   SearchService             ChatModelClient      PostgresSaver       audit tables
   (app/services/)           (app/llm/models.py)  (agent memory)      (app/db/)
     ├─► Qdrant (1,219 chunks, through vector_repository.py)
     ├─► Gemini embeddings (the question)
     └─► Cohere rerank
```

| Component | Responsibility | Talks to |
|---|---|---|
| `app/ingestion/` (offline) | PDFs → structured chunks → vectors | Docling, Gemini embeddings, Qdrant |
| `SearchService` | question → the 5 best passages | Qdrant, Gemini (query embedding), Cohere |
| `ChatModelClient` | every LLM call: rate limit, cache, retries, fallback, structured output | DeepSeek and Gemini via LangChain |
| graphs (`app/agent/`) | the decisions: which steps run, in what order | the three above |
| `TurnRunner` | one question end to end: the daily limit, PII masking (NRIC, card and account numbers), streaming, audit | graphs, Postgres |
| FastAPI / Streamlit | serving and the UI | `TurnRunner` |

Every external service has exactly one adapter with a timeout, retries and a
fallback: `vector_repository.py` (Qdrant), `embedding_service.py`,
`rerank_service.py` (Cohere), `llm/models.py` (DeepSeek and Gemini chat), `db/`
(Postgres). The models per role: DeepSeek `deepseek-flash` for the planner,
reviewer and generator (thinking off), each with a Gemini Flash-Lite fallback;
the judge is Gemini, a different family from the generator.
Settings come only from `app/core/config.py`.

## 2. Offline: from PDF to searchable chunks

```
PDF ─► parser.py (Docling) ─► Blocks ─► structure.py (rules) ─► chunker.py ─► chunks.jsonl ─► indexer.py ─► Qdrant
```

1. **Parse** (`parser.py`, the only module that imports Docling): text blocks in
   reading order with their page and vertical position. Page headers and footers
   are removed, but the printed page number is kept for citations. OCR is off
   (digital PDFs); the parse is cached by the PDF's SHA-256.
2. **Structure** (`structure.py`): rules on the text decide what each block is,
   because Docling's labels are inconsistent (S 8.1 came back as a list item,
   G 8.10 as text). A reader with a memory keeps the open paragraph, its S/G
   marker, the section path and the scope (main body or "Appendix 2"):
   - a line starting with a paragraph number (`G 8.10 …`, `S 14A.15.2 …`) opens a paragraph;
   - a heading closes it; anything else continues it (sub-items, page breaks);
   - AML prints S/G in a separate column that Docling emits out of order, so each
     marker is matched to the paragraph **at the same height on the page**
     (182 of 182 matched);
   - "tables" whose rows start with paragraph numbers are column layouts and are
     read back as paragraphs;
   - appendices restart their numbering, so every paragraph carries its scope.
3. **Chunk** (`chunker.py`): one chunk per numbered paragraph (a paragraph is
   what people ask about and cite), one per table, one per run of unnumbered
   text. Tables are stitched across pages with their lead-in sentence (the FAR
   table); footnotes are attached to the paragraph that references them; parts
   over 350 words are split at sentence boundaries; a context header
   (`[e-KYC Policy Document | … | para 8.10 (Guidance)]`) is prepended to the
   text that is embedded and BM25-indexed. Stable IDs: UUIDv5 of
   `doc|scope|type|paragraph|part`; a content hash tells the indexer what to
   re-embed. AML Part 14's sector copies are tagged with their sector
   (14A banking, 14B insurance/takaful, 14C money services, 14D e-money).
4. **Index** (`indexer.py`): each chunk becomes one Qdrant point with a dense
   vector (Gemini, 768 dimensions, `RETRIEVAL_DOCUMENT`), a BM25 sparse vector
   (our tokenizer keeps paragraph numbers such as `14a.9.6` whole; Qdrant
   applies IDF) and the whole chunk as payload, with keyword indexes for
   filters. Embeddings are cached by content hash, the upsert is idempotent and
   stale points are deleted, so the index is a sync, not a rebuild. Docker and
   Qdrant Cloud are loaded from the committed cache with 0 Gemini calls.

Quality checks (`python -m app.ingestion.pipeline` writes them to
`app/data/debug/report.txt`): e-KYC 57/57 and FTFC 120/120 main-body paragraphs
correct; in AML two paragraph boundaries are wrong (14A.14.1 and 14B.16.2 merged
into a neighbour, text kept); a per-page word-coverage check against PyMuPDF
flags only blank-form pages. Known labelling issue: FTFC "Good practices" and
"Poor practice" boxes are stored as their own chunks but keep the S marker of the
paragraph they illustrate (50 chunks); they should carry no S/G marker.

## 3. The search engine

`SearchService.search(question, sector, documents, paragraph_refs)` →
`SearchResult.final` (5 passages):

1. **Exact-lookup stage.** Paragraph numbers in the question are fetched by
   their `paragraph_id`; if the question names one document, its matches come
   first. Dense search alone doesn't have 14A.9.6 in its top 5 for "What does
   paragraph 14A.9.6 … require?" (it ranks 14A.9.16 first): a number has almost
   no meaning to an embedding.
2. **Hybrid retrieval.** The question is embedded (`RETRIEVAL_QUERY`) and
   BM25-encoded; Qdrant runs both searches in one round trip (top 30 each).
3. **RRF fusion** (k = 60): ranks, not scores, because cosine (~0.7) and BM25
   (~2–30) are on different scales.
4. **Rerank.** Cohere scores all 30 candidates by reading question and passage
   together. If Cohere fails, the fused order is kept (`reranked=False`).
5. **Selection** (pure code): one copy of each rule repeated across sectors (word
   overlap ≥ 0.85, ignoring tokens with digits), the user's sector preferred;
   other sectors' rules moved down; named paragraphs never dropped.

The traditional mode calls `search()` once with the question. The agentic mode's
tools call it per planned search (often limited to one document); the
`lookup_paragraphs` tool calls `lookup()` with explicit `(document, paragraph)`
pairs from the planner or reviewer.

## 4. The two graphs

Both are LangGraph `StateGraph`s compiled with a Postgres checkpointer.

```
traditional:   retrieve ─► generate_answer ─► verify_answer ─► finish
                                ▲                  │ serious or incomplete, once
                                └──── feedback ────┘

agentic:       understand_question (planner) ─► route_after_planning
                 ├─ out of scope / small talk ─► respond_directly ───────────┐
                 ├─ ambiguous ─► ask_clarification ─────────────────────────┤
                 └─ search_policies × N + lookup_paragraphs (parallel, Send) │
                        └─► review_evidence (reviewer) ─► route_after_review │
                               ▲                        │ not enough (≤ 2 rounds)
                               └──── more searches ─────┘                    │
                                                        │ enough             │
                        generate_answer ◄───────────────┘                    │
                              ▼                                              │
                        verify_answer ─ serious or incomplete, once ─► generate_answer
                              ▼                                              │
                            finish ◄──────────────────────────────────────────┘
```

- **Nodes** are plain functions: state in, the changed fields out. Tool nodes
  started with `Send` get only their own task (one search, or one set of
  paragraph references), not the whole state; a reducer merges what they return.
- **Conditional edges** (`app/agent/routing.py`) are pure code reading the LLM's
  structured decision (`plan.route`, `review.is_sufficient`) and returning the
  next node, or a list of `Send`s to run tools in parallel. After a review the
  graph answers when the reviewer is satisfied **or** when the limits leave no
  room for more searches.
- **Bounds** (code, not prompts): at most 3 searches per plan, 2 follow-up
  searches and 3 cross-references per review, 2 review rounds, and a
  per-question budget of 5 LLM calls and 6 searches, with no new round or
  regeneration after 60 s (`app/agent/budget.py`). The time limit stops new
  work; it doesn't cut off a call already running, so a slow answer can pass
  60 s (agentic p95: 69.9 s). LangGraph's recursion limit (25) is the last
  safety net.
- **Degradation:** planner down → a plain search; reviewer down → answer from
  all evidence; one failing tool → the others still answer; reranker down →
  fused order; embedding down → BM25 only.

## 5. State and schemas

`AgentState` (`app/agent/state.py`) is one dictionary per question: `messages`
(the conversation, the only field kept across turns), `question`, `sector`,
`plan`, `evidence` (merged by a reducer that de-duplicates and labels E1, E2…),
`review`, `rounds`, `answer`, `verifications`, `feedback`, `outcome`, counters
(`llm_calls`, tokens) and `search_log`. `turn_input()` resets every per-question
field, so counters never leak into the next question.

The LLM fills Pydantic schemas (`app/agent/schemas.py`), and code then sanitises
them:

| Schema | Filled by | Holds |
|---|---|---|
| `QueryPlan` | planner | a route, the standalone question, up to 3 searches (query, documents, sector), paragraph references `{document, paragraph_id}`, a clarifying question, a one-line reason |
| `EvidenceReview` | reviewer | relevant evidence IDs, is_sufficient, what is missing, follow-up searches, cross-references to fetch |
| `GroundedAnswer` | generator | status, summary, statements (text, obligation must / should_or_may / information, evidence IDs, an exact quote, topic), gaps |

Routes: `search`, `lookup`, `search_and_lookup`, `clarify`, `out_of_scope`,
`small_talk`. The real plan for "What does e-KYC paragraph 8.10 require?":

```json
{"route": "lookup", "standalone_question": "What does paragraph 8.10 of the e-KYC policy document require?",
 "searches": [], "paragraph_refs": [{"document": "ekyc", "paragraph_id": "8.10"}],
 "reasoning": "The user names a paragraph and its document."}
```

The **verifier** (`app/agent/verification.py`) is code. Every cited ID must be
evidence the generator was given; the quote must appear in a cited passage
(fragments of at least 3 words, in order); a "must" may not cite only guidance;
the status must match the content; a cited passage that depends on another
given passage ("as required under paragraph 8.4") must not be answered without
it. A serious problem gets one regeneration with the problems listed; whatever
still fails is removed. An answer that leaves out a cross-referenced passage
gets the same one regeneration; if it is still incomplete it is kept, because
nothing in it is wrong. The MUST check trusts the S/G marker from ingestion
(see the FTFC good-practice issue in §2).

## 6. Memory, audit and serving

- **Memory:** the checkpointer saves the state after every step per conversation
  thread (`<conversation>:<mode>`), so "what about takaful operators?" reads
  the earlier question. The planner rewrites the follow-up into a standalone
  question and switches the sector.
- **Audit:** our own tables (`conversations`, `turns`, `turn_events`,
  `feedback`) record the question, plan, evidence, answer, every verification
  report, models, prompt and index versions, tokens and latency, plus one row
  per step. `turns` and `turn_events` are append-only: database triggers reject
  UPDATE, DELETE and TRUNCATE. A turn is audited even when it fails or the
  client leaves mid-stream.
- **PII:** MyKad (NRIC), card and account numbers are masked to `[NRIC]`,
  `[CARD]` and `[ACCOUNT]` before anything reaches the model, the logs, a trace
  or the database.
- **Serving:** `TurnRunner.stream()` yields `start`, one `step` per node, then
  `answer` (or `error`); FastAPI sends them as Server-Sent Events and the UI draws a live
  step timeline. On Streamlit Community Cloud the UI calls the same runner in
  its own process (`UI_BACKEND=local`), with Qdrant Cloud and Neon, a daily
  question limit, a per-browser-session limit (`QUESTIONS_PER_SESSION`) and the
  response cache for repeated questions ([DEPLOY.md](DEPLOY.md)).

## 7. Design decisions and why

| Decision | Why |
|---|---|
| LLM for judgment, code for facts | Questions need language understanding; paragraph boundaries, retrieval maths, duplicate detection and citation checks have one right answer that code computes exactly, for free, and can be tested. |
| A bounded agent, not a free ReAct loop | A bank needs predictable, auditable behaviour: the LLM decides at fixed routing points, code enforces every limit. |
| Citations by evidence ID | The model can't invent a citation; code maps `E3` to the paragraph and page. |
| One engine, two modes | The difference between the modes then comes from the agent's decisions, including the extra passages those decisions fetch. Cost and latency: [the investigation, §4](agentic_rag_investigation.md#4-trade-offs). |
| LangGraph `StateGraph` | Typed shared state, conditional edges for the LLM's decisions, `Send` for parallel tool calls, a checkpointer for memory and per-node streaming for the UI; a hand-written loop would need all of these rebuilt and tested. |
| Tools chosen through a structured plan, not provider function calling | Code validates the plan (paragraph IDs, limits, route consistency) before any tool runs, and the same schema works with any model. |
| Paragraph chunks with a context header | People ask about and cite paragraphs; the header tells the embedding and BM25 which document and section a short rule belongs to. |
| Hybrid + exact lookup + rerank | Meaning search misses codes like 14A.9.6; BM25 misses paraphrases; the reranker reads question and passage together. |
| Sector copies collapsed by code | The AML copies are word-for-word the same; without collapsing, one rule fills all 5 slots. |
| A paid main model with a fallback from another provider, and a cooldown | One provider's outage or overload (the Gemini free tier's 503s) doesn't stop answers; a model that is overloaded is skipped for 120 s instead of retried on every step. |
| Structured output into Pydantic, then sanitised | Code can branch on fields safely and clip anything out of bounds. |
| Postgres checkpointer + separate audit tables | Working memory and the product's permanent record have different owners and rules. |
