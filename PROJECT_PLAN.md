# Project Plan: Bank Policy Compliance Assistant (Agentic RAG)

> **Read this first.** This file explains **what** we are building, **why**
> each design decision was made, and **in what order** we build it.
> [PROGRESS.md](PROGRESS.md) records what is done. **Revised 24 Oct** for the
> autonomous Claude Code build: see §5.6, §8 and §10.
>
> **Deadline:** live demo on **Mon 26 Oct 2026**, using Streamlit.
> **Plan written:** Tue 20 Oct 2026.

---

## 1. The product in one paragraph

Bank compliance staff ask questions in plain English about three Bank Negara
Malaysia policy documents: **e-KYC** (30 pages), **Fair Treatment of Financial
Consumers** (FTFC, 54 pages) and **AML/CFT/CPF & TFS** (181 pages). The
assistant answers using only those documents. It does three things beyond
answering:

- It **cites every statement** down to the paragraph and page, for example
  *e-KYC, para 8.4 (S), p. 6*.
- It says clearly whether something is an obligation ("**must**", an S
  paragraph) or guidance ("**may / should**", a G paragraph).
- When the documents don't cover a question, it **refuses** instead of
  guessing.

It remembers the conversation, so a follow-up such as *"what about takaful
operators?"* works. It shows **its reasoning steps** live. It also runs in two
modes, **traditional** and **agentic**, so the difference between them can be
demonstrated and measured.

### Where each assessment requirement is answered

| Assessment requirement | Where it is answered | Section |
|---|---|---|
| Agentic RAG that retrieves chunks correctly | LangGraph agent on top of our hybrid retrieval, measured with retrieval metrics | §4, §5, §8 |
| Working prototype demo | Streamlit app with chat, live agent steps, citations, and a compare page | §7, Day 4 |
| Thought process and implementation flow | Architecture and design-decision docs, plus the "why" tables in this file | §3–§6, `docs/` |
| Investigation of agentic RAG as a whole | `docs/agentic_rag_investigation.md`: patterns, trade-offs, when not to use it | §9 |
| Traditional vs agentic RAG | Both modes in the app, compared side by side and measured on the same test set | §2, §8 |
| Test cases that assure quality | Unit, graph-path and API tests, plus a labelled evaluation set and metrics | §8 |
| *Bonus:* citations | Paragraph-level citations, code checks that each citation is real, S/G consistency check | §5.4 |
| *Bonus:* optimised retrieval | Hybrid (dense + BM25), RRF, rerank, exact lookup, sector handling, with an ablation table proving each step helps | §8.3 |

Our own additional requirements: **Postgres chat history** and **Docker
packaging**.

---

## 2. Traditional RAG vs agentic RAG (the core idea)

**Traditional RAG** is a fixed pipeline. The code decides everything, in the
same order, every time:

```
question ──► retrieve top 5 ──► LLM writes answer ──► done
```

**Agentic RAG** lets the LLM **make decisions** inside the pipeline: what to
search for, whether the results are good enough, whether to search again, and
when to stop.

```
question ──► LLM plans ──► search / look up (in parallel) ──► LLM reviews evidence
                                    ▲                               │
                                    └──── not enough? search again ─┘ (max 2 rounds)
                                                                    │ enough
                                                                    ▼
                                       LLM writes cited answer ──► code verifies citations
```

### Why our documents need it: real cases from the PDFs

| Situation (real) | Traditional RAG | Agentic RAG |
|---|---|---|
| **Cross-references.** e-KYC 8.10 says *"as required under paragraph 8.4"*. AML 14A.15.2 says it *"shall be read together with the e-KYC policy document"*. | Retrieves 8.10 only. The answer is missing the rule it depends on. | The reviewer sees the reference and fetches 8.4, or the e-KYC rules, before answering. |
| **Multi-part or comparison questions.** *"How do the e-KYC verification rules differ from AML's non-face-to-face rules?"* | One search mixes both topics, so the top 5 favours one document. | The planner splits it into two searches, each filtered to one document, and they run in parallel. |
| **Follow-ups.** *"what about takaful operators?"* | Searches for those four words alone and gets junk. | Rewrites it using the chat history into a full question, and switches sector to insurance/takaful. |
| **Ambiguous paragraph numbers.** "para 8.2" exists in **several** documents. | The exact lookup returns all of them, or the wrong one. | The planner reads the context ("the e-KYC one") and looks up the right document. If it can't tell, it asks. |
| **Out of scope.** *"What's the interest rate cap on credit cards?"* | Retrieves the 5 "closest" chunks anyway, and the LLM may stretch them into an answer. | The planner routes it to a polite refusal before any search runs. The reviewer is a second safety net. |

### What it costs (we will measure this, not just claim it)

| | Traditional | Agentic |
|---|---|---|
| LLM calls per question | 1 | usually 3, at most 5 |
| Latency | ~3–5 s | ~6–12 s |
| Predictability | Same path every time | Path varies, so it must be **bounded** and **traced** |
| Best for | Simple, single-topic questions | Multi-part, cross-referenced, follow-up and borderline questions |

> **Honest conclusion we expect to show:** agentic RAG is not always better.
> On simple questions both give the same answer, and agentic is slower and
> costs more. It wins on hard questions. Saying this in the demo shows you
> understand the trade-off. It is not a weakness of the project.

---

## 3. Design principle: the LLM for judgment, code for facts

You said you don't want "regex-level" decisions, and you're right about
**understanding questions**. Matching keywords against a list (our current
`detect_sector`) breaks as soon as someone phrases a question differently.
That job moves to the LLM.

A senior engineer doesn't put the LLM everywhere, though. **An LLM call is
slow, costs money, and can be wrong.** If a task has one correct answer that
code can compute, code does it, and the code can be tested exactly. This is
what makes an AI system trustworthy for a bank. Learn this table well,
because interviewers ask this question.

| Decision | Who decides | Why |
|---|---|---|
| What is the user really asking? Which documents, which sector? Does it need splitting? Is it in scope? | **LLM** (planner) | Needs language understanding: paraphrases, follow-ups, implied context |
| Are the retrieved passages enough? What is missing? Which cross-references should be followed? | **LLM** (reviewer) | Needs reading comprehension |
| Writing the answer and choosing citations | **LLM** (generator) | Needs synthesis |
| Splitting PDFs into numbered paragraphs (ingestion) | **Code** (rules) | The numbering is a fixed format. The rules are verified against every paragraph (57/57, 120/120). An LLM would be slower, cost money on all 265 pages, and give results we can't verify |
| Finding the best chunks (dense + BM25 + RRF) | **Code** + embedding model | Math over vectors. It is fast and testable |
| Ranking chunks by relevance | **Cross-encoder model** (Cohere rerank) | A specialised model: faster and more accurate than asking a chat LLM to rank |
| Merging repeated sector copies (14A/14B/14C/14D) | **Code** | Copies are word-for-word the same, which code can check exactly for free |
| Is a cited paragraph actually one we retrieved? Does "must" match an S paragraph? | **Code** (verifier) | There is a correct answer. Never ask an LLM to check what code can check |
| Loop limits (max 2 search rounds) | **Code** | Safety. An agent must never be able to loop forever |

**What changes in the current code:**
- `detect_sector()` keyword matching is **removed from the question path**.
  The planner returns `sector`, and the user's institution type, chosen in the
  UI, is the default.
- Paragraph references become **document-aware**: the planner returns
  `{document: "ekyc", paragraph_id: "8.2"}`. That fixes the known issue that
  exact lookup ignored the document.
- The paragraph-number regex stays **only as validation**: it checks that what
  the LLM extracted is a real paragraph-ID shape, and the index confirms the
  paragraph exists. This guards against the LLM hallucinating an ID.

---

## 4. Architecture

```
┌──────────────────────┐   HTTP + SSE stream   ┌───────────────────────────────────────────────┐
│  Streamlit UI (ui/)  │ ────────────────────► │  FastAPI backend (app/api)                    │
│  chat · agent steps  │ ◄──────────────────── │  POST /chat  GET /conversations  /documents   │
│  citations · compare │   step events, answer └───────────────────┬───────────────────────────┘
└──────────────────────┘                                           │
                                                                   ▼
                                   ┌───────────────────────────────────────────────┐
                                   │  LangGraph agent (app/agent)                  │
                                   │  understand → route → search/lookup → review  │
                                   │  → generate → verify                          │
                                   └─────┬───────────────┬─────────────────┬───────┘
                         tools (code)    │               │ LLM calls       │ state per conversation
                                         ▼               ▼                 ▼
                    ┌─────────────────────────┐  ┌──────────────┐  ┌──────────────────────────┐
                    │ SearchService (built)   │  │ Gemini       │  │ PostgreSQL               │
                    │ embed → Qdrant hybrid   │  │ 3.1 lite:    │  │ • LangGraph checkpoints  │
                    │ → RRF → Cohere rerank   │  │  plan/review │  │   (agent memory)         │
                    │ → sector selection      │  │ 3.5 lite:    │  │ • conversations, turns,  │
                    │ get_paragraphs (built)  │  │  answer      │  │   feedback (product/audit)│
                    └───────────┬─────────────┘  └──────────────┘  └──────────────────────────┘
                                ▼
                    ┌─────────────────────────┐        LangSmith: a trace of every node and LLM call
                    │ Qdrant: 1,219 chunks    │        Docker Compose: qdrant, postgres, api, ui
                    │ dense 768 + BM25 sparse │
                    └─────────────────────────┘
```

**Why the UI is separate from the backend.** In a real bank, the Streamlit
demo would be replaced by the bank's own portal or a Teams bot. The agent
lives behind an API, so any front end can use it. This also keeps API keys on
the server.

### Tech stack: what we use and why

| Layer | Choice | Why this one | Alternatives considered |
|---|---|---|---|
| Agent framework | **LangGraph** | Explicit state machine (auditable, which matters for a bank), built-in checkpointing (memory), streaming, parallel fan-out (`Send`), LangSmith tracing. The industry standard for controllable agents | Plain Python loop (would work, but we'd rebuild memory, streaming and tracing ourselves). CrewAI (multi-agent role-play, too loose for compliance). LlamaIndex workflows (fine, but less widely used) |
| LLM access | **LangChain chat model** (`langchain-google-genai`) | Native structured output (Pydantic), token-usage metadata, automatic LangSmith traces, and the provider can be swapped in config | Raw `google-genai` SDK (we'd hand-wire tracing). The OpenAI-compatible endpoint (the old plan, now superseded) |
| Models | **`gemini-3.1-flash-lite`** for plan/review, **`gemini-3.5-flash-lite`** for answers, each the other's fallback | The free tier gives each model its own quota: 15 RPM and 500 RPD for the two Flash-Lite models, but only 20 RPD for every Flash model. Two Lite models means two quota buckets, about 1,000 calls/day. 3.5 is the newer, better writer. Model names are config only | `gemini-3.8-flash` (20/day: about 5 agentic questions a day, too few). 2.5 models (closed to new projects since Sep 2026) |
| Retrieval | **Built already:** Qdrant hybrid + RRF + Cohere rerank | Measured on real queries (see PROGRESS) | — |
| Memory | **LangGraph `PostgresSaver`** + our own tables | The checkpointer is the agent's working memory. Our tables are the product's record (history list, audit, feedback) | In-memory saver (lost on restart) |
| API | **FastAPI** + Server-Sent Events | Streams agent steps live to the UI | WebSockets (two-way, which we don't need) |
| UI | **Streamlit** | Required by the assessment, and quick to build a good chat UI | Gradio |
| DB migrations | **Alembic** | Schema changes are versioned like code, as real teams do | `create_all()` (no history) |
| Tracing | **LangSmith** (free developer plan) | See every node, prompt, token count and latency. Great to show live | Langfuse (also good, self-hostable) |
| Evaluation | Own retrieval metrics + deterministic answer checks + our own LLM judge (claim-level faithfulness) | Retrieval metrics need no LLM, so they're cheap and exact. Our answers are already claims, so one judge call per answer is enough | RAGAS (several calls per answer on a 500/day quota; kept as an optional cross-check), DeepEval |
| Packaging | **Docker Compose** | One command starts all four services | — |

---

## 5. The agent in detail

### 5.1 LangGraph in 60 seconds (read before Day 2)

- **State** is one Python dict (a `TypedDict`) that flows through the graph:
  the question, the plan, the evidence, the answer.
- A **node** is a plain function: `def review_evidence(state) -> dict`. It
  reads the state and returns **only the fields it changes**.
- An **edge** is "after node A, go to node B". A **conditional edge** is a
  function that looks at the state and returns the name of the next node.
  That is where the agent's decisions become control flow.
- A **reducer** says how a field is updated when a node returns a value.
  `messages` uses `add_messages`, which appends instead of overwriting.
  Evidence merges by chunk ID, so parallel searches don't overwrite each
  other.
- **`Send`** starts the same node several times **in parallel** with
  different inputs (map-reduce). One search per sub-question.
- **Compile, then `invoke` or `stream`.** `stream` emits an event after each
  node. That is how the UI shows live steps.
- A **checkpointer** saves the state after every step under a `thread_id`
  (our conversation ID). The next message in the same thread starts with the
  history already loaded. That is memory.
- **`recursion_limit`** is a hard stop on total steps. It is our second safety
  net after the round counter.

### 5.2 The graph

```
                ┌────────────────────┐
  user msg ───► │ understand_question│  LLM (3.1-flash-lite) → QueryPlan
                └─────────┬──────────┘
                          │ route_after_planning()   (conditional edge, code)
     ┌──────────────┬─────┴─────────┬───────────────────────┐
     ▼              ▼               ▼                       ▼
 respond_directly  ask_clarification  lookup_paragraphs   search_policies × N   (parallel via Send)
 (out of scope /   (vague question)   (doc-aware exact)   (hybrid + rerank + selection)
  greeting)            │                   └──────┬──────────┘
     │                 │                          ▼
     │                 │               ┌────────────────────┐
     │                 │               │  review_evidence   │  LLM (3.1-flash-lite) → EvidenceReview
     │                 │               └─────────┬──────────┘
     │                 │                         │ route_after_review()
     │                 │       not enough AND round < 2 ──► follow-up searches / cross-ref lookups ──┐
     │                 │                         │ enough (or out of rounds)                       │
     │                 │                         ▼                                       (back to review)
     │                 │               ┌────────────────────┐
     │                 │               │  generate_answer   │  LLM (3.5-flash-lite) → GroundedAnswer
     │                 │               └─────────┬──────────┘
     │                 │                         ▼
     │                 │               ┌────────────────────┐
     │                 │               │  verify_answer     │  CODE: citations real? S/G consistent?
     │                 │               └─────────┬──────────┘  fails once → regenerate with feedback
     ▼                 ▼                         ▼
                              END  (answer + citations + trace saved to Postgres)
```

**The traditional mode is a graph too:** `retrieve → generate_answer →
verify_answer`. It shares the same answer and verify nodes, so the
comparison is fair. The **only** difference is the decision-making.

### 5.3 Each node

| Node | Kind | Input → output | Agentic pattern it implements |
|---|---|---|---|
| `understand_question` | LLM | chat history + question + document catalog → `QueryPlan` | Query rewriting, **routing**, query decomposition |
| `route_after_planning` | code | plan → which nodes run next | Routing (Adaptive RAG) |
| `search_policies` | tool (code) | one `SearchRequest` (query, documents, sector) → evidence | Tool use, **parallelisation** |
| `lookup_paragraphs` | tool (code) | `ParagraphRef`s → evidence (or "not found" → falls back to search) | Tool use |
| `review_evidence` | LLM | question + numbered evidence → `EvidenceReview` | **Corrective RAG** (grade, then re-retrieve) |
| `route_after_review` | code | review + round count → loop or generate | Evaluator–optimiser loop, bounded |
| `generate_answer` | LLM | question + relevant evidence → `GroundedAnswer` | Grounded generation |
| `verify_answer` | code | answer + evidence → verified answer, or one retry | **Self-check** (Self-RAG idea, done with code) |
| `respond_directly` / `ask_clarification` | code | plan → refusal, greeting or clarifying question | Refusal / clarification |

### 5.4 The contracts (Pydantic schemas the LLM must fill)

**Structured output** means the LLM must return JSON matching a schema, and
we receive a validated Python object, not free text we have to parse. These
schemas are the most important design in the agent.

```python
class SearchRequest(BaseModel):
    query: str  # a self-contained search query
    documents: list[DocCode] = []  # [] = all documents; e.g. ["ekyc"]
    sector: Sector | None = None  # None = the user's own sector


class ParagraphRef(BaseModel):
    document: DocCode | None  # None only when the user didn't say
    paragraph_id: str  # "8.2", "14A.9.6"


class QueryPlan(BaseModel):
    route: Literal["search", "lookup", "search_and_lookup", "clarify", "out_of_scope", "small_talk"]
    standalone_question: str  # follow-up rewritten using history
    searches: list[SearchRequest]  # 1–3: decomposition happens here
    paragraph_refs: list[ParagraphRef]
    clarifying_question: str | None
    reasoning: str  # one sentence, shown in the UI steps


class EvidenceReview(BaseModel):
    relevant_evidence: list[str]  # ["E1", "E4"]: noise is dropped before answering
    is_sufficient: bool
    missing_information: str | None
    follow_up_searches: list[SearchRequest]
    cross_references: list[ParagraphRef]  # "as required under paragraph 8.4" → fetch 8.4


class Statement(BaseModel):
    text: str
    obligation: Literal["must", "should_or_may", "information"]
    evidence_ids: list[str]  # every statement must cite ≥ 1
    quote: str  # exact supporting words copied from the cited evidence
    topic: str  # which part of a multi-part question this statement answers


class GroundedAnswer(BaseModel):
    status: Literal["answered", "partially_answered", "not_covered"]
    summary: str  # the direct answer in 1–2 sentences
    statements: list[Statement]
    gaps: str | None  # what the documents don't say
```

**How citations work (the bonus point):**
1. Evidence is shown to the LLM as numbered blocks: `[E3] e-KYC, para 8.4
   (S), p. 6: <text>`. The LLM cites `E3`, a short ID. It never types a
   paragraph number or page, so it **cannot invent a citation format**.
2. `verify_answer` (code) checks that every `evidence_id` exists. Statements
   with no valid citation are removed. If nothing valid remains, the status
   becomes `not_covered`, so the user gets a refusal rather than an
   unsupported answer.
3. **Obligation check:** a statement marked `must` has to cite at least one
   **S** paragraph. If it cites only G paragraphs, that is an error, and the
   answer is regenerated once with feedback. This catches the worst
   compliance mistake: presenting guidance as an obligation.
4. **Quote check:** each statement carries the exact words it relies on.
   Code checks that the quote really appears in the cited evidence (after
   normalising spaces and case). A citation pointing at the wrong paragraph
   fails this check.
5. The UI maps `E3` back to the human citation, *"e-KYC Policy Document,
   para 8.4 (S), p. 6"*, with a link that opens the PDF at that page.

**What `verify_answer` does, in order:**

| # | Check | If it fails |
|---|---|---|
| 1 | Every cited ID is evidence we actually gave the model | Drop that ID (fixable) |
| 2 | Every statement still has ≥ 1 valid ID | Drop the statement (fixable) |
| 3 | The quote appears in one of its cited passages | Serious |
| 4 | A `must` statement does not cite only G paragraphs (unmarked appendix/table passages may state obligations) | Serious |
| 5 | Status agrees with content (`answered` needs statements; `not_covered` has none) | Fixable: correct the status |
| 6 | Something valid is left | No: becomes `not_covered` (refusal) |
| 7 | No cross-reference left out: a cited passage that depends on another given passage ("as required under paragraph 8.4") is answered with it (added 24 Oct) | Incomplete: one regeneration; if still incomplete, kept as is (nothing in it is wrong) |

- Fixable problems are corrected silently and recorded in the audit log.
- Serious problems mean one regeneration, with feedback listing the problems.
  If they persist, the bad statements are removed and the status is
  downgraded to `partially_answered`. The summary is then rebuilt by code
  from the surviving statements, because the model's summary is not
  verified and may repeat a removed claim.
- If the generator itself fails (both models down), the user gets a
  "temporarily unavailable" notice, or, on a regeneration, the verified part
  of the first answer.
- The full verification report is saved with the turn.

### 5.5 Safety and bounds

- **At most 2 review rounds** (counter in state) and a `recursion_limit` on
  the graph. A cost ceiling is set before the question starts.
- **At most 3 sub-searches** per plan (enforced in code, not just requested
  in the prompt).
- Evidence text goes into the prompt as **quoted data**. The system prompt
  says instructions inside the documents or the user's message never override
  the rules. Input length is capped.
- **Retry policy** for live requests: Gemini and Cohere calls get **3
  attempts**, with exponential backoff plus jitter (about 1 s, then 2 s,
  capped at 8 s). Retries happen only on 429, 5xx and timeouts, never on 400
  (a bad request won't fix itself). Each call has a timeout of about 30 s.
  Malformed structured output gets **1 repair retry**. Offline indexing keeps
  6 attempts, because it can afford to wait.
- If an LLM call fails after retries, the system **degrades** instead of
  crashing: a planner failure falls back to a plain search on the original
  question, a reviewer failure answers from all evidence, a reranker failure
  (or no Cohere key) falls back to fused order, and a query-embedding
  failure falls back to BM25 + exact lookup (`dense_used=False`).
- Answers carry a short notice: *"Based on BNM policy documents. Not legal
  advice."*

---

### 5.6 Production hardening (added 24 Oct)
- **One rate limiter per model,** shared by every role and every parallel
  search (15 RPM for each Lite model, 10/min for Cohere). Parallel nodes must
  never burst past the limit.
- **Response cache** (SQLite, keyed by model + prompt + input) for
  development and evaluation. Re-runs are free, and the cache can never
  answer live users: it is off in the API.
- **PII masking (a data-protection guardrail):** MyKad/NRIC numbers
  (`YYMMDD-PB-###G`), card numbers and account-number-like digit runs are
  masked **as soon as the question arrives**, before it reaches Gemini,
  LangSmith, logs or the audit tables (data minimisation). Policy questions
  never need a customer's real number, so masking costs nothing in answer
  quality. Regex is the right tool here, because these formats are fixed.
  The UI tells the user when something was masked.
- **Reproducible index:** commit `chunks.jsonl` and the embedding cache
  (gzipped), so `docker compose up` rebuilds Qdrant with **zero** API calls on
  any machine.
- **Request budget:** each question has a ceiling (5 LLM calls, 6 searches,
  60 s). Hitting it returns the best verified answer so far, with a note.
- **Readiness:** `/health` reports Qdrant (point count), Postgres, the index
  version, and which models are configured, without spending quota.
- **Structured logs:** every line carries `request_id`, `conversation_id` and
  `turn_id`, so one question can be traced across the API, the agent and the
  database.
- **Flash-Lite reliability:** flat schemas with `Literal` enums, a short
  example in each prompt, a 1-repair retry, and a code sanitiser after every
  LLM output (caps, valid IDs, known document codes).

### 5.7 Build-time changes (24 Oct, autonomous build)
Recorded here so the plan stays the source of truth:
- **Verifier check 4** flags `must` only when *all* cited passages are G.
  e-KYC Appendix 2 para 13 (the FAR rule) has no S/G marker but is an
  obligation; the strict "needs an S" rule rejected the correct FAR answer.
- **Quote check:** each `...`-joined fragment needs ≥ 3 words, in order.
- **Per-turn state:** a conversation is one checkpointer thread, but only
  `messages` carries over. `turn_input()` resets every per-turn field; summed
  and appended fields use reducers that accept `None` as "reset".
- **Live retry policy** `retry_live` (3 attempts, 1–8 s + jitter, timeouts
  retried) for Gemini, embeddings and Cohere; indexing keeps 6 attempts.
- **Rate limiting:** one thread-safe sliding-window limiter per model (and
  one for Cohere, one for query embeddings), shared across roles and
  parallel searches.
- **Sector:** `detect_sector` is gone. The caller passes the preferred sector
  (planner, else sidebar); `None` ("show all sectors") demotes nothing.
  Traditional mode keeps regex paragraph refs, limited to the searched
  documents.
- **Evidence across searches:** the evidence reducer also drops another
  sector's copy of a rule already found by a different search (lookups are
  always kept).
- **Cached LLM results** report the calls they originally cost, so the
  budget and graph path are identical with or without the cache.
- **Agentic graph details (M2):**
  - the generator reads the planner's `standalone_question`, so a follow-up
    like "what about takaful operators?" is answered as a full question;
  - the planner's sector applies to the whole answer only when *every*
    search names that one sector (one side of a banks-vs-takaful comparison
    must keep the user's sector);
  - a copy of a rule is kept when its search asked for that sector
    explicitly (both sides of a cross-sector comparison stay cited);
  - the review node, not the router, trims follow-ups to the budget: no new
    round after 2 reviews, or when fewer than 2 LLM calls (a review + the
    generator) or no time remain. A lookup needs 2 search slots (its
    fallback search), so the 6-search ceiling always holds;
  - a lookup miss searches the standalone question in the named documents
    (a bare "paragraph 99.9" query matched nothing meaningful);
  - one failing tool branch (Qdrant timeout) returns a note and no evidence;
    the other branches' evidence still reaches the answer;
  - failed LLM attempts count towards the call budget;
  - `out_of_scope`, `small_talk` and `clarify` answers carry
    `status="not_covered"` plus an `outcome` field the UI shows.
- **Persistence + API details (M3):**
  - the runner lives in `app/api/runner.py` (it builds the API's turn view);
    the route calls `runner.start()` (PII masked, conversation created, so
    a database error is an HTTP 503) before the SSE stream begins;
  - checkpointer thread = `<conversation_id>:<mode>`, so the two modes never
    share graph state; one turn at a time per thread (a second request gets
    an error event);
  - every turn is audited: answered, `error` (with the LLM calls spent so
    far) or `cancelled` (the client left mid-stream, with the steps reached);
  - `turn_events` stores the user-visible step `lines` (instead of an
    `input_summary`) plus the node's full state update and any tool error;
    `turns` also stores `outcome`, `sector`, `pii_masked`, `notes` and
    `error`;
  - append-only is enforced by Postgres triggers against UPDATE, DELETE and
    TRUNCATE (migrations 0001, 0002); Alembic ignores LangGraph's checkpoint
    tables, so autogenerate never proposes dropping agent memory;
  - PII: NRIC with dashes, spaces or nothing; cards by Luhn; any other 10-20
    digits (contiguous or in groups of 3+) as an account. Two-digit groups
    (phone area codes, paragraph lists) and amounts are left alone.
- **Gemini settings:** thinking level per role: planner, reviewer and judge
  `low` (default thinking made the planner take ~11 s; low takes ~3 s), the
  generator `medium` (at low it ignored a fetched cross-reference: e-KYC
  8.10 answered without 8.4). A 20 s timeout, which Gemini uses as the
  server deadline, so a stalled call (504) fails fast and is retried.
- **Cross-references are marked by code (M4):** when a passage depends on
  another given passage of the same document ("as required under / in
  relation to / subject to paragraph X", lists and ranges included,
  references into another document excluded), its evidence block says
  "Cross-reference: E2 (para 8.4)". The generator is told to include E2's
  rule, and verifier check 7 enforces it with one regeneration. A
  regeneration that fails a serious check never replaces a first answer
  that passed. Real run: agentic "What does e-KYC paragraph 8.10 require?"
  now answers MUST (8.4, S) + MAY (8.10, G); traditional gives only 8.10.
  Cost: one extra generator call when it triggers (to be measured in M5).
- **Streamlit UI (M4):** `ui/app.py` uses `st.navigation` with pages in
  `ui/views/` (chat, compare, evaluation); the sector and mode choices
  survive page switches; citation chips show short labels ("e-KYC 8.4
  (S)", "e-KYC App. 2 para 13") linking to the PDF page; each new rating
  is posted (feedback is append-only, the latest wins); a failed turn keeps
  its conversation. The UI reads settings from `app/core/config.py` and
  talks to the API over HTTP only. `scripts/screenshot_ui.py` (Playwright,
  dev only) checks the flow in a real browser.
- **Evaluation details (M5):**
  - gold labels are paragraph keys, `ekyc:8.4` or `ekyc:Appendix 2:13`
    (appendices reuse numbers); `python -m evals.golden --check/--show`;
    who checked what is recorded in `evals/golden_review.yaml` and printed
    in every report;
  - every retrieval stage is scored on its first 5 chunks with MRR@5, so
    the stages are compared on equal terms; the report lists every item a
    stage made worse, with its top 5 before and after;
  - a crash or model outage is an `error` outcome, reported apart, never a
    correct refusal;
  - the judge sees the user's own conversation (earlier turns + question),
    never the system's rewrite; it writes its reason before each verdict;
    "must" where the passage says "may" is "no"; statement numbers that
    don't match 1..n leave the answer unjudged; the cross-check sample is
    stratified by category and stops at the first quota refusal;
  - results are saved after each phase; the Evaluation page shows the
    latest full run first.
- **Packaging (M6):**
  - one image runs three services (index bootstrap, api, ui) instead of a
    separate UI image: built once, always in step;
  - the committed `chunks.jsonl.gz` and embedding-cache `.gz` are unpacked
    by the `index` job, which runs `indexer --require-cached` (it refuses
    to call Gemini) after Qdrant's healthcheck passes; the API runs
    `alembic upgrade head` first; the UI calls `API_URL` (inside Docker)
    and links PDFs through `PUBLIC_API_URL` (the browser's address);
  - a complete host indexer run rewrites both `.gz` snapshots, so a later
    `docker compose up` can't roll Qdrant back; for local development only
    the databases run in Docker (`docker compose up -d db qdrant`);
  - Postgres and Qdrant bind to 127.0.0.1 (default credentials, no auth);
    the containers run as a non-root user; host ports and the `.env` path
    are variables (`API_PORT`, `UI_PORT`, `ENV_FILE`, ...);
  - LangSmith tracing starts with the real API stack only (never in
    pytest) and is switched off explicitly when not configured; questions
    are PII-masked before any trace (API and `scripts/ask.py`);
  - CI installs `requirements-ci.txt` (runtime + docling-core + pinned
    pytest/ruff), not full Docling and torch.
- **Response cache keys** are `<namespace>:<hash>` (llm, embed_query,
  rerank), so a run can report its real API calls per kind (cache misses).
- **Public deployment (26 Oct, decided with Syakir):** the submission needs a
  public link, and Streamlit Community Cloud runs only one Streamlit
  process. So the UI gets a second backend:
  - `UI_BACKEND=api` (default, local and Docker): the UI calls FastAPI over
    HTTP, exactly as before;
  - `UI_BACKEND=local` (Streamlit Cloud): the UI calls the same `TurnRunner`
    in its own process, built by the same `build_services()`. Both
    backends share one interface (`ui/backend.py`), so the pages don't know
    which one they use. This is a deliberate exception to "the UI never
    imports the agent", for free hosting; the API stays the reference
    architecture;
  - storage: Qdrant Cloud (`QDRANT_URL` + `QDRANT_API_KEY`, indexed from the
    committed cache with `indexer --require-cached`: 0 Gemini calls) and
    Neon Postgres (`DATABASE_URL`); the local backend runs
    `alembic upgrade head` at startup;
  - PDFs are served by Streamlit's static file serving (copied into
    `ui/static/` at startup), so citations still open the PDF at the page;
  - a public link spends the free Gemini quota, so: a daily question limit
    shared by all visitors (`DAILY_QUESTION_LIMIT`, counted from the audit
    log since midnight Pacific time, when Gemini's quota resets, enforced by
    `TurnRunner.start`; the API answers 429), a per-visitor limit
    (`QUESTIONS_PER_SESSION`, counted when a question starts), and the
    response cache for live answers (`LIVE_RESPONSE_CACHE`), so repeated
    questions such as the example buttons cost no quota (they still count
    towards the limits). The daily limit is soft: questions still running
    aren't counted yet, so a burst can pass it by a few, which the quota's
    headroom absorbs;
  - the checkpointer's connection pool checks a connection before use, so
    one Neon dropped while idle is replaced; a failed start closes what it
    opened and shows a clear message;
  - the sidebar history shows only the visitor's own conversations from
    this browser session (there are no logins), so visitors never see each
    other's questions.
- **Model cooldown and cache across models (26 Oct, from the live logs):**
  on Streamlit Cloud `gemini-3.1-flash-lite` answered 503 "high demand" on
  almost every call, so each planner and reviewer step spent ~5–6 s on three
  failed tries before the fallback. Now a model that fails with an outage,
  overload or quota error is tried *last* for `LLM_COOLDOWN_S` (120 s), so
  only the first step pays; and before any call the cache is checked for an
  answer from either of the role's models, so a repeated question costs no
  call even when the earlier answer came from the fallback. A model that
  answers leaves its cooldown at once. **The evaluation turns both off**
  (`cooldown_s=0, reuse_fallback_answers=False` in `evals/run.py`): there,
  faithful attribution beats speed, and a fallback verdict cached during an
  outage must not stand in for the independent judge on later re-runs.
  Known: a daily-quota 429 cools the model for 120 s at a time (so it is
  retried every 2 minutes until the reset), and after a model's live
  retries give up its failed attempts count as one call in the request
  budget (`ChatModelClient.requests`, the quota ledger, stays exact).
- **UI redesign (26 Oct):** no emojis; Material icons, a restrained theme in
  `.streamlit/config.toml` (light and dark), example questions for a first
  visit, agent steps as a step timeline with per-step times (`TurnView.steps`
  is now structured: node, title, lines, ms), sources as a list with the S/G
  label, the quote and a PDF link, and friendly messages for limits and
  errors.

## 6. Memory and chat history (Postgres)

There are two kinds of storage, and you should be able to explain why both
exist:

| | LangGraph checkpointer (`PostgresSaver`) | Our tables (SQLAlchemy + Alembic) |
|---|---|---|
| What it is | The agent's **working memory**: the full graph state after each step | The **product's record** |
| Used for | Follow-up questions (`messages` history), resuming, debugging a past run | Sidebar conversation list, showing past answers, **audit trail**, feedback, analytics |
| Format | Internal, owned by LangGraph | Our schema, queryable with SQL |

**Our tables**
- `conversations(id, user_id NULL, title, user_sector, created_at,
  updated_at)`. `user_id` stays empty for now: there are no logins, but the
  column is ready for SSO later.
- `turns(id, conversation_id, mode, question, standalone_question, plan,
  answer, verification, evidence_ids, models, prompt_version, index_version,
  llm_calls, input_tokens, output_tokens, latency_ms, status, created_at)`:
  one row per question.
- `turn_events(id, turn_id, seq, node, input_summary, output, ms, error,
  created_at)`: one row per graph step. Every search query with its filters
  and returned chunk IDs and scores, every reviewer decision, fallbacks used,
  and errors.
- `feedback(turn_id, rating, comment, created_at)` records thumbs up or down
  from the UI.

**Audit log rules:**
- `turns` and `turn_events` are **append-only**: no updates, no deletes.
- They record enough to answer *"why did the assistant say X on 3 March?"*:
  the exact evidence, the models, the prompt version, and the index version
  (a hash of `chunks.jsonl`).

**Long conversations:** the planner sees only the last ~6 messages (trimmed),
which keeps prompts small and fast.

---

## 7. The demo app (Streamlit)

**Page 1, Chat (main):**
- Sidebar: institution type (Banking, the default; Insurance/takaful; Money
  services; E-money; or "Not sure: show all sectors"). The sector is decided
  in this order:
  1. a sector named or implied in the question (the planner);
  2. the sidebar choice;
  3. banking.

  Each answer states the assumption it made ("Applies to: banking
  institutions"). Also in the sidebar: mode (Agentic / Traditional),
  conversation history (from Postgres), and a "New chat" button.
- Each answer shows:
  - a status badge (Answered / Partially answered / Not covered);
  - the summary;
  - statements with **MUST** / **MAY** badges and citation chips;
  - a sources panel listing each citation, an S/G badge, a text snippet, and
    an "Open PDF at page" link;
  - thumbs up / down feedback.
- **"Agent steps"**, a live step timeline that fills while the agent works:
  *Plan → Search e-KYC… → Review: following cross-reference para 8.4 →
  Write → Verify: citations verified*, each step with its time. It also
  shows latency and LLM calls.

**Page 2, Compare:** one question, with traditional and agentic answers side
by side, plus their steps, latency and LLM calls. **This is the most
convincing part of the demo.**

**Page 3, Evaluation:** the measured results tables from `evals/results/`.

---

## 8. Quality: test strategy and evaluation

Quality is assured at **four levels**. This answers the "explain test case
build" requirement.

```
            ▲   4. Evaluation set (LLM, slow, run on demand)   — "is it actually good?"
            │   3. API tests (fake agent)                       — "does the service contract hold?"
  cost &    │   2. Agent graph tests (scripted fake LLM)        — "does each decision path work?"
  time      │   1. Unit tests (pure functions, no network)      — "is each piece correct?"
```

### 8.1 Automated tests (pytest, no API keys, run on every change)

| Level | Examples | Status |
|---|---|---|
| Unit | Chunking rules, BM25 tokenizer, RRF, sector collapse, rate limiter, retries | ✅ 85 passing |
| Unit (new) | `verify_answer`: invalid ID removed; "must" citing only G flagged; zero valid citations becomes `not_covered`. Routing functions: every route goes to the right node. Plan sanitising: more than 3 searches cut to 3; malformed paragraph ID dropped | Day 1–2 |
| Graph paths (new) | A **scripted fake LLM** returns fixed plans and reviews, so every path runs deterministically: lookup path, parallel search, review loop stops at 2 rounds, out-of-scope refusal, clarification, cross-reference fetch, regenerate after a failed verify, planner failure falls back to plain search | Day 2 |
| API (new) | `/chat` streams events in order; a conversation persists; feedback is saved | Day 3 |

**Why a fake LLM?** Real LLM output varies, so a test against it would pass
one day and fail the next. To test **our** logic, the routing and the loop
bounds, we script the LLM's answers. To test **the LLM's** quality, we use the
evaluation set below. Being able to explain this split is a strong signal in
an interview.

### 8.2 The evaluation set: `evals/golden_set.yaml` (~40 questions)

| Category | # | Example | What it proves |
|---|---|---|---|
| Direct lookup | 4 | "What does para 14A.9.6 require?" | Exact retrieval |
| Paraphrased single fact | 10 | "Can a customer be onboarded without meeting face to face?" | Semantic retrieval |
| Comparison / multi-part | 6 | "Compare e-KYC and AML non-FTF verification rules" | Decomposition |
| Cross-reference | 4 | "What does 8.10 require, including the rule it refers to?" | Following references |
| Sector-specific | 4 | "Can a takaful operator onboard remotely?" | Sector handling |
| Follow-up (2 turns) | 4 | "…and for money changers?" | Memory and rewriting |
| Out of scope | 6 | "Credit card interest cap?", "Write me a poem" | Refusal |
| Ambiguous | 2 | "What does para 8.2 say?" | Clarification |

**How it is built (autonomous build, 24 Oct):**
1. **Drafted from the chunks.** Claude Code reads `chunks.jsonl`, picks
   paragraphs across all three documents and categories, and writes
   questions. Each paragraph it drew from becomes the gold label.
2. **Hard cases written by hand.** Comparisons, cross-references, follow-ups,
   out-of-scope and ambiguous questions, written by Claude Code from the
   actual paragraph text.
3. **Checked twice.** Claude Code re-reads each gold paragraph to confirm it
   really answers the question. Then **Syakir spot-checks at least 10 items**
   against the PDFs.
4. **Disclosed.** The report states exactly how the set was built and who
   checked what.

Three different model families keep the evaluation honest: Claude writes the
questions, Gemini 3.5 answers them, and Gemini 3.1 judges the answers.

Each item records: `id, category, question, (previous turns),
expected_paragraphs: ["ekyc:8.4", …], answerable: true/false`.

### 8.3 Metrics

| Metric | Measures | Needs an LLM? |
|---|---|---|
| **Hit@5, Recall@5, MRR** | Did retrieval find the gold paragraphs, and how high? | No: exact and cheap |
| **Retrieval ablation** | Dense only → BM25 only → hybrid RRF → + exact lookup → + rerank → + sector selection → agentic. Proves each optimisation (the bonus) | No (rerank calls only) |
| **Refusal accuracy** | Refused all unanswerable questions? Answered all answerable ones? | No |
| **Citation validity** | Cited IDs that exist (should be 100% by construction) | No |
| **Citation recall** | Gold paragraphs that ended up cited | No |
| **Faithfulness** (our judge) | For each statement: is it supported by its quote and cited passage? Judged per answer in one call, scored as the share of supported statements | Yes (3.1-flash-lite) |
| **Answer relevancy** (our judge) | Does the answer address the question? 1–5 scale, in the same call | Yes |
| **Judge agreement** | The same judgments repeated with `gemini-3.8-flash` on ~15 answers, reported as % agreement | Yes (20/day) |
| **Cost and speed** | LLM calls, tokens, p50/p95 latency **per mode** | No |

Output: `evals/results/<date>/report.md` plus JSON files, shown on the
Evaluation page and in the slides. **These are the numbers your resume line
will quote.**

**Why our own judge instead of RAGAS (decided 24 Oct):**
- RAGAS's faithfulness metric first extracts claims, then checks each one,
  which costs several LLM calls per answer. Our answers already **are**
  claims: statements, each with a quote and a citation. So one judge call per
  answer covers it, about 3× fewer calls on a 500/day quota.
- The judge prompt and rubric live in the repo, so they can be read and
  explained.
- RAGAS stays an optional cross-check on a small subset if quota allows.
- **Resume wording:** "LLM-as-judge faithfulness", not "RAGAS".

**Judge bias:** the judge is a different model from the generator, and a
second, stronger model cross-checks a sample.

---

## 9. Documents to produce (for the "thought process" and "investigation" parts)

| File | Contents |
|---|---|
| `README.md` | What it is, a screenshot, the architecture diagram, quick start (`docker compose up`), results table |
| `docs/architecture.md` | Components, data flow from PDF to answer, the graph, the state and schemas, design decisions with reasons |
| `docs/agentic_rag_investigation.md` | What agentic RAG is; workflows vs agents; the patterns (routing, tool use, Corrective RAG, Self-RAG, Adaptive RAG, decomposition, ReAct); trade-offs; **when NOT to use it**; our measured comparison |
| `docs/testing_and_evaluation.md` | The test pyramid, how the golden set was built, metric definitions, results, known failure cases |
| `docs/demo_script.md` | The exact demo questions, in order, with what each one shows |
| `docs/BUILD_REPORT.md` | What Claude Code built, milestone by milestone: deviations from the plan and why, real inputs and outputs, test results, evaluation results with failure analysis, quota used, known limits |
| `docs/LEARNING_GUIDE.md` | The project explained from one LLM call up to the agent, for a beginner: every concept mapped to real files, one name per concept, with "explain it" questions and answers |
| Slides (Claude Slides) | ~10–12 slides built from the docs above |

---

## 10. Build schedule: autonomous build with Claude Code (24 → 26 Oct)

**Updated 24 Oct.** Claude Code builds everything, in the repo, as
milestones. Syakir approves the plan at the start, spot-checks the golden set,
reviews the demo, then learns from `docs/LEARNING_GUIDE.md` and
`docs/BUILD_REPORT.md`.

**Every milestone runs the same loop:**
1. Re-read the plan section for this milestone.
2. Implement it.
3. `ruff` + `pytest` green.
4. A fresh-context review by the `code-reviewer` subagent, comparing the
   code against this plan and CLAUDE.md.
5. Fix every finding.
6. A real-API smoke check, within the quota budget below.
7. Update PROGRESS.md.
8. Commit.

| # | Milestone | Done when |
|---|---|---|
| M1 | **LLM layer + traditional RAG:** `app/llm` (provider-agnostic factory, per-model rate limiter, response cache), `app/agent/schemas.py`, `evidence.py`, `prompts.py`, `generate_answer`, `verify_answer` (6 checks + quote check), the traditional graph, `SearchService` gets document and sector parameters, `scripts/ask.py` | `ask.py --mode traditional` answers the FAR question with verified citations and refuses "credit card interest cap"; tests green |
| M2 | **Agentic graph:** state + reducers, `understand_question`, routing with `Send`, `search_policies`, `lookup_paragraphs`, `review_evidence` loop, clarify/refuse nodes, fallbacks, request budget, fake-LLM graph-path tests | The five §2 cases behave as described, run via `ask.py --mode agentic --steps` |
| M3 | **Persistence + API:** Postgres checkpointer, Alembic (`conversations`, `turns`, `turn_events`, `feedback`), PII masking, `/chat` SSE, `/conversations`, `/turns/{id}/feedback`, `/documents`, `/health`; old echo app removed | A follow-up works across 2 requests; audit rows written and masked; API tests green |
| M4 | **Streamlit UI:** Chat (live steps, MUST/MAY badges, citations, open-PDF links, sector + mode sidebar, history, 👍/👎), Compare, Evaluation | Full demo flow works in the browser (Claude Code checks with Playwright screenshots) |
| M5 | **Evaluation harness + golden set:** ~40 items (§8.2), retrieval eval + ablation, answer eval for both modes, our judge, judge cross-check, cached runs, a `--smoke` 8-question subset, report with per-category failure analysis | `python -m evals.run --smoke` works; full-run command ready |
| M6 | **Packaging:** Dockerfile, compose (qdrant, postgres, api, ui, one-off index bootstrap from committed cache), `.env.example`, LangSmith env, GitHub Actions (ruff + pytest) | `docker compose up` from a clean clone serves the app with no embedding calls |
| M7 | **First full evaluation** (Syakir runs it when Claude Code says it's ready), then Claude Code analyses it, fixes the top failures, and re-runs only the affected items | Report shows both modes; fixes recorded |
| M8 | **Docs:** README, architecture, investigation, testing & evaluation, demo script, BUILD_REPORT, LEARNING_GUIDE; `docs-reviewer` subagent checks them for a beginner | Syakir can follow the LEARNING_GUIDE without needing outside help |
| M9 | **Final evaluation** (Syakir runs it), freeze the code (tag `v1.0-assessment`), demo rehearsal checklist | Final numbers in README, BUILD_REPORT and slides |

**Quota: Syakir runs the full evaluations himself.** Claude Code only runs
small smoke checks (≤ 30 calls per milestone). When an evaluation is ready,
it stops and tells him the command, the estimated calls, and whether to run
now or after the 3 pm reset. Rules of thumb:
- One full evaluation run is expected to use about 160 calls on
  gemini-3.1-flash-lite (planner, reviewer, judge), 100 on
  gemini-3.5-flash-lite (generator) and 15 on gemini-3.8-flash (judge
  cross-check): `python -m evals.run --full --estimate` prints the current
  figures (revised 24 Oct from "about 575", measured on the smoke runs).
  Both Lite models together allow about 1,000 a day, and the quota resets
  at 3 pm Malaysia time.
- Start a full run right after a reset, so the run doesn't stop halfway.
- On the day before the demo, keep **≥ 250 calls** unused for rehearsal and
  the live demo.
- Check usage on AI Studio's usage page before each run.

Cohere trial: about 1,000 calls/month in total. Rerank results are cached in
evaluation runs, and every run logs how many calls it used.

**If time runs short, cut in this order:** CI → Evaluation page → judge
cross-check → feedback buttons → history sidebar. **Never cut:** the agent,
citation verification, the Compare page, the evaluation numbers, Docker, or
the two learning documents.

## 11. The concept ladder: what you must be able to explain

Learn it bottom-up. Each level uses the one below it. Next to each level is
where it lives in **your** code, so every answer you give in the interview
can point to something real.

| # | Concept | In this project | Say it in one breath |
|---|---|---|---|
| 1 | **An LLM call** | `app/llm/models.py` | "A list of messages (system, user, assistant) goes in, text comes out; tokens cost money and time; temperature controls randomness." |
| 2 | **Structured output** | `app/agent/schemas.py` | "I give the model a schema; it returns JSON that is validated into a Pydantic object, so code can branch on it safely." |
| 3 | **Embeddings & vector search** | `embedding_service.py`, Qdrant | "Text becomes a vector; similar meaning means nearby vectors; Qdrant finds the nearest." |
| 4 | **Hybrid retrieval & reranking** | `retrieval_service.py`, `rerank_service.py` | "Dense finds meaning, BM25 finds exact terms like 14A.9.6; RRF merges the ranks; a cross-encoder reranks the top 30 by reading query and passage together." |
| 5 | **RAG** | traditional graph | "Retrieve evidence, put it in the prompt, generate an answer only from it, cite it, refuse if it isn't there." |
| 6 | **Tool use** | `search_policies`, `lookup_paragraphs` | "The model decides which function to call, with which arguments; my code executes it and returns the result." |
| 7 | **Workflows vs agents** | the two graphs | "A workflow follows code paths I defined; an agent lets the LLM choose the path. Mine is a bounded agent: the LLM decides, code enforces limits." |
| 8 | **Agentic patterns** | §5.3 table | "Routing, decomposition, parallel tool calls, Corrective RAG grading with re-retrieval, and a self-check: each is a node in my graph." |
| 9 | **LangGraph** | `app/agent/graph.py` | "State, nodes, conditional edges, reducers, Send for fan-out, and a checkpointer for memory, compiled into a state machine I can stream and trace." |
| 10 | **Memory** | `PostgresSaver` + `turns` | "A checkpointer saves state per thread, so follow-ups know the history; separate tables are the product's audit record." |
| 11 | **Serving** | FastAPI SSE + Streamlit | "The agent runs behind an API that streams each step as an event; the UI just renders events." |
| 12 | **Evaluation** | `evals/` | "A human-verified golden set; retrieval metrics without an LLM; LLM-as-judge for faithfulness; compared per mode, re-run after every change." |
| 13 | **Observability & ops** | LangSmith, logs, Docker | "Every node and prompt is traced with tokens and latency; the whole stack starts with one command." |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Free-tier quota runs out during evals or the demo | Response caching for evals, rate limiter, model tiering, billing as a fallback, the demo questions tested the day before |
| The agent loops or gets slow | Round counter + `recursion_limit` + max 3 sub-searches; latency shown in the UI |
| The live demo fails (network, API) | Backup video; `/health` checked before going on; graceful-degradation paths |
| The LLM invents a citation | Impossible by design: it cites evidence IDs, and code verifies them |
| An obligation is presented as guidance, or the reverse | The S/G consistency check in `verify_answer` |
| Not enough time | Priority labels and the cut order in §10 |

## 13. After the assessment (portfolio backlog)

- Async end-to-end (`ainvoke`, async Qdrant and HTTP clients) for concurrent
  users.
- Gemini Embedding 2 A/B test, and Qdrant server-side RRF vs our own (k=2 vs
  60).
- A ReAct-style free tool-calling agent as a third mode, to compare autonomy
  levels.
- Long-term user memory (the institution type remembered per user),
  authentication.
- Guardrails: PII detection in questions, rate limiting per user.
- Deployment (Cloud Run / ECS) with a managed Postgres and Qdrant Cloud.
- **Resume:** remove LlamaIndex unless it gets used. Replace the placeholder
  numbers with the Day 5 results.
