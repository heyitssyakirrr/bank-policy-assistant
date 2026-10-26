# Design Q&A

> Updated 24 Oct for the autonomous build: our own LLM judge instead of
> RAGAS, PII masking, shared rate limiters, a reproducible index. See
> PROJECT_PLAN.md §5.6 and §8.

These are the questions asked while planning the agentic version (20–22 Oct
2026) and the answers we agreed on. Read this with PROJECT_PLAN.md §2–§8.

## Concepts

**Are the planner, reviewer and generator agents or tools?**
Neither. They are three roles the same LLM plays inside **one** agent. Each is
a graph node that calls Gemini with its own prompt and output schema:
- `understand_question` is the planner;
- `review_evidence` is the reviewer;
- `generate_answer` is the generator.

The tools are `search_policies` and `lookup_paragraphs`: code that fetches
data. The agent is the whole graph working together. A multi-agent system
isn't needed here, because it would add cost and latency with no benefit.

**What is a node? Is it a tool?**
A node is a step: a Python function that receives the state and returns
updates to it. Some nodes call the LLM, some call a tool, and some are pure
code. An edge connects nodes. A conditional edge is a small function that
reads the state, including what the LLM decided, and picks the next node.

**What makes it agentic?**
The LLM decides the path:
1. **Decision 1:** the planner picks the route (search, lookup, clarify, or
   refuse) and which searches to run.
2. **Decision 2:** the reviewer judges the evidence. If something is
   missing, the graph loops back to search or lookup, at most twice.

Traditional RAG is one fixed line. The pattern names are routing (Adaptive
RAG), tool use and decomposition, Corrective RAG, and a self-check
(the Self-RAG idea).

**Multi-part questions: one by one, or together?**
Independent parts run **together**. The planner returns up to 3 searches,
each with its own document and sector filters. `Send` runs them in parallel.
LangGraph waits for all of them, then a reducer merges the evidence with
duplicates removed. The reviewer checks each part and searches again **only
for the missing part**. The generator writes one answer, with a `topic`
label on each statement so the UI can group them. **Dependent** steps go one
by one through the review loop (e.g. 8.10 refers to 8.4, so fetch 8.4).

**Does the planner search only part of the question?**
No. It reads the whole question plus the history and writes a
self-contained `standalone_question`, which the reviewer and generator
always see. The searches are just how evidence is fetched:
- one rewritten search in the documents' own wording;
- one search per part of a multi-part question;
- or no search at all (lookup or refusal).

## Citations and verification

**Who makes a citation?**
1. Code labels each retrieved chunk E1…En in the prompt.
2. The generator cites E-numbers, plus an exact `quote` from the passage.
3. Code verifies those choices.
4. Code swaps each E-number for the chunk's stored `citation` string, made
   at ingestion by `Chunk.citation()`.

The LLM never types a paragraph number or page. This "LLM proposes, code
verifies" pattern is the industry norm. Provider citation features
(Anthropic's Citations, Cohere's grounded generation) work the same way:
sources have IDs, the model references IDs, code owns the metadata.

**What does `verify_answer` check?**
1. Each cited ID was actually retrieved. If not, the ID is dropped.
2. Each statement still has a citation. If not, the statement is dropped.
3. The quote appears in a cited passage.
4. A `must` statement doesn't cite only G (guidance) paragraphs. Unmarked appendix
   and table rules (e.g. the FAR thresholds) can state obligations too.
5. The status matches the content.
6. Something valid is left. If not, the answer becomes `not_covered`
   (a refusal).
7. No cross-reference is left out: a cited passage that depends on another
   given passage ("as required under paragraph 8.4") is answered with it. If
   not, one regeneration; if still incomplete, the answer is kept as it is.

Checks 1, 2 and 5 are fixed silently and logged. Checks 3 and 4 trigger one
regeneration with feedback, then a downgrade to `partially_answered`. Code
can't check *meaning*. That is measured offline by the faithfulness judge.

## Safety
- Max 2 review rounds (a state counter) and a `recursion_limit`.
- Max 3 searches per plan.
- Paragraph IDs are checked for format, then against the index; a miss
  falls back to search.
- Citations by ID only, verified by code.
- All tools are read-only, so prompt injection can at worst produce a bad
  answer, never a harmful action.
- Message length cap.
- **PII masking** (added 24 Oct), a data-protection guardrail: MyKad/NRIC,
  card and account-like numbers are masked as soon as a question arrives,
  before Gemini, LangSmith, logs and the audit tables see it. Regex is right
  here, because the formats are fixed.
- **Retries:** 3 attempts on 429, 5xx and timeouts, with exponential backoff
  plus jitter (~1 s, 2 s, capped at 8 s) and a 30 s timeout per call. No
  retry on 400. 1 repair retry for malformed JSON. Offline indexing keeps 6
  attempts.
- **Fallbacks:** a planner failure becomes a plain search; a rerank failure
  uses fused order; a model whose quota runs out switches to the other Lite
  model.

## Memory, audit, users
- **Two stores, both in Postgres, both survive restarts.** They're lost only
  with `docker compose down -v`.
  - The LangGraph `PostgresSaver` checkpointer is the agent's working
    memory, used for follow-ups.
  - Our tables are the product record: `conversations`, `turns`,
    `turn_events` and `feedback`.
- **Audit log:** `turns` and `turn_events` are append-only. They record the
  question, plan, every search with its returned chunk IDs and scores,
  reviewer decisions, the evidence, the answer, the verification report,
  models, prompt version, index version, tokens, latency and errors. Enough
  to answer "why did it say X on that date".
- **Alembic** versions our table schema (migrations). LangGraph creates its
  own checkpoint tables with `setup()`.
- **No users or login for now.** `conversations.user_id` is nullable and
  ready for SSO later.
- **Thumbs up/down is not RLHF.** RLHF retrains model weights, which we
  can't do with an API model. Feedback is a product signal: find bad
  answers, fix them, add them to the evaluation set.
- **LangSmith** records a trace of every node, prompt, token count and
  latency. Free Developer plan: 1 seat, 5,000 traces a month, kept 14 days.
  It's optional, and the prompts leave the machine (a real bank would
  self-host something like Langfuse).

## Sidebar
- **Institution type** sets the user's sector. AML Part 14 repeats its rules
  per sector:
  - 14A banking;
  - 14B insurance/takaful;
  - 14C money services;
  - 14D e-money.

  Some rules exist for one sector only (14C.16.12; e-KYC 8.6, 8.11 and
  8.15).
- **Choices:** Banking (default), Insurance/takaful, Money services,
  E-money, "Not sure: show all sectors".
- **The sector is decided in this order:**
  1. a sector named or implied in the question;
  2. the sidebar choice;
  3. banking.
- The value feeds `selection.py`: the preferred copy is kept and other
  sectors' rules are demoted. Every answer states "Applies to: …". In a
  real deployment it would come from the user's login profile.
- **Mode** (Agentic / Traditional) exists to demonstrate and measure the
  difference. A real product would ship one mode.

## Quality
- **Tests (pytest):** does *our code* follow our rules? A scripted fake LLM
  makes the graph paths deterministic. Fast, free, written alongside the
  code.
- **Evaluation:** does the *whole system* answer well? Real models run on
  about 40 human-verified questions, each with an answer key
  (`expected_paragraphs`, `answerable`).
  - Retrieval metrics: Hit@5, Recall@5, MRR, plus an ablation of each
    optimisation.
  - Deterministic answer metrics: refusal accuracy, citation validity and
    citation recall.
  - Faithfulness and relevancy come from **our own LLM judge** (decided
    24 Oct): one call per answer checks each statement against its quote
    and cited passage. That's about 3× cheaper than RAGAS. RAGAS stays an
    optional cross-check.
  - The judge is 3.1-flash-lite, cross-checked on ~15 answers with
    3.8-flash.
  - Everything is compared for traditional vs agentic, including latency
    and LLM calls.
  - Responses are cached. The run starts right after the 3 pm quota reset.

## Models
- **Chosen:**
  - `gemini-3.1-flash-lite` for the planner, reviewer and judge;
  - `gemini-3.5-flash-lite` for the generator.
  - Each has its own quota bucket of 500 a day, and each is the other's
    fallback.
- **Rejected:**
  - The Flash models allow only 20 a day.
  - The 2.5 models are closed to new projects.
  - DeepSeek is cheap and capable, but its hosted API stores data in China,
    a concern for a bank. Kept as an interview talking point: the design is
    provider-agnostic, and a bank would use local data residency or
    self-host open weights.
- **Switching LLMs:** change the role's setting in `.env`, then re-run the
  evaluation, because prompts behave differently per model.
  - Structured-output method and rate limits are per-role settings.
  - The embedding model is not switchable without re-indexing.
