# CLAUDE.md: Bank Policy Compliance Assistant

Read at the start of every session, in this order:
1. **PROGRESS.md**: what is done and which milestone is next.
2. **PROJECT_PLAN.md**: the design (§2–§8) and the milestones (§10).
3. **docs/design_qa.md**: why each design decision was made.

The plan is the source of truth. If you find a better approach or a mistake
in the plan, don't silently diverge. Write the change and the reason into
PROJECT_PLAN.md (and the decisions log in PROGRESS.md), then follow the
updated plan.

## The project
An agentic RAG assistant over three Bank Negara Malaysia policy PDFs: e-KYC,
Fair Treatment of Financial Consumers (FTFC), and AML/CFT/CPF & TFS. It
returns paragraph-cited answers, labels obligations (S = must) vs guidance
(G = may/should), and refuses when the documents don't cover a question.

- **Two modes:**
  - **traditional:** retrieve → generate → verify;
  - **agentic:** a LangGraph graph: plan → parallel search/lookup → review
    loop (max 2 rounds) → generate → code verifier.

  Both share the search engine, the generator and the verifier, so the
  comparison is fair.
- **Live Streamlit demo on Mon 26 Oct 2026,** for an AI Engineer technical
  assessment. Also a portfolio project: GitHub + Docker Compose.

## Working mode: autonomous build
Syakir wants you to **build the whole plan end to end**, milestones M1→M9 in
PROJECT_PLAN.md §10, test and evaluate it yourself, and only then hand it to
him. Work without waiting for approval between milestones. Stop and ask only
for:
- a missing key or an environment problem you can't fix;
- a decision that changes the plan's scope;
- anything in "Hard rules" that says ask.

**Every milestone runs this loop:**
1. Re-read the plan section for the milestone. List the files you will
   add, change or remove.
2. Implement it, with tests alongside the code.
3. Run `ruff check .`, `ruff format --check .` and `pytest -q`, all green.
4. Run the **code-reviewer** subagent on the milestone's changes (fresh
   context: it compares the code against the plan and these standards).
5. Fix every finding, or record why you didn't.
6. Run a small real-API smoke check if the milestone needs one (≤ 30
   Gemini calls), and log it in the PROGRESS.md quota ledger.
7. Update PROGRESS.md: the checkbox, what was built, decisions made, and
   the results of any real runs.
8. `git commit` with a clear message (`feat(agent): …`, `test: …`,
   `docs: …`).

Never mark a milestone done with failing tests, unreviewed code, or
untested behaviour.

## Code standards
- Write code the way a senior AI engineer would: maintainable, scalable,
  efficient, readable. Typed throughout (Pydantic for every contract).
  Small pure functions where possible. No dead code. No placeholder files.
- Self-explanatory names. Short comments only where the *why* isn't
  obvious; no long comment blocks.
- **One name per concept, everywhere:** code, docs, UI, logs. Use the terms
  in docs/design_qa.md: planner, reviewer, generator, verifier, tool,
  evidence (E1…), round. Never call the same thing by two names.
- **LLM for judgment, code for facts.** The LLM handles language
  understanding, planning, reviewing and writing. Code handles parsing,
  retrieval math, de-duplication, citation checks, PII masking and limits.
- Every external call (Gemini, Cohere, Qdrant, Postgres) goes through one
  adapter, with a timeout, retries (PROJECT_PLAN §5.5) and a fallback.
- Prompts live in `app/agent/prompts.py` with a `PROMPT_VERSION`. Every LLM
  output is validated and then sanitised by code.
- **Tests:** unit tests for pure logic, scripted fake-LLM tests for every
  graph path, API tests with a fake graph. Real-model quality is measured
  by `evals/`, never by pytest.

## Layout rules (keep them)
- `app/ingestion/` is offline only and never imported by the API.
  `parser.py` is the only module that imports Docling.
- `app/db/vector_repository.py` is the only module that talks to Qdrant.
- Settings come only from `app/core/config.py` (pydantic-settings). Nothing
  else reads environment variables.
- Where things live:
  - `app/services/`: the search engine (built and verified, change it only
    where the plan says);
  - `app/llm/`: the LLM factory, rate limiters and cache;
  - `app/agent/`: the graphs;
  - `app/api/`: FastAPI;
  - `app/db/` + `migrations/`: Postgres;
  - `ui/`: Streamlit;
  - `evals/`: evaluation;
  - `docs/`: documentation.

## Hard rules
- **Never read, print or commit `.env`.** The app loads it itself.
  Configuration examples go in `.env.example`.
- **Never** run `indexer --recreate`, `docker compose down -v`, or delete
  `app/data/interim/embeddings/` or the Qdrant volume. Re-embedding 1,219
  chunks takes two days on the free tier.
- **Never** `git push`, rewrite history, or work on `main`. Use the branch
  `feat/agentic-rag`.
- **Quota.** Gemini Lite models allow 15 RPM and 500 RPD each, and quotas
  reset at 3 pm Malaysia time. Cohere trial: 10/min and about 1,000/month.
  - Smoke checks: ≤ 30 Gemini calls per milestone, ≤ 300 per day.
  - **Never run a full evaluation yourself. Syakir runs it.** When the
    evaluation is ready (M7, M9), stop and tell him:
    - the exact command;
    - the estimated calls;
    - the quota already used today;
    - whether he should run it now or wait for the 3 pm reset.

    After he runs it, analyse the results and fix the failures.
  - Use the response cache for every repeated run. Log every real run in
    the quota ledger.
- **Models:**
  - planner, reviewer, judge: `gemini-3.1-flash-lite`;
  - generator: `gemini-3.5-flash-lite`;
  - each is the other's fallback;
  - judge cross-check only: `gemini-3.8-flash` (20/day);
  - embeddings: `gemini-embedding-001`, 768 dimensions (fixed).

## Environment
- Windows, PowerShell, Python 3.12, `.venv`. Docker Desktop runs Postgres
  (5432) and Qdrant (6333).
```powershell
.venv\Scripts\activate
pytest -q
ruff check . ; ruff format .
python -m scripts.try_search "question"     # uses 1 embed + 1 rerank
```

## Explaining to Syakir
He is a fresh CS graduate who knows basic LLM workflows: send a prompt, get
text back. He is **new to agents, tool calling, LangGraph, orchestration and
evaluation**. Everything you explain or write for him must:
- start from what he knows (one LLM call) and build up one concept at a
  time;
- point to the real file and function in this repo for every concept;
- use concrete examples from these PDFs (e-KYC 8.10 → 8.4, the 14A/14B/14C/14D
  sector copies, the FAR table);
- give small doses with clear headings: no walls of text, no unexplained
  jargon, no synonyms for the same thing;
- end each topic with 2–3 "explain it in your own words" questions, with
  model answers.

**The final deliverable (M8) includes two documents written for him:**
- `docs/BUILD_REPORT.md`: what you built and why, deviations from the
  plan, real example inputs and outputs (planner, reviewer, final answer,
  SSE events), test results, evaluation results with failure analysis,
  quota used, known limits, how to run everything.
- `docs/LEARNING_GUIDE.md`: the concept ladder (PROJECT_PLAN §11) taught
  with this codebase, and a walkthrough of one real question through every
  node with the real state at each step.

Run the **docs-reviewer** subagent on both before calling M8 done.
