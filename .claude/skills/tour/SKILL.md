---
name: tour
description: Teach Syakir the whole project before the demo, from one LLM call up to the agent, then the built system (docs/HANDOFF.md Part A), with the real code, free commands and check-yourself questions.
argument-hint: "[empty = from the start | ladder N | A1..A10]"
---

Teach Syakir the project. Start at **$ARGUMENTS**; if empty, start at
step 1 of the concept ladder. He is new to agents, tool calling, LangGraph,
orchestration and evaluation, so go slowly and follow CLAUDE.md
"Explaining to Syakir" exactly.

**Stage 1: the concept ladder** (PROJECT_PLAN.md §11, 13 steps). One step at
a time, in order: an LLM call → structured output → embeddings and vector
search → hybrid retrieval and reranking → RAG → tool use → workflows vs
agents → agentic patterns → LangGraph → memory → serving → evaluation →
observability and ops. For each step:
1. Explain the idea in plain words, building on the previous step.
2. Show where it lives in this repo: open the real file and quote only the
   few lines that matter (the §11 table and docs/HANDOFF.md A3 name them).
3. Show a real input and output from these PDFs (e-KYC 8.10 → 8.4, the FAR
   table, the AML 14A/14B/14C/14D copies, the NRIC masking). Prefer free
   ways to show it: cached `python -m scripts.ask` replays, `pytest -q`,
   `python -m evals.golden --show`, SQL on the audit tables, reading
   `evals/results/*/report.md`.
4. Ask 2–3 "explain it in your own words" questions. Wait for his answers,
   then correct gently with short model answers.
5. Ask whether to go deeper or move on. Never continue until he says so.

**Stage 2: the built system** (docs/HANDOFF.md Part A, A1 to A10): where the
build stands, what happened during the automated build, the architecture,
one question end to end, the design decisions, tests, evaluation, his
hands-on plan, known limits. Same rhythm as stage 1.

**Stage 3: demo readiness.** Run a mock assessor Q&A from PROJECT_PLAN §1
(the assessment requirements), HANDOFF A5, A7 and A9. Ask one question at
a time and give feedback on each answer.

Rules: keep each message short, with headings; use the project's names for
things (docs/design_qa.md); never run anything that spends Gemini or Cohere
quota without saying the cost first and getting his yes; if he seems lost,
step back one rung instead of adding more.
