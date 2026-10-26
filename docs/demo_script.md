# Demo script

About 10 minutes, on the live app (https://bnm-policy-assistant.streamlit.app/).
Each step names what to click, what to say, and what the audience should see.
Agentic answers take 25–40 s on the free tier: talk through the step timeline
while it fills.

**Before starting (10 minutes early):**
1. Open the link: an app idle for days shows a wake-up button and takes a minute
   to start.
2. Check the sidebar says *Ready · 1,219 passages indexed*; set **Your
   institution** to Banking and **Answer mode** to Agentic.
3. Ask each demo question once, in both modes. The response cache
   (`LIVE_RESPONSE_CACHE`) then replays them in about a second if the model is
   slow during the demo. The cache is lost when the app restarts, so do this on
   the day.

## 1. The problem (30 s, the Ask page)
*"Compliance staff need answers from three BNM policy documents, with the exact
paragraph, and they need to know whether each rule is an obligation or
guidance. If the documents don't cover a question, the assistant must say so."*
Point at the three document cards.

## 2. A cross-reference: why an agent (2 min)
Click the example **"What does e-KYC paragraph 8.10 require?"**

While it runs, read the timeline:
- **Plan:** the planner sees a named paragraph and routes it to a lookup;
- **Look up:** e-KYC 8.10;
- **Review:** *"8.10 refers to the requirements in paragraph 8.4" → following
  cross-reference e-KYC para 8.4*;
- **Look up:** 8.4; **Review:** enough to answer;
- **Write**, then **Verify**. Verify may say *"Answer misses a cross-referenced
  passage: regenerating"*: the verifier saw that 8.10 depends on 8.4 and sent
  the generator back once. Then **Write** again and **Verify: citations
  verified**.

Show the answer: **MUST** from 8.4 (S) and **MAY** from 8.10 (G). Open
**Sources** and click **Open in the PDF**.

Say: *"8.10 is guidance, but it serves an obligation in 8.4. One search finds
8.10 only. The reviewer noticed the reference and fetched 8.4; the verifier made
sure the answer actually uses it."*

## 3. The same question both ways (2 min, Compare modes)
Click **"What does e-KYC paragraph 8.10 require?"** in the examples, then **Compare**.

- Traditional: about 5–7 s, 1 LLM call, only 8.10 (MAY).
- Agentic: 4–5 LLM calls, 25–40 s, both 8.4 and 8.10.

Point at the table underneath: *"Same search engine, same generator, same
verifier. The difference comes from the agent's decisions, including the extra
passages those decisions fetch, and it costs more calls and time."*

## 4. A follow-up and a sector (2 min, Ask)
**New conversation**, then:
1. *"Can a customer be onboarded without meeting face to face?"* → banking rules (AML 14A.15.x);
2. *"what about takaful operators?"*

Show: *"Applies to insurers and takaful operators"* and the **14B** paragraphs.
Say: *"The planner rewrote the follow-up from the conversation and switched the
sector. AML repeats these rules for each sector: code keeps one copy, the one
for your institution."*

## 5. Refusal and ambiguity (1 min, Compare modes)
1. **"What is the interest rate cap on credit cards?"** → agentic refuses before
   any search (1 call); traditional searches first, then says not covered.
2. **"What does para 8.2 say?"** → agentic asks which document (8.2 exists in all
   three); traditional just answers.

## 6. Privacy and audit (30 s, Ask)
Ask *"Can a bank onboard customer 900101-14-5678 without meeting face to face?"*
→ *"Masked before processing: NRIC"*. Say: *"The number never reaches the model,
the logs or the database. Every answer and every step is stored in append-only
audit tables."*

## 7. How we know it works (2 min, Evaluation)
Open the **Evaluation** page:
- 40 questions in 8 categories, both modes: agentic 100% outcome accuracy vs
  95%, citation recall 91% vs 80%;
- the cost: 2.85 vs 1.05 LLM calls, about 25 s vs 4 s;
- the retrieval chart: on this small set the stages add little on average (MRR
  ends where dense search starts). Say why they stay: exact codes like 14A.9.6
  that meaning search misses, and paraphrases that keyword search misses;
- the honest part: where agentic lost (a money-changer question the planner
  narrowed to e-KYC), that it isn't worth it for simple lookups, and the known
  FTFC labelling issue.

Mention the 335 automated tests: a scripted fake LLM runs every graph path
without any API key.

## Likely questions, with short answers

| Question | Answer | Where |
|---|---|---|
| Why LangGraph? | Typed shared state, conditional edges for the model's decisions, `Send` for parallel tools, a checkpointer for memory, per-step streaming. | `app/agent/graph.py`, [architecture §4](architecture.md) |
| Why a bounded agent, not ReAct? | A bank needs predictable, auditable behaviour: the model decides at fixed points, code enforces every limit. | `app/agent/budget.py`, `routing.py` |
| How are citations verified? | The model cites labels (E1…); code checks each label exists, the quote is in the passage, a MUST doesn't rest only on G, and a referenced passage isn't left out. | `app/agent/verification.py` |
| How do you measure quality? | A 40-question golden set, both modes, code metrics plus an LLM judge; small, so read it as direction. | [testing and evaluation](testing_and_evaluation.md) |
| What does the agent cost? | About 2.7× the LLM calls and 6× the time of traditional. | [investigation §6](agentic_rag_investigation.md#6-what-we-measured) |
| What fails today? | sector-4 (the planner narrowed the search), the FTFC good-practice boxes carrying S, a golden set not yet checked by a person, and a judge that wasn't cross-checked. | [README, known limits](../README.md#known-limits) |

## If something goes wrong
- **Slow or "high demand" errors:** the free Gemini model is overloaded; the app
  falls back to the other model and skips the overloaded one for 2 minutes. The
  questions you warmed up replay from the cache; otherwise use the screenshots
  in `docs/screenshots/`.
- **Daily limit reached:** the demo's free quota is used up; it resets at 3 pm
  Malaysia time (4 pm from November).
