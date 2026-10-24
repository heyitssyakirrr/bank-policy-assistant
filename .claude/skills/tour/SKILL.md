---
name: tour
description: Walk Syakir through the built system before the demo, one section of docs/HANDOFF.md Part A at a time, with the real code, free commands and check-yourself questions.
argument-hint: "[section, e.g. A4, or empty to start at A1]"
---

Guide Syakir through **docs/HANDOFF.md Part A**, starting at section
**$ARGUMENTS** (A1 if empty). Follow CLAUDE.md "Explaining to Syakir".

For each section:
1. Summarise it in a few sentences, starting from what he already knows
   (one LLM call).
2. Open the real files it names and show only the few lines that matter.
3. Run the free commands it mentions with him (pytest, `--estimate`,
   cached `scripts.ask` replays, `evals.golden --show`, SQL on the audit
   tables). Before anything that spends Gemini or Cohere quota, say how many
   calls it costs and ask first.
4. Ask 2–3 "explain it in your own words" questions and wait for his
   answers. Correct gently, using the model answers in the doc.
5. Ask whether to go deeper or move on. Don't continue to the next section
   until he says so.

Keep each step short. At the end, offer a mock assessor Q&A based on
sections A5, A7 and A9.
