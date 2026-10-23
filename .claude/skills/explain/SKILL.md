---
name: explain
description: Teach a concept or a part of this codebase to Syakir at beginner level, using the real code and the real PDFs.
argument-hint: "<topic, file or question>"
---

Explain **$ARGUMENTS** to Syakir, following the "Explaining to Syakir"
section of CLAUDE.md:
- Start from one LLM call, which he already understands. Add one idea at a
  time.
- Show the real file and function in this repo, quoting only the few lines
  that matter.
- Walk through one concrete example from these PDFs (e-KYC 8.10 → 8.4, the
  AML 14A/14B/14C/14D sector copies, the FAR table), showing the real
  inputs and outputs.
- Use the project's names for things (docs/design_qa.md), and never a
  synonym.
- Keep it short, with headings. Offer to go deeper instead of dumping
  everything.
- End with 2–3 "explain it in your own words" questions, plus short model
  answers he can check himself against.
