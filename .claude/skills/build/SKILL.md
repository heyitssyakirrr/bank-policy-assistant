---
name: build
description: Build the next milestone(s) of PROJECT_PLAN.md §10 with the full implement-test-review-commit loop.
argument-hint: "[M1 | M1-M6 | next]"
disable-model-invocation: true
---

Build milestone(s) **$ARGUMENTS** from PROJECT_PLAN.md §10. If no argument
was given, build the next unchecked milestone in PROGRESS.md and keep going
through the following ones, until M6 is done or a "stop and ask" condition
in CLAUDE.md occurs.

For each milestone, follow the "Every milestone runs this loop" section of
CLAUDE.md exactly:
1. Plan: list the files you will add, change or remove.
2. Implement, with tests alongside the code.
3. `ruff check .` + `ruff format --check .` + `pytest -q`, all green.
4. Use the code-reviewer subagent on the milestone's changes.
5. Fix every finding.
6. Real-API smoke check within the quota budget; log it in the quota
   ledger.
7. Update PROGRESS.md.
8. Commit.

When a milestone is done, print a short summary:
- what was built;
- test count;
- the reviewer's verdict and what you fixed;
- quota used;
- anything that deviated from the plan, and why.
