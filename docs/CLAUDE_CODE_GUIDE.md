# Building this project with Claude Code

This is how you hand the build to Claude Code and stay in control: what to
set up, what to type, when to check in, and how to learn afterwards.

## What's already in the repo for Claude Code

| File | What it does |
|---|---|
| `CLAUDE.md` | Loaded automatically every session. The project, the working mode, the build loop, code standards, hard rules, quota limits, and how to explain things to you |
| `PROJECT_PLAN.md` | The design and milestones M1–M9 (§10) |
| `PROGRESS.md` | Where the build is. Claude Code updates it after every milestone, so any new session can resume |
| `docs/design_qa.md` | Why each decision was made, and the official name for every concept |
| `.claude/settings.json` | **Permissions.** Allowed without asking: tests, ruff, git commit, docker up. Asks you first: indexer, ingestion, full evaluation. Blocked: reading `.env`, `git push`, `docker compose down`, `--recreate`. Also a hook that formats every Python file after each edit |
| `.claude/agents/code-reviewer.md` | A reviewer subagent with **fresh context**, so it hasn't seen the conversation that wrote the code. It checks each milestone against the plan |
| `.claude/agents/docs-reviewer.md` | Reads the final docs as you would (a beginner) and flags jargon, gaps and wrong facts |
| `.claude/skills/build` | `/build M1-M6` runs milestones with the full loop |
| `.claude/skills/explain` | `/explain <topic>` teaches any part of the code at your level |
| `.claude/skills/status` | `/status` reports progress, tests, quota used and what's next |

## Before you start (about 15 minutes)

1. **Extract the zip** into your repo root:
   ```powershell
   cd "C:\AI Projects\bank-policy-assistant"
   Expand-Archive -Path "$HOME\Downloads\claude-code-setup.zip" -DestinationPath . -Force
   ```
2. **Commit everything** and **create the work branch:**
   ```powershell
   git status
   git add -A
   git commit -m "Add Claude Code setup and revised plan"
   git switch -c feat/agentic-rag
   ```
3. **Check that `.env` has** `GEMINI_API_KEY` and `COHERE_API_KEY`. Add
   `LANGSMITH_API_KEY` if you want tracing (free account at
   smith.langchain.com).
4. **Start Docker Desktop,** then run `docker compose up -d db qdrant` (the databases
   only; `docker compose up --build` starts the whole app in Docker).
5. **Activate the venv** (so the hook and the tests use the right Python),
   then **start Claude Code:**
   ```powershell
   .venv\Scripts\activate
   claude
   ```

## Step 1: let it review the plan first (plan mode)

Press **Shift+Tab** until the status bar shows **plan mode**. In plan mode it
can read and think, but it can't edit anything. Paste:

> Read CLAUDE.md, PROGRESS.md, PROJECT_PLAN.md and docs/design_qa.md, then
> explore the repo. Before any code: (1) check the plan against the existing
> code and tell me anything that won't fit or that you would do differently,
> and why; (2) list exactly what M1–M6 will add, change and remove;
> (3) confirm the quota plan for today. Keep it short.

Read the answer. If it suggests a change you agree with, say so, and it will
write the change into the plan. Then accept the plan.

## Step 2: let it build (M1 → M6)

Press **Shift+Tab** to switch to **accept edits** mode, or **auto** mode if
your plan offers it. Then type:

```
/build M1-M6
```

It works through each milestone: implement → tests → reviewer subagent →
fixes → smoke check → PROGRESS.md → commit. Every milestone is a commit you
can go back to.

**While it runs:**
- It will **ask** before the indexer, ingestion or a full evaluation. Answer
  those prompts.
- **Esc** stops it mid-step if it's going the wrong way. Type a correction,
  and it continues.
- **Esc Esc** (or `/rewind`) restores the code and the conversation to an
  earlier point.
- `/status` at any time shows where it is.
- **If the session gets long or hits a usage limit:** start again with
  `claude --continue`. Or run `/clear` and say *"Continue the build from
  PROGRESS.md"*. Everything it needs is in the files.

## Step 3: your checkpoints

| When | You do | Time |
|---|---|---|
| After M5 | Open `evals/golden_set.yaml` and spot-check **at least 10 items** against the PDFs: is the question clear, is the gold paragraph right? Tell it what to fix | 20 min |
| After M4–M6 | Run the app yourself (`docker compose up`, then open the Streamlit page) and try the demo questions. Report anything that feels wrong as a user | 20 min |
| When it says **"evaluation ready"** | Run the command it gives you, at the time it recommends (usually right after the 3 pm reset). Then say *"Results are in, analyse and fix"* | ~30 min run |
| After M8 | Read `docs/BUILD_REPORT.md`, then `docs/LEARNING_GUIDE.md`. Use `/explain` for anything unclear | 2–3 h |
| After the fixes | Run the **final** evaluation it gives you (M9). Then it freezes the code | ~30 min run |
| Sun evening | Slides (built from the docs), two rehearsals, a backup video | 3 h |

**Quota rules:**
- You run the full evaluations. Claude Code only does small smoke checks.
- One full run is about 575 calls. You get about 1,000 a day, and the quota
  resets at 3 pm Malaysia time, so start a run right after the reset.
- Before demo day, keep at least 250 calls unused.
- Check usage in AI Studio before every run.

## Step 4: learn it

Once the build is done, in this order:
1. `docs/BUILD_REPORT.md`: what exists and what it does, with real outputs.
2. `docs/LEARNING_GUIDE.md`: from one LLM call up to the agent, using your
   code.
3. `/explain` for each node in `app/agent/graph.py`, one at a time:
   `/explain review_evidence`, `/explain the Send fan-out`,
   `/explain the checkpointer`.
4. Ask it to quiz you: *"Ask me 10 interview questions about this project,
   one at a time, and correct my answers."*

## How AI engineers actually use Claude Code

- **Context lives in files, not in chat.** CLAUDE.md, the plan and
  PROGRESS.md mean any session can resume. Keep them updated (it does this
  itself).
- **Plan first, then build.** Plan mode for anything big. Reviewing the plan
  is cheaper than reviewing the wrong code.
- **Small verified steps.** Milestones with tests and a commit each. Never
  one giant change.
- **Tests are the feedback loop.** Claude Code is most reliable when it can
  run something that tells it whether it's right: pytest, ruff, the smoke
  check, the evaluation.
- **Independent review.** A subagent with fresh context reviews the code,
  so the author isn't grading its own work.
- **Guardrails in config, not in hope.** Permissions block the dangerous
  commands. The hook formats every file, every time.
- **You own the decisions.** Let it propose. You accept or reject, and you
  check the parts that need human judgment: the golden set, and the UI as a
  user.
- **Read the diffs on the important parts:** `app/agent/graph.py`,
  `verify_answer`, the prompts. You'll be asked about them.
