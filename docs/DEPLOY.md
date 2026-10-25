# Deploying the demo to Streamlit Community Cloud

The public link runs the UI and the agent in one Streamlit process
(`UI_BACKEND=local`), with the index in **Qdrant Cloud** and the conversation
memory and audit log in **Neon** (Postgres). Locally and in Docker the UI still
calls the FastAPI service (`UI_BACKEND=api`). Why: PROJECT_PLAN.md §5.7
"Public deployment".

```
Browser ──► Streamlit Cloud: ui/app.py ──► TurnRunner ──► agentic / traditional graph
                                              ├─► Qdrant Cloud   (1,219 passages)
                                              ├─► Gemini, Cohere (your keys, from Secrets)
                                              └─► Neon Postgres  (memory + audit log)
```

Never paste a key into a chat, an issue or a committed file. Keys live only in
your local `.env` and in Streamlit's **Secrets** page.

## 1. Create the accounts (free)

1. **Qdrant Cloud** (cloud.qdrant.io): create a free cluster, then copy its
   **URL** (`https://….cloud.qdrant.io:6333`) and create an **API key**.
2. **Neon** (neon.tech): create a project (region: Singapore if offered).
   Copy the connection string with **connection pooling turned off** (the
   direct connection: LangGraph's checkpointer runs DDL that a pooler can
   refuse). Change its start from `postgresql://` to
   `postgresql+psycopg://` and keep `?sslmode=require` at the end.
3. **Streamlit Community Cloud** (share.streamlit.io): sign in with GitHub.

## 2. Copy the index into Qdrant Cloud (0 Gemini calls)

In PowerShell, in the project folder with the venv active. The variables
apply to this terminal only and override `.env`:

```powershell
$env:QDRANT_URL = "https://<your-cluster>.cloud.qdrant.io:6333"
$env:QDRANT_API_KEY = "<your Qdrant API key>"
python -m app.ingestion.indexer --require-cached
```

It reads `app/data/processed/chunks.jsonl` and the embedding cache, and
refuses to call Gemini (`--require-cached`). Expect `upserted: 1219` and
`points in 'policy_chunks': 1219`. Close the terminal afterwards so the
cloud address isn't used by accident.

The Neon tables are created by the app itself on its first start
(`alembic upgrade head`, then LangGraph's own checkpoint tables).

## 3. Push the branch

```powershell
git push -u origin feat/agentic-rag
```

## 4. Create the app

On share.streamlit.io: **Create app** → deploy from a GitHub repository:

| Field | Value |
|---|---|
| Repository | your repo |
| Branch | `feat/agentic-rag` |
| Main file path | `ui/app.py` |
| App URL | e.g. `bnm-policy-assistant` |
| Advanced settings → Python | 3.12 |
| Advanced settings → Secrets | the block below, with your values |

```toml
GEMINI_API_KEY = "<your Gemini key>"
COHERE_API_KEY = "<your Cohere key>"
QDRANT_URL = "https://<your-cluster>.cloud.qdrant.io:6333"
QDRANT_API_KEY = "<your Qdrant API key>"
DATABASE_URL = "postgresql+psycopg://<user>:<password>@<host>.neon.tech/<db>?sslmode=require"
UI_BACKEND = "local"
LIVE_RESPONSE_CACHE = "true"
DAILY_QUESTION_LIMIT = "80"
QUESTIONS_PER_SESSION = "20"
```

Streamlit passes top-level secrets to the app as environment variables, and
`app/core/config.py` reads them like `.env`.

**The limits:** an agentic question costs about 3 calls of
`gemini-3.1-flash-lite` (planner and reviews, 500 a day) and 1–2 of
`gemini-3.5-flash-lite` (the generator). 80 questions a day leaves room for an
evaluation run on the same quota day. Raise `DAILY_QUESTION_LIMIT` after
the evaluation is done. A question asked before is answered from the cache
for free, though it still counts towards both limits.
The per-visitor limit counts questions in one browser session: a reload
starts a new session. The daily limit resets at midnight Pacific time, when
Gemini's quota does (3 pm Malaysia time, 4 pm from 1 November).

## 5. Check the live app

The first start takes a few minutes (installing, then building the stack).
Then, on the live link:

1. The sidebar says **Ready · 1,219 passages indexed**.
2. Ask "What does e-KYC paragraph 8.10 require?" in agentic mode: the steps
   show *Review … Following cross-reference e-KYC para 8.4*, and the answer
   has a MUST from 8.4 and a MAY from 8.10.
3. Open a source with **Open in the PDF**: the PDF opens at the page.
4. Ask a follow-up ("what about takaful operators?") in the same
   conversation.
5. **Compare modes** with "What does para 8.2 say?": agentic asks which
   document, traditional answers.

If something fails, open **Manage app → Logs** on Streamlit Cloud.

## 6. Before sending the link

- Open it yourself shortly before sending: an app idle for several days is
  put to sleep and shows a wake-up button.
- Send two links: the app and the GitHub repository.
