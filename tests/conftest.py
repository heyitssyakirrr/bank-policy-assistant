"""Test-wide guard: tests never send anything to LangSmith, even when the
developer's .env turns tracing on (environment variables override .env)."""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
