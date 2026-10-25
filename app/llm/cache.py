"""A response cache for development and evaluation runs (and a public demo).

Re-running the evaluation after changing one prompt should only pay for the
calls whose input changed. Every external result (LLM answer, query
embedding, rerank scores) is stored under a hash of everything that
determines it: the namespace, the model, and the exact input (the full
prompt text, so a prompt change is a cache miss).

Live answers skip it by default, so a user gets a fresh answer. A public
demo turns it on (LIVE_RESPONSE_CACHE), so a repeated question costs no quota.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any


def cache_key(namespace: str, model: str, payload: Any) -> str:
    """'<namespace>:<hash>': the namespace ("llm", "embed_query", "rerank")
    lets the cache count real API calls (misses) per kind of call."""
    raw = json.dumps([namespace, model, payload], sort_keys=True, ensure_ascii=False, default=str)
    return f"{namespace}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)  # guarded by our own lock
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._conn.commit()
        self.hits = 0
        self.misses = 0
        self.misses_by_namespace: Counter[str] = Counter()  # each miss becomes one real API call

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                self.misses_by_namespace[key.split(":", 1)[0]] += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, json.dumps(value, default=str))
            )
            self._conn.commit()
