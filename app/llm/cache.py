"""A response cache for development and evaluation runs.

Re-running the evaluation after changing one prompt should only pay for the
calls whose input changed. Every external result (LLM answer, query
embedding, rerank scores) is stored under a hash of everything that
determines it: the namespace, the model, and the exact input (the full
prompt text, so a prompt change is a cache miss).

The API never uses it: a live user always gets a fresh answer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def cache_key(namespace: str, model: str, payload: Any) -> str:
    raw = json.dumps([namespace, model, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, json.dumps(value, default=str))
            )
            self._conn.commit()
