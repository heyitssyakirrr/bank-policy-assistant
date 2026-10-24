"""The golden set: loading, validating, and printing it for review.

    python -m evals.golden --check          every gold paragraph exists in chunks.jsonl
    python -m evals.golden --show           each question with its gold paragraphs' text
    python -m evals.golden --show fact-6    one item

A gold label is a paragraph key: "ekyc:8.4" for the main body, or
"ekyc:Appendix 2:13" for an appendix (appendices reuse paragraph numbers).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.agent.schemas import Sector
from app.core.config import settings

GOLDEN_PATH = Path("evals/golden_set.yaml")
Category = Literal[
    "direct_lookup",
    "paraphrased",
    "comparison",
    "cross_reference",
    "sector_specific",
    "follow_up",
    "out_of_scope",
    "ambiguous",
]


class GoldenItem(BaseModel):
    id: str
    category: Category
    question: str
    expected_paragraphs: list[str] = Field(default_factory=list)
    expected_outcome: Literal["answer", "refuse", "clarify"]
    sector: Sector | None = "banking"
    previous_turns: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.expected_outcome == "answer"


def paragraph_key(payload: dict[str, Any]) -> str:
    """The gold-label form of a chunk (search hit payload, evidence or chunks.jsonl row)."""
    document = payload.get("doc_code") or payload.get("document")
    scope = payload.get("scope", "main")
    paragraph = payload.get("paragraph_id") or "-"
    return f"{document}:{paragraph}" if scope == "main" else f"{document}:{scope}:{paragraph}"


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = [GoldenItem.model_validate(raw) for raw in yaml.safe_load(path.read_text(encoding="utf-8"))]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden item ids")
    return items


@lru_cache(maxsize=1)
def chunk_texts(chunks_path: Path = settings.chunks_path) -> dict[str, list[str]]:
    """Every paragraph key -> the texts of its chunks (a long paragraph can have several parts)."""
    texts: dict[str, list[str]] = defaultdict(list)
    handle = (
        chunks_path.open(encoding="utf-8")
        if chunks_path.exists()
        else gzip.open(chunks_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8")  # a clean clone: the gzip
    )
    with handle:
        for line in handle:
            row = json.loads(line)
            texts[paragraph_key(row)].append(row["text"])
    return dict(texts)


def missing_labels(items: list[GoldenItem], known: dict[str, list[str]]) -> list[str]:
    return [f"{item.id}: {key}" for item in items for key in item.expected_paragraphs if key not in known]


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--show", nargs="*", metavar="ID")
    args = cli.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    items = load_golden()
    known = chunk_texts()

    missing = missing_labels(items, known)
    print(
        f"{len(items)} items; {sum(i.answerable for i in items)} answerable; missing gold labels: {missing or 'none'}"
    )
    if args.show is not None:
        for item in items:
            if args.show and item.id not in args.show:
                continue
            turns = " -> ".join([*item.previous_turns, item.question])
            print(f"\n[{item.id}] ({item.category}, expect {item.expected_outcome}) {turns}")
            for key in item.expected_paragraphs:
                for text in known.get(key, ["(MISSING)"]):
                    print(f"  {key}: {text[:400]}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
