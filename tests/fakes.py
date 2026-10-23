"""Test doubles for the agent: a scripted LLM and an in-memory search.

The scripted LLM returns fixed outputs per role, in order, so every graph
path runs deterministically. It tests OUR logic (routing, bounds,
verification); the real model's quality is measured by evals/, not pytest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from app.llm.models import LLMResult
from app.services.retrieval_service import ParagraphLookup, RankedChunk, RetrievalResult
from app.services.search_service import SearchResult
from app.services.selection import Candidate, select_final


class ScriptedLLM:
    def __init__(self, **scripts: list[Any]) -> None:
        self._scripts = {role: list(outputs) for role, outputs in scripts.items()}
        self.calls: list[tuple[str, list[Any]]] = []  # (role, messages)

    def invoke(self, role, messages, schema):
        self.calls.append((role, list(messages)))
        if not self._scripts.get(role):
            raise AssertionError(f"unexpected {role} call")
        output = self._scripts[role].pop(0)
        if isinstance(output, Exception):
            raise output
        assert isinstance(output, schema), f"{role} script returned {type(output).__name__}, not {schema.__name__}"
        return LLMResult(output, f"fake-{role}", 1, 100, 20, 1.0)

    def roles(self) -> list[str]:
        return [role for role, _ in self.calls]


def payload(
    pid: str,
    text: str,
    doc: str = "ekyc",
    provision: str | None = "S",
    sector: str | None = None,
    page: int = 6,
) -> dict[str, Any]:
    marker = f" ({provision})" if provision else ""
    return {
        "doc_code": doc,
        "paragraph_id": pid,
        "provision": provision,
        "sector": sector,
        "text": text,
        "embed_text": text,
        "footnotes": [],
        "page_start": page,
        "citation": f"{doc} para {pid}{marker}, p. {page}",
    }


def candidate(chunk_id: str, data: dict[str, Any], rank: int = 1, exact: bool = False) -> Candidate:
    return Candidate(RankedChunk(chunk_id, data, 0.0, None, None, exact_match=exact), rank, 0.9)


class FakeSearch:
    """Search over a fixed corpus: {chunk_id: payload}. A query matches the
    chunks whose text contains any of its words longer than 3 letters."""

    def __init__(self, corpus: dict[str, dict[str, Any]], top_n: int = 5) -> None:
        self.corpus = corpus
        self.top_n = top_n
        self.searches: list[dict[str, Any]] = []
        self.lookups: list[list[ParagraphLookup]] = []

    def search(
        self,
        question: str,
        sector: str | None,
        documents: Sequence[str] = (),
        paragraph_refs: Sequence[ParagraphLookup] | None = None,
    ) -> SearchResult:
        self.searches.append({"query": question, "sector": sector, "documents": list(documents)})
        words = {w for w in question.lower().split() if len(w) > 3}
        hits = [
            candidate(cid, data, rank)
            for rank, (cid, data) in enumerate(self.corpus.items(), start=1)
            if (not documents or data["doc_code"] in documents) and words & set(data["text"].lower().split())
        ]
        selection = select_final(hits, self.top_n, sector)
        retrieval = RetrievalResult(question, [c.chunk for c in hits], [], [], [], {"embed": 1.0, "search": 1.0})
        return SearchResult(question, selection.final, hits, selection, retrieval, sector, list(documents), True, {})

    def lookup(self, refs: Sequence[ParagraphLookup]) -> list[Candidate]:
        self.lookups.append(list(refs))
        found: dict[str, Candidate] = {}
        wanted = defaultdict(set)
        for ref in refs:
            wanted[ref.paragraph_id].add(ref.document)
        for rank, (cid, data) in enumerate(self.corpus.items(), start=1):
            docs = wanted.get(data["paragraph_id"])
            if docs is not None and (None in docs or data["doc_code"] in docs):
                found[cid] = candidate(cid, data, rank, exact=True)
        return list(found.values())
