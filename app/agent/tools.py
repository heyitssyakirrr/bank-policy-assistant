"""The tools: code the agent calls to fetch evidence. Both are read-only, so
a prompt injection can at worst produce a bad answer, never an action.

    search_policies    one SearchRequest -> hybrid search + rerank + selection
    lookup_paragraphs  named paragraphs -> fetched directly by number and
                       document; if any is not found, the question is
                       searched in those documents instead

Each runs as its own graph branch: `Send` starts one per search, in
parallel, and the evidence reducer merges what they return. A branch that
fails (Qdrant timeout) reports the failure and returns no evidence, so the
other branches' evidence still reaches the answer.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.runtime import Runtime

from app.agent.evidence import to_evidence
from app.agent.nodes import search_record
from app.agent.schemas import ParagraphRef, SearchRequest
from app.agent.state import AgentContext
from app.services.retrieval_service import ParagraphLookup
from app.services.selection import Candidate

logger = logging.getLogger("app.agent")


class SearchTask(TypedDict):
    request: SearchRequest
    sector: str | None  # used when the request names no sector


class LookupTask(TypedDict):
    refs: list[ParagraphRef]
    sector: str | None
    question: str  # searched in the named documents if a paragraph isn't found


def _failed(kind: str, query: str, exc: Exception) -> dict[str, Any]:
    logger.error(f"{kind} '{query}' failed: {type(exc).__name__}: {exc}")
    return {
        "search_log": [{"kind": kind, "query": query, "chunks": [], "error": type(exc).__name__}],
        "notes": [f"The {kind} for '{query}' failed; answered with the other evidence."],
    }


def search_policies(task: SearchTask, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    request = task["request"]
    try:
        result = runtime.context.search.search(
            request.query,
            sector=request.sector or task["sector"],
            documents=request.documents,
            paragraph_refs=[],  # paragraphs are looked up by lookup_paragraphs, with their document
        )
    except Exception as exc:  # one failed branch must not lose the others' evidence
        return _failed("search", request.query, exc)
    evidence = to_evidence(result.final, "search", request.query, requested_sector=request.sector)
    return {"evidence": evidence, "search_log": [search_record(result)]}


def _found(ref: ParagraphLookup, candidates: list[Candidate]) -> bool:
    return any(
        c.chunk.payload.get("paragraph_id") == ref.paragraph_id
        and ref.document in (None, c.chunk.payload.get("doc_code"))
        for c in candidates
    )


def _label(ref: ParagraphLookup) -> str:
    return f"{ref.document}:{ref.paragraph_id}" if ref.document else ref.paragraph_id


def lookup_paragraphs(task: LookupTask, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    search = runtime.context.search
    refs = [ParagraphLookup(r.document, r.paragraph_id.upper()) for r in task["refs"]]
    query = ", ".join(_label(ref) for ref in refs)
    try:
        found = search.lookup(refs)
    except Exception as exc:
        return _failed("lookup", query, exc)
    missing = [ref for ref in refs if not _found(ref, found)]
    evidence = to_evidence(found, "lookup", query)
    log: list[dict[str, Any]] = [
        {
            "kind": "lookup",
            "query": query,
            "chunks": [{"id": c.id, "citation": c.chunk.citation} for c in found],
            "missing": [_label(ref) for ref in missing],
        }
    ]
    if not missing:
        return {"evidence": evidence, "search_log": log}
    # A number the index doesn't have (or the LLM got wrong): search for the
    # question itself in the documents named. A bare "paragraph 99.9" query
    # would match nothing meaningful.
    documents = sorted({ref.document for ref in missing if ref.document})
    try:
        result = search.search(task["question"], sector=task["sector"], documents=documents, paragraph_refs=[])
    except Exception as exc:
        failed = _failed("search", task["question"], exc)
        return {"evidence": evidence, "search_log": log + failed["search_log"], "notes": failed["notes"]}
    evidence += to_evidence(result.final, "search", task["question"])
    return {"evidence": evidence, "search_log": [*log, search_record(result)]}
