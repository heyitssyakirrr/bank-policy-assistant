"""Offline ingestion: PDFs -> chunks.jsonl (+ debug files).

    python -m app.ingestion.pipeline                 # all documents in the manifest
    python -m app.ingestion.pipeline --doc ekyc      # one document (fast iteration)
    python -m app.ingestion.pipeline --reparse       # ignore the Docling cache

Output:
    app/data/processed/chunks.jsonl   one chunk per line; the input of the embedding step
    app/data/debug/<code>.1_blocks.txt / .2_structure.txt / .3_chunks.txt
    app/data/debug/report.txt         counts, warnings, pages to eyeball

This step makes NO API calls. Embedding is a separate step that reads
chunks.jsonl, so we can re-embed (new model, new dimensions) without
re-parsing, and re-chunk (Phase 2 experiments) without touching Qdrant
until we're happy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import setup_logging

from app.core.config import settings
from app.ingestion.chunker import ChunkingConfig, build_chunks
from app.ingestion.debug import (
    report_lines,
    write_blocks,
    write_chunks,
    write_structure,
)
from app.ingestion.manifest import DocumentSpec, load_manifest
from app.ingestion.models import Chunk
from app.ingestion.parser import DoclingParser
from app.ingestion.quality import check_coverage
from app.ingestion.structure import annotate

logger = logging.getLogger("app.ingestion.pipeline")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # app/data, wherever you run from


@dataclass(frozen=True, slots=True)
class IngestionPaths:
    manifest: Path = DATA_DIR / "ingest_manifest.yaml"
    raw_pdfs: Path = DATA_DIR / "raw_pdfs"
    docling_cache: Path = DATA_DIR / "interim" / "docling"
    chunks_out: Path = DATA_DIR / "processed" / "chunks.jsonl"
    debug_dir: Path = DATA_DIR / "debug"


def ingest_document(
    spec: DocumentSpec, parser: DoclingParser, paths: IngestionPaths, config: ChunkingConfig, debug: bool
) -> tuple[list[Chunk], list[str]]:
    started = time.perf_counter()
    pdf_path = paths.raw_pdfs / spec.file

    parsed = parser.parse(pdf_path, doc_code=spec.code)
    annotated = annotate(parsed.blocks, spec)
    chunks, stats = build_chunks(parsed, annotated, spec, config)
    coverage = check_coverage(parsed, pdf_path)

    if debug:
        paths.debug_dir.mkdir(parents=True, exist_ok=True)
        write_blocks(paths.debug_dir / f"{spec.code}.1_blocks.txt", parsed)
        write_structure(paths.debug_dir / f"{spec.code}.2_structure.txt", annotated)
        write_chunks(paths.debug_dir / f"{spec.code}.3_chunks.txt", chunks)

    logger.info(
        f"{spec.code}: {len(parsed.blocks)} blocks -> {len(chunks)} chunks in {time.perf_counter() - started:.1f}s"
    )
    return chunks, report_lines(parsed, annotated, chunks, stats, coverage)


def write_jsonl_atomically(path: Path, chunks: list[Chunk]) -> None:
    """Write to a temp file, then rename. A crash mid-write never leaves a
    half-written chunks.jsonl for the embedding step to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run(only: list[str] | None, reparse: bool, debug: bool, paths: IngestionPaths | None = None) -> list[Chunk]:
    paths = paths or IngestionPaths()
    specs = load_manifest(paths.manifest)
    if only:
        unknown = set(only) - {s.code for s in specs}
        if unknown:
            raise SystemExit(f"Unknown document code(s): {sorted(unknown)}. Known: {[s.code for s in specs]}")
        specs = [s for s in specs if s.code in only]

    parser = DoclingParser(paths.docling_cache, use_cache=not reparse)
    config = ChunkingConfig()
    all_chunks: list[Chunk] = []
    report: list[str] = []
    for spec in specs:
        chunks, lines = ingest_document(spec, parser, paths, config, debug)
        all_chunks.extend(chunks)
        report += lines + [""]

    # A partial run (--doc) must not overwrite the full corpus file.
    out = paths.chunks_out if not only else paths.chunks_out.with_name(f"chunks.{'_'.join(sorted(only))}.jsonl")
    write_jsonl_atomically(out, all_chunks)
    report.append(f"TOTAL chunks: {len(all_chunks)} -> {out}")

    if debug:
        (paths.debug_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return all_chunks


def main() -> None:
    cli = argparse.ArgumentParser(description="Parse and chunk the policy PDFs (no API calls).")
    cli.add_argument("--doc", action="append", help="document code from the manifest; repeatable")
    cli.add_argument("--reparse", action="store_true", help="ignore the cached Docling parse")
    cli.add_argument("--no-debug", action="store_true", help="skip writing the debug .txt files")
    args = cli.parse_args()

    setup_logging(settings.log_level)
    run(only=args.doc, reparse=args.reparse, debug=not args.no_debug)


if __name__ == "__main__":
    main()
