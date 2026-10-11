"""Human-readable dumps of every stage, for debugging.

For each document, three files you can open side by side:

  <code>.1_blocks.txt     what the PARSER produced (full text, reading order)
  <code>.2_structure.txt  what the RULES decided for each block, and why
  <code>.3_chunks.txt     the final chunks, exactly as they will be embedded

If an answer is wrong later, you trace it backwards: chunk -> the blocks
it came from (block numbers are shown in every file) -> the PDF page.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from app.ingestion.models import AnnotatedBlock, Chunk, ParsedDocument
from app.ingestion.quality import PageCoverage

RULE = "=" * 100


def write_blocks(path: Path, parsed: ParsedDocument) -> None:
    lines = [
        f"PARSER OUTPUT: {parsed.source_file} ({parsed.page_count} pages, {len(parsed.blocks)} blocks)",
        f"Printed page numbers found on {len(parsed.printed_pages)} pages (from headers/footers, which are not indexed)",
        "Format: #block pPDFpage [printed page] kind marker y=position | text",
        RULE,
    ]
    for b in parsed.blocks:
        printed = parsed.printed_pages.get(b.page, "-")
        marker = f" marker={b.marker!r}" if b.marker else ""
        top = f" y={b.top:.0f}" if b.top is not None else ""  # distance from page top: same y = same line
        head = f"#{b.index:04d} p{b.page:<3} [{printed:>3}] {b.kind.value:<9}{marker}{top}"
        if b.table_rows:
            caption = f" caption={b.caption!r}" if b.caption else ""
            lines.append(f"{head}{caption} ({len(b.table_rows)} rows)\n{b.text}")
        else:
            lines.append(f"{head} | {b.text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure(path: Path, annotated: list[AnnotatedBlock]) -> None:
    lines = [
        "STRUCTURE: the role our rules gave each block, the state after it, and which rule fired",
        "Format: #block pPage ROLE para=id S/G scope | section path | text (first 160 chars)",
        RULE,
    ]
    last_path: tuple[str, ...] | None = None
    for ab in annotated:
        if ab.section_path != last_path:
            lines.append(f"\n  >> SECTION: {' > '.join(ab.section_path) or '(none)'}")
            last_path = ab.section_path
        text = ab.block.full_text.replace("\n", " ")
        text = text if len(text) <= 160 else text[:160] + "…"
        para = f"para={ab.paragraph_id}" if ab.paragraph_id else "para=-"
        tags = f" tags={','.join(ab.tags)}" if ab.tags else ""
        lines.append(
            f"#{ab.block.index:04d} p{ab.block.page:<3} {ab.role.value.upper():<16} {para:<14} "
            f"{ab.provision or ' '} {ab.scope:<12}{tags} | {text}\n{'':>30}rule: {ab.reason}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    lines = [f"CHUNKS: {len(chunks)} (embed_text is exactly what gets embedded and BM25-indexed)", RULE]
    for i, c in enumerate(chunks, start=1):
        pages = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
        lines += [
            f"[{i}] {c.citation()}",
            f"    key={c.chunk_key}  id={c.chunk_id}",
            f"    type={c.chunk_type.value}  pdf_pages={pages}  words={c.word_count}  "
            f"part={c.part_index}/{c.part_count}  tags={c.tags or '-'}  blocks={c.block_indexes}",
            "-" * 100,
            c.embed_text,
            RULE,
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_lines(
    parsed: ParsedDocument,
    annotated: list[AnnotatedBlock],
    chunks: list[Chunk],
    stats: dict,
    coverage: list[PageCoverage] | None,
    skip_pages: frozenset[int] = frozenset(),
) -> list[str]:
    roles = Counter(ab.role.value for ab in annotated)
    types = Counter(c.chunk_type.value for c in chunks)
    words = [c.word_count for c in chunks] or [0]
    content_pages = {b.page for b in parsed.blocks}
    missing_printed = sorted(content_pages - set(parsed.printed_pages))

    lines = [
        f"## {parsed.doc_code} ({parsed.source_file})",
        f"pages={parsed.page_count} blocks={len(parsed.blocks)} chunks={len(chunks)}",
        f"block roles: {dict(roles)}",
        f"chunk types: {dict(types)}",
        f"chunk words: min={min(words)} median={int(statistics.median(words))} max={max(words)}",
        f"tagged chunks: {dict(Counter(tag for c in chunks for tag in c.tags))}",
        f"units split for size: {stats['split_units']}",
        "S/G markers split from their paragraph: "
        f"{sum('matched by position' in ab.reason for ab in annotated)} matched by position, "
        f"{sum('no position match' in ab.reason for ab in annotated)} by reading order (less reliable)",
        f"tables re-read as numbered paragraphs (rows): {sum('table row re-read' in ab.reason for ab in annotated)}",
        f"pages with no printed page number: {missing_printed or 'none'}",
    ]
    if stats["duplicate_keys"]:
        lines.append(f"WARNING duplicate paragraph keys (check structure file): {stats['duplicate_keys']}")
    if stats["restarted_numbering"]:
        lines.append(f"appendix numbering restarts (expected, keys suffixed ~n): {len(stats['restarted_numbering'])}")
    if stats["unreferenced_footnotes"]:
        lines.append(f"footnotes not matched to a chunk (kept as own chunks): {len(stats['unreferenced_footnotes'])}")
        lines += [f"    - {text}" for text in stats["unreferenced_footnotes"]]
    if coverage is None:
        lines.append("coverage check: skipped (pymupdf not installed)")
    else:
        # Pages the manifest skips (cover, contents) are excluded on purpose.
        flagged = [c for c in coverage if c.flagged and c.page not in skip_pages]
        lines.append(f"coverage check: {len(flagged)} of {len(coverage)} pages flagged")
        lines += [
            f"    - p{c.page}: parser {c.parser_words} words vs raw text {c.reference_words} ({c.ratio:.0%})"
            for c in flagged
        ]
    return lines
