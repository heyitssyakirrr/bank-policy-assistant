"""Cheap safety net: did the parser silently drop or duplicate text?

For every page we compare the words Docling gave us with the words in the
PDF's raw text layer (read by PyMuPDF, which is fast and exact but has no
structure). A big gap means a passage was lost or scrambled; a big excess
usually means a table with merged cells was duplicated. Flagged pages are
listed in the report so a human looks at them, instead of discovering the
problem when the bot gives a wrong answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.models import BlockKind, ParsedDocument

logger = logging.getLogger("app.ingestion.quality")

WORD_RE = re.compile(r"\w+")  # counts words, ignores markdown pipes and punctuation
LOW_RATIO, HIGH_RATIO = 0.85, 1.25
MIN_WORDS_TO_CHECK = 40  # near-empty pages give meaningless ratios


@dataclass(frozen=True, slots=True)
class PageCoverage:
    page: int
    parser_words: int
    reference_words: int

    @property
    def ratio(self) -> float:
        return self.parser_words / self.reference_words if self.reference_words else 1.0

    @property
    def flagged(self) -> bool:
        return self.reference_words >= MIN_WORDS_TO_CHECK and not (LOW_RATIO <= self.ratio <= HIGH_RATIO)


def check_coverage(parsed: ParsedDocument, pdf_path: Path) -> list[PageCoverage] | None:
    """Returns per-page coverage, or None if PyMuPDF isn't installed."""
    try:
        import pymupdf
    except ImportError:
        logger.warning("pymupdf not installed: skipping the coverage check")
        return None

    parser_words: dict[int, int] = {}
    for block in parsed.blocks:
        if block.kind is BlockKind.OTHER:
            continue
        text = " ".join(" ".join(row) for row in block.table_rows) if block.table_rows else block.full_text
        parser_words[block.page] = parser_words.get(block.page, 0) + len(WORD_RE.findall(text))

    coverage: list[PageCoverage] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            raw = len(WORD_RE.findall(page.get_text("text")))
            # Headers/footers are in the raw text but deliberately not in ours.
            furniture = parsed.furniture_words.get(page_no, 0)
            coverage.append(PageCoverage(page_no, parser_words.get(page_no, 0), max(raw - furniture, 0)))
    return coverage
