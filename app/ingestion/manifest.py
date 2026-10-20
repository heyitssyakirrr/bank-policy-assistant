"""Loads and validates ingest_manifest.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_LEADING_NUMBER_RE = re.compile(r"^(?:[SG]\s+)?\d{1,2}[A-Z]?(?:\.\d{1,2})+\s*")


class ManifestError(ValueError):
    """Raised for a malformed manifest, with a message that says where."""


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    code: str
    file: str
    title: str
    skip_pages: frozenset[int] = frozenset()
    tag_pages: dict[str, frozenset[int]] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)  # paragraph-number prefix -> sector, e.g. "14A" -> "banking"
    sector_phrases: dict[str, str] = field(default_factory=dict)  # opening phrase -> sector, for docs not split by part

    def tags_for_page(self, page: int) -> tuple[str, ...]:
        return tuple(sorted(tag for tag, pages in self.tag_pages.items() if page in pages))

    def sector_for(
        self, paragraph_id: str | None, section_path: tuple[str, ...] | list[str], text: str = ""
    ) -> str | None:
        """Which type of institution a chunk applies to (None = everyone).
        Matches, in order: the paragraph number ("14A.15.1"); for tables and
        text without one, the section heading ("14A CDD: Banking..."); and for
        documents not split into sector parts, how the paragraph opens
        ("S 8.11 For the money services business sector, ...")."""
        for prefix, sector in self.sectors.items():
            if paragraph_id and paragraph_id.startswith(prefix + "."):
                return sector
            if any(heading.startswith(prefix + " ") for heading in section_path):
                return sector
        if self.sector_phrases:
            opening = _LEADING_NUMBER_RE.sub("", text).lower()
            for phrase, sector in self.sector_phrases.items():
                if opening.startswith(phrase.lower()):
                    return sector
        return None


def parse_page_spec(items: list[int | str], *, where: str) -> frozenset[int]:
    """[1, "3-5"] -> {1, 3, 4, 5}"""
    pages: set[int] = set()
    for item in items or []:
        try:
            if isinstance(item, int):
                pages.add(item)
                continue
            start, _, end = str(item).partition("-")
            first, last = int(start), int(end or start)
        except ValueError as exc:
            raise ManifestError(f"{where}: bad page value {item!r}") from exc
        if first < 1 or last < first:
            raise ManifestError(f"{where}: bad page range {item!r}")
        pages.update(range(first, last + 1))
    return frozenset(pages)


def load_manifest(path: Path) -> list[DocumentSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path}: expected a non-empty 'documents' list")

    specs: list[DocumentSpec] = []
    seen_codes: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"{path.name} documents[{i}]"
        missing = [key for key in ("code", "file", "title") if not entry.get(key)]
        if missing:
            raise ManifestError(f"{where}: missing {', '.join(missing)}")
        code = str(entry["code"])
        if code in seen_codes:
            raise ManifestError(f"{where}: duplicate code {code!r}")
        seen_codes.add(code)

        tag_pages = {
            str(tag): parse_page_spec(pages, where=f"{where}.tag_pages.{tag}")
            for tag, pages in (entry.get("tag_pages") or {}).items()
        }
        sectors = entry.get("sectors") or {}
        sector_phrases = entry.get("sector_phrases") or {}
        for name, mapping in (("sectors", sectors), ("sector_phrases", sector_phrases)):
            if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
                raise ManifestError(f"{where}.{name}: expected a mapping of text -> sector name")
        specs.append(
            DocumentSpec(
                code=code,
                file=str(entry["file"]),
                title=str(entry["title"]),
                skip_pages=parse_page_spec(entry.get("skip_pages"), where=f"{where}.skip_pages"),
                tag_pages=tag_pages,
                sectors={str(prefix): sector for prefix, sector in sectors.items()},
                sector_phrases={str(phrase): sector for phrase, sector in sector_phrases.items()},
            )
        )
    return specs
