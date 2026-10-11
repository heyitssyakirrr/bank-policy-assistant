"""Loads and validates ingest_manifest.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ManifestError(ValueError):
    """Raised for a malformed manifest, with a message that says where."""


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    code: str
    file: str
    title: str
    skip_pages: frozenset[int] = frozenset()
    tag_pages: dict[str, frozenset[int]] = field(default_factory=dict)

    def tags_for_page(self, page: int) -> tuple[str, ...]:
        return tuple(sorted(tag for tag, pages in self.tag_pages.items() if page in pages))


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
        specs.append(
            DocumentSpec(
                code=code,
                file=str(entry["file"]),
                title=str(entry["title"]),
                skip_pages=parse_page_spec(entry.get("skip_pages"), where=f"{where}.skip_pages"),
                tag_pages=tag_pages,
            )
        )
    return specs
