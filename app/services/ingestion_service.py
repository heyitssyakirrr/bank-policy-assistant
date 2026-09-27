import re
from collections import Counter
from dataclasses import dataclass

import pymupdf


@dataclass
class Chunk:
    document: str
    page: int
    content: str


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_number, text))
    return pages


def _normalize_word(word: str) -> str:
    """Digits vary per page (page numbers) — normalize so a repeated
    header is still recognized as identical across pages."""
    return "#" if re.fullmatch(r"\d+", word) else word


def detect_header_length(
    pages: list[tuple[int, str]], max_words: int = 40, threshold: float = 0.5
) -> int:
    """Grow a matched leading-word prefix across pages (digits
    normalized) until it stops being consistent. Returns how many
    leading words to strip — 0 if no header is detected. Works for
    headers of any length, and tolerates a page number anywhere
    within the header text, not just at the very start."""
    word_lists = [
        [_normalize_word(w) for w in text.split()[:max_words]] for _, text in pages
    ]
    header_len = 0
    for n in range(1, max_words + 1):
        prefixes = Counter(
            tuple(words[:n]) for words in word_lists if len(words) >= n
        )
        if not prefixes:
            break
        _, count = prefixes.most_common(1)[0]
        if count / len(pages) >= threshold:
            header_len = n
        else:
            break
    return header_len


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    header_len = detect_header_length(pages)
    if header_len == 0:
        return pages

    word_lists = [
        [_normalize_word(w) for w in text.split()[:header_len]] for _, text in pages
    ]
    prefixes = Counter(tuple(words) for words in word_lists if len(words) == header_len)
    header_pattern, _ = prefixes.most_common(1)[0]

    cleaned = []
    for page_number, text in pages:
        words = text.split()
        normalized_prefix = tuple(_normalize_word(w) for w in words[:header_len])
        if normalized_prefix == header_pattern:
            words = words[header_len:]
        cleaned.append((page_number, " ".join(words)))
    return cleaned


def chunk_pages(
    document_name: str,
    pages: list[tuple[int, str]],
    chunk_words: int = 350,
    overlap_words: int = 50,
) -> list[Chunk]:
    chunks = []
    for page_number, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_words
            content = " ".join(words[start:end])
            chunks.append(Chunk(document=document_name, page=page_number, content=content))
            if end >= len(words):
                break
            start = end - overlap_words
    return chunks