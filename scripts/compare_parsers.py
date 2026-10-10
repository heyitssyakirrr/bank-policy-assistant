"""Trial: run PyMuPDF and Docling on the same page range and save both outputs.

Usage:  python -m scripts.compare_parsers <pdf_path> <first_page> <last_page>
Pages are 1-based PDF indexes (not the printed page numbers).
Outputs go to scripts/_parser_trial/ so you can open them side by side.
"""
import sys
from pathlib import Path

import pymupdf

OUT_DIR = Path("scripts/_parser_trial")


def run_pymupdf(pdf_path: str, first: int, last: int) -> str:
    parts = []
    with pymupdf.open(pdf_path) as doc:
        for number in range(first, last + 1):
            page = doc[number - 1]
            parts.append(f"\n=== PDF page {number} ===\n")
            parts.append(page.get_text("text"))
            tables = page.find_tables().tables
            parts.append(f"\n[find_tables: {len(tables)} table(s) on this page]\n")
            for i, table in enumerate(tables, start=1):
                parts.append(f"--- table {i} ---\n")
                for row in table.extract():
                    parts.append(" | ".join((cell or "").replace("\n", " ") for cell in row) + "\n")
    return "".join(parts)


def run_docling(pdf_path: str, first: int, last: int) -> str:
    # Imported here so the PyMuPDF half still works if Docling isn't installed.
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path, page_range=(first, last))
    return result.document.export_to_markdown()


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    pdf_path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem

    pymupdf_out = OUT_DIR / f"{stem}_p{first}-{last}_pymupdf.txt"
    pymupdf_out.write_text(run_pymupdf(pdf_path, first, last), encoding="utf-8")
    print(f"PyMuPDF  -> {pymupdf_out}")

    docling_out = OUT_DIR / f"{stem}_p{first}-{last}_docling.md"
    docling_out.write_text(run_docling(pdf_path, first, last), encoding="utf-8")
    print(f"Docling  -> {docling_out}")


if __name__ == "__main__":
    main()