# scripts/inspect_tables.py
from pathlib import Path

import pymupdf

RAW_PDF_DIR = Path("app/data/raw_pdfs")

for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
    table_count = 0
    pages_with_tables = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            found = page.find_tables().tables
            if found:
                table_count += len(found)
                pages_with_tables.append(page_number)
    print(f"{pdf_path.name}: {table_count} tables on {len(pages_with_tables)} pages")
    print(f"  first pages: {pages_with_tables[:10]}")