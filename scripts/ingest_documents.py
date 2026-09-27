"""Step 1: extract + chunk only. No API calls yet — inspect the output
here before spending any embedding budget on it."""

from pathlib import Path

from app.services.ingestion_service import chunk_pages, clean_pages, detect_header_length, extract_pages

RAW_PDF_DIR = Path("app/data/raw_pdfs")

DOCUMENTS = {
    "pd-ekyc-apr2024.pdf": "e-KYC Policy",
    "pd-ftfc-mar24.pdf": "Fair Treatment of Financial Consumers Policy",
    "PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf": "AML/CFT/CPF/TFS Policy for FIs",
}


def main():
    all_chunks = []
    for filename, label in DOCUMENTS.items():
        path = RAW_PDF_DIR / filename
        pages = extract_pages(str(path))
        header_len = detect_header_length(pages)

        if header_len:
            sample_page = pages[len(pages) // 2][1]  # a mid-document page, safer than page 1
            header_preview = " ".join(sample_page.split()[:header_len])
        else:
            header_preview = "(none detected)"

        pages = clean_pages(pages)
        chunks = chunk_pages(filename, pages)
        page_1_chunks = [c for c in chunks if c.page == 1]
        if page_1_chunks:
            print(f"  page 1 preview: {page_1_chunks[0].content[:200]!r}")
        else:
            print("  page 1 preview: (no chunks — page 1 produced no extractable text)")
        print(f"{label}: {len(pages)} pages -> {len(chunks)} chunks")
        print(f"  header stripped ({header_len} words): {header_preview!r}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    print("\n--- Sample chunk ---")
    sample = all_chunks[5]
    print(f"Document: {sample.document} | Page: {sample.page}")
    print(sample.content[:500])


if __name__ == "__main__":
    main()