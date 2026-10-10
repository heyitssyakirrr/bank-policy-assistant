"""Dump Docling's structured items (not the markdown) for a page range.

Usage:  python -m scripts.inspect_docling_items <pdf_path> <first_page> <last_page>
Pages are 1-based PDF indexes. OCR is off (our PDFs have a real text layer).

What to check in the output:
  1. Do list items keep their original marker, e.g. "(i)", "(ii)", "5."?
  2. Does every item have a page number?
  3. Are page headers/footers labelled as furniture (so we can drop them)?
  4. Are footnotes labelled "footnote"?
"""
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

OUT_DIR = Path("scripts/_parser_trial")
MAX_TEXT = 160  # truncate long text so the dump stays readable


def build_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def iterate_all_items(doc):
    """Yield body items AND furniture (headers/footers) so we can see both.
    Falls back to the default (body only) on older docling-core versions."""
    try:
        from docling_core.types.doc import ContentLayer

        yield from doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        )
    except (ImportError, TypeError):
        yield from doc.iterate_items()


def describe(item, level: int, doc) -> str:
    label = getattr(item.label, "value", str(item.label))
    page = item.prov[0].page_no if getattr(item, "prov", None) else "?"
    layer = getattr(getattr(item, "content_layer", None), "value", "")
    marker = getattr(item, "marker", "") or ""

    if label == "table":
        try:
            text = "\n" + item.export_to_markdown(doc=doc)
        except TypeError:  # older signature without doc
            text = "\n" + item.export_to_markdown()
    else:
        text = (getattr(item, "text", "") or "").replace("\n", " ")
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "…"

    indent = "  " * level
    marker_part = f" marker={marker!r}" if marker else ""
    layer_part = f" [{layer}]" if layer and layer != "body" else ""
    return f"p{page} {indent}{label}{layer_part}{marker_part}: {text}"


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    pdf_path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    result = build_converter().convert(pdf_path, page_range=(first, last))
    doc = result.document

    lines = [describe(item, level, doc) for item, level in iterate_all_items(doc)]
    output = "\n".join(lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{Path(pdf_path).stem}_p{first}-{last}_docling_items.txt"
    out_file.write_text(output, encoding="utf-8")
    print(output)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()