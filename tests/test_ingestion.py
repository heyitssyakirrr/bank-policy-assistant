"""Tests for the ingestion rules, built from REAL Docling output.

The blocks below are copied from Docling's structured dump of the e-KYC PDF
(pages 8-9 and 18-20). Tests run in milliseconds and need no models or
PDFs, so they protect the regex rules every time you change them.

Run:  python -m pytest -q
"""

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import TableCell, TableData

from app.ingestion.chunker import ChunkingConfig, build_chunks, references_footnote
from app.ingestion.manifest import DocumentSpec, parse_page_spec
from app.ingestion.models import Block, BlockKind, ChunkType, ParsedDocument, Role
from app.ingestion.parser import to_parsed_document
from app.ingestion.structure import annotate

SPEC = DocumentSpec(code="ekyc", file="pd-ekyc-apr2024.pdf", title="e-KYC Policy Document")


def make_blocks(rows: list[tuple]) -> tuple[Block, ...]:
    """rows: (kind, page, text) or (kind, page, text, marker) or a ready Block."""
    blocks = []
    for i, row in enumerate(rows):
        if isinstance(row, Block):
            blocks.append(Block(**{**{s: getattr(row, s) for s in row.__slots__}, "index": i}))
            continue
        kind, page, text, *rest = row
        blocks.append(Block(index=i, kind=kind, text=text, page=page, marker=rest[0] if rest else ""))
    return tuple(blocks)


def parsed_doc(blocks: tuple[Block, ...], printed: dict[int, str] | None = None) -> ParsedDocument:
    return ParsedDocument(
        doc_code="ekyc",
        source_file="pd-ekyc-apr2024.pdf",
        source_sha256="0" * 64,
        page_count=30,
        blocks=blocks,
        printed_pages=printed or {},
        furniture_words={},
    )


H, T, L, F, TB = BlockKind.HEADING, BlockKind.TEXT, BlockKind.LIST_ITEM, BlockKind.FOOTNOTE, BlockKind.TABLE

# e-KYC pp. 8-9, as Docling returned them (page headers already removed as furniture)
MAIN_BODY = make_blocks(
    [
        (H, 8, "PART B POLICY REQUIREMENTS"),
        (H, 8, "8 e-KYC implementation"),
        (H, 8, "Role and responsibility of the Board"),
        (
            L,
            8,
            "S 8.1 A financial institution shall obtain Board approval on the overall risk appetite and internal framework. The framework shall address-",
        ),
        (L, 8, "high risk or material risk scenarios that require subsequent Board approval;", "i."),
        (L, 8, "other instances that require Board approval.", "iv."),
        (
            L,
            8,
            "S 8.2 The Board of financial institutions shall be responsible for ensuring satisfactory measures are undertaken.",
        ),
        (H, 8, "Identification and verification (IDV) of customers through e-KYC"),
        (L, 8, "General requirements", "A."),
        (
            L,
            8,
            "S 8.4 In line with requirements under the AML/CFT/CPF and TFS for FIs policy document, a financial institution shall ensure measures are secure and effective.",
        ),
        (
            L,
            8,
            "G 8.5 In relation to paragraph 8.4, where reference is made to face-to-face processes, this should mainly serve as guidance.",
        ),
        (
            L,
            9,
            "G 8.8 In respect of paragraph 8.7, a financial institution may give regard to the key basic authentication factors, which include:",
        ),
        (L, 9, "(i) something the customer possesses (e.g. national identity document);"),
        (
            T,
            9,
            "An e-KYC solution that depends on more than one factor is typically more difficult to compromise than a single factor system.",
        ),
        (H, 9, "B. IDV through e-KYC for individuals"),
        (
            T,
            9,
            "G 8.10 In identifying and verifying an individual's identity through e-KYC, a financial institution may undertake measures, including-",
        ),
        (
            L,
            9,
            "(ii) Biometric matching - i.e., verifying the customer against a government issued ID 3 by utilising biometric technology; and/or",
        ),
        (F, 9, "3 i.e, National Registration Identity Card (NRIC), passport, or any other official documents."),
    ]
)

FAR_ROWS_P19 = (
    ("FAR thresholds", "Action to be taken when breached"),
    (
        "Level 1 (0% - 3%)",
        "A financial institution shall continuously monitor and improve the capability of e-KYC solutions.",
    ),
)
FAR_ROWS_P20 = (
    ("Level 2 (> 3%)", "A financial institution shall conduct a risk assessment for notification to the Board."),
    (
        "Level 3 (> 5%)",
        "Where FAR is measured to be more than 5% for any two months within a four-month period, notify the Bank.",
    ),
)

# e-KYC Appendix 2, pp. 18-20: numbered "1." paragraphs, a list split by a page
# break with footnotes in between, and the FAR table split across two pages.
APPENDIX = make_blocks(
    [
        (H, 18, "Appendix 2: False Acceptance Rate and sampling"),
        (
            L,
            18,
            "FAR shall be measured based on the number of complete 20 identification and verification cases processed under e-KYC.",
            "2.",
        ),
        (
            L,
            18,
            "In determining the appropriate sampling approach, a financial institution shall consider the following sampling dimensions:",
            "5.",
        ),
        (
            L,
            18,
            "(i) Minimum sample size: The sample size shall minimally meet a 95% confidence level and 3% margin of error or 400 cases per month, whichever is higher.",
        ),
        (
            F,
            18,
            "20 A complete identification and verification case processed under e-KYC is defined as a case where the customer has completed only the e-KYC checks.",
        ),
        (F, 18, "22 For avoidance of doubt, the higher or stricter of considerations in (i) - (iv) shall apply."),
        (
            L,
            19,
            "(iv) Progressive sampling: Where one or more false positives are detected in the initial sample, consider a higher sample size.",
        ),
        (
            L,
            19,
            "To strengthen solution performance, a financial institution shall adopt a tiered approach for FAR threshold monitoring, as follows:",
            "10.",
        ),
        Block(
            index=0,
            kind=TB,
            text="",
            page=19,
            table_rows=FAR_ROWS_P19,
            caption="Table 1: FAR thresholds and actions to be taken",
        ),
        (F, 19, "23 Reasons may include but are not limited to blurry images."),
        Block(index=0, kind=TB, text="", page=20, table_rows=FAR_ROWS_P20),
        (
            L,
            20,
            "In respect of paragraph 10 of this Appendix, the notification to the Bank shall include the following-",
            "11.",
        ),
    ]
)


def roles(annotated):
    return [(ab.role, ab.paragraph_id, ab.provision) for ab in annotated]


# ---------------------------------------------------------------- structure rules


def test_paragraph_numbers_and_sg_markers_are_read_from_text():
    annotated = annotate(MAIN_BODY, SPEC)
    starts = [(ab.paragraph_id, ab.provision) for ab in annotated if ab.role is Role.PARAGRAPH_START]
    # 8.10 is a plain "text" item in Docling, the others are list items: both must work.
    assert starts == [("8.1", "S"), ("8.2", "S"), ("8.4", "S"), ("8.5", "G"), ("8.8", "G"), ("8.10", "G")]


def test_lettered_list_item_is_treated_as_heading():
    annotated = annotate(MAIN_BODY, SPEC)
    general = next(ab for ab in annotated if ab.block.text == "General requirements")
    assert general.role is Role.HEADING
    para_84 = next(ab for ab in annotated if ab.paragraph_id == "8.4")
    assert para_84.section_path == (
        "PART B POLICY REQUIREMENTS",
        "8 e-KYC implementation",
        "Identification and verification (IDV) of customers through e-KYC",
        "A. General requirements",
    )


def test_lettered_heading_replaces_previous_lettered_heading():
    annotated = annotate(MAIN_BODY, SPEC)
    para_810 = next(ab for ab in annotated if ab.paragraph_id == "8.10")
    assert para_810.section_path[-1] == "B. IDV through e-KYC for individuals"
    assert "A. General requirements" not in para_810.section_path


def test_numbered_title_is_a_heading_not_a_paragraph():
    blocks = make_blocks(
        [
            (H, 126, "29 Targeted Financial Sanctions"),
            (T, 126, "29.1 Definition and Interpretation"),
            (T, 126, "29.1.1 For the purpose of paragraph 29, the following definitions apply."),
        ]
    )
    annotated = annotate(blocks, SPEC)
    assert roles(annotated)[1][0] is Role.HEADING
    assert roles(annotated)[2] == (Role.PARAGRAPH_START, "29.1.1", None)
    assert annotated[2].section_path == ("29 Targeted Financial Sanctions", "29.1 Definition and Interpretation")


def test_wrapped_cross_reference_is_not_a_new_paragraph():
    blocks = make_blocks(
        [
            (L, 9, "S 8.9 In verifying the identity of an individual or beneficial owner under paragraphs"),
            (T, 9, "8.10 and 8.13 of this policy document, financial institutions shall:"),
        ]
    )
    annotated = annotate(blocks, SPEC)
    assert roles(annotated) == [(Role.PARAGRAPH_START, "8.9", "S"), (Role.CONTINUATION, "8.9", "S")]


def test_standalone_sg_marker_is_applied_to_the_next_paragraph():
    blocks = make_blocks(
        [
            (T, 9, "S"),
            (T, 9, "8.6 For the money services business sector, money services businesses shall also comply."),
        ]
    )
    annotated = annotate(blocks, SPEC)
    assert roles(annotated) == [(Role.SKIP, None, None), (Role.PARAGRAPH_START, "8.6", "S")]


def test_appendix_numbering_opens_its_own_scope():
    annotated = annotate(APPENDIX, SPEC)
    starts = [(ab.paragraph_id, ab.scope) for ab in annotated if ab.role is Role.PARAGRAPH_START]
    assert starts == [("2", "Appendix 2"), ("5", "Appendix 2"), ("10", "Appendix 2"), ("11", "Appendix 2")]


def test_manifest_skip_and_tag_pages():
    spec = DocumentSpec(
        code="x", file="x.pdf", title="X", skip_pages=frozenset({8}), tag_pages={"form_template": frozenset({9})}
    )
    annotated = annotate(MAIN_BODY, spec)
    assert all(ab.role is Role.SKIP for ab in annotated if ab.block.page == 8)
    assert all(ab.tags == ("form_template",) for ab in annotated if ab.block.page == 9)


def test_parse_page_spec():
    assert parse_page_spec([1, "3-5"], where="t") == frozenset({1, 3, 4, 5})


# ---------------------------------------------------------------- chunking


def chunks_for(blocks, max_words=350):
    parsed = parsed_doc(blocks, printed={8: "6", 9: "7", 18: "16", 19: "17", 20: "18"})
    chunks, stats = build_chunks(parsed, annotate(blocks, SPEC), SPEC, ChunkingConfig(max_words=max_words))
    return chunks, stats


def test_one_chunk_per_paragraph_with_sub_items_and_trailing_text():
    chunks, _ = chunks_for(MAIN_BODY)
    para_88 = next(c for c in chunks if c.paragraph_id == "8.8")
    assert "(i) something the customer possesses" in para_88.text
    assert "more than one factor" in para_88.text  # loose text after the list stays with 8.8
    para_81 = next(c for c in chunks if c.paragraph_id == "8.1")
    assert "i. high risk" in para_81.text  # marker re-attached to the text
    assert para_81.provision == "S" and para_81.printed_page_start == "6"


def test_footnote_attached_to_the_paragraph_that_references_it():
    chunks, stats = chunks_for(MAIN_BODY)
    para_810 = next(c for c in chunks if c.paragraph_id == "8.10")
    assert para_810.footnotes == [
        "[3] i.e, National Registration Identity Card (NRIC), passport, or any other official documents."
    ]
    assert "Footnotes:" in para_810.embed_text
    assert stats["unreferenced_footnotes"] == []


def test_paragraph_continues_across_page_break_and_footnotes():
    chunks, _ = chunks_for(APPENDIX)
    para_5 = next(c for c in chunks if c.paragraph_id == "5" and c.chunk_type is ChunkType.PARAGRAPH)
    assert "(i) Minimum sample size" in para_5.text and "(iv) Progressive sampling" in para_5.text
    assert (para_5.page_start, para_5.page_end) == (18, 19)
    assert "complete identification" not in para_5.text  # footnote 20 belongs to para 2, not 5


def test_table_split_across_pages_is_stitched_into_one_chunk():
    chunks, _ = chunks_for(APPENDIX)
    tables = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
    assert len(tables) == 1
    far = tables[0]
    assert all(level in far.text for level in ("Level 1", "Level 2", "Level 3"))
    lines = far.text.splitlines()
    assert lines[0].startswith("10. To strengthen solution performance")  # lead-in says what the table is for
    assert lines[1] == "Table 1: FAR thresholds and actions to be taken"
    assert far.paragraph_id == "10" and (far.page_start, far.page_end) == (19, 20)
    # header row stays the header; the continuation's first row is data
    assert lines[2].startswith("| FAR thresholds")


def test_reference_numbers_are_not_mistaken_for_footnote_markers():
    assert references_footnote("against a government issued ID 3 by utilising", "3")
    assert not references_footnote("In respect of paragraph 3 of this Appendix", "3")
    assert not references_footnote("within seven (7) working days", "7")


def test_oversized_paragraph_is_split_on_block_boundaries_and_ids_are_stable():
    first, _ = chunks_for(MAIN_BODY, max_words=30)
    para_88 = [c for c in first if c.paragraph_id == "8.8"]
    assert len(para_88) == 2 and [c.part_index for c in para_88] == [1, 2]
    assert all(c.word_count <= 30 for c in para_88)
    assert "part 1 of 2" in para_88[0].embed_text
    again, _ = chunks_for(MAIN_BODY, max_words=30)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]  # re-running = same ids = idempotent upserts


def test_chunk_keys_are_unique():
    chunks, stats = chunks_for(MAIN_BODY + APPENDIX)
    keys = [c.chunk_key for c in chunks]
    assert len(keys) == len(set(keys)) and stats["duplicate_keys"] == []


def test_citation_format():
    chunks, _ = chunks_for(MAIN_BODY)
    para_82 = next(c for c in chunks if c.paragraph_id == "8.2")
    assert para_82.citation() == "e-KYC Policy Document, para 8.2 (S), p. 6"


# ---------------------------------------------------------------- parser translation (no models needed)


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0))


def test_docling_document_is_translated_to_blocks():
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    doc.add_text(
        label=DocItemLabel.PAGE_HEADER, text="Page 6 of 28", prov=_prov(1), content_layer=ContentLayer.FURNITURE
    )
    doc.add_heading(text="8 e-KYC implementation", prov=_prov(1))
    doc.add_list_item(text="high risk scenarios &amp; others;", marker="i.", prov=_prov(1))
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="3 i.e, NRIC.", prov=_prov(1))
    data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            TableCell(
                text=t,
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
            )
            for r, row in enumerate([("A", "B"), ("1", "2")])
            for c, t in enumerate(row)
        ],
    )
    doc.add_table(data=data, prov=_prov(1))

    parsed = to_parsed_document(doc, doc_code="t", source_file="t.pdf", source_sha256="x")
    kinds = [b.kind for b in parsed.blocks]
    assert kinds == [BlockKind.HEADING, BlockKind.LIST_ITEM, BlockKind.FOOTNOTE, BlockKind.TABLE]
    assert parsed.printed_pages == {1: "6"}  # read from the header, which is not a block
    assert parsed.blocks[1].marker == "i." and parsed.blocks[1].text == "high risk scenarios & others;"
    assert parsed.blocks[3].table_rows == (("A", "B"), ("1", "2"))
    assert parsed.page_count == 1


# ---------------------------------------------------------------- FTFC and AML layouts

FTFC_SPEC = DocumentSpec(code="ftfc", file="pd-ftfc-mar24.pdf", title="FTFC Policy Document")

# FTFC p.12 (printed 11): G 10.4, then a boxed "Good practices" list, then the next paragraph.
FTFC_CALLOUT = make_blocks(
    [
        (H, 12, "Role of senior management"),
        (
            T,
            12,
            "G 10.4 In implementing paragraph 10.3(g), measures taken by senior management should promote a business culture.",
        ),
        (T, 12, "Good practices"),
        (
            L,
            12,
            "Senior management deliberates and ensures appropriate and timely escalation of material issues.",
            "1.",
        ),
        (L, 12, "Senior management requires business units to complete a self-assessment compliance checklist.", "2."),
        (T, 12, "Poor practices 8"),
        (L, 12, "Absence of deliberation by the Board on matters of importance.", "1."),
        (T, 13, "S 10.5 Senior management shall ensure that adequate resources are allocated."),
    ]
)


def test_callout_box_is_tagged_and_tied_to_the_paragraph_above():
    chunks, _ = chunks_for_spec(FTFC_CALLOUT, FTFC_SPEC)
    box = next(c for c in chunks if "callout" in c.tags)
    assert box.paragraph_id == "10.4" and box.provision == "G"
    assert box.section_path[-1] == "Good practices"
    assert "1. Senior management deliberates" in box.text and "2. Senior management requires" in box.text
    assert box.citation() == "FTFC Policy Document, para 10.4 (G), Good practices, p. pdf 12"
    para_104 = next(c for c in chunks if c.paragraph_id == "10.4" and "callout" not in c.tags)
    assert "Good practices" not in para_104.text  # the box is a separate chunk


def test_paragraph_after_a_callout_box_is_clean():
    chunks, _ = chunks_for_spec(FTFC_CALLOUT, FTFC_SPEC)
    para_105 = next(c for c in chunks if c.paragraph_id == "10.5")
    assert para_105.tags == [] and "Good practices" not in para_105.section_path


def test_glossary_row_cut_by_page_break_is_glued_back():
    rows_p8 = (
        ("Term", "Definition"),
        ("\u201cbeneficial owner\u201d", "Refers to the natural person(s) who ultimately own or control a customer"),
    )
    rows_p9 = (
        ("", "power; and (v) any other natural person(s) exercising ultimate effective control over the arrangement."),
        (
            "\u201cbeneficiary\u201d",
            "Depending on the context: in trust law, a beneficiary refers to the person entitled to the benefit.",
        ),
    )
    blocks = make_blocks(
        [
            (H, 8, "6 Interpretation"),
            (T, 8, "6.1 The terms and expressions used in this policy document shall have the following meanings:"),
            Block(index=0, kind=TB, text="", page=8, table_rows=rows_p8),
            Block(index=0, kind=TB, text="", page=9, table_rows=rows_p9),
        ]
    )
    chunks, _ = chunks_for(blocks)
    table = next(c for c in chunks if c.chunk_type is ChunkType.TABLE)
    row_lines = [line for line in table.text.splitlines() if line.startswith("| \u201c")]
    assert len(row_lines) == 2  # 2 terms, not 3 rows
    assert "control a customer power; and (v)" in row_lines[0]


def test_appendices_divider_resets_the_main_body_path():
    blocks = make_blocks(
        [
            (H, 128, "30 Other Matters"),
            (H, 132, "APPENDICES"),
            (H, 132, "APPENDIX 1 Guidance on Application of Risk Based Approach"),
            (H, 132, "1.0 Introduction"),
            (T, 132, "1.1 The risk-based approach (RBA) is central to the effective implementation."),
        ]
    )
    para = next(ab for ab in annotate(blocks, SPEC) if ab.role is Role.PARAGRAPH_START)
    assert para.scope == "Appendix 1" and para.paragraph_id == "1.1"
    assert para.section_path == ("APPENDIX 1 Guidance on Application of Risk Based Approach", "1.0 Introduction")


def chunks_for_spec(blocks, spec):
    parsed = parsed_doc(blocks)
    return build_chunks(parsed, annotate(blocks, spec), spec)


def test_two_boxes_after_one_paragraph_get_distinct_keys():
    chunks, stats = chunks_for_spec(FTFC_CALLOUT, FTFC_SPEC)
    boxes = [c for c in chunks if "callout" in c.tags]
    assert [b.section_path[-1] for b in boxes] == ["Good practices", "Poor practices 8"]
    assert len({b.chunk_key for b in boxes}) == 2 and stats["duplicate_keys"] == []


# ---------------------------------------------------------------- fixes from the first real Docling run (e-KYC)


def test_footnote_marker_after_a_paragraph_number():
    # e-KYC p.12: "...fulfil the requirements in paragraph 8.22 11 ."
    assert references_footnote("shall fulfil the requirements in paragraph 8.22 11 .", "11")


def test_multi_block_footnote_is_merged_and_attached():
    # e-KYC p.12: footnote 11 continues with its own (i) and (ii) blocks
    blocks = make_blocks(
        [
            (T, 12, "S 8.24 Financial institutions shall ensure that:"),
            (
                L,
                12,
                "(ii) Prior to implementing the e-KYC solution, financial institutions shall fulfil paragraph 8.22 11 .",
            ),
            (F, 12, "11 This requirement must be completed prior to implementation of the e-KYC solution unless:"),
            (
                F,
                12,
                "(i) Such assessment has already been conducted by the technology provider within the past two (2) years; or",
            ),
            (F, 12, "(ii) Where the technology provider has experience in applying the e-KYC solution effectively."),
        ]
    )
    chunks, stats = chunks_for(blocks)
    para = next(c for c in chunks if c.paragraph_id == "8.24")
    assert len(para.footnotes) == 1
    assert (
        para.footnotes[0].startswith("[11] This requirement")
        and "(ii) Where the technology provider" in para.footnotes[0]
    )
    assert stats["unreferenced_footnotes"] == []


def test_footnote_on_a_heading_goes_to_the_first_chunk_under_it():
    # e-KYC p.10: "C. IDV through e-KYC for legal person 5"
    blocks = make_blocks(
        [
            (H, 10, "C. IDV through e-KYC for legal person 5"),
            (T, 10, "G 8.12 A financial institution may implement e-KYC to identify and verify legal persons."),
            (
                F,
                10,
                "5 For avoidance of doubt, a sole proprietor is not deemed as legal person under this policy document.",
            ),
        ]
    )
    chunks, stats = chunks_for(blocks)
    para = next(c for c in chunks if c.paragraph_id == "8.12")
    assert para.footnotes and para.footnotes[0].startswith("[5] For avoidance of doubt")
    assert stats["unreferenced_footnotes"] == []


def test_numbered_list_misread_as_table_becomes_a_paragraph_again():
    # e-KYC p.20: Docling returned Appendix 2 para 11 as a 3-column table
    lead = "In respect of paragraph 10 of this Appendix, the notification to the Bank shall include the following-"
    rows = (
        ("11.", lead, lead),
        ("", "(i)", "an assessment on the current performance of the e-KYC solution;"),
        ("", "(ii)", "proposed rectification actions to reduce the FAR going forward."),
    )
    blocks = make_blocks(
        [
            (H, 18, "Appendix 2: False Acceptance Rate and sampling"),
            (
                L,
                19,
                "To strengthen solution performance, a financial institution shall adopt a tiered approach:",
                "10.",
            ),
            Block(index=0, kind=TB, text="", page=20, table_rows=rows),
            (L, 21, "In respect of paragraph 11 (iv) of this Appendix, the mitigating actions may include:", "12."),
        ]
    )
    chunks, _ = chunks_for(blocks)
    assert [c.paragraph_id for c in chunks] == ["10", "11", "12"]
    para_11 = chunks[1]
    assert para_11.chunk_type is ChunkType.PARAGRAPH
    assert para_11.text.splitlines() == [
        f"11. {lead}",
        "(i) an assessment on the current performance of the e-KYC solution;",
        "(ii) proposed rectification actions to reduce the FAR going forward.",
    ]
