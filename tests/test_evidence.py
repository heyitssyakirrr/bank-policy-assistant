"""Evidence labelling, merging across searches, and the prompt blocks."""

from app.agent.evidence import format_evidence, merge_evidence, to_evidence
from app.agent.prompts import generator_messages
from tests.fakes import candidate, payload

A = candidate("a", payload("8.4", "S 8.4 shall obtain approval"))
B = candidate("b", payload("8.10", "G 8.10 may use biometrics", provision="G"))
C = candidate("c", payload("13", "13. Where FAR exceeds 5%", provision=None))
RULE = "Reporting institutions may establish non face-to-face business relationships with its customers."
BANKING = candidate("bank", payload("14A.15.1", f"G 14A.15.1 {RULE}", "amlcft", "G", sector="banking"))
INSURANCE = candidate("ins", payload("14B.17.1", f"G 14B.17.1 {RULE}", "amlcft", "G", sector="insurance_takaful"))


def labelled(*candidates, source="search"):
    return merge_evidence([], to_evidence(candidates, source, "q"))


def test_labels_are_sequential_and_carry_the_metadata():
    evidence = labelled(A, B)
    assert [e.id for e in evidence] == ["E1", "E2"]
    assert evidence[1].provision == "G" and evidence[1].citation == "ekyc para 8.10 (G), p. 6"


def test_new_passages_get_the_next_labels_and_duplicates_are_skipped():
    merged = merge_evidence(labelled(A, B), to_evidence([B, C], "lookup", "13"))
    assert [(e.id, e.chunk_id) for e in merged] == [("E1", "a"), ("E2", "b"), ("E3", "c")]
    assert merged[2].source == "lookup"


def test_none_resets_the_evidence_for_a_new_turn():
    assert merge_evidence(labelled(A), None) == []


def test_a_sector_copy_found_by_another_search_is_dropped():
    merged = merge_evidence(labelled(BANKING), to_evidence([INSURANCE, C], "search", "takaful"))
    assert [e.chunk_id for e in merged] == ["bank", "c"]


def test_a_named_paragraph_is_kept_even_if_it_copies_another_sector():
    merged = merge_evidence(labelled(BANKING), to_evidence([INSURANCE], "lookup", "14B.17.1"))
    assert [e.chunk_id for e in merged] == ["bank", "ins"]


def test_footnotes_are_part_of_the_passage():
    data = payload("5.2", "5.2 text") | {"footnotes": ["[1] a footnote"]}
    assert labelled(candidate("f", data))[0].text == "5.2 text\n[1] a footnote"


def test_passages_are_fenced_and_labelled_in_the_prompt():
    block = format_evidence(labelled(A))
    assert block.startswith("[E1] ekyc para 8.4 (S), p. 6\n") and '"""' in block


def test_generator_prompt_includes_sector_and_feedback():
    evidence = labelled(A)
    system, user = generator_messages("Q?", evidence, "insurance_takaful", feedback="fix the quote")
    assert "insurers and takaful operators" in user.content and user.content.endswith("fix the quote")
    assert "all sectors" in generator_messages("Q?", evidence, None)[1].content
    assert "DATA" in system.content  # the prompt-injection rule is always present
