"""Evidence labelling, merging across searches, and the prompt blocks."""

import pytest

from app.agent.evidence import cross_reference_ids, format_evidence, merge_evidence, to_evidence
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


def test_a_passage_referring_to_another_given_passage_says_so():
    e810 = candidate("e810", payload("8.10", "G 8.10 as required under paragraph 8.4 of this policy", provision="G"))
    e84 = candidate("e84", payload("8.4", "S 8.4 shall ensure measures are secure"))
    ftfc84 = candidate("f84", payload("8.4", "S 8.4 fair treatment", doc="ftfc"))
    block = format_evidence(labelled(e810, e84, ftfc84))
    assert "[E1] ekyc para 8.10 (G), p. 6\nCross-reference: E2 (para 8.4)\n" in block  # same document only
    assert "Cross-reference" not in block.split("[E2]")[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("G 8.10 as required under paragraph 8.4 of this policy document", {"8.4"}),
        ("S 8.16 In relation to paragraph 8.13 (ii), where the authorised person", {"8.13"}),
        ("in relation to paragraphs 14A.9.4, 14A.9.5 and 14A.9.6, institutions", {"14A.9.4", "14A.9.5", "14A.9.6"}),
        ("S 10.1 Subject to paragraphs 8.1 to 8.3, where a licensed person", {"8.1", "8.3"}),
        ("as specified under paragraph 6.2 of the AML/CFT/CPF and TFS for FIs policy document", set()),
        ("as required under paragraph 6.2 of the AML/CFT policy document", set()),  # another document
        ("G 8.26 Notwithstanding paragraphs 8.19 to 8.21, a financial institution may", set()),  # a mention
    ],
)
def test_cross_reference_ids(text, expected):
    assert cross_reference_ids(text) == expected
