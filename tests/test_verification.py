"""The verifier: every check in PROJECT_PLAN §5.4, on hand-made answers."""

import pytest

from app.agent.evidence import Evidence
from app.agent.schemas import GroundedAnswer, Statement
from app.agent.verification import (
    NOT_COVERED_SUMMARY,
    PARTIAL_SUMMARY_PREFIX,
    check_answer,
    finalize,
    normalise_id,
    quote_found,
    regeneration_feedback,
)


def ev(eid: str, text: str, provision: str | None = "S") -> Evidence:
    return Evidence(
        id=eid,
        chunk_id=f"c{eid}",
        document="ekyc",
        paragraph_id=eid,
        provision=provision,
        sector=None,
        citation=f"e-KYC para {eid}",
        text=text,
        page=6,
        source="search",
        query="q",
    )


EVIDENCE = [
    ev("E1", "S 8.4 A financial institution shall obtain the approval of its Board before implementing e-KYC."),
    ev("E2", "G 8.10 A financial institution may use biometric technology to verify identity.", "G"),
    ev("E3", "13. Where FAR exceeds 5%, the financial institution shall notify the Bank immediately.", None),
]


def stmt(text="x", obligation="information", ids=("E1",), quote="obtain the approval of its Board") -> Statement:
    return Statement(text=text, obligation=obligation, evidence_ids=list(ids), quote=quote, topic="t")


def answer(*statements: Statement, status="answered") -> GroundedAnswer:
    return GroundedAnswer(status=status, summary="summary", statements=list(statements))


def test_a_correct_answer_passes_unchanged():
    report = check_answer(answer(stmt(obligation="must")), EVIDENCE)
    assert report.passed and not report.fixes and report.answer.status == "answered"


@pytest.mark.parametrize(
    ("raw", "expected"), [("E3", "E3"), ("[E3]", "E3"), ("e 3", "E3"), ("E03", "E3"), ("8.4", None)]
)
def test_evidence_ids_are_normalised(raw, expected):
    assert normalise_id(raw) == expected


def test_unknown_id_is_dropped_and_the_valid_one_kept():
    report = check_answer(answer(stmt(ids=["E9", "[E1]"])), EVIDENCE)
    assert report.answer.statements[0].evidence_ids == ["E1"]
    assert [i.check for i in report.fixes] == ["unknown_id"] and report.passed


def test_statement_left_with_no_valid_citation_is_dropped():
    report = check_answer(answer(stmt(ids=["E9"]), stmt(text="kept")), EVIDENCE)
    assert [s.text for s in report.answer.statements] == ["kept"]
    assert "uncited" in [i.check for i in report.fixes]


def test_no_valid_statement_left_becomes_a_refusal():
    report = check_answer(answer(stmt(ids=["E9"])), EVIDENCE)
    assert report.answer.status == "not_covered" and report.answer.summary == NOT_COVERED_SUMMARY
    assert report.answer.statements == [] and "status" in [i.check for i in report.fixes]


def test_quote_not_in_the_cited_passage_is_serious():
    report = check_answer(answer(stmt(quote="approval of its Board")), [ev("E1", "unrelated text entirely")])
    assert [i.check for i in report.serious] == ["quote_not_found"]


def test_quote_from_a_different_passage_than_cited_is_serious():
    # The quote is real, but it is in E1, and the statement cites E2.
    report = check_answer(answer(stmt(ids=["E2"])), EVIDENCE)
    assert [i.check for i in report.serious] == ["quote_not_found"]


@pytest.mark.parametrize(
    "quote",
    [
        "Obtain the approval  of its BOARD",  # case and spacing
        "shall obtain the ... before implementing e-KYC",  # ellipsis joins fragments
        "approval of its Board.",  # punctuation
    ],
)
def test_quote_matching_ignores_formatting(quote):
    assert quote_found(quote, EVIDENCE[0].text)


def test_quote_must_have_at_least_three_words():
    assert not quote_found("Board", EVIDENCE[0].text)


def test_ellipsis_fragments_of_single_words_do_not_match():
    assert not quote_found("Board ... the ... shall", EVIDENCE[0].text)


def test_ellipsis_fragments_must_appear_in_order():
    assert not quote_found("before implementing e-KYC ... shall obtain the", EVIDENCE[0].text)


def test_the_same_citation_twice_is_kept_once_silently():
    report = check_answer(answer(stmt(ids=["E1", "E1"])), EVIDENCE)
    assert report.answer.statements[0].evidence_ids == ["E1"] and not report.fixes


def test_must_citing_only_guidance_is_serious():
    report = check_answer(answer(stmt(obligation="must", ids=["E2"], quote="may use biometric technology")), EVIDENCE)
    assert [i.check for i in report.serious] == ["must_cites_only_guidance"]


def test_must_citing_an_unmarked_appendix_passage_is_allowed():
    # e-KYC Appendix 2 para 13 (the FAR rule) has no S/G marker but states an obligation.
    report = check_answer(answer(stmt(obligation="must", ids=["E3"], quote="shall notify the Bank")), EVIDENCE)
    assert report.passed


def test_not_covered_with_valid_statements_becomes_partially_answered():
    report = check_answer(answer(stmt(), status="not_covered"), EVIDENCE)
    assert report.answer.status == "partially_answered"


def test_not_covered_without_statements_keeps_the_models_explanation():
    plain = GroundedAnswer(status="not_covered", summary="The documents don't set credit card interest caps.")
    report = check_answer(plain, EVIDENCE)
    assert report.passed and report.answer.summary == plain.summary


def test_finalize_removes_bad_statements_downgrades_and_rewrites_the_summary():
    good = stmt(text="Board approval is required.")
    bad = stmt(text="Banks must use biometrics.", obligation="must", ids=["E2"], quote="may use biometric technology")
    final = finalize(check_answer(answer(good, bad), EVIDENCE))
    assert [s.text for s in final.statements] == ["Board approval is required."]
    assert final.status == "partially_answered"
    # the model's summary may repeat the removed claim, so it is rebuilt from what survived
    assert final.summary == PARTIAL_SUMMARY_PREFIX + "Board approval is required."


def test_finalize_with_nothing_valid_left_refuses():
    final = finalize(check_answer(answer(stmt(quote="words that are nowhere")), EVIDENCE))
    assert final.status == "not_covered" and final.summary == NOT_COVERED_SUMMARY


def test_feedback_names_the_failing_statement():
    report = check_answer(answer(stmt(text="Banks must X", quote="nowhere at all here")), EVIDENCE)
    assert "Banks must X" in regeneration_feedback(report)


LINKED = [
    ev("E1", "G 8.10 As required under paragraph 8.4, a financial institution may use biometric technology.", "G"),
    ev("E2", "S 8.4 A financial institution shall ensure measures are secure and effective."),
]
for item, paragraph in zip(LINKED, ("8.10", "8.4"), strict=True):
    item.paragraph_id = paragraph


def test_answer_skipping_a_passage_its_citation_relies_on_is_incomplete():
    only_810 = stmt(obligation="should_or_may", ids=["E1"], quote="may use biometric technology")
    report = check_answer(answer(only_810), LINKED)
    assert report.passed and not report.complete
    assert "E1 depends on E2" in report.incomplete[0].detail
    assert "E1 depends on E2" in regeneration_feedback(report)


def test_answer_using_both_linked_passages_is_complete():
    both = [
        stmt(obligation="should_or_may", ids=["E1"], quote="may use biometric technology"),
        stmt(obligation="must", ids=["E2"], quote="shall ensure measures are secure"),
    ]
    assert check_answer(answer(*both), LINKED).complete


def test_finalize_never_removes_anything_for_an_incomplete_answer():
    only_810 = stmt(obligation="should_or_may", ids=["E1"], quote="may use biometric technology")
    assert finalize(check_answer(answer(only_810), LINKED)).status == "answered"
