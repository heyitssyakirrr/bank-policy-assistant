"""PII masking: real-shaped Malaysian identifiers are masked, policy text isn't."""

import pytest

from app.core.pii import mask_pii


@pytest.mark.parametrize(
    ("question", "expected", "kinds"),
    [
        ("Customer 900101-14-5678 wants e-KYC", "Customer [NRIC] wants e-KYC", ["NRIC"]),
        ("IC 900101145678 verified?", "IC [NRIC] verified?", ["NRIC"]),
        ("card 4111 1111 1111 1111 declined", "card [CARD] declined", ["CARD"]),
        ("card 4111-1111-1111-1111", "card [CARD]", ["CARD"]),
        ("account 5140123456789 frozen", "account [ACCOUNT] frozen", ["ACCOUNT"]),
        ("customer IC-900101-14-5678 failed e-KYC", "customer IC-[NRIC] failed e-KYC", ["NRIC"]),
        ("IC 900101 14 5678", "IC [NRIC]", ["NRIC"]),
        ("account 1640-1234-5678", "account [ACCOUNT]", ["ACCOUNT"]),
        ("account 5140 1234 5678", "account [ACCOUNT]", ["ACCOUNT"]),
        ("card 4111 1111 1111 1112", "card [ACCOUNT]", ["ACCOUNT"]),  # mistyped card: still masked
        (
            "IC 850230-10-1234 and account 1234567890",  # Feb 30 still counts: the format is what matters
            "IC [NRIC] and account [ACCOUNT]",
            ["NRIC", "ACCOUNT"],
        ),
    ],
)
def test_identifiers_are_masked(question, expected, kinds):
    result = mask_pii(question)
    assert result.text == expected and result.masked == kinds


@pytest.mark.parametrize(
    "question",
    [
        "What does para 14A.9.6 require?",
        "What must a bank do if FAR exceeds 5%?",
        "Transactions above RM50,000 or RM 3,000,000",
        "Policy document dated 2024-02-01, BNM/RH/PD 030-1",
        "Call 03-2698 8044",
        "Transactions of RM 1,000,000,000 or 20240101",
        "paragraphs 10 11 12 13 14",
    ],
)
def test_policy_text_is_untouched(question):
    result = mask_pii(question)
    assert result.text == question and result.masked == []


def test_a_long_number_failing_the_card_checksum_is_an_account_not_a_card():
    assert mask_pii("ref 1234567812345678").masked == ["ACCOUNT"]


def test_twelve_digits_that_are_not_a_birth_date_are_an_account():
    assert mask_pii("ref 991399123456").masked == ["ACCOUNT"]
