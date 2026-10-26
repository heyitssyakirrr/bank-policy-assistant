"""What the Guide page and the Ask page say about the three documents.

Every expected answer was checked against the paragraph it cites
(app/data/processed/chunks.jsonl); docs/policy_guide.md lists the same
questions, and a test keeps the two in step.
"""

from __future__ import annotations

from typing import NamedTuple


class GuideQuestion(NamedTuple):
    question: str
    expected: str  # what a correct answer says
    check: str  # the paragraphs to read


class QuestionSet(NamedTuple):
    questions: list[GuideQuestion]
    agentic_only: bool = False  # these show what the agent adds: asked in agentic mode


DOCUMENTS = {
    "e-KYC": (
        "How a financial institution may verify a new customer's identity online, without meeting them: "
        "Board approval, identity checks, AI and its error rate (FAR), human-only e-KYC, weaknesses, reporting."
    ),
    "Fair Treatment of Financial Consumers": (
        "How a financial service provider must treat customers fairly: the Treat Customers Fairly Charter, "
        "fair contract terms, clear information, fair dealing and advice, complaints, vulnerable consumers."
    ),
    "AML/CFT/CPF and TFS": (
        "How a reporting institution must stop money laundering and terrorism financing: customer due "
        "diligence (repeated for each sector in 14A to 14D), PEPs, cash and suspicious transaction reports, "
        "record keeping, sanctions."
    ),
}

QUESTIONS: dict[str, QuestionSet] = {
    "e-KYC": QuestionSet(
        [
            GuideQuestion(
                "Who has to approve a financial institution's risk appetite for e-KYC?",
                "MUST: the Board approves the overall risk appetite and the internal framework for e-KYC.",
                "e-KYC 8.1 (S)",
            ),
            GuideQuestion(
                "What does e-KYC paragraph 8.10 require?",
                "MAY: document verification, biometric matching, liveness detection. They serve the MUST in 8.4: "
                "identity checks must be secure and effective.",
                "e-KYC 8.10 (G), 8.4 (S)",
            ),
            GuideQuestion(
                "What must a bank do if FAR exceeds 5%?",
                "MUST: above 5% for any two months in four, notify BNM in writing within 7 working days with an "
                "assessment and mitigation measures; the notification must cover the items in para 11.",
                "e-KYC App. 2 paras 10-11",
            ),
            GuideQuestion(
                "How often must a bank revalidate its e-KYC technology provider?",
                "MUST: review or revalidate the requirements on the provider in 8.22 (independent assessment, "
                "certification) at least once every 3 years, or when the solution changes materially.",
                "e-KYC 8.24, 8.22 (S)",
            ),
            GuideQuestion(
                "Can e-KYC be done entirely by a human, without AI?",
                "MAY: yes, solely by a human representative, but it may give lower identity assurance and may not "
                "suit all circumstances. MUST: watch for a higher risk of misidentification and set safeguards.",
                "e-KYC 8.26, 8.27 (G), 8.28 (S)",
            ),
            GuideQuestion(
                "What must a bank do when it finds vulnerabilities in its e-KYC solution?",
                "MUST: keep monitoring for vulnerabilities and adopt immediate mitigation measures.",
                "e-KYC 8.29 (S)",
            ),
        ]
    ),
    "Fair Treatment": QuestionSet(
        [
            GuideQuestion(
                "Does a bank have to publish its commitment to treating customers fairly?",
                "MUST: yes, prominently on its website, with how it will carry it out.",
                "FTFC 9.1 (S)",
            ),
            GuideQuestion(
                "What must senior management ensure for handling complaints?",
                "MUST: enough resources, trained staff, and set timeframes for resolving complaints and claims.",
                "FTFC 15.4 (S)",
            ),
            GuideQuestion(
                "What should a bank do for vulnerable consumers?",
                "MUST: assess their needs and put in place policies and procedures to meet them.",
                "FTFC 16.2 (S)",
            ),
        ]
    ),
    "AML/CFT": QuestionSet(
        [
            GuideQuestion(
                "How long must a bank keep customer and transaction records?",
                "MUST: at least six years after the transaction or the end of the relationship.",
                "AML 24.3 (S)",
            ),
            GuideQuestion(
                "When must a bank report a cash transaction to BNM?",
                "MUST: cash transactions of RM25,000 and above in a day, in the same account, in a cash threshold "
                "report to BNM.",
                "AML 21.3.1, 21.1.1 (S)",
            ),
            GuideQuestion(
                "What must a bank do if it suspects a transaction?",
                "MUST: promptly submit a suspicious transaction report to BNM, and not disclose it or related "
                "information (no tipping off), except as AMLA section 14A(3) allows.",
                "AML 22.1.1, 23.1, 23.2 (S)",
            ),
            GuideQuestion(
                "What checks apply to a foreign politically exposed person?",
                "MUST: have a system to identify foreign PEPs, and apply enhanced due diligence to them.",
                "AML 15.2.1, 15.2.3 (S)",
            ),
            GuideQuestion(
                "Can a bank onboard a customer without meeting them face to face?",
                "MAY: yes. MUST: with Board approval, read together with the e-KYC document, with identity checks "
                "that are secure and effective.",
                "AML 14A.15.1 (G), 14A.15.2, 14A.15.3, 14A.15.7 (S)",
            ),
        ]
    ),
    "What the agent adds": QuestionSet(
        [
            GuideQuestion(
                "How do the e-KYC verification rules differ from AML's non-face-to-face rules?",
                "One search per document, and the answer cites both documents.",
                "Agent steps: Plan",
            ),
            GuideQuestion(
                "What does para 8.2 say?",
                "8.2 exists in all three documents: the agent asks which one.",
                "Agent steps: Clarify",
            ),
            GuideQuestion(
                "What is the interest rate cap on credit cards?",
                "Not in these documents: the agent refuses before searching.",
                "Agent steps: Respond",
            ),
        ],
        agentic_only=True,
    ),
}

TERMS = {
    "S / MUST": "a Standard: an obligation (the paragraph says shall or must)",
    "G / MAY": "Guidance: advice or an example (may or should)",
    "the Bank": "how the documents refer to Bank Negara Malaysia (BNM)",
    "FAR": "False Acceptance Rate: how often e-KYC wrongly accepts someone",
    "CDD": "Customer Due Diligence: identifying and verifying customers",
    "Non-face-to-face": "opening an account without meeting the customer in person",
    "PEP": "Politically Exposed Person: needs extra checks",
    "CTR / STR": "Cash Threshold Report / Suspicious Transaction Report, both sent to BNM",
    "TFS": "Targeted Financial Sanctions: freezing funds of people on UN sanctions lists and Malaysia's Domestic List",
    "14A / 14B / 14C / 14D": "the same AML rules for banks, insurers and takaful, money services, e-money",
}

READING_AN_ANSWER = (
    "Reading an answer: the badge says whether it was answered; **In short** gives the direct answer; "
    "**MUST: obligations** lists what the institution must do; **MAY: options and guidance** what it may or "
    "should do, or what BNM may do (e.g. enforcement); **INFO: background** gives definitions and scope. "
    "**Sources** show each passage with a link to the PDF page: the passage must support the statement. "
    "*Citations verified* means code checked that every statement's supporting quote is in a passage it cites "
    "and that no MUST rests only on guidance (G) paragraphs. It doesn't check that the answer is complete or "
    "that the meaning is right."
)
