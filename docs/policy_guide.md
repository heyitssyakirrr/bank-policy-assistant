# A plain-English guide to the three policy documents

The assistant answers only from three Bank Negara Malaysia (BNM) policy
documents. You don't need to read their 265 pages to use it or to judge its
answers: this guide says what each one is about, the words they use, and
questions to try, each with what a correct answer says and where to check it.
The same questions are on the app's **Guide: what to ask** page
(`ui/guide_content.py`), each with an **Ask** button.

## How these documents work

- **BNM** issues them; inside the documents BNM calls itself **"the Bank"**.
- They tell financial institutions what they **must** do and what they **may**
  or **should** do. Every numbered paragraph is marked:
  - **S (Standard)** = an obligation. The paragraph says *shall* or *must*.
    The assistant labels these **MUST**.
  - **G (Guidance)** = advice or an example. The paragraph says *may* or
    *should*. The assistant labels these **MAY**.
  - Appendices and tables carry no S/G marker, but can still state
    obligations (the e-KYC FAR table is one).
- Rules are cited by paragraph: "e-KYC para 8.4 (S), p. 6" means paragraph 8.4
  of the e-KYC document, a Standard, on printed page 6.

## The three documents at a glance

| | e-KYC | Fair Treatment of Financial Consumers (FTFC) | AML/CFT/CPF and TFS |
|---|---|---|---|
| **In one line** | How a financial institution may verify a new customer's identity **online**, without meeting them | How a financial service provider must **treat its customers fairly** | How a reporting institution must **stop money laundering and terrorism financing**, and apply sanctions |
| **Who it covers** | licensed banks and investment banks, life insurers, Islamic banks and family takaful operators, development financial institutions, e-money (payment instrument) issuers, money services businesses (e-KYC 5.2) | "financial service providers" (FSPs) | "reporting institutions" in the financial sector |
| **Size** | 30 pages, 108 passages | 54 pages, 232 passages | 181 pages, 879 passages |
| **Main topics** | Board approval and oversight (8.1–8.3), identity checks (8.4–8.18), AI and FAR (8.19–8.25), human-only e-KYC (8.26–8.28), weaknesses (8.29–8.30), reporting: monthly performance records submitted half-yearly (9), the FAR table (Appendix 2) | the Treat Customers Fairly Charter (9), culture (10), fair contract terms (11), information (12), fair dealing (13), advice (14), complaints (15), vulnerable consumers (16) | risk-based approach (10), compliance programme (11), customer due diligence for each sector (14A–14D), PEPs (15), wire transfers (19), cash reports (21), suspicious transaction reports (22), record keeping (24), sanctions (27–29) |

## Words you'll meet

| Term | Meaning |
|---|---|
| **e-KYC** | electronic Know-Your-Customer: checking a customer's identity remotely (ID document, face match, liveness check) |
| **FAR** | False Acceptance Rate: how often the e-KYC system wrongly accepts someone it should have rejected (e-KYC 8.21) |
| **Board** | the institution's board of directors |
| **CDD** | Customer Due Diligence: identifying and verifying customers and understanding their business |
| **Non-face-to-face (non-FTF)** | opening a relationship with a customer without meeting them in person |
| **PEP** | Politically Exposed Person: someone with a prominent public function, who needs extra checks |
| **CTR** | Cash Threshold Report: a report to BNM of large cash transactions |
| **STR** | Suspicious Transaction Report: a report to BNM when a transaction looks suspicious |
| **TFS** | Targeted Financial Sanctions: freezing funds of people and entities on UN Security Council sanctions lists and Malaysia's Domestic List (AML 27.1.1) |
| **Vulnerable consumer** | a customer who is more likely to be harmed (e.g. elderly, low digital skills) |
| **14A / 14B / 14C / 14D** | the same AML rules, repeated for banks, insurers and takaful, money services, and e-money issuers |

## Questions to try, and what a correct answer says

### e-KYC
| Question | A correct answer says | Check |
|---|---|---|
| Who has to approve a financial institution's risk appetite for e-KYC? | **MUST:** the **Board** approves the overall risk appetite and the internal framework for e-KYC | e-KYC 8.1 (S) |
| What does e-KYC paragraph 8.10 require? | **MAY:** document verification, biometric matching, liveness detection. These serve the **MUST** in 8.4: identity checks must be secure and effective | e-KYC 8.10 (G), 8.4 (S) |
| What must a bank do if FAR exceeds 5%? | **MUST:** if FAR is above 5% for any two months in a four-month period, notify BNM in writing within **7 working days** with an assessment and mitigation measures; the notification must cover the items in para 11 (performance, the false positives, fixes, new controls, the Board's view) | e-KYC App. 2 paras 10–11 |
| How often must a bank revalidate its e-KYC technology provider? | **MUST:** review or revalidate the requirements on the provider in 8.22 (independent assessment, certification) at least **once every 3 years**, or when the solution changes materially | e-KYC 8.24, 8.22 (S) |
| Can e-KYC be done entirely by a human, without AI? | **MAY:** yes, solely by a human representative, but it may give lower identity assurance and may not suit all circumstances. **MUST:** watch for a higher risk of misidentification and set internal safeguards | e-KYC 8.26, 8.27 (G), 8.28 (S) |
| What must a bank do when it finds vulnerabilities in its e-KYC solution? | **MUST:** keep monitoring for vulnerabilities, and adopt immediate mitigation measures | e-KYC 8.29 (S) |

### Fair Treatment of Financial Consumers
| Question | A correct answer says | Check |
|---|---|---|
| Does a bank have to publish its commitment to treating customers fairly? | **MUST:** yes, prominently on its website, with how it will carry it out | FTFC 9.1 (S) |
| What must senior management ensure for handling complaints? | **MUST:** enough resources, trained staff, and set timeframes for resolving complaints and claims | FTFC 15.4 (S) |
| What should a bank do for vulnerable consumers? | **MUST:** assess the needs of vulnerable consumers and put in place policies and procedures to meet them | FTFC 16.2 (S) |

### AML/CFT/CPF and TFS
| Question | A correct answer says | Check |
|---|---|---|
| How long must a bank keep customer and transaction records? | **MUST:** at least **six years** after the transaction or the end of the relationship | AML 24.3 (S) |
| When must a bank report a cash transaction to BNM? | **MUST:** cash transactions of **RM25,000 and above** in a day, in the same account, in a cash threshold report to BNM | AML 21.3.1, 21.1.1 (S) |
| What must a bank do if it suspects a transaction? | **MUST:** promptly submit a suspicious transaction report to BNM, and not disclose it or related information (no tipping off), except as AMLA section 14A(3) allows | AML 22.1.1, 23.1, 23.2 (S) |
| What checks apply to a foreign politically exposed person? | **MUST:** have a system to identify foreign PEPs, and apply enhanced due diligence to them | AML 15.2.1, 15.2.3 (S) |
| Can a bank onboard a customer without meeting them face to face? | **MAY:** yes. **MUST:** with Board approval, read together with the e-KYC document, with identity checks that are secure and effective | AML 14A.15.1 (G), 14A.15.2, 14A.15.3, 14A.15.7 (S) |

### What the agent adds (asked in Agentic mode)
| Question | What to look for |
|---|---|
| How do the e-KYC verification rules differ from AML's non-face-to-face rules? | the planner splits it into one search per document; the answer cites both documents |
| What does para 8.2 say? | 8.2 exists in all three documents: the agent asks which one |
| What is the interest rate cap on credit cards? | not in these documents: the agent refuses before searching |

A follow-up also shows the agent: ask *"Can a customer be onboarded without
meeting face to face?"*, then *"what about takaful operators?"*. The follow-up
is rewritten from the conversation and the answer switches to the **14B**
(insurance and takaful) rules.

## How to read an answer

1. **The badge** at the top: *Answered*, *Partly answered*, *Not in the documents*,
   *Needs clarification* or *Outside the documents*.
2. **In short:** the direct answer in one or two sentences.
3. **MUST: obligations:** what the institution must do, each with its paragraph.
4. **MAY: options and guidance:** what it may or should do, or what BNM may do
   (e.g. enforcement).
5. **INFO: background:** definitions and scope.
6. **Sources:** each cited paragraph with the passage and a link that opens the
   PDF at that page. To check an answer, open the source and read the
   paragraph: it must support the statement.
7. **Agent steps:** what the agent did (plan, search, review, write, verify).

*Citations verified* means code checked that every statement's supporting
quote is in a passage it cites, and that no MUST rests only on guidance (G)
paragraphs. It doesn't check that the answer is complete or that the meaning is
right: compare it with the expected answers above.
