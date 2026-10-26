# Evaluation: 20261026-0338-full
2026-10-26 03:38 · commit `c565c5d` · prompts `2026-10-24.8` · 40 items evaluated (40 in the golden set)

| Role | Model |
|---|---|
| planner | google_genai:gemini-3.1-flash-lite |
| reviewer | google_genai:gemini-3.1-flash-lite |
| generator | google_genai:gemini-3.5-flash-lite |
| judge | google_genai:gemini-3.1-flash-lite |
| judge cross-check | google_genai:gemini-3.8-flash |

**How the golden set was built:** questions written by Claude Code from the paragraphs in `chunks.jsonl` (gold labels = those paragraphs); hard cases (comparison, cross-reference, follow-up, out-of-scope, ambiguous) written by hand from the paragraph text. Claude writes the questions, Gemini 3.5 Flash-Lite answers, Gemini 3.1 Flash-Lite judges. Claude Code re-read the gold paragraphs of 32 items against their questions. Syakir spot-checked 0 item(s) against the PDFs (not done yet: the plan asks for at least 10).

## How to read this
- **Hit@5**: did any gold paragraph appear in the first 5 passages? **Recall@5**: what share of the gold
  paragraphs did. **MRR@5**: 1 / the position of the first gold paragraph (1.00 = first place).
- **Outcome**: answered when the documents cover the question, refused when they don't, asked a clarifying
  question when it is ambiguous. A crash or model outage is an **error**, counted apart, never as a refusal.
- **Evidence recall**: gold paragraphs the mode *retrieved*. **Citation recall**: gold paragraphs the answer
  *cited*. Retrieved-but-not-cited points at the generator; never-retrieved points at the search.
- **Faithfulness**: the judge's share of statements supported by their cited passages (partly = 0.5).
  A "must" citing a passage that only says "may" counts as unsupported.

## Retrieval ablation (first 5 passages, single-turn answerable questions)
| Stage | n | Hit@5 | Recall@5 | MRR@5 |
|---|---|---|---|---|
| dense only | 28 | 93% | 83% | 0.86 |
| BM25 only | 28 | 86% | 70% | 0.72 |
| hybrid (RRF) | 28 | 96% | 81% | 0.85 |
| + exact lookup | 28 | 96% | 81% | 0.82 |
| + rerank | 28 | 96% | 84% | 0.86 |
| + selection | 28 | 96% | 83% | 0.86 |

**Where adding a stage made an item worse** (from hybrid onwards, each stage builds on the last):
- lookup-3 (direct_lookup): hybrid (RRF) → + exact lookup: MRR@5 1.00 → 0.33; top 5 before ['ftfc:9.1', 'ftfc:9.2', 'ftfc:9.4', 'ftfc:15.1', 'ftfc:8.2'], after ['ekyc:9.1', 'amlcft:9.1', 'ftfc:9.1', 'ftfc:9.2', 'ftfc:9.4']
- compare-1 (comparison): + exact lookup → + rerank: MRR@5 1.00 → 0.50; top 5 before ['amlcft:14A.15.7', 'amlcft:14C.16.8', 'amlcft:14B.17.7', 'amlcft:14A.15.2', 'amlcft:14D.16.2'], after ['ekyc:6.2', 'ekyc:8.10', 'amlcft:14A.15.7', 'amlcft:14C.16.8', 'amlcft:14A.15.2']
- compare-4 (comparison): + exact lookup → + rerank: MRR@5 0.50 → 0.20; top 5 before ['ekyc:Appendix 4:11', 'ekyc:9.1', 'ekyc:9.2', 'amlcft:24.3', 'amlcft:24.5'], after ['ekyc:8.16', 'ekyc:Appendix 4:11', 'ekyc:8.9', 'ekyc:8.4', 'amlcft:24.1']
- sector-1 (sector_specific): + exact lookup → + rerank: MRR@5 1.00 → 0.50; top 5 before ['amlcft:14B.17.1', 'amlcft:14B.17.9', 'amlcft:14B.17.2', 'amlcft:14B.17.5', 'amlcft:14B.17.8'], after ['amlcft:14B.17.9', 'amlcft:14B.17.1', 'amlcft:14B.17.2', 'amlcft:14B.17.7', 'amlcft:14B.17.5']

Agentic mode, all evidence it gathered (planner searches, lookups, follow-ups): recall 92% over 32 answerable questions (includes follow-ups, so a different set from the table above; not limited to 5 passages).

## Answers by mode
| Metric | traditional | agentic |
|---|---|---|
| Outcome accuracy | 95% | 100% |
| Answered, when answerable | 100% | 100% |
| Refused, when out of scope | 100% | 100% |
| Asked to clarify, when ambiguous | 0% | 100% |
| Citation recall | 80% | 91% |
| Evidence recall | 82% | 92% |
| Answers passing every verifier check | 100% | 100% |
| Faithfulness (judge) | 98% | 99% |
| Relevancy 1-5 (judge) | 4.84 | 5.00 |
| Judge agreement with gemini-3.8-flash | - | - |
| Answers cross-checked | 0 | 0 |
| Answers judged by the generator's own model (fallback) | 1 | 3 |
| LLM calls per question | 1.05 | 2.85 |
| Tokens per question | 2778 | 4907 |
| Latency p50 (s) | 4.38 | 25.08 |
| Latency p95 (s) | 11.03 | 69.89 |
| Errors (outages, crashes; excluded above) | 0 | 0 |

Latency counts only questions answered with live API calls (traditional: 35, agentic: 35); cached replays are excluded.

## By category (outcome accuracy · citation recall)
| Category | traditional | agentic |
|---|---|---|
| direct_lookup | 100% · 100% | 100% · 100% |
| paraphrased | 100% · 87% | 100% · 92% |
| comparison | 100% · 56% | 100% · 83% |
| cross_reference | 100% · 62% | 100% · 100% |
| sector_specific | 100% · 100% | 100% · 75% |
| follow_up | 100% · 75% | 100% · 100% |
| out_of_scope | 100% · - | 100% · - |
| ambiguous | 0% · - | 100% · - |

## Failures to analyse, by category

**paraphrased**
- **fact-3** (traditional): retrieved but not cited: ['ekyc:8.27']
- **fact-3** (agentic): never retrieved: ['ekyc:8.27']
- **fact-6** (agentic): never retrieved: ['ekyc:Appendix 2:11']
- **fact-10** (traditional): never retrieved: ['ftfc:16.2']; faithfulness 0.43 (verdicts ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes'])

**comparison**
- **compare-1** (traditional): retrieved but not cited: ['amlcft:14A.15.7']; never retrieved: ['ekyc:8.4', 'amlcft:14A.15.9']
- **compare-1** (agentic): never retrieved: ['amlcft:14A.15.7']
- **compare-3** (traditional): never retrieved: ['ekyc:8.10']
- **compare-4** (traditional): never retrieved: ['ekyc:9.1', 'ekyc:9.2', 'amlcft:24.3']
- **compare-5** (agentic): never retrieved: ['ekyc:8.27']
- **compare-6** (traditional): retrieved but not cited: ['amlcft:14A.15.1']; never retrieved: ['amlcft:14C.16.1', 'amlcft:14C.16.10', 'amlcft:14C.16.11', 'amlcft:14C.16.12']
- **compare-6** (agentic): retrieved but not cited: ['amlcft:14C.16.10']; never retrieved: ['amlcft:14A.15.1', 'amlcft:14C.16.1']

**cross_reference**
- **xref-1** (traditional): never retrieved: ['ekyc:8.4']
- **xref-2** (agentic): faithfulness 0.75 (verdicts ['yes', 'partly'])
- **xref-3** (traditional): never retrieved: ['ekyc:8.13']
- **xref-4** (traditional): never retrieved: ['amlcft:14A.15.7']

**sector_specific**
- **sector-4** (agentic): never retrieved: ['amlcft:14C.16.12']

**follow_up**
- **followup-1** (traditional): never retrieved: ['amlcft:14B.17.1']

**ambiguous**
- **ambiguous-1** (traditional): expected clarify, got answer (answered)
- **ambiguous-2** (traditional): expected clarify, got answer (partially_answered)

## Cost of this run
| Calls | Count |
|---|---|
| google_genai:gemini-3.5-flash-lite | 89 |
| google_genai:gemini-3.1-flash-lite | 185 |
| google_genai:gemini-3.8-flash | 3 |
| query embeddings | 68 |
| Cohere reranks | 68 |
| cache hits (free) | 211 |
