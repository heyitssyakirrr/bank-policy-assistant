# Evaluation: 20261024-1457-smoke
2026-10-24 14:57 · commit `fdc8d1e` · prompts `2026-10-24.8` · 11 items evaluated (40 in the golden set)

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
| dense only | 8 | 88% | 72% | 0.75 |
| BM25 only | 8 | 100% | 78% | 0.74 |
| hybrid (RRF) | 8 | 100% | 78% | 0.94 |
| + exact lookup | 8 | 100% | 78% | 0.85 |
| + rerank | 8 | 100% | 88% | 0.81 |
| + selection | 8 | 100% | 88% | 0.81 |

**Where adding a stage made an item worse** (from hybrid onwards, each stage builds on the last):
- lookup-3 (direct_lookup): hybrid (RRF) → + exact lookup: MRR@5 1.00 → 0.33; top 5 before ['ftfc:9.1', 'ftfc:9.2', 'ftfc:9.4', 'ftfc:15.1', 'ftfc:8.2'], after ['ekyc:9.1', 'amlcft:9.1', 'ftfc:9.1', 'ftfc:9.2', 'ftfc:9.4']
- compare-1 (comparison): + exact lookup → + rerank: MRR@5 1.00 → 0.50; top 5 before ['amlcft:14A.15.7', 'amlcft:14C.16.8', 'amlcft:14B.17.7', 'amlcft:14A.15.2', 'amlcft:14D.16.2'], after ['ekyc:6.2', 'ekyc:8.10', 'amlcft:14A.15.7', 'amlcft:14C.16.8', 'amlcft:14A.15.2']
- sector-1 (sector_specific): + exact lookup → + rerank: MRR@5 1.00 → 0.50; top 5 before ['amlcft:14B.17.1', 'amlcft:14B.17.9', 'amlcft:14B.17.2', 'amlcft:14B.17.5', 'amlcft:14B.17.8'], after ['amlcft:14B.17.9', 'amlcft:14B.17.1', 'amlcft:14B.17.2', 'amlcft:14B.17.7', 'amlcft:14B.17.5']

Agentic mode, all evidence it gathered (planner searches, lookups, follow-ups): recall 83% over 3 answerable questions (includes follow-ups, so a different set from the table above; not limited to 5 passages).

## Answers by mode
| Metric | traditional | agentic |
|---|---|---|
| Outcome accuracy | 80% | 100% |
| Answered, when answerable | 100% | 100% |
| Refused, when out of scope | 100% | 100% |
| Asked to clarify, when ambiguous | 0% | 100% |
| Citation recall | 50% | 83% |
| Evidence recall | 50% | 83% |
| Answers passing every verifier check | 100% | 100% |
| Faithfulness (judge) | 100% | 100% |
| Relevancy 1-5 (judge) | 3.67 | 5.00 |
| Judge agreement with gemini-3.8-flash | - | - |
| Answers cross-checked | 0 | 0 |
| Answers judged by the generator's own model (fallback) | 0 | 0 |
| LLM calls per question | 1.00 | 2.60 |
| Tokens per question | 2798 | 4744 |
| Latency p50 (s) | - | - |
| Latency p95 (s) | - | - |
| Errors (outages, crashes; excluded above) | 0 | 0 |

Latency counts only questions answered with live API calls (traditional: 0, agentic: 0); cached replays are excluded.

## By category (outcome accuracy · citation recall)
| Category | traditional | agentic |
|---|---|---|
| paraphrased | 100% · 100% | 100% · 50% |
| cross_reference | 100% · 50% | 100% · 100% |
| follow_up | 100% · 0% | 100% · 100% |
| out_of_scope | 100% · - | 100% · - |
| ambiguous | 0% · - | 100% · - |

## Failures to analyse, by category

**paraphrased**
- **fact-6** (agentic): never retrieved: ['ekyc:Appendix 2:11']

**cross_reference**
- **xref-1** (traditional): never retrieved: ['ekyc:8.4']

**follow_up**
- **followup-1** (traditional): never retrieved: ['amlcft:14B.17.1']

**ambiguous**
- **ambiguous-1** (traditional): expected clarify, got answer (answered)

## Cost of this run
| Calls | Count |
|---|---|
| LLM (all cached) | 0 |
| query embeddings | 0 |
| Cohere reranks | 0 |
| cache hits (free) | 93 |
