# Evaluation: 20261026-1615-only
2026-10-26 16:15 · commit `c565c5d` · prompts `2026-10-24.8` · 8 items evaluated (40 in the golden set)

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
| dense only | 8 | 88% | 75% | 0.88 |
| BM25 only | 8 | 100% | 81% | 1.00 |
| hybrid (RRF) | 8 | 100% | 81% | 1.00 |
| + exact lookup | 8 | 100% | 81% | 1.00 |
| + rerank | 8 | 100% | 81% | 1.00 |
| + selection | 8 | 100% | 81% | 1.00 |

**Where adding a stage made an item worse** (from hybrid onwards, each stage builds on the last):
- none

## Answers by mode
| Metric | traditional |
|---|---|
| Outcome accuracy | 100% |
| Answered, when answerable | 100% |
| Refused, when out of scope | - |
| Asked to clarify, when ambiguous | - |
| Citation recall | 81% |
| Evidence recall | 81% |
| Answers passing every verifier check | 100% |
| Faithfulness (judge) | 100% |
| Relevancy 1-5 (judge) | 5.00 |
| Judge agreement with gemini-3.8-flash | - |
| Answers cross-checked | 0 |
| Answers judged by the generator's own model (fallback) | 0 |
| LLM calls per question | 1.12 |
| Tokens per question | 3189 |
| Latency p50 (s) | - |
| Latency p95 (s) | - |
| Errors (outages, crashes; excluded above) | 0 |

Latency counts only questions answered with live API calls (traditional: 0); cached replays are excluded.

## By category (outcome accuracy · citation recall)
| Category | traditional |
|---|---|
| direct_lookup | 100% · 100% |
| cross_reference | 100% · 62% |

## Failures to analyse, by category

**cross_reference**
- **xref-1** (traditional): never retrieved: ['ekyc:8.4']
- **xref-3** (traditional): never retrieved: ['ekyc:8.13']
- **xref-4** (traditional): never retrieved: ['amlcft:14A.15.7']

## Cost of this run
| Calls | Count |
|---|---|
| LLM (all cached) | 0 |
| query embeddings | 0 |
| Cohere reranks | 1 |
| cache hits (free) | 72 |
