"""BM25 as sparse vectors, for Qdrant.

How BM25 maps onto Qdrant
-------------------------
BM25 score(query, doc) = sum over query terms t of  IDF(t) * TF_part(t, doc)

  TF_part(t, doc) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
  IDF(t)          = how rare t is across all chunks

We split the work:
  * Here (at index time) each chunk becomes a sparse vector {term_id: TF_part}.
  * A query becomes {term_id: 1.0} for each of its terms.
  * Qdrant, with `Modifier.IDF` on the sparse field, multiplies in IDF(t)
    itself, computed from the whole collection, and keeps it up to date as
    chunks are added or removed.
  Dot product of the two vectors = the BM25 score. No search engine needed.

Why our own tokenizer
---------------------
Generic tokenizers split "14A.9.6" into "14a", "9", "6" and drop the
paragraph number people actually type. This one keeps paragraph ids and
compound terms ("e-kyc", "aml/cft/cpf") whole, and also indexes their parts
so "AML" still matches "AML/CFT/CPF".
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Paragraph ids first (14a.9.6, 8.10), then words, keeping - and / inside compounds.
TOKEN_RE = re.compile(r"\d{1,2}[a-z]?(?:\.\d{1,2})+|[a-z0-9]+(?:[-/][a-z0-9]+)*")
PARAGRAPH_ID_RE = re.compile(r"\d{1,2}[a-z]?(?:\.\d{1,2})+")

# Words too common to help ranking. Modal verbs ("shall", "may", "must") are
# kept on purpose: in a policy they separate obligations from guidance.
STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in into is it its of on or that the
    their them these this those to was were which will with within such any all other than
    under where when who whom whose also""".split()
)


def _light_stem(word: str) -> str:
    """'requirements' -> 'requirement', 'institutions' -> 'institution'.
    Deliberately minimal: a full stemmer would also merge words that mean
    different things in regulation."""
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if PARAGRAPH_ID_RE.fullmatch(raw):
            tokens.append(raw)  # "14a.9.6" stays one exact token
            continue
        if raw in STOPWORDS:
            continue
        tokens.append(_light_stem(raw))
        if "-" in raw or "/" in raw:  # "aml/cft/cpf" also counts as "aml", "cft", "cpf"
            tokens.extend(
                _light_stem(part) for part in re.split(r"[-/]", raw) if len(part) > 1 and part not in STOPWORDS
            )
    return tokens


def term_id(token: str) -> int:
    """Stable 32-bit id for a token. Python's built-in hash() changes between
    runs, which would scramble an index built yesterday; blake2b never does."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


@dataclass(frozen=True, slots=True)
class SparseVec:
    indices: list[int]
    values: list[float]


def _to_sparse(weights: dict[int, float]) -> SparseVec:
    ordered = sorted(weights.items())
    return SparseVec(indices=[i for i, _ in ordered], values=[v for _, v in ordered])


@dataclass(frozen=True, slots=True)
class Bm25Encoder:
    avg_doc_len: float
    k1: float = 1.2  # how fast repeated words stop adding score (standard value)
    b: float = 0.75  # how much long chunks are penalised (standard value)

    @classmethod
    def fit(cls, documents: Iterable[str]) -> Bm25Encoder:
        lengths = [len(tokenize(doc)) for doc in documents]
        return cls(avg_doc_len=(sum(lengths) / len(lengths)) if lengths else 1.0)

    def encode_document(self, text: str) -> SparseVec:
        tokens = tokenize(text)
        length_norm = 1 - self.b + self.b * len(tokens) / self.avg_doc_len
        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            tf_part = tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
            tid = term_id(token)
            weights[tid] = weights.get(tid, 0.0) + tf_part  # (sum on the rare hash collision)
        return _to_sparse(weights)

    @staticmethod
    def encode_query(text: str) -> SparseVec:
        return _to_sparse({term_id(token): 1.0 for token in set(tokenize(text))})
