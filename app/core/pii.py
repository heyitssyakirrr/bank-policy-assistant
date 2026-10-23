"""PII masking: a data-protection guardrail (PROJECT_PLAN §5.6).

A question like "customer 900101-14-5678 wants to open an account remotely"
must not send the MyKad number to Gemini, LangSmith, logs or the audit
tables. Policy questions never need a real number, so masking happens as
soon as the question arrives and costs nothing in answer quality.

Regex is the right tool here: the formats are fixed.
  NRIC     YYMMDD-PB-###G (dashes, spaces or nothing between parts; a
           plausible month and day)
  CARD     13-19 digits, optionally grouped by spaces or dashes, passing the
           Luhn checksum
  ACCOUNT  any other 10-20 digits, contiguous or in groups of 3+ (a mistyped
           card number is still masked, as an account)
Paragraph numbers ("14A.9.6"), percentages and amounts are never touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NRIC_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})[- ]?(\d{2})[- ]?(\d{4})(?!\d)")
# Digit groups of 3+ joined by a space or dash: "5140 1234 5678", "4111-1111-1111-1111".
# Two-digit groups ("03-2698 8044", "paragraphs 10 11 12") and amounts ("1,000,000") don't match.
_GROUPED_DIGITS_RE = re.compile(r"(?<![\d,.])\d{3,}(?:[ -]\d{3,}){0,5}(?![\d,.])")


@dataclass(frozen=True, slots=True)
class MaskResult:
    text: str
    masked: list[str] = field(default_factory=list)  # kinds found, e.g. ["NRIC", "CARD"]


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _plausible_birth_date(month: str, day: str) -> bool:
    return 1 <= int(month) <= 12 and 1 <= int(day) <= 31


def mask_pii(text: str) -> MaskResult:
    found: list[str] = []

    def nric(match: re.Match[str]) -> str:
        if not _plausible_birth_date(match.group(2), match.group(3)):
            return match.group(0)
        found.append("NRIC")
        return "[NRIC]"

    def card_or_account(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 10:
            return raw
        kind = "CARD" if 13 <= len(digits) <= 19 and _luhn_valid(digits) else "ACCOUNT"
        found.append(kind)
        return f"[{kind}]"

    masked = _NRIC_RE.sub(nric, text)
    masked = _GROUPED_DIGITS_RE.sub(card_or_account, masked)
    return MaskResult(masked, list(dict.fromkeys(found)))
