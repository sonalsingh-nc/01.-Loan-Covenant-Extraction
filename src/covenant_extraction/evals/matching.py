"""Matches extracted financial covenants against a hand-annotated gold set.

Matching is keyed on citation-text overlap, not name equality: the model
may phrase a covenant's `name` differently than the gold annotation (e.g.
"Consolidated Leverage Ratio" vs. "Leverage Ratio"), but the actual clause
it quoted in `citation.source_text` either is or isn't the right one -- so
each gold entry carries a short, distinctive `citation_contains` substring
picked by the human annotator from the real clause, and a match requires
that substring to literally appear in an extracted covenant's citation.

Gold entries may also carry:

- `citation_contains` as a list of alternative anchors (any one matching is
  enough), for clauses that different models quote with different
  boundaries.
- `schedule`: a step-down/step-up list of `{"period", "threshold"}` levels.
  The extraction schema holds one threshold per row, so a correct extractor
  emits one row per level. Every extracted row whose citation hits the
  anchor is absorbed into the same gold covenant (one hit, not several false
  positives), and the set of extracted thresholds is scored against the
  schedule in field_metrics.py.
- `required: false`: a borderline item (a springing cash-deposit trigger, a
  formula-only minimum with no fixed number, a CMBS cash-management trigger)
  that a reasonable extractor may or may not report. Extracting it is not a
  false positive and missing it is not a miss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_WHITESPACE_RE = re.compile(r"\s+")
# HTML-to-text conversion often leaves a space before punctuation
# ("Ratio . The Borrower", "( x )"), and different loaders do this
# differently -- so drop those spaces on both sides of the comparison.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})


def normalize_text(text: str) -> str:
    text = (text or "").translate(_QUOTES)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _SPACE_AFTER_OPEN_RE.sub(r"\1", text)


# Kept for backwards compatibility with earlier callers.
_normalize = normalize_text


def gold_anchors(gold: Dict[str, Any]) -> List[str]:
    anchors = gold.get("citation_contains") or []
    if isinstance(anchors, str):
        anchors = [anchors]
    return [normalize_text(a) for a in anchors if a and a.strip()]


def is_required(gold: Dict[str, Any]) -> bool:
    return gold.get("required", True) is not False


@dataclass(frozen=True)
class CovenantMatch:
    """One gold covenant's match result. `extracted` is None if no
    extracted covenant's citation contained the gold's anchor text at all
    (a miss) -- in that case the correctness flags are meaningless (False).

    `extra_extracted` holds additional rows absorbed by a scheduled
    (step-down) gold covenant, one per extra threshold level."""

    gold: Dict[str, Any]
    extracted: Optional[Dict[str, Any]]
    threshold_correct: bool
    comparison_correct: bool
    covenant_type_correct: bool
    extra_extracted: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def all_extracted(self) -> List[Dict[str, Any]]:
        if self.extracted is None:
            return []
        return [self.extracted, *self.extra_extracted]

    @property
    def required(self) -> bool:
        return is_required(self.gold)


def _citation_hits(anchors: List[str], extracted: Dict[str, Any]) -> bool:
    source_text = normalize_text((extracted.get("citation") or {}).get("source_text", ""))
    return any(anchor in source_text for anchor in anchors)


def _pick_primary(gold: Dict[str, Any], rows: List[Tuple[int, Dict[str, Any]]]) -> Tuple[int, Dict[str, Any]]:
    """Among several anchor hits for a scheduled covenant, prefer the row
    whose threshold equals the gold's headline (first-level) threshold."""
    for i, row in rows:
        if _thresholds_match(row.get("threshold"), gold.get("threshold")):
            return i, row
    return rows[0]


def match_covenants(
    gold_covenants: List[Dict[str, Any]], extracted_covenants: List[Dict[str, Any]]
) -> Tuple[List[CovenantMatch], List[Dict[str, Any]]]:
    """Match each gold covenant to extracted covenants.

    Returns:
        (matches, unmatched_extracted) -- `matches` has exactly one
        CovenantMatch per gold covenant (`extracted=None` means it was
        missed entirely -- a false negative). `unmatched_extracted` is
        every extracted covenant that didn't match any gold entry (false
        positives: fabricated, duplicate, or out-of-scope extractions).
    """
    claimed = set()
    matches: List[CovenantMatch] = []

    for gold in gold_covenants:
        anchors = gold_anchors(gold)
        hits = [
            (i, e)
            for i, e in enumerate(extracted_covenants)
            if i not in claimed and anchors and _citation_hits(anchors, e)
        ]
        if not hits:
            matches.append(
                CovenantMatch(
                    gold=gold,
                    extracted=None,
                    threshold_correct=False,
                    comparison_correct=False,
                    covenant_type_correct=False,
                )
            )
            continue

        if gold.get("schedule"):
            primary_index, found = _pick_primary(gold, hits)
            extra = tuple(e for i, e in hits if i != primary_index)
            claimed.update(i for i, _ in hits)
        else:
            primary_index, found = hits[0]
            extra = ()
            claimed.add(primary_index)

        matches.append(
            CovenantMatch(
                gold=gold,
                extracted=found,
                threshold_correct=_thresholds_match(found.get("threshold"), gold.get("threshold")),
                comparison_correct=found.get("comparison") == gold.get("comparison"),
                covenant_type_correct=found.get("covenant_type") == gold.get("covenant_type"),
                extra_extracted=extra,
            )
        )

    unmatched_extracted = [e for i, e in enumerate(extracted_covenants) if i not in claimed]
    return matches, unmatched_extracted


def _thresholds_match(extracted_threshold: Any, gold_threshold: Any, tolerance: float = 1e-6) -> bool:
    if extracted_threshold is None or gold_threshold is None:
        return False
    return abs(float(extracted_threshold) - float(gold_threshold)) <= tolerance
