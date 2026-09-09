"""Matches extracted financial covenants against a hand-annotated gold set.

Matching is keyed on citation-text overlap, not name equality: the model
may phrase a covenant's `name` differently than the gold annotation (e.g.
"Consolidated Leverage Ratio" vs. "Leverage Ratio"), but the actual clause
it quoted in `citation.source_text` either is or isn't the right one -- so
each gold entry carries a short, distinctive `citation_contains` substring
picked by the human annotator from the real clause, and a match requires
that substring to literally appear in an extracted covenant's citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class CovenantMatch:
    """One gold covenant's match result. `extracted` is None if no
    extracted covenant's citation contained the gold's anchor text at all
    (a miss) -- in that case the correctness flags are meaningless (False)."""

    gold: Dict[str, Any]
    extracted: Optional[Dict[str, Any]]
    threshold_correct: bool
    comparison_correct: bool
    covenant_type_correct: bool


def match_covenants(
    gold_covenants: List[Dict[str, Any]], extracted_covenants: List[Dict[str, Any]]
) -> Tuple[List[CovenantMatch], List[Dict[str, Any]]]:
    """Match each gold covenant to at most one extracted covenant.

    Returns:
        (matches, unmatched_extracted) -- `matches` has exactly one
        CovenantMatch per gold covenant (`extracted=None` means it was
        missed entirely -- a false negative). `unmatched_extracted` is
        every extracted covenant that didn't match any gold entry (false
        positives: fabricated, duplicate, or out-of-scope extractions).
    """
    matched_indices = set()
    matches: List[CovenantMatch] = []

    for gold in gold_covenants:
        anchor = _normalize(gold["citation_contains"])
        found = None
        found_index = None
        for i, extracted in enumerate(extracted_covenants):
            if i in matched_indices:
                continue
            source_text = _normalize(extracted.get("citation", {}).get("source_text", ""))
            if anchor and anchor in source_text:
                found = extracted
                found_index = i
                break

        if found is None:
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

        matched_indices.add(found_index)
        threshold_correct = _thresholds_match(found.get("threshold"), gold.get("threshold"))
        comparison_correct = found.get("comparison") == gold.get("comparison")
        covenant_type_correct = found.get("covenant_type") == gold.get("covenant_type")
        matches.append(
            CovenantMatch(
                gold=gold,
                extracted=found,
                threshold_correct=threshold_correct,
                comparison_correct=comparison_correct,
                covenant_type_correct=covenant_type_correct,
            )
        )

    unmatched_extracted = [e for i, e in enumerate(extracted_covenants) if i not in matched_indices]
    return matches, unmatched_extracted


def _thresholds_match(extracted_threshold: Any, gold_threshold: Any, tolerance: float = 1e-6) -> bool:
    if extracted_threshold is None or gold_threshold is None:
        return False
    return abs(float(extracted_threshold) - float(gold_threshold)) <= tolerance
