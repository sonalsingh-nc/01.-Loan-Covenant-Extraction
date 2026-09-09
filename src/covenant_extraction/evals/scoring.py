"""Aggregate precision/recall/threshold-accuracy metrics from matched
covenants (see matching.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from covenant_extraction.evals.matching import CovenantMatch


@dataclass(frozen=True)
class EvalScore:
    num_gold: int
    num_extracted: int
    num_hits: int  # gold covenants that WERE found (citation matched)
    num_misses: int  # gold covenants not found at all (false negatives)
    num_false_positives: int  # extracted covenants matching no gold entry
    recall: float  # hits / gold -- did we find everything that's really there?
    precision: float  # hits / extracted -- is everything we extracted real?
    threshold_accuracy: float  # of hits, fraction with correct threshold + comparison
    covenant_type_accuracy: float  # of hits, fraction with correct covenant_type


def score_matches(matches: List[CovenantMatch], unmatched_extracted: List[Dict[str, Any]]) -> EvalScore:
    hits = [m for m in matches if m.extracted is not None]
    num_gold = len(matches)
    num_hits = len(hits)
    num_false_positives = len(unmatched_extracted)
    num_extracted = num_hits + num_false_positives

    recall = num_hits / num_gold if num_gold else 1.0
    precision = num_hits / num_extracted if num_extracted else 1.0

    threshold_correct = sum(1 for m in hits if m.threshold_correct and m.comparison_correct)
    threshold_accuracy = threshold_correct / num_hits if num_hits else 1.0

    type_correct = sum(1 for m in hits if m.covenant_type_correct)
    covenant_type_accuracy = type_correct / num_hits if num_hits else 1.0

    return EvalScore(
        num_gold=num_gold,
        num_extracted=num_extracted,
        num_hits=num_hits,
        num_misses=num_gold - num_hits,
        num_false_positives=num_false_positives,
        recall=recall,
        precision=precision,
        threshold_accuracy=threshold_accuracy,
        covenant_type_accuracy=covenant_type_accuracy,
    )
