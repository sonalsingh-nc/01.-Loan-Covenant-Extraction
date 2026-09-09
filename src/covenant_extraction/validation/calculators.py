"""Deterministic, regex-based re-extraction of numeric covenant thresholds.

This acts as an independent cross-check on the LLM's extracted `threshold`
and `comparison` fields: we re-parse the cited source text ourselves and
flag any mismatch, rather than trusting the model's numbers outright.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import re

from covenant_extraction.extraction.schema import Comparison, FinancialCovenant

_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?::|to)\s*1(?:\.0+)?")
_DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_NOT_EXCEED_RE = re.compile(r"\b(exceed|greater than)\b", re.IGNORECASE)
_NOT_LESS_THAN_RE = re.compile(r"\b(less than|not less than)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RecomputedThreshold:
    threshold: float
    comparison: Comparison


def recompute_threshold(source_text: str) -> Optional[RecomputedThreshold]:
    """Independently parse a numeric threshold + direction out of raw text.

    Tries a ratio pattern (e.g. "3.50:1.00" or "3.50 to 1.00") first, then a
    dollar-amount pattern (e.g. "$10,000,000"). Direction is inferred from
    nearby "exceed" / "less than" language.

    Returns:
        RecomputedThreshold if a confident parse was possible, else None.
    """
    comparison = (
        Comparison.NOT_TO_EXCEED
        if _NOT_EXCEED_RE.search(source_text)
        else Comparison.NOT_LESS_THAN
        if _NOT_LESS_THAN_RE.search(source_text)
        else None
    )
    if comparison is None:
        return None

    ratio_match = _RATIO_RE.search(source_text)
    if ratio_match:
        return RecomputedThreshold(threshold=float(ratio_match.group(1)), comparison=comparison)

    dollar_match = _DOLLAR_RE.search(source_text)
    if dollar_match:
        value = float(dollar_match.group(1).replace(",", ""))
        return RecomputedThreshold(threshold=value, comparison=comparison)

    return None


def _threshold_string_variants(threshold: float) -> set:
    """Plausible literal string forms `threshold` might appear as in prose
    -- e.g. 4.5 as "4.5", "4.50", or comma-grouped "4.50"; 10000000.0 as
    "10,000,000" or "10000000"."""
    variants = {f"{threshold:.2f}", f"{threshold:g}", f"{threshold:,.2f}"}
    if threshold == int(threshold):
        variants.add(str(int(threshold)))
        variants.add(f"{int(threshold):,}")
    return variants


def citation_contains_threshold(threshold: float, source_text: str) -> bool:
    """True if some plausible string form of `threshold` literally appears
    in `source_text` -- a blunt but effective check that the number wasn't
    pulled from somewhere outside the quoted clause (e.g. confused with a
    different, nearby covenant's threshold). Independent of, and a lower
    bar than, `recompute_threshold`'s fuller ratio/dollar-amount parse --
    this still catches a fabricated number even when the citation lacks a
    direction keyword ("exceed"/"less than") for that fuller parse to run
    against at all."""
    return any(variant in source_text for variant in _threshold_string_variants(threshold))


@dataclass(frozen=True)
class CovenantCheck:
    covenant: FinancialCovenant
    recomputed: Optional[RecomputedThreshold]
    matches: bool  # True if recomputed is None (nothing to cross-check) or values agree
    citation_grounded: bool  # True if `covenant.threshold` literally appears in its own citation text


def check_financial_covenant(covenant: FinancialCovenant, tolerance: float = 1e-6) -> CovenantCheck:
    """Cross-check an extracted FinancialCovenant against an independent,
    regex-based re-parse of its own citation source text."""
    recomputed = recompute_threshold(covenant.citation.source_text)
    citation_grounded = citation_contains_threshold(covenant.threshold, covenant.citation.source_text)

    if recomputed is None:
        return CovenantCheck(covenant=covenant, recomputed=None, matches=True, citation_grounded=citation_grounded)

    matches = (
        recomputed.comparison == covenant.comparison
        and abs(recomputed.threshold - covenant.threshold) <= tolerance
    )
    return CovenantCheck(covenant=covenant, recomputed=recomputed, matches=matches, citation_grounded=citation_grounded)
