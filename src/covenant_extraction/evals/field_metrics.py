"""Field-level scoring for matched financial covenants.

matching.py decides *which* extracted row belongs to which gold covenant;
this module asks how good each matched row is, one field at a time:

- threshold, two ways: exact (to 1e-6) and within a numeric tolerance
  (default 1% relative), so "3.5 vs 3.50" and rounding noise don't hide
  real errors, while sign flips or a dropped "000" still fail.
- comparison (not_to_exceed / not_less_than), covenant_type, unit, name,
  and test frequency (maintenance covenants only).
- step-down/step-up schedules: share of schedule levels whose threshold
  appears among the rows matched to that covenant, and whether the whole
  schedule was captured.

Only required gold entries are scored here (see matching.is_required).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from covenant_extraction.evals.matching import CovenantMatch

DEFAULT_REL_TOL = 0.01

_FIELDS = (
    "threshold_exact",
    "threshold_tolerance",
    "comparison",
    "covenant_type",
    "unit",
    "name",
    "test_frequency",
    "all_fields",
)

_NAME_STOPWORDS = {"the", "of", "and", "consolidated", "minimum", "maximum", "min", "max", "ratio", "covenant"}


def thresholds_close(a: Any, b: Any, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def thresholds_exact(a: Any, b: Any) -> bool:
    return thresholds_close(a, b, rel_tol=0.0)


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.strip().lower()
    if u in {"usd", "$", "dollars", "us dollars", "u.s. dollars"} or "dollar" in u or "usd" in u:
        return "USD"
    if "ratio" in u or u in {"x", "times", ":1", "to 1"}:
        return "ratio"
    if u in {"%", "percent", "percentage"} or "percent" in u:
        return "percent"
    return u


def normalize_frequency(freq: Optional[str]) -> Optional[str]:
    """Collapse free-text test frequencies into a small set of buckets."""
    if not freq:
        return None
    f = freq.lower()
    if re.search(r"all times|any time|at any date|as of any date|continuous|daily", f):
        return "continuous"
    if "quarter" in f:
        return "quarterly"
    if "month" in f:
        return "monthly"
    if re.search(r"annual|fiscal year|year", f):
        return "annual"
    return None


def _name_tokens(name: Optional[str]) -> set:
    tokens = re.findall(r"[a-z]+", (name or "").lower())
    return {t for t in tokens if t not in _NAME_STOPWORDS}


def names_match(extracted: Optional[str], gold: Optional[str], aliases: Iterable[str] = ()) -> bool:
    """Token-overlap name match that ignores filler words like
    "Consolidated"/"Minimum", so "Leverage Ratio" matches "Consolidated
    Leverage Ratio" but "Interest Coverage" doesn't match "Fixed Charge
    Coverage"."""
    ext = _name_tokens(extracted)
    for candidate in (gold, *aliases):
        ref = _name_tokens(candidate)
        if not ref and not ext:
            return True
        if ref and ext and len(ref & ext) / len(ref | ext) >= 0.5:
            return True
    return False


@dataclass
class CovenantFieldResult:
    gold_name: str
    covenant_type: str
    threshold_exact: bool
    threshold_tolerance: bool
    comparison: bool
    covenant_type_correct: bool
    unit: bool
    name: bool
    test_frequency: Optional[bool]  # None when not applicable (incurrence, or gold has none)
    all_fields: bool
    extracted_threshold: Any
    gold_threshold: Any
    schedule_levels: int = 0
    schedule_levels_found: int = 0

    @property
    def schedule_complete(self) -> Optional[bool]:
        if not self.schedule_levels:
            return None
        return self.schedule_levels_found == self.schedule_levels


def score_fields(match: CovenantMatch, rel_tol: float = DEFAULT_REL_TOL) -> Optional[CovenantFieldResult]:
    """Field-by-field correctness of one matched covenant, or None if it
    was missed."""
    if match.extracted is None:
        return None
    gold, ext = match.gold, match.extracted

    gold_threshold = gold.get("threshold")
    extracted_thresholds = [row.get("threshold") for row in match.all_extracted]
    if gold.get("schedule"):
        # For a scheduled covenant, credit the headline threshold if ANY
        # absorbed row carries it.
        exact = any(thresholds_exact(t, gold_threshold) for t in extracted_thresholds)
        close = any(thresholds_close(t, gold_threshold, rel_tol) for t in extracted_thresholds)
    else:
        exact = thresholds_exact(ext.get("threshold"), gold_threshold)
        close = thresholds_close(ext.get("threshold"), gold_threshold, rel_tol)

    comparison = ext.get("comparison") == gold.get("comparison")
    covenant_type = ext.get("covenant_type") == gold.get("covenant_type")
    unit = normalize_unit(ext.get("unit")) == normalize_unit(gold.get("unit"))
    name = names_match(ext.get("name"), gold.get("name"), gold.get("name_aliases", ()))

    test_frequency: Optional[bool] = None
    gold_freq = gold.get("test_frequency")
    if gold_freq and gold.get("covenant_type") == "maintenance":
        test_frequency = any(normalize_frequency(row.get("test_frequency")) == gold_freq for row in match.all_extracted)

    schedule = gold.get("schedule") or []
    levels_found = sum(
        1 for level in schedule if any(thresholds_close(t, level["threshold"], rel_tol) for t in extracted_thresholds)
    )

    return CovenantFieldResult(
        gold_name=gold.get("name", ""),
        covenant_type=gold.get("covenant_type", ""),
        threshold_exact=exact,
        threshold_tolerance=close,
        comparison=comparison,
        covenant_type_correct=covenant_type,
        unit=unit,
        name=name,
        test_frequency=test_frequency,
        all_fields=exact and comparison and covenant_type and unit,
        extracted_threshold=ext.get("threshold"),
        gold_threshold=gold_threshold,
        schedule_levels=len(schedule),
        schedule_levels_found=levels_found,
    )


@dataclass
class FieldAccuracy:
    """Accuracy per field over matched required covenants. Each value is
    (correct, applicable)."""

    counts: Dict[str, List[int]]
    schedule_covenants: int
    schedule_complete: int
    schedule_levels: int
    schedule_levels_found: int

    def accuracy(self, field_name: str) -> Optional[float]:
        correct, total = self.counts[field_name]
        return correct / total if total else None

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["accuracy"] = {f: self.accuracy(f) for f in _FIELDS}
        return out


def aggregate_fields(results: Iterable[CovenantFieldResult]) -> FieldAccuracy:
    counts = {f: [0, 0] for f in _FIELDS}
    sched_cov = sched_complete = sched_levels = sched_found = 0
    for r in results:
        values = {
            "threshold_exact": r.threshold_exact,
            "threshold_tolerance": r.threshold_tolerance,
            "comparison": r.comparison,
            "covenant_type": r.covenant_type_correct,
            "unit": r.unit,
            "name": r.name,
            "test_frequency": r.test_frequency,
            "all_fields": r.all_fields,
        }
        for f, v in values.items():
            if v is None:
                continue
            counts[f][1] += 1
            counts[f][0] += int(bool(v))
        if r.schedule_levels:
            sched_cov += 1
            sched_complete += int(bool(r.schedule_complete))
            sched_levels += r.schedule_levels
            sched_found += r.schedule_levels_found
    return FieldAccuracy(counts, sched_cov, sched_complete, sched_levels, sched_found)
