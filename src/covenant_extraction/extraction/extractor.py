"""Orchestrates LLM extraction over candidate clauses and stitches together
final results with citations pinned to the actual chunk metadata (not
whatever the model may have hallucinated for page/section_label).
"""
from __future__ import annotations

import re
import sys
import time
from typing import List, TypeVar

from covenant_extraction.extraction.llm_client import LLMClient
from covenant_extraction.extraction.schema import CovenantExtractionResult
from covenant_extraction.retrieval.candidate_selector import Candidate

_PREVIEW_CHARS = 80

_WHITESPACE_RE = re.compile(r"\s+")

T = TypeVar("T")  # a FinancialCovenant, EbitdaAddback, or NegativeCovenant


def _dedupe_by_citation(items: List[T]) -> List[T]:
    """Drop items whose citation.source_text is a whitespace-normalized
    duplicate of an earlier item's.

    The same clause sometimes gets extracted more than once -- e.g. it
    falls inside two overlapping candidate chunks, or a single call's JSON
    response repeats an item -- which isn't a second real covenant, just
    the same one seen twice. Keeps the first occurrence."""
    seen = set()
    deduped: List[T] = []
    for item in items:
        key = _WHITESPACE_RE.sub(" ", item.citation.source_text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _candidate_location(candidate: Candidate) -> str:
    if candidate.chunk.page:
        return f"page {candidate.chunk.page}"
    return candidate.chunk.section_label or "unknown location"


def _log_candidate_summary(candidates: List[Candidate]) -> None:
    """Print the number of candidate clauses and a one-line summary of each,
    before any LLM calls are made, so a reviewer can see up front what will
    be sent to the model."""
    print(f"Selected {len(candidates)} candidate clause(s) for LLM extraction:", file=sys.stderr, flush=True)
    for i, candidate in enumerate(candidates, start=1):
        location = _candidate_location(candidate)
        basis = "keyword+embedding" if candidate.keyword_match else "embedding"
        preview = candidate.chunk.text[:_PREVIEW_CHARS].replace("\n", " ")
        if len(candidate.chunk.text) > _PREVIEW_CHARS:
            preview += "..."
        print(
            f"  [{i}] {location} | basis={basis} | score={candidate.score:.3f} | \"{preview}\"",
            file=sys.stderr,
            flush=True,
        )


def _pin_citation_metadata(result: CovenantExtractionResult, candidate: Candidate) -> CovenantExtractionResult:
    """Overwrite page/section_label on every citation with the candidate
    chunk's real metadata, since only that is trustworthy (the model's
    citation fields for page/section_label are advisory at best)."""
    for group in (result.financial_covenants, result.ebitda_addbacks, result.negative_covenants):
        for item in group:
            item.citation.page = candidate.chunk.page
            item.citation.section_label = candidate.chunk.section_label
    return result


def extract_from_candidates(candidates: List[Candidate], llm_client: LLMClient) -> CovenantExtractionResult:
    """Run extraction over each candidate clause and merge results.

    Args:
        candidates: ranked candidate chunks from candidate_selector.
        llm_client: an LLMClient (real LMStudioClient or a test fake).

    Returns:
        A single CovenantExtractionResult combining items found across all
        candidates.
    """
    _log_candidate_summary(candidates)

    merged = CovenantExtractionResult()
    total = len(candidates)

    for i, candidate in enumerate(candidates, start=1):
        location = _candidate_location(candidate)
        print(f"[{i}/{total}] extracting from {location} ...", file=sys.stderr, flush=True)
        start = time.monotonic()

        result = llm_client.extract(candidate.chunk.text)

        elapsed = time.monotonic() - start
        print(f"[{i}/{total}] done in {elapsed:.1f}s", file=sys.stderr, flush=True)

        result = _pin_citation_metadata(result, candidate)
        merged.financial_covenants.extend(result.financial_covenants)
        merged.ebitda_addbacks.extend(result.ebitda_addbacks)
        merged.negative_covenants.extend(result.negative_covenants)

    merged.financial_covenants = _dedupe_by_citation(merged.financial_covenants)
    merged.ebitda_addbacks = _dedupe_by_citation(merged.ebitda_addbacks)
    merged.negative_covenants = _dedupe_by_citation(merged.negative_covenants)

    return merged
