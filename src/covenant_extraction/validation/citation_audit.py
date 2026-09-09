"""Citation auditing: verifies that an LLM-produced `source_text` quote
actually appears (verbatim or near-verbatim) in the chunk it was supposedly
extracted from. This is the safety net against hallucinated citations.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class AuditResult:
    verified: bool
    similarity: float  # 0.0-1.0; 1.0 means exact substring match


def audit_citation(quoted_text: str, source_text: str, min_similarity: float = 0.85) -> AuditResult:
    """Check that `quoted_text` genuinely appears in `source_text`.

    Uses exact substring match after whitespace normalization first (fast
    path); falls back to a fuzzy ratio (difflib) against the best-matching
    window of the source, to tolerate minor whitespace/punctuation drift
    from the model without accepting fabricated quotes.

    Args:
        quoted_text: the citation.source_text produced by the LLM.
        source_text: the actual chunk text the LLM was given.
        min_similarity: minimum fuzzy-match ratio to count as verified when
            an exact substring match isn't found.

    Returns:
        AuditResult with `verified` True/False and the similarity score used.
    """
    norm_quote = _normalize(quoted_text)
    norm_source = _normalize(source_text)

    if not norm_quote:
        return AuditResult(verified=False, similarity=0.0)

    if norm_quote in norm_source:
        return AuditResult(verified=True, similarity=1.0)

    matcher = difflib.SequenceMatcher(None, norm_source, norm_quote)
    match = matcher.find_longest_match(0, len(norm_source), 0, len(norm_quote))
    if match.size == 0:
        return AuditResult(verified=False, similarity=0.0)

    window = norm_source[max(0, match.a - 10) : match.a + match.size + 10]
    similarity = difflib.SequenceMatcher(None, window, norm_quote).ratio()
    return AuditResult(verified=similarity >= min_similarity, similarity=similarity)
