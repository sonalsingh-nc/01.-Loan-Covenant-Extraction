"""Extraction of loan-level deal terms (borrowers, lenders, loan type,
amount, interest rate, start/maturity date, principal repayment terms).

These attributes are structurally different from the ongoing covenants
extraction.py handles: each one is typically stated exactly once, in a
specific part of the document -- parties/amount/type/date usually together
in the preamble, while interest rate, maturity, and repayment terms each
tend to live in their own dedicated section. So rather than reusing
candidate_selector.py's embedding-ranked search (tuned for financial-ratio
covenant language), this module targets the document's opening chunks plus
a few keyword-matched chunks, and merges whatever each one finds.
"""
from __future__ import annotations

import re
from typing import List, Optional

from covenant_extraction.extraction.llm_client import StructuredLLMClient
from covenant_extraction.extraction.prompts import LOAN_TERMS_SYSTEM_PROMPT, build_loan_terms_user_prompt
from covenant_extraction.extraction.schema import CitedText, LoanTerms
from covenant_extraction.ingestion.chunker import Chunk

# Preamble/recitals almost always name the parties, facility type/amount,
# and effective date in the opening paragraph(s) -- accumulate chunks up to
# this character budget rather than guessing how many chunks that takes.
# The headline loan amount and type are often in there too (e.g.
# "$3,250,000,000 ... CREDIT AGREEMENT"), but not always -- unlike interest
# rate/maturity/repayment, a bare keyword like "commitment" is too common
# throughout a credit agreement (referring to individual lenders' pieces,
# not the deal's headline amount) to search for directly, so the loan_type
# and loan_amount patterns below stick to specific, low-noise phrasing as a
# fallback for when the preamble alone doesn't state them.
PREAMBLE_CHAR_BUDGET = 3000

# Each entry is a priority-ordered tuple of patterns for one attribute:
# the first pattern with ANY match across the document wins (searched in
# document order for that pattern only) -- e.g. prefer "Applicable Margin"
# language (the actual pricing grid) over a merely rate-mechanics-adjacent
# definition like "ABR" that happens to mention "Base Rate" much earlier in
# an alphabetical Definitions section.
_INTEREST_RATE_PATTERNS = (
    re.compile(r"\b(applicable margin|applicable rate|term sofr)\b", re.IGNORECASE),
    re.compile(r"\b(interest rate|base rate|libor|per annum)\b", re.IGNORECASE),
)
_MATURITY_PATTERNS = (
    re.compile(r"\b(maturity date|termination date)\b", re.IGNORECASE),
    re.compile(r"\b(final maturity|scheduled maturity|stated maturity|expiration date)\b", re.IGNORECASE),
)
# "dated as of" is the standard execution-date boilerplate at the very top
# of a credit agreement and is reliable enough to trust on its own; "Closing
# Date"/"Effective Date" are a weaker fallback since they're defined terms
# that don't always carry the literal date value in the same sentence.
_START_DATE_PATTERNS = (
    re.compile(r"\bdated as of\b", re.IGNORECASE),
    re.compile(r"\b(closing date|effective date)\b", re.IGNORECASE),
)
# Deliberately NOT bare "amortization" -- that word is just as commonly
# accounting/EBITDA-addback amortization (depreciation & amortization) as
# it is loan-principal amortization, and the two are easy to conflate.
_REPAYMENT_PATTERNS = (
    re.compile(
        r"\b(repayment of (?:the )?(?:loans?|principal)|principal (?:installment|payment|amortization)s?|"
        r"amortization (?:schedule|of (?:the )?(?:term )?loans?)|scheduled (?:repayments?|installments?)|"
        r"quarterly installments? of)\b",
        re.IGNORECASE,
    ),
)
# Prefer a named facility-type phrase over a bare secured/unsecured mention
# elsewhere (e.g. a security-interest clause) that says nothing about the
# facility structure itself.
_LOAN_TYPE_PATTERNS = (
    re.compile(
        r"\b(senior secured|senior unsecured|secured|unsecured)\s+"
        r"(?:revolving\s+(?:credit\s+)?(?:facility|loan)s?|term\s+loan(?:\s+facility)?|credit\s+facility)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(revolving credit facility|term loan facility|revolving loan|term loan|credit facility)\b", re.IGNORECASE),
)
# Bare "commitment" is excluded on purpose (see PREAMBLE_CHAR_BUDGET comment
# above) -- these phrases only match the deal's headline aggregate figure,
# not an individual lender's piece of it. "credit|loan|facility" is
# required (not optional) after maximum/total -- a bare "maximum amount"
# also matches ordinary late-fee/usury boilerplate ("...the maximum amount
# permitted by Applicable Law...") that has nothing to do with the deal's
# headline figure, and that false match would use up this category's
# one-shot-per-document budget before ever reaching the real clause.
#
# Both tiers also require an actual "$" within the same sentence (a
# trailing lookahead, not consumed) -- "aggregate amount"/"maximum credit
# amount" alone still shows up in unrelated baskets and carve-outs (e.g.
# "the aggregate amount of the indebtedness described in (B) and (C) shall
# not exceed at any time two percent (2%) of..." -- a percentage cap, no
# dollar figure at all), and since only the first *tier* to match anywhere
# in the document wins, one such false match would permanently block the
# fallback tier's "principal amount of $" from ever being tried, even when
# it would have found the real, dollar-denominated clause.
_AMOUNT_HAS_DOLLAR_SIGN = r"(?=[^.]{0,80}\$)"
_LOAN_AMOUNT_PATTERNS = (
    re.compile(
        r"\b(aggregate (?:principal )?(?:amount|commitments?)|"
        r"(?:maximum|total) (?:revolving )?(?:credit|loan|facility) (?:amount|commitments?))\b" + _AMOUNT_HAS_DOLLAR_SIGN,
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(not to exceed \$|up to \$|principal amount of \$|"
        r"(?:maximum|original) principal amount" + _AMOUNT_HAS_DOLLAR_SIGN + r")\b",
        re.IGNORECASE,
    ),
)
_ATTRIBUTE_PATTERN_GROUPS = (
    _INTEREST_RATE_PATTERNS,
    _MATURITY_PATTERNS,
    _START_DATE_PATTERNS,
    _REPAYMENT_PATTERNS,
    _LOAN_TYPE_PATTERNS,
    _LOAN_AMOUNT_PATTERNS,
)


def select_loan_terms_candidates(chunks: List[Chunk]) -> List[Chunk]:
    """Select chunks likely to state loan-level deal terms: the document's
    opening chunks up to PREAMBLE_CHAR_BUDGET, plus one representative
    chunk per attribute (interest rate, maturity date, start date,
    repayment terms, loan type, loan amount) not already covered by the
    preamble -- trying each attribute's patterns in priority order and
    stopping at the first one that matches anywhere."""
    if not chunks:
        return []

    candidates: List[Chunk] = []
    seen_ids = set()

    char_count = 0
    for chunk in chunks:
        if char_count >= PREAMBLE_CHAR_BUDGET:
            break
        candidates.append(chunk)
        seen_ids.add(id(chunk))
        char_count += len(chunk.text)

    for pattern_tiers in _ATTRIBUTE_PATTERN_GROUPS:
        for pattern in pattern_tiers:
            matched = False
            for chunk in chunks:
                if id(chunk) in seen_ids:
                    continue
                if pattern.search(chunk.text):
                    candidates.append(chunk)
                    seen_ids.add(id(chunk))
                    matched = True
                    break  # one representative chunk per attribute is enough
            if matched:
                break  # this attribute is covered -- don't also try the fallback tier

    return candidates


def _pin_citation(cited: Optional[CitedText], chunk: Chunk) -> None:
    if cited is not None:
        cited.citation.page = chunk.page
        cited.citation.section_label = chunk.section_label


def _dedupe_cited_values(items: List[CitedText]) -> List[CitedText]:
    """Drop entries whose (whitespace-normalized, case-insensitive) value
    duplicates an earlier one -- the same party is often named more than
    once across the preamble and later sections."""
    seen = set()
    deduped: List[CitedText] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.value).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge(a: LoanTerms, b: LoanTerms) -> LoanTerms:
    """Combine two partial LoanTerms results: extend the list fields, and
    for each single-value field keep whichever result already has one
    (each such attribute is expected to be stated once, in one place)."""
    return LoanTerms(
        borrowers=a.borrowers + b.borrowers,
        lenders=a.lenders + b.lenders,
        loan_type=a.loan_type or b.loan_type,
        loan_amount=a.loan_amount or b.loan_amount,
        interest_rate=a.interest_rate or b.interest_rate,
        start_date=a.start_date or b.start_date,
        end_date=a.end_date or b.end_date,
        principal_repayment_terms=a.principal_repayment_terms or b.principal_repayment_terms,
    )


def extract_loan_terms(chunks: List[Chunk], llm_client: StructuredLLMClient) -> LoanTerms:
    """Extract loan-level deal terms by scanning the document's preamble
    plus a few keyword-matched clauses, merging whatever each call finds.

    Args:
        chunks: all chunks from the document (see ingestion/chunker.py).
        llm_client: a StructuredLLMClient (real LMStudioClient or a test fake).

    Returns:
        A single LoanTerms combining every candidate's findings, with
        citation page/section_label pinned to the real chunk metadata
        (not whatever the model may have hallucinated for those fields).
    """
    merged = LoanTerms()

    for chunk in select_loan_terms_candidates(chunks):
        result = llm_client.complete(
            system_prompt=LOAN_TERMS_SYSTEM_PROMPT,
            user_prompt=build_loan_terms_user_prompt(chunk.text),
            schema=LoanTerms,
        )

        for item in result.borrowers:
            _pin_citation(item, chunk)
        for item in result.lenders:
            _pin_citation(item, chunk)
        for cited in (
            result.loan_type,
            result.loan_amount,
            result.interest_rate,
            result.start_date,
            result.end_date,
            result.principal_repayment_terms,
        ):
            _pin_citation(cited, chunk)

        merged = _merge(merged, result)

    merged.borrowers = _dedupe_cited_values(merged.borrowers)
    merged.lenders = _dedupe_cited_values(merged.lenders)
    return merged
