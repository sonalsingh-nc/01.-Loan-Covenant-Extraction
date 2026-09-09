"""Tests for extraction/extractor.py, using FakeLLMClient from conftest."""
from __future__ import annotations

from covenant_extraction.extraction.extractor import extract_from_candidates
from covenant_extraction.ingestion.chunker import Chunk
from covenant_extraction.retrieval.candidate_selector import Candidate


def _candidate(text: str, page=None, section_label=None) -> Candidate:
    chunk = Chunk(text=text, page=page, section_label=section_label, start_char=0, end_char=len(text))
    return Candidate(chunk=chunk, keyword_match=True, embedding_score=1.0, score=1.0)


def test_extract_merges_results_across_candidates(fake_llm_client):
    candidates = [
        _candidate(
            "The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00.",
            page=5,
        ),
        _candidate(
            "The Borrower shall not create, incur, assume or suffer to exist any Indebtedness.",
            page=6,
        ),
        _candidate("Irrelevant boilerplate notices clause.", page=7),
    ]

    result = extract_from_candidates(candidates, fake_llm_client)

    assert len(result.financial_covenants) == 1
    assert result.financial_covenants[0].name == "Consolidated Leverage Ratio"
    assert len(result.negative_covenants) == 1
    assert result.negative_covenants[0].name == "Indebtedness"
    assert len(fake_llm_client.calls) == 3


def test_extract_pins_citation_metadata_from_candidate_chunk(fake_llm_client):
    candidates = [
        _candidate(
            "The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00.",
            page=42,
            section_label=None,
        )
    ]

    result = extract_from_candidates(candidates, fake_llm_client)

    covenant = result.financial_covenants[0]
    assert covenant.citation.page == 42
    assert covenant.citation.section_label is None


def test_extract_with_no_matches_returns_empty_result(fake_llm_client):
    candidates = [_candidate("Nothing relevant here at all.")]

    result = extract_from_candidates(candidates, fake_llm_client)

    assert result.financial_covenants == []
    assert result.ebitda_addbacks == []
    assert result.negative_covenants == []


def test_extract_dedupes_the_same_clause_seen_in_two_overlapping_candidates(fake_llm_client):
    # Two overlapping candidate chunks both containing the same "Leverage
    # Ratio ... 3.50" clause -- the fake (like a real LLM) extracts an
    # identical covenant + citation from each. That's the same real clause
    # seen twice, not two distinct covenants, so it should collapse to one.
    candidates = [
        _candidate("Preamble text. The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00.", page=5),
        _candidate("The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00. Trailing text.", page=5),
    ]

    result = extract_from_candidates(candidates, fake_llm_client)

    assert len(result.financial_covenants) == 1
    assert len(fake_llm_client.calls) == 2  # both candidates were still sent to the LLM


def test_extract_keeps_distinct_covenants_with_different_citations(fake_llm_client):
    candidates = [
        _candidate("The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00.", page=5),
        _candidate("The Borrower shall not create, incur, assume or suffer to exist any Indebtedness.", page=6),
    ]

    result = extract_from_candidates(candidates, fake_llm_client)

    assert len(result.financial_covenants) == 1
    assert len(result.negative_covenants) == 1
