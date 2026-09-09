"""Tests for retrieval/candidate_selector.py (and implicitly keyword_filter)."""
from __future__ import annotations

from covenant_extraction.ingestion.chunker import Chunk
from covenant_extraction.retrieval.candidate_selector import select_candidates
from covenant_extraction.retrieval.keyword_filter import matches_keyword


def _chunk(text: str, page: int = 1) -> Chunk:
    return Chunk(text=text, page=page, section_label=None, start_char=0, end_char=len(text))


def test_financial_keyword_chunk_is_selected(fake_embedder):
    chunks = [
        _chunk("The Consolidated Leverage Ratio shall not exceed 3.50:1.00."),
        _chunk("This is an unrelated boilerplate notices clause about addresses."),
    ]

    candidates = select_candidates(chunks, fake_embedder, max_candidates=5)

    texts = [c.chunk.text for c in candidates]
    assert any("Leverage Ratio" in t for t in texts)
    assert all(c.keyword_match for c in candidates if "Leverage Ratio" in c.chunk.text)


def test_negative_covenant_text_does_not_match_financial_keywords():
    text = "The Borrower shall not create, incur, assume or suffer to exist any Indebtedness."

    assert matches_keyword(text) is False


def test_negative_covenant_only_chunks_are_excluded_when_similarity_is_low(fake_embedder):
    chunks = [
        _chunk("Consolidated Leverage Ratio financial covenant shall not exceed 3.50:1.00."),
        _chunk("The Borrower shall not create, incur, assume or suffer to exist any Indebtedness."),
    ]

    candidates = select_candidates(chunks, fake_embedder, max_candidates=5)

    assert all(c.keyword_match or "Indebtedness" not in c.chunk.text for c in candidates)


def test_max_candidates_caps_total_selected(fake_embedder):
    chunks = [
        _chunk("Consolidated Leverage Ratio financial covenant shall not exceed 3.50:1.00."),
        _chunk("Fixed Charge Coverage Ratio shall not be less than 1.25:1.00."),
        _chunk("Minimum Liquidity financial covenants of not less than $10,000,000."),
        _chunk("Some vaguely related financial covenant discussion of ratios and testing."),
        _chunk("Totally unrelated notices and counterparts boilerplate section."),
        _chunk("Another totally unrelated governing law boilerplate section."),
    ]

    candidates = select_candidates(chunks, fake_embedder, max_candidates=2)

    assert len(candidates) == 2


def test_empty_chunks_returns_empty(fake_embedder):
    assert select_candidates([], fake_embedder) == []


def test_fewer_candidates_returned_when_fewer_qualify(fake_embedder):
    chunks = [
        _chunk("Consolidated Leverage Ratio financial covenant shall not exceed 3.50:1.00."),
        _chunk("Totally unrelated notices and counterparts boilerplate section."),
        _chunk("Another totally unrelated governing law boilerplate section."),
    ]

    candidates = select_candidates(chunks, fake_embedder, max_candidates=5)

    # Only the one financial-metric chunk qualifies; the cap of 5 shouldn't
    # be padded out with irrelevant boilerplate.
    assert len(candidates) == 1
    assert "Leverage Ratio" in candidates[0].chunk.text


def test_candidates_sorted_by_combined_score_descending(fake_embedder):
    chunks = [
        _chunk("Indebtedness restrictions and negative covenants on Liens."),
        _chunk("Consolidated Leverage Ratio financial covenant minimum liquidity."),
    ]

    candidates = select_candidates(chunks, fake_embedder, max_candidates=5)

    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
