"""Tests for retrieval/embeddings.py's TfidfEmbedder -- the no-download
alternative to BgeM3Embedder for network-restricted environments."""
from __future__ import annotations

from covenant_extraction.retrieval.embeddings import TfidfEmbedder, cosine_similarity


def test_tfidf_embedder_returns_one_vector_per_text():
    embedder = TfidfEmbedder()
    texts = [
        "Consolidated Leverage Ratio shall not exceed 3.50:1.00.",
        "Notices shall be delivered by hand or overnight courier.",
        "Financial covenant: minimum liquidity requirement.",
    ]

    vectors = embedder.embed(texts)

    assert vectors.shape[0] == len(texts)
    # Every vector must live in the same fitted vocabulary space (shared
    # dimensionality) for cosine similarity between any two of them to be
    # meaningful -- see candidate_selector.py's single combined embed() call.
    assert len({vectors.shape[1]}) == 1


def test_tfidf_embedder_ranks_lexically_similar_text_higher():
    embedder = TfidfEmbedder()
    query = "Financial covenant: Consolidated Leverage Ratio shall not exceed a maximum ratio"
    related = "The Consolidated Leverage Ratio shall not exceed 3.50 to 1.00 as of any fiscal quarter."
    unrelated = "Notices shall be delivered by hand or overnight courier to the addresses set forth herein."

    vectors = embedder.embed([related, unrelated, query])
    related_vec, unrelated_vec, query_vec = vectors[0], vectors[1], vectors[2]

    related_score = cosine_similarity(query_vec, related_vec.reshape(1, -1))[0]
    unrelated_score = cosine_similarity(query_vec, unrelated_vec.reshape(1, -1))[0]

    assert related_score > unrelated_score


def test_tfidf_embedder_works_end_to_end_with_select_candidates():
    from covenant_extraction.ingestion.chunker import Chunk
    from covenant_extraction.retrieval.candidate_selector import select_candidates

    chunks = [
        Chunk(
            text="Fixed Charge Coverage Ratio shall not be less than 1.25:1.00 for any period.",
            page=1,
            section_label=None,
            start_char=0,
            end_char=10,
        ),
        Chunk(
            text="This Agreement may be executed in counterparts, each of which is an original.",
            page=1,
            section_label=None,
            start_char=0,
            end_char=10,
        ),
    ]

    candidates = select_candidates(chunks, TfidfEmbedder(), max_candidates=5)

    assert len(candidates) == 1
    assert "Fixed Charge Coverage Ratio" in candidates[0].chunk.text
