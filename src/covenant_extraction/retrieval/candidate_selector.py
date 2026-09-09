"""Combines keyword filtering and embedding similarity to select the
candidate clauses passed to the LLM for structured extraction.

Scope is restricted to financial-metric covenant clauses (leverage ratio,
coverage ratios, minimum liquidity, etc.) -- negative covenants and generic
definitions are excluded so the LLM only sees the clauses most likely to
contain a financial covenant threshold. Each chunk's combined score is its
best embedding similarity to a set of financial-covenant reference queries,
plus a bonus if it also matches a financial-metric keyword. A chunk only
qualifies as a candidate if it matches a keyword OR clears a minimum
embedding-similarity floor -- otherwise irrelevant clauses (e.g. negative
covenants, boilerplate) would get padded in just to fill the cap. Up to
`max_candidates` qualifying chunks are selected, by combined score (fewer
if fewer qualify).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from covenant_extraction.config import settings
from covenant_extraction.ingestion.chunker import Chunk
from covenant_extraction.retrieval.embeddings import Embedder, cosine_similarity
from covenant_extraction.retrieval.keyword_filter import matches_keyword

REFERENCE_QUERIES = [
    "Financial covenant: Consolidated Leverage Ratio shall not exceed a maximum ratio",
    "Financial covenant: minimum liquidity or Fixed Charge Coverage Ratio requirement",
]

KEYWORD_MATCH_BONUS = 1.0

# Cosine-similarity floor a non-keyword-matched chunk must clear to qualify
# as a candidate at all (tune once running against the real BGE-M3 embedder).
MIN_EMBEDDING_SCORE_FOR_MATCH = 0.4

# TfidfEmbedder is purely lexical, so it consistently under-ranks a genuine
# covenant clause phrased differently from REFERENCE_QUERIES (e.g. "will not
# permit ... to be greater than" vs. "shall not exceed") below an
# unrelated-but-lexically-similar false positive. Verified against real
# documents: the true covenant clause landed at rank 6-8 of 16-21 eligible
# candidates -- comfortably outside settings.max_candidates' default of 5,
# comfortably inside 10. Callers using TfidfEmbedder should pass this (or
# higher) as `max_candidates` rather than the bge-m3-tuned default; BGE-M3's
# semantic ranking doesn't need it (the same clauses ranked in the top 5).
TFIDF_RECOMMENDED_MAX_CANDIDATES = 10


@dataclass(frozen=True)
class Candidate:
    """A chunk selected for LLM extraction, with the reason(s) it was picked."""

    chunk: Chunk
    keyword_match: bool
    embedding_score: float
    score: float


def select_candidates(
    chunks: List[Chunk],
    embedder: Embedder,
    max_candidates: int | None = None,
) -> List[Candidate]:
    """Rank and select candidate chunks for LLM extraction.

    Args:
        chunks: all chunks from the document.
        embedder: an Embedder (real BgeM3Embedder or a test fake).
        max_candidates: max number of candidates to return overall. Defaults
            to settings.max_candidates.

    Returns:
        Up to `max_candidates` Candidates that matched a financial-metric
        keyword or cleared the embedding-similarity floor, ordered by
        combined score descending. Fewer than `max_candidates` are returned
        if fewer chunks qualify.
    """
    max_candidates = max_candidates if max_candidates is not None else settings.max_candidates

    if not chunks:
        return []

    # One combined embed() call, not two separate ones -- some Embedder
    # implementations (e.g. TfidfEmbedder) fit a fresh vector space per
    # call, so chunk vectors and query vectors from two different calls
    # wouldn't be comparable by cosine similarity at all. A single call
    # keeps every Embedder implementation (fixed-space like BGE-M3, or
    # fit-per-call like TF-IDF) working correctly.
    chunk_texts = [c.text for c in chunks]
    combined_vecs = embedder.embed(chunk_texts + REFERENCE_QUERIES)
    chunk_vecs = combined_vecs[: len(chunk_texts)]
    query_vecs = combined_vecs[len(chunk_texts) :]
    # Best similarity across all reference queries, per chunk.
    embedding_scores = np.max(
        np.stack([cosine_similarity(q, chunk_vecs) for q in query_vecs]), axis=0
    )

    keyword_flags = [matches_keyword(c.text) for c in chunks]
    combined_scores = [
        float(embedding_scores[i]) + (KEYWORD_MATCH_BONUS if keyword_flags[i] else 0.0)
        for i in range(len(chunks))
    ]

    eligible_idx = [
        i
        for i in range(len(chunks))
        if keyword_flags[i] or embedding_scores[i] >= MIN_EMBEDDING_SCORE_FOR_MATCH
    ]
    eligible_idx.sort(key=lambda i: combined_scores[i], reverse=True)
    selected_idx = eligible_idx[:max_candidates]

    return [
        Candidate(
            chunk=chunks[i],
            keyword_match=keyword_flags[i],
            embedding_score=float(embedding_scores[i]),
            score=combined_scores[i],
        )
        for i in selected_idx
    ]
