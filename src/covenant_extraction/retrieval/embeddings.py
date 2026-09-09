"""Embedding-based similarity scoring for candidate_selector.py.

An `Embedder` protocol/interface is used so tests can swap in a fake,
deterministic embedder without downloading or running a real model -- and so
production code can swap between:

- `BgeM3Embedder`: BAAI/bge-m3 via sentence-transformers, run locally on
  CPU. Best semantic recall, but downloads a multi-GB pretrained model from
  the Hugging Face Hub on first use (and needs `torch`/`sentence-transformers`
  installed) -- not viable in a network-restricted corporate environment.
- `TfidfEmbedder`: scikit-learn TF-IDF, fit fresh on whatever texts are
  passed to `embed()` each call. No pretrained weights, no network access,
  no extra heavyweight dependency (`scikit-learn` is already a transitive
  dependency of `sentence-transformers`, but works standalone too). Purely
  lexical, so it misses semantic near-misses a keyword regex and BGE-M3
  would both catch -- see candidate_selector.py's keyword-match path, which
  qualifies a chunk independently of the embedding score.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np

from covenant_extraction.config import settings

_QUERY_PROMPT = (
    "Represent this loan/credit agreement covenant clause for retrieval: "
)


class Embedder(Protocol):
    """Minimal interface required by candidate_selector."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) array of embedding vectors for `texts`."""
        ...


class BgeM3Embedder:
    """Loads BAAI/bge-m3 via sentence-transformers on first use."""

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        self._model_name = model_name or settings.embed_model_name
        self._cache_dir = cache_dir or str(settings.embed_model_cache_dir)
        self._model = None  # lazy-loaded

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, cache_folder=self._cache_dir)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(model.encode(texts, normalize_embeddings=True))


class TfidfEmbedder:
    """TF-IDF vectors via scikit-learn -- no pretrained model to download or
    load, so this works fully offline. Fits a fresh vectorizer on whatever
    `texts` is passed to `embed()`, so callers that need vectors from two
    separate `embed()` calls to be comparable (e.g. candidate_selector.py
    scoring document chunks against reference queries) MUST pass both sets
    of texts to a single `embed()` call together -- a TF-IDF vector's
    dimensions are indexed by that call's own fitted vocabulary, so vectors
    from two different fits aren't in the same space and cosine similarity
    between them would be meaningless."""

    def embed(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()


def cosine_similarity(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and many candidate vectors.

    Assumes vectors may not be pre-normalized.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    candidate_norms = candidate_vecs / (
        np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-12
    )
    return candidate_norms @ query_norm
