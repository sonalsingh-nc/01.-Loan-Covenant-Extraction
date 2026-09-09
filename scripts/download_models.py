"""One-time helper: downloads and caches the BGE-M3 embedding model locally
so the pipeline can run fully offline afterward.

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

from covenant_extraction.config import settings


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"Downloading {settings.embed_model_name} into {settings.embed_model_cache_dir} ...")
    SentenceTransformer(settings.embed_model_name, cache_folder=str(settings.embed_model_cache_dir))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
