"""CLI entry point: extract covenants from a single contract file.

Usage:
    python scripts/run_extraction.py path/to/contract.pdf [--out output.json]
    python scripts/run_extraction.py path/to/contract.pdf --embedder tfidf
    python scripts/run_extraction.py path/to/contract.pdf --backend claude --embedder tfidf

Requires:
    - --backend lmstudio (default): LM Studio running locally with a model
      loaded (see README.md).
    - --backend claude: an Anthropic API credential resolvable by the SDK
      (ANTHROPIC_API_KEY, `ant auth login`, etc.) -- calls the real API and
      incurs real cost, unlike the free local LM Studio backend.
    - --embedder bge-m3 (default): the BGE-M3 embedding model, downloaded
      from the Hugging Face Hub on first use and cached locally thereafter.
    - --embedder tfidf: no download, no network access -- scikit-learn TF-IDF
      fit fresh per document. Use this in network-restricted environments
      (see retrieval/embeddings.py for the recall tradeoff).

    --backend claude --embedder tfidf together make no calls to any local
    server or external model host except the Anthropic API itself.
"""
from __future__ import annotations

import argparse
import json
import sys

from covenant_extraction.extraction.llm_client import ClaudeClient, LMStudioClient
from covenant_extraction.pipeline import run_pipeline
from covenant_extraction.retrieval.candidate_selector import TFIDF_RECOMMENDED_MAX_CANDIDATES
from covenant_extraction.retrieval.embeddings import BgeM3Embedder, TfidfEmbedder

_EMBEDDERS = {"bge-m3": BgeM3Embedder, "tfidf": TfidfEmbedder}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contract_path", help="Path to a .pdf or .html/.htm loan contract")
    parser.add_argument("--out", default=None, help="Path to write JSON output (default: stdout)")
    parser.add_argument("--backend", choices=["lmstudio", "claude"], default="lmstudio")
    parser.add_argument("--model", default=None, help="Override the backend's default model name")
    parser.add_argument(
        "--embedder",
        choices=sorted(_EMBEDDERS),
        default="bge-m3",
        help="bge-m3 (default): best recall, downloads a model on first use. "
        "tfidf: no download, no network access, lexical-only.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Clauses sent to the LLM. Default: settings.max_candidates for "
        f"bge-m3, {TFIDF_RECOMMENDED_MAX_CANDIDATES} for tfidf (verified needed "
        "for comparable recall -- see retrieval/candidate_selector.py).",
    )
    args = parser.parse_args()

    embedder = _EMBEDDERS[args.embedder]()
    llm_client = ClaudeClient(model=args.model) if args.backend == "claude" else LMStudioClient(model=args.model)

    max_candidates = args.max_candidates
    if max_candidates is None and args.embedder == "tfidf":
        max_candidates = TFIDF_RECOMMENDED_MAX_CANDIDATES

    result = run_pipeline(args.contract_path, embedder=embedder, llm_client=llm_client, max_candidates=max_candidates)
    output = json.dumps(result, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote results to {args.out}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
