"""CLI: backfill `loan_terms` into an extraction result JSON produced before
extraction/loan_terms.py existed -- without re-running the (already
verified) covenant extraction, dedup, or definitions-chaining already in
that file.

Usage:
    python scripts/backfill_loan_terms.py "data/sample/contract.html" --json data/processed/Result.json
    python scripts/backfill_loan_terms.py "data/sample/contract.html" --json data/processed/Result.json --backend lmstudio

Requires:
    --backend claude (default): an Anthropic API credential resolvable by
        the SDK (ANTHROPIC_API_KEY, `ant auth login`, etc.).
    --backend lmstudio: LM Studio running locally with a model loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from covenant_extraction.extraction.llm_client import ClaudeClient, LMStudioClient
from covenant_extraction.extraction.loan_terms import extract_loan_terms
from covenant_extraction.ingestion.chunker import chunk_blocks
from covenant_extraction.pipeline import _export_loan_terms, load_document
from covenant_extraction.retrieval.definitions_lookup import extract_definitions_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contract_path", help="Path to the source .pdf/.html/.htm contract")
    parser.add_argument("--json", required=True, help="Path to the existing result JSON to backfill")
    parser.add_argument("--backend", choices=["lmstudio", "claude"], default="claude")
    parser.add_argument("--model", default=None, help="Override the backend's default model name")
    args = parser.parse_args()

    blocks = load_document(args.contract_path)
    chunks = chunk_blocks(blocks)
    full_text = "\n".join(b.text for b in blocks)
    definitions_text = extract_definitions_text(chunks)

    llm_client = ClaudeClient(model=args.model) if args.backend == "claude" else LMStudioClient(model=args.model)

    print(f"Extracting loan terms from {args.contract_path} via backend={args.backend} ...", file=sys.stderr, flush=True)
    loan_terms = extract_loan_terms(chunks, llm_client)
    loan_terms_export = _export_loan_terms(loan_terms, full_text, definitions_text, llm_client)

    json_path = Path(args.json)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["loan_terms"] = loan_terms_export
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Backfilled loan_terms into {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
