"""CLI: run the extraction pipeline against one or all gold-annotated
documents and score the result (recall / precision / threshold accuracy /
covenant_type accuracy), so extraction quality can be compared across LLM
backends (LM Studio/qwen vs. Claude Haiku) on the same ground truth.

Usage:
    python evals/run_eval.py --backend lmstudio
    python evals/run_eval.py --backend claude --model claude-haiku-4-5
    python evals/run_eval.py --backend lmstudio --gold evals/gold/mizuho_qvc.json

Requires:
    --backend lmstudio: LM Studio running locally with a model loaded.
    --backend claude: an Anthropic API credential resolvable by the SDK
        (ANTHROPIC_API_KEY, `ant auth login`, etc.) -- calls real API and
        incurs real cost, unlike the free local LM Studio backend.
    Either way: --embedder bge-m3 (default) needs the BGE-M3 embedding
        model cached locally (downloaded from the Hugging Face Hub on
        first use); --embedder tfidf needs neither a download nor network
        access -- see retrieval/embeddings.py for the recall tradeoff.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covenant_extraction.evals.matching import match_covenants  # noqa: E402
from covenant_extraction.evals.scoring import EvalScore, score_matches  # noqa: E402
from covenant_extraction.extraction.llm_client import ClaudeClient, LMStudioClient  # noqa: E402
from covenant_extraction.pipeline import run_pipeline  # noqa: E402
from covenant_extraction.retrieval.candidate_selector import TFIDF_RECOMMENDED_MAX_CANDIDATES  # noqa: E402
from covenant_extraction.retrieval.embeddings import BgeM3Embedder, TfidfEmbedder  # noqa: E402

GOLD_DIR = PROJECT_ROOT / "evals" / "gold"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
_EMBEDDERS = {"bge-m3": BgeM3Embedder, "tfidf": TfidfEmbedder}


def _build_llm_client(backend: str, model: str | None):
    if backend == "lmstudio":
        return LMStudioClient(model=model) if model else LMStudioClient()
    if backend == "claude":
        return ClaudeClient(model=model) if model else ClaudeClient()
    raise ValueError(f"Unknown backend '{backend}'. Use 'lmstudio' or 'claude'.")


def run_one(
    gold_path: Path, backend: str, model: str | None, embedder, max_candidates: int | None
) -> Dict[str, Any]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    document_path = PROJECT_ROOT / gold["document"]

    llm_client = _build_llm_client(backend, model)
    print(f"[{gold_path.stem}] extracting with backend={backend} ...", file=sys.stderr, flush=True)
    result = run_pipeline(document_path, embedder=embedder, llm_client=llm_client, max_candidates=max_candidates)

    matches, unmatched = match_covenants(gold["financial_covenants"], result["financial_covenants"])
    score = score_matches(matches, unmatched)

    return {
        "document": gold["document"],
        "backend": backend,
        "model": model,
        "score": score.__dict__,
        "misses": [m.gold for m in matches if m.extracted is None],
        "false_positives": unmatched,
        # Full raw extraction, so a later gold_contains anchor fix can be
        # re-scored offline (via match_covenants + score_matches directly)
        # without spending another LLM call to re-extract.
        "extracted_financial_covenants": result["financial_covenants"],
    }


def _print_score(gold_stem: str, score: EvalScore) -> None:
    print(
        f"{gold_stem:35s} recall={score.recall:.2f} precision={score.precision:.2f} "
        f"threshold_acc={score.threshold_accuracy:.2f} type_acc={score.covenant_type_accuracy:.2f} "
        f"(gold={score.num_gold} hits={score.num_hits} misses={score.num_misses} "
        f"false_pos={score.num_false_positives})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["lmstudio", "claude"], required=True)
    parser.add_argument("--model", default=None, help="Override the backend's default model name")
    parser.add_argument(
        "--gold", default=None, help="Path to one gold JSON file (default: run every file in evals/gold/)"
    )
    parser.add_argument("--out", default=None, help="Path to write the full JSON report (default: auto-named)")
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

    gold_paths: List[Path] = [Path(args.gold)] if args.gold else sorted(GOLD_DIR.glob("*.json"))
    if not gold_paths:
        print(f"No gold files found in {GOLD_DIR}", file=sys.stderr)
        return 1

    max_candidates = args.max_candidates
    if max_candidates is None and args.embedder == "tfidf":
        max_candidates = TFIDF_RECOMMENDED_MAX_CANDIDATES

    embedder = _EMBEDDERS[args.embedder]()
    reports = []
    for gold_path in gold_paths:
        report = run_one(gold_path, args.backend, args.model, embedder, max_candidates)
        reports.append(report)
        _print_score(gold_path.stem, EvalScore(**report["score"]))

    model_slug = (args.model or args.backend).replace("/", "-")
    if args.embedder != "bge-m3":
        # Keeps existing bge-m3 report filenames unchanged; only tags the
        # embedder on when it's not the default, so a tfidf run doesn't
        # silently overwrite a same-model bge-m3 report (or vice versa).
        model_slug += f"-{args.embedder}"
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{model_slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nWrote full report to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
