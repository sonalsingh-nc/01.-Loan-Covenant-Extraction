"""CLI: score covenant extraction against the hand-labeled gold set in
evals/gold/ and write a JSON report plus a markdown results report.

Three ways to get the extractions being scored:

    # Re-score the committed pipeline outputs in data/processed/ (no LLM,
    # no network -- each gold file names its `cached_output`).
    covenant-eval --backend cached

    # Re-score a saved run from an earlier live eval (no LLM).
    covenant-eval --from-results evals/results/claude.json

    # Run the full pipeline live, then score it.
    covenant-eval --backend lmstudio
    covenant-eval --backend claude --model claude-haiku-4-5

    # Check that every gold anchor really appears in its source document.
    covenant-eval --check-gold

`python evals/run_eval.py ...` takes the same arguments.

Live runs need: --backend lmstudio, LM Studio running locally with a model
loaded; --backend claude, an Anthropic API credential (real cost). Either
way, --embedder bge-m3 (default) needs the BGE-M3 model cached locally;
--embedder tfidf needs no download.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from covenant_extraction.evals.field_metrics import DEFAULT_REL_TOL, aggregate_fields, score_fields
from covenant_extraction.evals.matching import gold_anchors, match_covenants, normalize_text
from covenant_extraction.evals.report import render_markdown
from covenant_extraction.evals.scoring import EvalScore, score_matches

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = PROJECT_ROOT / "evals" / "gold"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

Extractor = Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]


def load_gold(paths: List[Path]) -> List[Dict[str, Any]]:
    golds = []
    for p in paths:
        gold = json.loads(p.read_text(encoding="utf-8"))
        gold["_id"] = p.stem
        golds.append(gold)
    return golds


def score_document(
    gold: Dict[str, Any], extracted: List[Dict[str, Any]], rel_tol: float = DEFAULT_REL_TOL
) -> Dict[str, Any]:
    """Score one document's extracted financial covenants against its gold."""
    matches, unmatched = match_covenants(gold["financial_covenants"], extracted)
    score = score_matches(matches, unmatched)
    field_results = [score_fields(m, rel_tol) for m in matches if m.required and m.extracted is not None]
    return {
        "gold_id": gold["_id"],
        "document": gold["document"],
        "score": asdict(score),
        "fields": [asdict(r) | {"schedule_complete": r.schedule_complete} for r in field_results],
        "field_accuracy": aggregate_fields(field_results).as_dict(),
        "misses": [m.gold for m in matches if m.required and m.extracted is None],
        "optional_found": [m.gold for m in matches if not m.required and m.extracted is not None],
        "false_positives": unmatched,
        "extracted_financial_covenants": extracted,
    }


def summarize(doc_reports: List[Dict[str, Any]], rel_tol: float) -> Dict[str, Any]:
    """Micro-averaged totals across every scored document."""
    scored = [r for r in doc_reports if r.get("score")]
    tp = sum(r["score"]["num_hits"] for r in scored)
    fn = sum(r["score"]["num_misses"] for r in scored)
    fp = sum(r["score"]["num_false_positives"] for r in scored)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_type: Dict[str, Dict[str, int]] = {}
    for r in scored:
        for g in r["misses"]:
            by_type.setdefault(g["covenant_type"], {"gold": 0, "hits": 0})["gold"] += 1
        for f in r["fields"]:
            t = by_type.setdefault(f["covenant_type"], {"gold": 0, "hits": 0})
            t["gold"] += 1
            t["hits"] += 1

    # Re-aggregate field accuracy across documents.
    counts: Dict[str, List[int]] = {}
    sched = {"schedule_covenants": 0, "schedule_complete": 0, "schedule_levels": 0, "schedule_levels_found": 0}
    for r in scored:
        fa = r["field_accuracy"]
        for k, (c, n) in fa["counts"].items():
            counts.setdefault(k, [0, 0])
            counts[k][0] += c
            counts[k][1] += n
        for k in sched:
            sched[k] += fa[k]

    no_covenant_docs = [r for r in scored if r["score"]["num_gold"] == 0]
    return {
        "documents_scored": len(scored),
        "documents_skipped": [r["gold_id"] for r in doc_reports if not r.get("score")],
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_by_covenant_type": {
            k: {**v, "recall": v["hits"] / v["gold"] if v["gold"] else None} for k, v in sorted(by_type.items())
        },
        "field_counts": counts,
        "field_accuracy": {k: (c / n if n else None) for k, (c, n) in counts.items()},
        **sched,
        "no_covenant_documents": len(no_covenant_docs),
        "no_covenant_documents_clean": sum(1 for r in no_covenant_docs if r["score"]["num_false_positives"] == 0),
        "rel_tol": rel_tol,
    }


def run_eval(golds: List[Dict[str, Any]], extractor: Extractor, rel_tol: float) -> Dict[str, Any]:
    doc_reports = []
    for gold in golds:
        extracted = extractor(gold)
        if extracted is None:
            doc_reports.append({"gold_id": gold["_id"], "document": gold["document"], "score": None})
            continue
        doc_reports.append(score_document(gold, extracted, rel_tol))
    return {"summary": summarize(doc_reports, rel_tol), "documents": doc_reports}


# --- extraction sources -------------------------------------------------------


def cached_extractor(gold: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    rel = gold.get("cached_output")
    if not rel or not (PROJECT_ROOT / rel).exists():
        return None
    return json.loads((PROJECT_ROOT / rel).read_text(encoding="utf-8")).get("financial_covenants", [])


def results_extractor(results_path: Path) -> Extractor:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    by_doc = {r["document"]: r.get("extracted_financial_covenants") for r in rows}

    def _extract(gold: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        return by_doc.get(gold["document"])

    return _extract


def live_extractor(backend: str, model: Optional[str], embedder_name: str, max_candidates: Optional[int]) -> Extractor:
    # Imported lazily so offline scoring never needs torch/openai/anthropic.
    from covenant_extraction.extraction.llm_client import ClaudeClient, LMStudioClient
    from covenant_extraction.pipeline import run_pipeline
    from covenant_extraction.retrieval.candidate_selector import TFIDF_RECOMMENDED_MAX_CANDIDATES
    from covenant_extraction.retrieval.embeddings import BgeM3Embedder, TfidfEmbedder

    embedder = {"bge-m3": BgeM3Embedder, "tfidf": TfidfEmbedder}[embedder_name]()
    if max_candidates is None and embedder_name == "tfidf":
        max_candidates = TFIDF_RECOMMENDED_MAX_CANDIDATES
    client_cls = {"lmstudio": LMStudioClient, "claude": ClaudeClient}[backend]

    def _extract(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"[{gold['_id']}] extracting with backend={backend} ...", file=sys.stderr, flush=True)
        llm_client = client_cls(model=model) if model else client_cls()
        result = run_pipeline(
            PROJECT_ROOT / gold["document"], embedder=embedder, llm_client=llm_client, max_candidates=max_candidates
        )
        return result["financial_covenants"]

    return _extract


# --- gold-set sanity check ----------------------------------------------------


def check_gold(golds: List[Dict[str, Any]]) -> int:
    """Every anchor must appear in its source document, or matching can
    never succeed and the item silently becomes a guaranteed miss."""
    from covenant_extraction.pipeline import load_document

    problems = 0
    for gold in golds:
        blocks = load_document(PROJECT_ROOT / gold["document"])
        full_text = normalize_text(" ".join(b.text for b in blocks))
        for cov in gold["financial_covenants"]:
            for anchor in gold_anchors(cov):
                if anchor not in full_text:
                    problems += 1
                    print(f"[{gold['_id']}] anchor not found in document: {anchor!r}")
        print(f"[{gold['_id']}] checked {len(gold['financial_covenants'])} covenant(s)")
    print("OK" if not problems else f"{problems} anchor(s) not found")
    return 1 if problems else 0


# --- main ---------------------------------------------------------------------


def _print_doc(r: Dict[str, Any]) -> None:
    if not r.get("score"):
        print(f"{r['gold_id']:35s} (no extraction available -- skipped)")
        return
    s = EvalScore(**r["score"])
    print(
        f"{r['gold_id']:35s} recall={s.recall:.2f} precision={s.precision:.2f} "
        f"threshold_acc={s.threshold_accuracy:.2f} type_acc={s.covenant_type_accuracy:.2f} "
        f"(gold={s.num_gold} hits={s.num_hits} misses={s.num_misses} false_pos={s.num_false_positives})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--backend", choices=["cached", "lmstudio", "claude"])
    source.add_argument("--from-results", type=Path, help="Re-score a saved eval results JSON")
    source.add_argument("--check-gold", action="store_true", help="Verify gold anchors appear in their documents")
    parser.add_argument("--model", default=None, help="Override the live backend's default model name")
    parser.add_argument("--gold", type=Path, default=None, help="One gold JSON file (default: all of evals/gold/)")
    parser.add_argument("--embedder", choices=["bge-m3", "tfidf"], default="bge-m3")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL, help="Relative tolerance for thresholds")
    parser.add_argument("--label", default=None, help="Run name used in output filenames and the report")
    parser.add_argument("--out", type=Path, default=None, help="JSON report path (default: evals/results/<label>.json)")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path (default: next to --out)")
    args = parser.parse_args(argv)

    gold_paths = [args.gold] if args.gold else sorted(GOLD_DIR.glob("*.json"))
    if not gold_paths:
        print(f"No gold files found in {GOLD_DIR}", file=sys.stderr)
        return 1
    golds = load_gold(gold_paths)

    if args.check_gold:
        return check_gold(golds)

    if args.from_results:
        extractor = results_extractor(args.from_results)
        label = args.label or f"{args.from_results.stem}-rescored"
        source_desc = f"saved run `{args.from_results.as_posix()}`"
    elif args.backend == "cached":
        extractor = cached_extractor
        label = args.label or "cached"
        source_desc = "committed pipeline outputs in `data/processed/`"
    else:
        extractor = live_extractor(args.backend, args.model, args.embedder, args.max_candidates)
        label = args.label or (args.model or args.backend).replace("/", "-")
        if args.embedder != "bge-m3":
            label += f"-{args.embedder}"
        source_desc = f"live run, backend `{args.backend}`, model `{args.model or 'default'}`, embedder `{args.embedder}`"

    report = run_eval(golds, extractor, args.rel_tol)
    report["meta"] = {"label": label, "source": source_desc, "date": date.today().isoformat(), "gold_files": len(golds)}

    for r in report["documents"]:
        _print_doc(r)
    s = report["summary"]
    print(f"\nOVERALL precision={s['precision']:.2f} recall={s['recall']:.2f} f1={s['f1']:.2f}")

    out_path = args.out or RESULTS_DIR / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path = args.report or out_path.with_suffix(".md")
    report_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {out_path} and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
