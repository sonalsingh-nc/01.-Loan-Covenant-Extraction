"""CLI: run the covenant extraction pipeline on a loan agreement and write
the result as a human-readable HTML report.

Usage:
    python scripts/generate_html_report.py path/to/contract.pdf [--out report.html]
    python scripts/generate_html_report.py path/to/contract.pdf --embedder tfidf
    python scripts/generate_html_report.py path/to/contract.pdf --backend claude --embedder tfidf

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
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Dict, List

from covenant_extraction.extraction.llm_client import ClaudeClient, LMStudioClient
from covenant_extraction.pipeline import run_pipeline
from covenant_extraction.retrieval.candidate_selector import TFIDF_RECOMMENDED_MAX_CANDIDATES
from covenant_extraction.retrieval.embeddings import BgeM3Embedder, TfidfEmbedder

_EMBEDDERS = {"bge-m3": BgeM3Embedder, "tfidf": TfidfEmbedder}

_STYLE = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
h1 { border-bottom: 2px solid #1a1a1a; padding-bottom: 0.5rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }
.item { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; background: #fafafa; }
.item h3 { margin-top: 0; }
.citation { font-style: italic; color: #444; background: #f0f0f0; border-left: 3px solid #888; padding: 0.5rem 0.75rem; margin: 0.5rem 0; }
.citation .location { font-style: normal; font-weight: bold; color: #000; }
.definition { color: #333; }
.badge { display: inline-block; font-size: 0.85rem; padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.4rem; }
.badge.ok { background: #d4edda; color: #155724; }
.badge.warn { background: #f8d7da; color: #721c24; }
.badge.type { background: #e2e3ff; color: #2d2a7a; }
.exceptions { margin: 0.25rem 0 0 1.25rem; }
.empty { color: #777; font-style: italic; }
.chain { margin-top: 0.5rem; }
.chain .formula { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #eef; padding: 0.1rem 0.3rem; border-radius: 3px; }
.chain-components { list-style: none; margin: 0.4rem 0 0 1rem; padding-left: 1rem; border-left: 2px solid #ccd; }
.chain-components li { margin-bottom: 0.4rem; }
.loan-terms { display: grid; grid-template-columns: 12rem 1fr; row-gap: 0.75rem; column-gap: 1rem; }
.loan-terms dt { font-weight: bold; }
.loan-terms dd { margin: 0; }
.loan-terms dd .citation { margin: 0.25rem 0 0; }
"""


def _citation_html(citation: Dict[str, Any]) -> str:
    location = citation.get("page") and f"page {citation['page']}" or citation.get("section_label") or "unknown location"
    return (
        '<div class="citation">'
        f'<span class="location">{html.escape(str(location))}</span>: '
        f'&ldquo;{html.escape(citation.get("source_text", ""))}&rdquo;'
        "</div>"
    )


def _audit_badges_html(audit: Dict[str, Any]) -> str:
    verified = audit.get("citation_verified")
    badges = [
        f'<span class="badge {"ok" if verified else "warn"}">'
        f'citation {"verified" if verified else "unverified"} ({audit.get("citation_similarity", 0):.2f})</span>'
    ]
    if audit.get("calculator_checked"):
        match = audit.get("calculator_match")
        badges.append(f'<span class="badge {"ok" if match else "warn"}">calculator {"match" if match else "mismatch"}</span>')
    if "citation_grounded" in audit:
        grounded = audit["citation_grounded"]
        badges.append(
            f'<span class="badge {"ok" if grounded else "warn"}">'
            f'threshold {"grounded" if grounded else "NOT found in citation"}</span>'
        )
    return "\n".join(badges)


def _resolved_formula_html(resolved: Dict[str, Any]) -> str:
    """Render one node of a ResolvedFormula tree (see
    definitions/schema.py) -- its formula, resolution status, any
    carve-outs, and (recursively) the components it was composed from."""
    status = resolved.get("status", "resolved")
    if status == "resolved":
        status_badge = '<span class="badge ok">resolved</span>'
    else:
        reason = resolved.get("reason")
        reason_suffix = f": {html.escape(reason)}" if reason else ""
        status_badge = f'<span class="badge warn">requires review{reason_suffix}</span>'

    formula = html.escape(resolved.get("formula") or "")
    formula_html = f'<span class="formula">{formula}</span> ' if formula else ""

    carve_outs = resolved.get("carve_outs") or []
    carve_outs_html = (
        '<ul class="exceptions">' + "".join(f"<li>{html.escape(c)}</li>" for c in carve_outs) + "</ul>"
        if carve_outs
        else ""
    )

    components = resolved.get("components") or {}
    components_html = ""
    if components:
        component_items = "".join(
            f"<li><strong>{html.escape(name)}:</strong> {_resolved_formula_html(component)}</li>"
            for name, component in components.items()
        )
        components_html = f'<ul class="chain-components">{component_items}</ul>'

    return f"{formula_html}{status_badge}{carve_outs_html}{components_html}"


def _financial_covenants_html(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<p class="empty">No financial covenants extracted.</p>'

    blocks = []
    for item in items:
        definition = item.get("definition")
        definition_html = html.escape(definition) if definition else '<span class="empty">not found in Definitions section</span>'
        frequency = f' &mdash; {html.escape(item["test_frequency"])}' if item.get("test_frequency") else ""

        resolved_definition = item.get("resolved_definition")
        chain_html = (
            f'<div class="chain"><strong>Resolved chain:</strong><br>{_resolved_formula_html(resolved_definition)}</div>'
            if resolved_definition
            else ""
        )

        covenant_type = item.get("covenant_type")
        type_badge = f' <span class="badge type">{html.escape(covenant_type)}</span>' if covenant_type else ""

        blocks.append(
            '<div class="item">'
            f'<h3>{html.escape(item["name"])}{type_badge}</h3>'
            f'<p><strong>{html.escape(item["comparison"])}</strong> {item["threshold"]} {html.escape(item["unit"])}{frequency}</p>'
            f'<p class="definition"><strong>Definition:</strong> {definition_html}</p>'
            f"{chain_html}"
            f'{_citation_html(item["citation"])}'
            f'<p>{_audit_badges_html(item["audit"])}</p>'
            "</div>"
        )
    return "\n".join(blocks)


def _ebitda_addbacks_html(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<p class="empty">No EBITDA add-backs extracted.</p>'

    blocks = []
    for item in items:
        description = f"<p>{html.escape(item['description'])}</p>" if item.get("description") else ""
        blocks.append(
            '<div class="item">'
            f'<h3>{html.escape(item["name"])}</h3>'
            f"{description}"
            f'{_citation_html(item["citation"])}'
            f'<p>{_audit_badges_html(item["audit"])}</p>'
            "</div>"
        )
    return "\n".join(blocks)


def _cited_text_html(item: Any) -> str:
    """Render one CitedText export (see extraction/schema.py) -- its value
    plus citation, or a "not stated" placeholder if the field is null."""
    if not item:
        return '<span class="empty">not stated</span>'
    return f'{html.escape(item["value"])}{_citation_html(item["citation"])}'


def _loan_terms_html(loan_terms: Dict[str, Any]) -> str:
    borrowers = loan_terms.get("borrowers") or []
    lenders = loan_terms.get("lenders") or []

    borrowers_html = (
        "".join(f"<div>{_cited_text_html(b)}</div>" for b in borrowers)
        if borrowers
        else '<span class="empty">not stated</span>'
    )
    lenders_html = (
        "".join(f"<div>{_cited_text_html(l)}</div>" for l in lenders)
        if lenders
        else '<span class="empty">not stated</span>'
    )

    rows = [
        ("Borrower(s)", borrowers_html),
        ("Lender(s)", lenders_html),
        ("Loan Type", _cited_text_html(loan_terms.get("loan_type"))),
        ("Loan Amount", _cited_text_html(loan_terms.get("loan_amount"))),
        ("Interest Rate", _cited_text_html(loan_terms.get("interest_rate"))),
        ("Start Date", _cited_text_html(loan_terms.get("start_date"))),
        ("Maturity Date", _cited_text_html(loan_terms.get("end_date"))),
        ("Principal Repayment Terms", _cited_text_html(loan_terms.get("principal_repayment_terms"))),
    ]
    dl_items = "".join(f"<dt>{html.escape(label)}</dt><dd>{value_html}</dd>" for label, value_html in rows)
    return f'<dl class="loan-terms">{dl_items}</dl>'


def _negative_covenants_html(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<p class="empty">No negative covenants extracted.</p>'

    blocks = []
    for item in items:
        exceptions = item.get("exceptions") or []
        exceptions_html = (
            "<ul class=\"exceptions\">" + "".join(f"<li>{html.escape(e)}</li>" for e in exceptions) + "</ul>"
            if exceptions
            else ""
        )
        blocks.append(
            '<div class="item">'
            f'<h3>{html.escape(item["name"])}</h3>'
            f'<p>{html.escape(item["summary"])}</p>'
            f"{exceptions_html}"
            f'{_citation_html(item["citation"])}'
            f'<p>{_audit_badges_html(item["audit"])}</p>'
            "</div>"
        )
    return "\n".join(blocks)


def build_report_html(contract_path: str, result: Dict[str, Any]) -> str:
    """Render a pipeline result dict (see pipeline.to_export_dict) as a
    standalone HTML report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Covenant Extraction Report - {html.escape(Path(contract_path).name)}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Covenant Extraction Report</h1>
<p><strong>Source document:</strong> {html.escape(str(contract_path))}</p>

<h2>Loan Terms</h2>
{_loan_terms_html(result.get("loan_terms", {}))}

<h2>Financial Covenants</h2>
{_financial_covenants_html(result.get("financial_covenants", []))}

<h2>EBITDA Add-backs</h2>
{_ebitda_addbacks_html(result.get("ebitda_addbacks", []))}

<h2>Negative Covenants</h2>
{_negative_covenants_html(result.get("negative_covenants", []))}

</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contract_path", help="Path to a .pdf or .html/.htm loan contract")
    parser.add_argument(
        "--out", default=None, help="Path to write the HTML report (default: <contract_stem>_report.html)"
    )
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
    report_html = build_report_html(args.contract_path, result)

    out_path = Path(args.out) if args.out else Path(args.contract_path).with_name(Path(args.contract_path).stem + "_report.html")
    out_path.write_text(report_html, encoding="utf-8")
    print(f"Wrote HTML report to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
