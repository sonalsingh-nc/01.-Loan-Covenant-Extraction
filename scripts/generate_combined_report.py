"""CLI: combine every extraction result in data/processed/*.json into one
HTML table -- one row per financial covenant, with the covenant's document's
loan-level deal terms (borrower, lenders, loan type, amount, interest terms,
maturity date) repeated alongside it.

Usage:
    python scripts/generate_combined_report.py
    python scripts/generate_combined_report.py --out report.html
    python scripts/generate_combined_report.py --dir data/processed

A document contributes one row per financial_covenant it has, or a single
row with covenant columns blank if it has none, so every processed document
is represented at least once. Older result files that predate the
`loan_terms` field (see extraction/loan_terms.py) show blank borrower/lender/
etc. cells -- flagged in a note rather than silently left unexplained.

Any covenant or loan-term field whose own audit flags came back False
(citation_grounded, citation_verified) is also surfaced in a "flagged for
review" list -- generated from the data, not hand-maintained, so it stays
accurate as results change.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_STYLE = """
:root {
  --paper: #f3f5f7;
  --paper-raised: #ffffff;
  --ink: #1c2531;
  --ink-soft: #5b6675;
  --line: #c7cdd6;
  --line-soft: #dde1e7;
  --accent: #3a5a78;
  --accent-soft: #e4ebf1;
  --warn: #9a5b12;
  --warn-soft: #f6e9d6;
  --warn-line: #c4791f;
  --badge-bg: #e2e9f0;
  --badge-ink: #3e5c76;
  --ok: #2e7d4f;
  --ok-soft: #e3f2e8;
  --ok-line: #4d9b6c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #14181f;
    --paper-raised: #1b212b;
    --ink: #e8eaee;
    --ink-soft: #9aa4b2;
    --line: #333b48;
    --line-soft: #262d38;
    --accent: #8fb3d1;
    --accent-soft: #1d2a35;
    --warn: #e6b567;
    --warn-soft: #2f2313;
    --warn-line: #e0a94d;
    --badge-bg: #202b38;
    --badge-ink: #a9c3de;
    --ok: #7fd9a4;
    --ok-soft: #17281e;
    --ok-line: #4d9b6c;
  }
}
:root[data-theme="dark"] {
  --paper: #14181f;
  --paper-raised: #1b212b;
  --ink: #e8eaee;
  --ink-soft: #9aa4b2;
  --line: #333b48;
  --line-soft: #262d38;
  --accent: #8fb3d1;
  --accent-soft: #1d2a35;
  --warn: #e6b567;
  --warn-soft: #2f2313;
  --warn-line: #e0a94d;
  --badge-bg: #202b38;
  --badge-ink: #a9c3de;
  --ok: #7fd9a4;
  --ok-soft: #17281e;
  --ok-line: #4d9b6c;
}

* { box-sizing: border-box; }

body {
  background: var(--paper);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 1400px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
  line-height: 1.5;
}

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin: 0 0 0.5rem;
}

h1 {
  font-size: clamp(1.5rem, 2.2vw, 1.9rem);
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 0.6rem;
  text-wrap: balance;
}

.meta {
  color: var(--ink-soft);
  font-size: 0.9rem;
  margin: 0 0 1.8rem;
}

.review {
  background: var(--warn-soft);
  border: 1px solid var(--warn-line);
  border-radius: 10px;
  padding: 1rem 1.3rem;
  margin: 0 0 1.8rem;
}
.review h2 {
  margin: 0 0 0.6rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--warn);
}
.review ul { margin: 0; padding-left: 1.2rem; }
.review li { margin-bottom: 0.35rem; font-size: 0.87rem; color: var(--ink); }
.review code {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.88em;
  background: var(--paper-raised);
  border: 1px solid var(--line-soft);
  border-radius: 4px;
  padding: 0.03em 0.32em;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line-soft);
  border-radius: 10px;
  background: var(--paper-raised);
}

table { border-collapse: collapse; width: 100%; font-size: 0.85rem; min-width: 1100px; }
th, td { padding: 0.55rem 0.75rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line-soft); }
th {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
  font-size: 0.76rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  position: sticky;
  top: 0;
}
tbody tr:hover { background: var(--accent-soft); }
td.num { font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.83rem; }
td.doc-cell { font-weight: 600; }

.empty { color: var(--ink-soft); font-style: italic; }
.note-ref {
  color: var(--accent);
  text-decoration: none;
  font-size: 0.82em;
  white-space: nowrap;
  border-bottom: 1px dotted var(--accent);
}
.note-ref:hover { border-bottom-style: solid; }
.footnote li:target { background: var(--accent-soft); }
.badge {
  display: inline-block;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  background: var(--badge-bg);
  color: var(--badge-ink);
  margin-left: 0.4rem;
  white-space: nowrap;
}

.footnote { margin-top: 1.5rem; font-size: 0.85rem; color: var(--ink-soft); }
.footnote ul { margin: 0.4rem 0 0; padding-left: 1.3rem; }

.covenant-link {
  color: inherit;
  text-decoration: none;
  border-bottom: 1px dotted var(--accent);
}
.covenant-link:hover { color: var(--accent); border-bottom-style: solid; }

.detail-section { margin-top: 2.5rem; }
.detail-section > h2 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 0 0 1rem;
}
.detail-card {
  border: 1px solid var(--line-soft);
  border-radius: 10px;
  background: var(--paper-raised);
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
  scroll-margin-top: 1rem;
}
.detail-card:target { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.detail-card .eyebrow { margin-bottom: 0.3rem; }
.detail-card h3 { margin: 0 0 0.5rem; font-size: 1.05rem; }
.detail-card .terms { margin: 0 0 0.6rem; color: var(--ink-soft); }
.detail-card .terms strong { color: var(--ink); font-weight: 600; }
.detail-card .citation {
  font-style: italic;
  border-left: 3px solid var(--line);
  background: var(--accent-soft);
  padding: 0.5rem 0.75rem;
  margin: 0.6rem 0;
  border-radius: 0 6px 6px 0;
}
.detail-card .citation .location { font-style: normal; font-weight: 600; color: var(--ink); }
.detail-card .definition { margin: 0.6rem 0; }
.detail-card .definition-label { font-weight: 600; display: block; margin-bottom: 0.2rem; }
.detail-card .chain { margin: 0.6rem 0; }
.detail-card .chain .formula {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.85rem;
  background: var(--accent-soft);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
}
.detail-card .chain-components { list-style: none; margin: 0.5rem 0 0 0.2rem; padding-left: 1rem; border-left: 2px solid var(--line-soft); }
.detail-card .chain-components li { margin-bottom: 0.5rem; font-size: 0.9rem; }
.detail-card .exceptions { margin: 0.3rem 0 0 1.25rem; font-size: 0.88rem; color: var(--ink-soft); }
.detail-card .badge.ok { background: var(--ok-soft); color: var(--ok); }
.detail-card .badge.warn { background: var(--warn-soft); color: var(--warn); }
.back-to-table { display: inline-block; margin-top: 0.7rem; font-size: 0.82rem; }
"""

EMPTY = '<span class="empty">&mdash;</span>'
MAX_CELL_CHARS = 100


def _cited_value(item: Optional[Dict[str, Any]]) -> str:
    """Prefer `resolved_value.formula` when present (pipeline.py's
    definitions-chaining for end_date/loan_amount/interest_rate -- see
    pipeline._export_cited_text) over the raw extracted `value`, since a
    present resolved_value means `value` was itself just a defined term's
    name (e.g. "Term Loan Maturity Date") rather than the actual figure.
    Falls back to `value` for every other field, which never carries a
    resolved_value key."""
    if not item:
        return EMPTY
    resolved = item.get("resolved_value")
    if resolved and resolved.get("formula"):
        return html.escape(resolved["formula"])
    if not item.get("value"):
        return EMPTY
    return html.escape(item["value"])


def _cited_values_joined(items: List[Dict[str, Any]]) -> str:
    if not items:
        return EMPTY
    return "; ".join(html.escape(i["value"]) for i in items if i.get("value"))


def _truncated_cell(items: List[Dict[str, Any]], field_label: str, note_id: str) -> tuple:
    """Render a cell for a list-valued field (borrowers/lenders): the full,
    semicolon-joined list if it's under MAX_CELL_CHARS, otherwise just the
    first value plus a link to a footnote listing the rest.

    Returns (cell_html, footnote_html_or_None). The footnote (when present)
    is meant to be collected once per document/field, not once per row --
    every covenant row for the same document shares the same loan_terms, so
    the caller is responsible for only emitting it once per (doc, field).
    """
    values = [i["value"] for i in items if i.get("value")]
    if not values:
        return EMPTY, None

    joined = "; ".join(values)
    if len(joined) <= MAX_CELL_CHARS:
        return "; ".join(html.escape(v) for v in values), None

    first, rest = values[0], values[1:]
    cell = (
        f'{html.escape(first)} '
        f'<a href="#{note_id}" class="note-ref">+{len(rest)} more &rarr;</a>'
    )
    footnote = (
        f'<li id="{note_id}"><strong>{html.escape(field_label)}:</strong> '
        + "; ".join(html.escape(v) for v in rest)
        + "</li>"
    )
    return cell, footnote


def _format_threshold(covenant: Dict[str, Any]) -> str:
    threshold = covenant.get("threshold")
    unit = covenant.get("unit") or ""
    if threshold is None:
        return EMPTY
    if unit.strip().upper() == "USD":
        return f"${threshold:,.0f}"
    formatted = f"{threshold:g}"
    return f"{formatted} {unit}".strip() if unit else formatted


def _format_comparison(covenant: Dict[str, Any]) -> str:
    comparison = covenant.get("comparison")
    if not comparison:
        return EMPTY
    return html.escape(comparison.replace("_", " "))


def _citation_block_html(citation: Dict[str, Any]) -> str:
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
    return " ".join(badges)


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


def _covenant_detail_html(doc_name: str, anchor_id: str, covenant: Dict[str, Any]) -> str:
    """One master-detail "detail" card for a single financial covenant: its
    full definition (if found in the contract's Definitions section) and
    resolved formula chain, alongside its citation and audit badges --
    everything the summary table row doesn't have room for."""
    covenant_type = covenant.get("covenant_type")
    type_badge = f' <span class="badge">{html.escape(covenant_type)}</span>' if covenant_type else ""

    definition = covenant.get("definition")
    definition_html = (
        f'<div class="definition"><span class="definition-label">Definition</span>{html.escape(definition)}</div>'
        if definition
        else '<div class="definition"><span class="definition-label">Definition</span>'
        '<span class="empty">not found in Definitions section</span></div>'
    )

    resolved_definition = covenant.get("resolved_definition")
    chain_html = (
        f'<div class="chain"><span class="definition-label">Resolved chain</span>{_resolved_formula_html(resolved_definition)}</div>'
        if resolved_definition
        else ""
    )

    frequency = covenant.get("test_frequency")
    frequency_suffix = f" &mdash; {html.escape(frequency)}" if frequency else ""

    return (
        f'<div class="detail-card" id="{anchor_id}">'
        f'<p class="eyebrow">{html.escape(doc_name)}</p>'
        f'<h3>{html.escape(covenant.get("name", ""))}{type_badge}</h3>'
        f'<p class="terms"><strong>{_format_comparison(covenant)}</strong> {_format_threshold(covenant)}{frequency_suffix}</p>'
        f"{definition_html}"
        f"{chain_html}"
        f'{_citation_block_html(covenant.get("citation") or {})}'
        f'<p>{_audit_badges_html(covenant.get("audit") or {})}</p>'
        f'<a class="back-to-table" href="#report-table">&uarr; Back to table</a>'
        "</div>"
    )


def _rows_for_document(doc_name: str, doc_slug: str, data: Dict[str, Any]) -> tuple:
    """Returns (rows, footnotes, details) -- footnotes has 0-2 entries
    (borrowers, lenders) emitted once for the document regardless of how
    many covenant rows it contributes; details has one detail-card entry
    per covenant row, in the same order, for the master-detail section."""
    loan_terms = data.get("loan_terms") or {}
    borrower, borrower_note = _truncated_cell(
        loan_terms.get("borrowers") or [], "Other borrowers", f"note-{doc_slug}-borrowers"
    )
    lenders, lenders_note = _truncated_cell(
        loan_terms.get("lenders") or [], "Other lenders", f"note-{doc_slug}-lenders"
    )
    footnotes = [n for n in (borrower_note, lenders_note) if n]

    loan_type = _cited_value(loan_terms.get("loan_type"))
    amount = _cited_value(loan_terms.get("loan_amount"))
    interest_terms = _cited_value(loan_terms.get("interest_rate"))
    maturity_date = _cited_value(loan_terms.get("end_date"))

    loan_level = [borrower, lenders, loan_type, amount, interest_terms, maturity_date]
    covenants = data.get("financial_covenants") or []

    if not covenants:
        return [[html.escape(doc_name), *loan_level, EMPTY, EMPTY, EMPTY, EMPTY]], footnotes, []

    rows = []
    details = []
    for index, covenant in enumerate(covenants):
        anchor_id = f"detail-{doc_slug}-{index}"
        type_badge = (
            f'<span class="badge">{html.escape(covenant["covenant_type"])}</span>'
            if covenant.get("covenant_type")
            else ""
        )
        section = covenant.get("citation", {}).get("section_label") or EMPTY
        rows.append(
            [
                html.escape(doc_name),
                *loan_level,
                f'<a class="covenant-link" href="#{anchor_id}">{html.escape(covenant.get("name", ""))}</a>{type_badge}',
                _format_comparison(covenant),
                _format_threshold(covenant),
                html.escape(str(section)) if section != EMPTY else EMPTY,
            ]
        )
        details.append(_covenant_detail_html(doc_name, anchor_id, covenant))
    return rows, footnotes, details


_LOAN_TERM_LABELS = {
    "loan_type": "loan type",
    "loan_amount": "loan amount",
    "interest_rate": "interest rate",
    "start_date": "start date",
    "end_date": "maturity date",
    "principal_repayment_terms": "principal repayment terms",
}


def _collect_review_flags(doc_name: str, data: Dict[str, Any]) -> List[str]:
    """Surface anything whose own audit came back False -- generated from
    the data (citation_grounded / citation_verified), not hand-maintained,
    so this stays accurate as results are re-run or backfilled."""
    flags: List[str] = []

    for covenant in data.get("financial_covenants") or []:
        audit = covenant.get("audit") or {}
        name = covenant.get("name", "?")
        if audit.get("citation_grounded") is False:
            flags.append(
                f'<strong>{html.escape(doc_name)}</strong> &mdash; <code>{html.escape(name)}</code> '
                f"(threshold {covenant.get('threshold')}): threshold does not appear in its own citation text"
            )
        if audit.get("citation_verified") is False:
            flags.append(
                f'<strong>{html.escape(doc_name)}</strong> &mdash; <code>{html.escape(name)}</code>: '
                f"citation quote not verified against the source document (similarity {audit.get('citation_similarity')})"
            )

    loan_terms = data.get("loan_terms") or {}
    for field_key, label in _LOAN_TERM_LABELS.items():
        item = loan_terms.get(field_key)
        if item and (item.get("audit") or {}).get("citation_verified") is False:
            similarity = (item.get("audit") or {}).get("citation_similarity")
            flags.append(
                f'<strong>{html.escape(doc_name)}</strong> &mdash; {label} '
                f'(<code>{html.escape(str(item.get("value", "")))}</code>): '
                f"citation quote not verified against the source document (similarity {similarity})"
            )

    return flags


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_report_html(processed_dir: Path) -> str:
    json_paths = sorted(processed_dir.glob("*.json"))
    all_rows: List[List[str]] = []
    all_details: List[str] = []
    stale_docs: List[str] = []
    review_flags: List[str] = []
    party_footnotes: List[str] = []

    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_name = path.stem
        if "loan_terms" not in data:
            stale_docs.append(doc_name)
        rows, footnotes, details = _rows_for_document(doc_name, _slugify(doc_name), data)
        all_rows.extend(rows)
        all_details.extend(details)
        party_footnotes.extend(footnotes)
        review_flags.extend(_collect_review_flags(doc_name, data))

    columns = [
        "Document",
        "Borrower",
        "Lenders",
        "Loan Type",
        "Amount",
        "Interest Terms",
        "Maturity Date",
        "Covenant Name",
        "Comparison",
        "Threshold",
        "Section",
    ]
    num_columns = {"Threshold", "Section"}
    header_html = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body_html = "\n".join(
        "<tr>"
        + "".join(
            f'<td class="{"doc-cell" if col == "Document" else ("num" if col in num_columns else "")}">{cell}</td>'
            for col, cell in zip(columns, row)
        )
        + "</tr>"
        for row in all_rows
    )

    review_html = ""
    if review_flags:
        items = "".join(f"<li>{flag}</li>" for flag in review_flags)
        review_html = (
            '<div class="review"><h2>Flagged for review</h2>'
            f"<ul>{items}</ul></div>"
        )

    party_footnotes_html = ""
    if party_footnotes:
        party_footnotes_html = (
            f'<div class="footnote"><strong>Full party lists</strong> (truncated in the table above at '
            f"{MAX_CELL_CHARS} characters):<ul>{''.join(party_footnotes)}</ul></div>"
        )

    stale_note = ""
    if stale_docs:
        items = "".join(f"<li>{html.escape(d)}</li>" for d in stale_docs)
        stale_note = (
            '<div class="footnote"><strong>Note:</strong> the following result files predate loan-level '
            f"term extraction, so Borrower/Lenders/Loan Type/Amount/Interest Terms/Maturity Date are blank "
            f"for their rows (not missing data -- just not yet re-extracted):<ul>{items}</ul></div>"
        )

    details_html = ""
    if all_details:
        details_html = (
            '<div class="detail-section"><h2>Covenant details</h2>' + "\n".join(all_details) + "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Combined Covenant Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{_STYLE}</style>
</head>
<body>
<p class="eyebrow">covenant_extraction &middot; combined across data/processed</p>
<h1>Combined Covenant Report</h1>
<p class="meta">{len(json_paths)} document(s) &middot; {len(all_rows)} row(s) &middot; source: {html.escape(str(processed_dir))} &middot; click a covenant name for its full definition</p>
{review_html}
<div class="table-wrap" id="report-table">
<table>
<thead><tr>{header_html}</tr></thead>
<tbody>
{body_html}
</tbody>
</table>
</div>
{party_footnotes_html}
{stale_note}
{details_html}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/processed", help="Directory of extraction JSON files to combine")
    parser.add_argument("--out", default="data/processed/combined_report.html", help="Path to write the HTML report")
    args = parser.parse_args()

    processed_dir = Path(args.dir)
    if not processed_dir.is_dir():
        print(f"No such directory: {processed_dir}", file=sys.stderr)
        return 1

    report_html = build_report_html(processed_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_html, encoding="utf-8")
    print(f"Wrote combined report to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
