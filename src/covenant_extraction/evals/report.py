"""Render an eval report (see cli.run_eval) as markdown."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_FIELD_LABELS = [
    ("threshold_exact", "Threshold, exact"),
    ("threshold_tolerance", "Threshold, within tolerance"),
    ("comparison", "Comparison direction"),
    ("covenant_type", "Covenant type (maintenance vs incurrence)"),
    ("unit", "Unit"),
    ("name", "Name"),
    ("test_frequency", "Test frequency (maintenance only)"),
    ("all_fields", "All of threshold + direction + type + unit"),
]


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: Any) -> str:
    if x is None:
        return "none"
    x = float(x)
    if abs(x) >= 1000:
        return f"${x:,.0f}" if x >= 0 else f"$({abs(x):,.0f})"
    return f"{x:g}"


def _short(text: str, n: int = 110) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_markdown(report: Dict[str, Any]) -> str:
    meta, s, docs = report["meta"], report["summary"], report["documents"]
    out: List[str] = []
    out.append(f"# Covenant extraction eval: {meta['label']}")
    out.append("")
    out.append(f"- Source of extractions: {meta['source']}")
    out.append(f"- Gold set: {meta['gold_files']} documents in `evals/gold/`, {s['documents_scored']} scored")
    if s["documents_skipped"]:
        out.append(f"- Skipped (no extraction available): {', '.join(s['documents_skipped'])}")
    out.append(f"- Numeric tolerance: {s['rel_tol']:.0%} relative")
    out.append(f"- Date: {meta['date']}")
    out.append("")

    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Precision | {_pct(s['precision'])} ({s['true_positives']} of {s['true_positives'] + s['false_positives']}) |")
    out.append(f"| Recall | {_pct(s['recall'])} ({s['true_positives']} of {s['true_positives'] + s['false_negatives']}) |")
    out.append(f"| F1 | {s['f1']:.2f} |")
    fc = s["field_counts"]
    if fc.get("all_fields"):
        c, n = fc["all_fields"]
        out.append(f"| Matched covenants fully correct | {_pct(c / n if n else None)} ({c} of {n}) |")
    if s["schedule_covenants"]:
        out.append(
            f"| Step-down schedule levels captured | {s['schedule_levels_found']} of {s['schedule_levels']} "
            f"({s['schedule_complete']} of {s['schedule_covenants']} schedules complete) |"
        )
    if s["no_covenant_documents"]:
        out.append(
            f"| Documents with no required covenants and zero false positives | "
            f"{s['no_covenant_documents_clean']} of {s['no_covenant_documents']} |"
        )
    out.append("")

    out.append("## Recall by covenant type")
    out.append("")
    out.append("| Type | Found | Gold | Recall |")
    out.append("|---|---|---|---|")
    for t, v in s["recall_by_covenant_type"].items():
        out.append(f"| {t} | {v['hits']} | {v['gold']} | {_pct(v['recall'])} |")
    out.append("")

    out.append("## Field-level accuracy (matched, required covenants)")
    out.append("")
    out.append("| Field | Correct | Scored | Accuracy |")
    out.append("|---|---|---|---|")
    for key, label in _FIELD_LABELS:
        c, n = fc.get(key, [0, 0])
        out.append(f"| {label} | {c} | {n} | {_pct(c / n if n else None)} |")
    out.append("")

    out.append("## Per document")
    out.append("")
    out.append("| Document | Gold | Found | Missed | False pos. | Precision | Recall | Fully correct |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in docs:
        if not r.get("score"):
            out.append(f"| {r['gold_id']} | – | – | – | – | – | – | not run |")
            continue
        sc = r["score"]
        c, n = r["field_accuracy"]["counts"]["all_fields"]
        out.append(
            f"| {r['gold_id']} | {sc['num_gold']} | {sc['num_hits']} | {sc['num_misses']} | "
            f"{sc['num_false_positives']} | {_pct(sc['precision'])} | {_pct(sc['recall'])} | {c}/{n} |"
        )
    out.append("")

    out.append("## Errors")
    out.append("")
    any_errors = False
    for r in docs:
        if not r.get("score"):
            continue
        lines: List[str] = []
        for g in r["misses"]:
            lines.append(f"- **Missed**: {g['name']} ({g['covenant_type']}, {_num(g.get('threshold'))})")
        for f in r["fields"]:
            wrong = [
                label
                for key, label in [
                    ("threshold_exact", "threshold"),
                    ("comparison", "direction"),
                    ("covenant_type_correct", "type"),
                    ("unit", "unit"),
                    ("name", "name"),
                    ("test_frequency", "frequency"),
                ]
                if f.get(key) is False
            ]
            if wrong:
                lines.append(
                    f"- **Wrong field(s)** on {f['gold_name']}: {', '.join(wrong)} "
                    f"(extracted {_num(f['extracted_threshold'])}, gold {_num(f['gold_threshold'])})"
                )
            if f["schedule_levels"] and not f["schedule_complete"]:
                lines.append(
                    f"- **Incomplete schedule** on {f['gold_name']}: "
                    f"{f['schedule_levels_found']} of {f['schedule_levels']} step levels extracted"
                )
        for e in r["false_positives"]:
            src = _short((e.get("citation") or {}).get("source_text", ""))
            lines.append(
                f"- **False positive**: {e.get('name')} {_num(e.get('threshold'))} "
                f"({e.get('covenant_type')}) citing \"{src}\""
            )
        if lines:
            any_errors = True
            out.append(f"### {r['gold_id']}")
            out.append("")
            out.extend(lines)
            out.append("")
    if not any_errors:
        out.append("None.")
        out.append("")

    optional = [(r["gold_id"], g) for r in docs if r.get("score") for g in r["optional_found"]]
    if optional:
        out.append("## Optional items the extractor reported")
        out.append("")
        out.append("These are borderline items (gold `required: false`). They count neither for nor against the scores.")
        out.append("")
        for gid, g in optional:
            out.append(f"- {gid}: {g['name']} ({_num(g.get('threshold'))})")
        out.append("")

    return "\n".join(out)
