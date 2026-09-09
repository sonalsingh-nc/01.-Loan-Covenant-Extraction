"""HTML loading using BeautifulSoup.

Extracts text per top-level "section-like" element (h1/h2/h3/h4/p/li),
tracking the nearest preceding heading (h1-h4) as each element's location
label -- falling back to an explicit "Section X.Y" (or bare "X.Y") prefix
parsed from the element's own text when present, since that is more precise
than the enclosing heading, or to a bare "ARTICLE N." paragraph (common in
CMBS/commercial mortgage loans that have no heading tags at all). Element
ids are not used since they are frequently missing/non-descriptive and
don't help a reviewer locate the clause.

Some real-world contract exports (e.g. SEC EDGAR filings converted from
Word) contain no semantic h1-4/p/li tags at all -- every paragraph is a
plain <div> of <font> runs, with section numbers embedded as ordinary
leading text (e.g. "9.4 Financial Covenant."). When a document has no
h1-4/p/li tags at all, we fall back to leaf-level <div> elements (divs with
no nested <div>) as the paragraph unit.

Other exports are a *hybrid* of the two: most paragraphs are semantic <p>
tags, but some sections (schedules, riders, exhibits) are plain <div> runs
with no enclosing <p> at all. Leaf <div> elements that fall outside every
matched heading/p/li element's subtree are picked up as paragraphs too, so
that content isn't silently dropped just because the rest of the document
happens to use <p> tags.

Covenant threshold *schedules* (a step-down/step-up table of test dates and
corresponding numbers) are commonly rendered as an HTML <table> -- none of
which (<table>/<tr>/<td>) are section-like tags, so without explicit
handling every number in such a table is invisible to the rest of the
pipeline: the lead-in sentence ("...the corresponding amount set forth
below:") gets captured, the table that sentence refers to does not. Each
<table> becomes one Block, its rows flattened to one "cell | cell | cell"
line per row -- see `_table_rows_text`.
"""
from __future__ import annotations

import html as html_entities
import re
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from bs4.element import Tag

from covenant_extraction.ingestion import Block

_HEADING_TAGS = ("h1", "h2", "h3", "h4")
_SECTION_TAGS = _HEADING_TAGS + ("p", "li")

# Matches a "Section 6.1" or bare "6.1"-style heading prefix, optionally
# followed by a trailing period before the whitespace (e.g. "SECTION 6.10.
# Financial Covenants", common in Word-exported agreements). Requires a
# genuinely capital letter after the number so we don't false-match numeric
# values like "1.25 to 1.00" or "3.50% per annum" that happen to start a
# clause -- note the "Section" keyword is matched case-insensitively via a
# *scoped* inline flag rather than a global one, so that guard doesn't also
# end up (silently) accepting a lowercase letter there too.
_EXPLICIT_SECTION_RE = re.compile(r"^((?:(?i:section)\s+)?\d+\.\d+(?:\.\d+)*)\.?\s+[A-Z]")

# Some agreements (common in CMBS/commercial mortgage loans) have no h1-4
# heading tags at all and number their top-level sections as a bare
# "ARTICLE 4." on its own line/paragraph, with the descriptive title in a
# *separate* following paragraph (e.g. "ARTICLE 4." then "BORROWER
# COVENANTS") rather than inline like "SECTION 6.10. Financial Covenants."
# above -- so unlike _EXPLICIT_SECTION_RE, this must match the *entire*
# element text (no title expected in the same paragraph) and doesn't
# attempt to also capture that following title.
_ARTICLE_HEADER_RE = re.compile(r"^((?i:article)\s+\d+[a-z]?)\.?$")


def _orphan_leaf_divs(soup: BeautifulSoup) -> List[Tag]:
    """Leaf <div> elements (no nested <div>) that aren't already covered by
    a matched heading/p/li element -- excludes a div containing one (it'll
    be captured via that descendant instead) and a div inside one (it's
    already part of that ancestor's captured text). What's left is either
    the paragraph unit in a div-only export with no semantic tags at all,
    or the "extra" div-wrapped sections of a hybrid export that otherwise
    uses <p> tags for everything else. Also excludes a div inside a <table>
    -- that cell's text is captured via the table's own block instead (see
    `_table_rows_text`), so including it here would duplicate it."""
    return [
        d
        for d in soup.find_all("div")
        if d.find("div") is None
        and d.find(_SECTION_TAGS) is None
        and d.find_parent(_SECTION_TAGS) is None
        and d.find_parent("table") is None
    ]


def _table_rows_text(table: Tag) -> List[str]:
    """One text line per <tr>, its cells joined with " | " -- a plain-text
    approximation of a table (typically a covenant threshold schedule: one
    row per test date/period) since there is otherwise no representation
    for <table>/<tr>/<td> content anywhere in this pipeline. Empty cells
    (e.g. a bare "&nbsp;" spacer column) are dropped rather than kept as
    empty " | " segments."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def load_html(path: str | Path) -> List[Block]:
    """Load an HTML file and return one Block per section-like element.

    Args:
        path: path to the HTML file.

    Returns:
        List of Block objects with `section_label` set to the nearest
        preceding heading's text (e.g. "SECTION 6. FINANCIAL COVENANTS"), or
        an explicit "Section X.Y"/"X.Y" prefix found in the element's own
        text if present (more precise than the enclosing heading), and
        offsets into the concatenation of all block texts (joined with a
        single newline).
    """
    path = Path(path)
    raw_html = path.read_text(encoding="utf-8", errors="replace")
    # Some Word-exported documents encode curly quotes as numeric character
    # references in the 128-159 range (e.g. "&#147;"/"&#148;") -- a Windows
    # -1252 legacy convention that the HTML5 spec requires re-mapping to the
    # correct Unicode punctuation (U+201C/U+201D), but lxml's HTML parser
    # doesn't apply that override and instead decodes them literally into
    # C1 control characters. That silently breaks every downstream
    # quote-matching regex (retrieval/definitions_lookup.py's
    # find_definition and _looks_like_definition) for the *entire*
    # document -- not a parse error, just an empty definitions_text.
    # html.unescape() implements the correct HTML5 mapping, so pre-decode
    # entities before lxml ever sees them.
    raw_html = html_entities.unescape(raw_html)
    soup = BeautifulSoup(raw_html, "lxml")

    # A <p>/<li> nested inside a <table> cell is excluded here -- its text
    # is captured via the table's own block instead (see _table_rows_text),
    # so including it separately would both duplicate it and lose its row/
    # column position (surfacing "(4,750,000)" with no idea which test date
    # it belongs to).
    structured = [el for el in soup.find_all(_SECTION_TAGS) if el.find_parent("table") is None]
    orphan_divs = _orphan_leaf_divs(soup)
    # Outermost <table> elements only -- a nested <table> (rare, but not
    # unheard of in Word-exported HTML) is already part of its parent
    # table's own get_text()-based row/cell extraction.
    tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]

    if not structured and not tables:
        # No semantic structure at all -- fall back to leaf <div> paragraphs
        # (e.g. SEC EDGAR filings exported from Word).
        elements = orphan_divs
    else:
        # Hybrid document: interleave the div-only sections and tables back
        # into document order among the <p>/<li>/heading elements, rather
        # than dropping them (they aren't nested in each other, so neither
        # list alone reflects the real reading order).
        position = {id(tag): i for i, tag in enumerate(soup.find_all(True))}
        elements = sorted(list(structured) + orphan_divs + tables, key=lambda tag: position[id(tag)])

    blocks: List[Block] = []
    offset = 0
    current_heading: Optional[str] = None

    for element in elements:
        if element.name == "table":
            text = "\n".join(_table_rows_text(element))
            if not text:
                continue
            if blocks:
                # A table almost always completes/illustrates the prose
                # right before it (a covenant threshold schedule, pricing
                # grid, amortization table) -- merge into the immediately
                # preceding block rather than emitting it as its own.
                # chunk_blocks() only merges *short* neighboring blocks, so
                # a table long enough to clear that bar would otherwise
                # land in its own chunk with no covenant-name/keyword
                # context at all -- unfindable by keyword/embedding scoring,
                # and meaningless to an LLM shown it in isolation (this is
                # exactly what broke Flux Power's Minimum EBITDA schedule).
                prev = blocks[-1]
                merged_text = f"{prev.text}\n{text}"
                end = prev.start_char + len(merged_text)
                blocks[-1] = Block(
                    text=merged_text,
                    page=prev.page,
                    section_label=prev.section_label,
                    start_char=prev.start_char,
                    end_char=end,
                )
                offset = end + 1
            else:
                # No preceding block (table is the very first content) --
                # nothing to merge into, so it becomes its own block.
                start = offset
                end = start + len(text)
                blocks.append(Block(text=text, section_label=current_heading, start_char=start, end_char=end))
                offset = end + 1
            continue

        text = element.get_text(" ", strip=True)
        if not text:
            continue

        if element.name in _HEADING_TAGS:
            current_heading = text
        else:
            explicit_match = _EXPLICIT_SECTION_RE.match(text)
            if explicit_match:
                # More precise than the enclosing heading -- also becomes the
                # running label for subsequent unlabeled sub-clauses (e.g.
                # "(A) Fixed Charge Coverage Ratio...") until the next match.
                current_heading = explicit_match.group(1)
            else:
                article_match = _ARTICLE_HEADER_RE.fullmatch(text)
                if article_match:
                    current_heading = article_match.group(1)
        section_label = current_heading

        start = offset
        end = start + len(text)
        blocks.append(Block(text=text, section_label=section_label, start_char=start, end_char=end))
        offset = end + 1

    return blocks
