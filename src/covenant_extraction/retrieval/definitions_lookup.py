"""Deterministic lookup of defined terms (e.g. "Consolidated EBITDA") from a
contract's Definitions / Additional Definitions section, so each extracted
financial metric can be paired with its actual contractual definition
without an extra LLM call.

Real-world HTML exports vary in two ways this module has to tolerate:
- Quoting style: some documents use straight quotes ("Term" means ...),
  others use HTML curly-quote entities (&ldquo;/&rdquo;, i.e. Unicode
  U+201C/U+201D once BeautifulSoup decodes them).
- Term markup: a defined term wrapped in its own tag (e.g. <U>Term</U>)
  becomes a separate text node from the surrounding quote characters, so
  after `element.get_text(" ", strip=True)` there's a stray space between
  the quote and the term; and if the source line-wraps the term across the
  tag's text content, that wrap survives as a literal newline *inside* the
  term (e.g. "Consolidated Leverage\nRatio").
"""
from __future__ import annotations

import re
from typing import List, Optional

from covenant_extraction.ingestion.chunker import Chunk

_DEFINITIONS_HEADING_RE = re.compile(r"\b(additional\s+)?definitions\b", re.IGNORECASE)

_OPEN_QUOTES = '"“'  # straight " or curly left double quote
_CLOSE_QUOTES = '"”'  # straight " or curly right double quote
_ALL_QUOTES = _OPEN_QUOTES + _CLOSE_QUOTES

# The defining verb: most agreements use "means", but some (commonly CMBS
# /commercial-mortgage-style exports, e.g. a Wells Fargo sample document
# that uses it exclusively -- 141 occurrences of "shall mean", zero of bare
# "means") use "shall mean" instead. A document using only the latter
# previously yielded a completely empty definitions_text -- not a partial
# miss, every definition in the document was invisible.
_MEANS_VERB_RE = r"(?:means|shall\s+mean)"

# Matches the start of a defined-term declaration, tolerating: a space
# between the quote and the term (from tag-split text nodes); a missing
# closing quote (some HTML exports drop it -- the closing-quote requirement
# is relaxed to optional, so the up-to-80-char "term body" class just keeps
# absorbing text, including any qualifier like "for any period,", until it
# reaches "means"/"shall mean"); and a qualifying clause between the term
# and the defining verb (e.g. `"Fixed Charges for any period, means ..."`,
# vs. the more common `"Term" means, for any period, ...`) -- used both to
# detect a chunk that IS itself a definition (extract_definitions_text's
# content-based fallback) and to bound where a definition's text ends
# (find_definition's lookahead).
_DEFINITION_START_RE = rf"[{_OPEN_QUOTES}]\s*[A-Z][^{_ALL_QUOTES}]{{0,80}}[{_CLOSE_QUOTES}]?\s*{_MEANS_VERB_RE}\b"
_DEFINITION_START_ANCHORED_RE = re.compile(rf"^\s*{_DEFINITION_START_RE}", re.IGNORECASE)

# Optional short qualifying clause between a term and "means" -- e.g. "for
# any period,". Only used where the term itself is already known precisely
# (find_definition), since there the generic 80-char body-absorption trick
# above doesn't apply.
_MEANS_QUALIFIER_RE = rf"(?:for\s+[^,{_ALL_QUOTES}]{{1,40}},\s*)?"


def _chunk_heading(chunk: Chunk) -> str:
    return chunk.section_label or ""


def _looks_like_definition(text: str) -> bool:
    """True if `text` itself opens with a `"<Term>" means` declaration --
    a strong, content-based signal that this chunk is a definition, used as
    a fallback for documents where the definitions section's heading isn't
    captured in `section_label` (e.g. it's rendered as a plain, unstyled
    paragraph rather than a real heading tag)."""
    return bool(_DEFINITION_START_ANCHORED_RE.search(text))


def extract_definitions_text(chunks: List[Chunk]) -> str:
    """Concatenate the text of every chunk whose nearest section heading
    mentions "Definitions"/"Additional Definitions", plus any chunk that
    itself opens with a `"<Term>" means` declaration (covers documents whose
    definitions live under a heading that isn't tagged as such).

    Also pulls in every OTHER chunk sharing the same (non-empty) section
    label as one of those matches -- a single definition frequently spans
    multiple chunks (e.g. a lead-in sentence plus an itemized (a)/(b)/(c)...
    list, each its own HTML paragraph), and those continuation chunks won't
    individually look like a definition opener or carry the word
    "definitions" themselves. Without this, a long definition (e.g.
    "Consolidated EBITDA") would get silently truncated to just its first
    sentence.
    """
    target_labels = {
        _chunk_heading(c)
        for c in chunks
        if _chunk_heading(c) and (_DEFINITIONS_HEADING_RE.search(_chunk_heading(c)) or _looks_like_definition(c.text))
    }
    matching = [
        c.text
        for c in chunks
        if (_chunk_heading(c) and _chunk_heading(c) in target_labels) or _looks_like_definition(c.text)
    ]
    return "\n".join(matching)


def _pluralize_pattern(word: str) -> str:
    """Regex fragment matching `word` in either singular or simple plural
    form, whichever direction `word` itself happens to be given -- e.g.
    "Subsidiary" and "Subsidiaries" (consonant+y -> ies) or "Lender" and
    "Lenders" (plain trailing s) both match each other. Needed because a
    reference discovered inside one definition's prose (e.g.
    regex_discovery's candidate-phrase scan) is often the plural of the
    term as it's actually declared (or vice versa)."""
    if word.endswith("ies") and len(word) > 3 and word[-4].lower() not in "aeiou":
        stem = word[:-3]
        return rf"(?:{re.escape(word)}|{re.escape(stem)}y)"
    if word.endswith("y") and len(word) > 1 and word[-2].lower() not in "aeiou":
        stem = word[:-1]
        return rf"(?:{re.escape(word)}|{re.escape(stem)}ies)"
    if word.endswith("s") and len(word) > 1:
        return rf"(?:{re.escape(word)}|{re.escape(word[:-1])})"
    return rf"{re.escape(word)}s?"


def _term_pattern(term: str) -> str:
    """Build a regex fragment matching `term`'s words joined by `\\s+`
    (rather than the literal escaped string) so it still matches if the
    source text wraps the term across a newline. The last word additionally
    tolerates simple singular/plural variation (see `_pluralize_pattern`)."""
    words = term.split()
    if not words:
        return ""
    *head, last = words
    return r"\s+".join([re.escape(w) for w in head] + [_pluralize_pattern(last)])


def find_definition(term: str, definitions_text: str) -> Optional[str]:
    """Regex-search `definitions_text` for `"<term>" means/shall mean ...`,
    capturing up to (but not including) the next quoted defined term or end
    of text."""
    if not definitions_text:
        return None

    pattern = re.compile(
        rf"[{_OPEN_QUOTES}]\s*{_term_pattern(term)}\s*[{_CLOSE_QUOTES}]?\s*{_MEANS_QUALIFIER_RE}{_MEANS_VERB_RE}\s*,?\s*(.*?)"
        rf"(?=(?:{_DEFINITION_START_RE})|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(definitions_text)
    if not match:
        return None

    definition = match.group(1).strip()
    definition = re.sub(r"\s+", " ", definition)
    return definition or None
