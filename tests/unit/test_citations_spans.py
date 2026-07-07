"""
Unit tests for evidence highlighting (pure functions, no network).

The invariants: spans index the ORIGINAL snippet exactly, matching is
case/punctuation-tolerant at the word level, short overlaps never count
as evidence, and HTML escaping composes with highlighting without
corrupting offsets.
"""

import html

from ragstone.rag.citations import find_supporting_spans, highlight_spans

SNIPPET = (
    "The Helios MK-3 carries a 28-year linear power output warranty, "
    "which is voided by pressure washing."
)


class TestSpanFinding:
    """Alignment: verbatim word runs light up; paraphrase does not."""

    def test_verbatim_reuse_is_found_with_exact_offsets(self):
        answer = "It carries a 28-year linear power output warranty."
        [(start, end)] = find_supporting_spans(answer, SNIPPET)
        # Spans cover whole source tokens, trailing punctuation included.
        assert SNIPPET[start:end] == "carries a 28-year linear power output warranty,"

    def test_case_and_punctuation_drift_still_match(self):
        answer = "...carries a 28-YEAR linear power output warranty!"
        [(start, end)] = find_supporting_spans(answer, SNIPPET)
        assert "28-year linear power output" in SNIPPET[start:end]

    def test_short_overlaps_are_not_evidence(self):
        # "the warranty is" style fragments are English, not provenance.
        assert find_supporting_spans("The warranty is long.", SNIPPET) == []

    def test_paraphrase_does_not_highlight(self):
        answer = "Its guarantee lasts nearly three decades."
        assert find_supporting_spans(answer, SNIPPET) == []

    def test_multiple_disjoint_matches(self):
        answer = (
            "The Helios MK-3 carries a warranty which is voided by " "pressure washing."
        )
        spans = find_supporting_spans(answer, SNIPPET, min_words=4)
        assert len(spans) == 2
        assert SNIPPET[spans[0][0] : spans[0][1]].startswith("The Helios MK-3")
        assert SNIPPET[spans[1][0] : spans[1][1]].endswith("pressure washing.")

    def test_empty_inputs_are_safe(self):
        assert find_supporting_spans("", SNIPPET) == []
        assert find_supporting_spans("anything", "") == []


class TestHighlighting:
    """Rendering: markers wrap spans; escaping never shifts offsets."""

    def test_wraps_spans_with_markers(self):
        text = "warranty is 28 years"
        out = highlight_spans(text, [(12, 20)], "<<", ">>")
        assert out == "warranty is <<28 years>>"

    def test_html_escape_composes_without_corrupting_offsets(self):
        text = "a < b & the 28-year warranty holds"
        spans = [(12, 28)]  # "28-year warranty" in the ORIGINAL text
        out = highlight_spans(text, spans, "<mark>", "</mark>", escape=html.escape)
        assert out == "a &lt; b &amp; the <mark>28-year warranty</mark> holds"

    def test_no_spans_is_plain_escaped_text(self):
        assert highlight_spans("a < b", [], "<m>", "</m>", escape=html.escape) == (
            "a &lt; b"
        )
