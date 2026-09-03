"""
Unit tests for the dedicated staffing-match UI's logic helpers.

The Streamlit render path is exercised by the headless boot smoke (the
app must start and serve /_stcore/health); here we lock the pure logic:
bundled-brief loading, evidence highlighting (citations reuse, HTML
escaping, span merging), and the progress-event formatter.
"""

from pathlib import Path

from ragstone.ui.staffing_app import (
    EXAMPLES_PATH,
    MARK_OPEN,
    _event_line,
    highlight_evidence,
    load_example_briefs,
)


class TestExampleBriefs:
    def test_bundled_golden_briefs_load(self):
        briefs = load_example_briefs()
        # One entry per committed golden record; keys are "aNN: title".
        # a09 is the Swedish cross-lingual brief, a10/a11 the long-form
        # RFQ-style ones.
        golden = EXAMPLES_PATH.read_text(encoding="utf-8").splitlines()
        assert len(briefs) == sum(1 for line in golden if line.strip())
        assert any(key.startswith("a01:") for key in briefs)
        assert any(key.startswith("a09:") for key in briefs)
        assert any(key.startswith("a10:") for key in briefs)
        assert all(brief.strip() for brief in briefs.values())

    def test_missing_file_yields_empty_dict(self):
        assert load_example_briefs(Path("/nonexistent/golden.jsonl")) == {}


class TestHighlightEvidence:
    CV = (
        "## Profile\n"
        "Astrid has deep experience with AUTOSAR Classic platforms for "
        "truck ECUs and ensures compliance with ISO 26262 standards.\n"
        "Unrelated line about <html> escaping."
    )

    def test_quote_is_highlighted(self):
        quote = "deep experience with AUTOSAR Classic platforms for truck"
        out = highlight_evidence(self.CV, [quote])
        assert MARK_OPEN in out
        assert "AUTOSAR Classic" in out

    def test_no_quotes_returns_escaped_text(self):
        out = highlight_evidence(self.CV, [])
        assert MARK_OPEN not in out
        assert "&lt;html&gt;" in out  # HTML is escaped, not rendered

    def test_html_is_escaped_even_with_highlights(self):
        quote = "deep experience with AUTOSAR Classic platforms for truck"
        out = highlight_evidence(self.CV, [quote])
        assert "&lt;html&gt;" in out

    def test_overlapping_quotes_merge_without_nesting(self):
        q1 = "deep experience with AUTOSAR Classic platforms"
        q2 = "experience with AUTOSAR Classic platforms for truck ECUs"
        out = highlight_evidence(self.CV, [q1, q2])
        assert out.count(MARK_OPEN) == 1  # merged into one span

    def test_empty_and_short_quotes_are_harmless(self):
        out = highlight_evidence(self.CV, ["", "too short"])
        assert MARK_OPEN not in out


class TestPersistUploads:
    class _Upload:
        def __init__(self, name, data):
            self.name = name
            self._data = data

        def getvalue(self):
            return self._data

    def test_content_addressed_and_written(self):
        from ragstone.ui.staffing_app import persist_uploads

        files = [
            self._Upload("Maria_Larsson_CV.txt", b"cv text"),
            self._Upload("john.md", b"other"),
        ]
        first = Path(persist_uploads(files))
        assert (first / "Maria_Larsson_CV.txt").read_bytes() == b"cv text"
        # Same content -> same directory (cache key stability)...
        assert Path(persist_uploads(files)) == first
        # ...different content -> different directory.
        changed = [self._Upload("Maria_Larsson_CV.txt", b"edited")]
        assert Path(persist_uploads(changed)) != first


class TestEventLines:
    def test_known_events_render(self):
        assert "Requirements" in _event_line({"event": "extract", "must": ["A"]})
        assert "Searching" in _event_line({"event": "discover", "query": "X"})
        assert "Verifying 2" in _event_line(
            {"event": "shortlist", "candidates": ["a", "b"]}
        )
        assert "Screening" in _event_line({"event": "verify", "candidate": "N"})
        assert "Ranking" in _event_line({"event": "result"})

    def test_unknown_event_is_skipped(self):
        assert _event_line({"event": "mystery"}) is None
