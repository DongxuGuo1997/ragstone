"""Evidence highlighting: align an answer with its source snippets.

Citations v1 (ROADMAP 6.1) done the zero-risk way: instead of prompting
the model to emit citation markers (a generation change that would need
its own eval gate), the alignment is computed AFTER the fact — maximal
word-sequence matches between the answer and each retrieved snippet,
mapped back to exact character offsets in the snippet. The UIs render
those offsets as highlights: the reader sees precisely which source
words the answer is built from.

Deterministic, dependency-free (difflib), and tolerant of case and
punctuation drift between the source text and the model's phrasing.
Paraphrases don't highlight — by design: a highlight is a verbatim-level
claim, and showing fewer, true highlights beats showing fuzzy ones.
"""

import re
from difflib import SequenceMatcher
from typing import List, Tuple

# A match must span at least this many words to count as evidence —
# shorter overlaps ("of the", "it is a") are English, not provenance.
MIN_MATCH_WORDS = 4

_TOKEN = re.compile(r"\S+")
_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def _tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Normalized word tokens plus each token's (start, end) char span."""
    words: List[str] = []
    spans: List[Tuple[int, int]] = []
    for match in _TOKEN.finditer(text):
        normalized = _STRIP.sub("", match.group()).lower()
        if normalized:
            words.append(normalized)
            spans.append((match.start(), match.end()))
    return words, spans


def find_supporting_spans(
    answer: str, snippet: str, min_words: int = MIN_MATCH_WORDS
) -> List[Tuple[int, int]]:
    """Character spans in `snippet` whose wording the answer reuses.

    Matching is word-level (case- and punctuation-insensitive) via
    difflib's maximal matching blocks; only runs of at least `min_words`
    words qualify. Returned spans index into the ORIGINAL snippet and
    never overlap; adjacent matches separated by whitespace are merged.
    """
    answer_words, _ = _tokenize(answer)
    snippet_words, snippet_spans = _tokenize(snippet)
    if not answer_words or not snippet_words:
        return []

    matcher = SequenceMatcher(a=answer_words, b=snippet_words, autojunk=False)
    spans: List[Tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        if block.size < min_words:
            continue
        start = snippet_spans[block.b][0]
        end = snippet_spans[block.b + block.size - 1][1]
        spans.append((start, end))

    # Merge touching/overlapping spans (blocks can abut across gaps).
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def highlight_spans(
    text: str,
    spans: List[Tuple[int, int]],
    before: str,
    after: str,
    escape=None,
) -> str:
    """Wrap each span of `text` in before/after markers.

    `escape` (e.g. html.escape) is applied to every segment — including
    the unhighlighted ones — so callers can render into HTML safely
    while the offsets keep referring to the ORIGINAL text.
    """
    escape = escape or (lambda segment: segment)
    pieces: List[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(escape(text[cursor:start]))
        pieces.append(before + escape(text[start:end]) + after)
        cursor = end
    pieces.append(escape(text[cursor:]))
    return "".join(pieces)
