"""
Unit tests for the optional reranker (no model download required).
"""

import pytest

from ragstone.rag.memory import SimpleTextRetriever
from ragstone.rag.reranker import _INSTALL_HINT, wrap_with_reranker


def test_missing_extra_raises_helpful_error(monkeypatch):
    """Without sentence-transformers, the wrapper must fail with install help."""
    import builtins

    from ragstone.rag import reranker

    # A previously cached encoder would bypass construction entirely; clear
    # it so the simulated missing import is actually exercised.
    monkeypatch.setattr(reranker, "_encoder_cache", {})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    retriever = SimpleTextRetriever.from_texts(["some content"])
    with pytest.raises(ImportError, match="rerank"):
        wrap_with_reranker(retriever)


def test_wrap_with_reranker_reorders_results():
    """With the extra installed, the wrapper returns a working retriever."""
    pytest.importorskip("sentence_transformers")

    retriever = SimpleTextRetriever.from_texts(
        [
            "The annual gala dinner is held in October.",
            "Bolts must be torqued to 14 newton-meters.",
            "Coffee beans rest for nine days after roasting.",
        ]
    )
    reranked = wrap_with_reranker(retriever, top_k=1)
    docs = reranked.invoke("What torque do the bolts need?")
    assert len(docs) == 1
    assert "14 newton-meters" in docs[0].page_content


def test_install_hint_mentions_pip_extra():
    assert "rerank" in _INSTALL_HINT and "pip install" in _INSTALL_HINT
