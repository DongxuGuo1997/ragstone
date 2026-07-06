"""
Unit tests for corpus starter-question generation (no network required).
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.pipeline import OpenAIPipeline


class _CountingFakeChatModel(FakeListChatModel):
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)


class _FakeLLMProxy:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _pipeline_with_corpus(llm, texts=("Paris is the capital.", "Mars is red.")):
    pipeline = OpenAIPipeline(model="gpt-4o-mini")
    pipeline.LLM = _FakeLLMProxy(llm)
    pipeline.texts = [Document(page_content=t) for t in texts]
    return pipeline


class TestSuggestQuestions:
    def test_generates_and_parses_questions(self):
        llm = _CountingFakeChatModel(
            responses=["1. What is the capital?\n2) Why is Mars red?\n- How big?"]
        )
        pipeline = _pipeline_with_corpus(llm)

        questions = pipeline.suggest_questions(n=3)

        # List markers stripped, one question per line.
        assert questions == [
            "What is the capital?",
            "Why is Mars red?",
            "How big?",
        ]

    def test_cached_per_corpus_one_llm_call(self):
        llm = _CountingFakeChatModel(responses=["Q1\nQ2\nQ3"])
        pipeline = _pipeline_with_corpus(llm)

        first = pipeline.suggest_questions()
        second = pipeline.suggest_questions()  # UI reruns call this freely

        assert first == second
        assert llm.calls == 1  # the cache absorbed the second call

    def test_new_corpus_regenerates(self):
        llm = _CountingFakeChatModel(responses=["Q1\nQ2\nQ3", "R1\nR2\nR3"])
        pipeline = _pipeline_with_corpus(llm)

        pipeline.suggest_questions()
        pipeline.texts = [Document(page_content="Entirely new content.")]
        regenerated = pipeline.suggest_questions()

        assert regenerated == ["R1", "R2", "R3"]
        assert llm.calls == 2

    def test_no_corpus_or_llm_returns_empty(self):
        llm = _CountingFakeChatModel(responses=["unused"])
        pipeline = _pipeline_with_corpus(llm, texts=())
        pipeline.texts = None
        assert pipeline.suggest_questions() == []
        assert llm.calls == 0

        pipeline_no_llm = OpenAIPipeline(model="gpt-4o-mini")
        pipeline_no_llm.LLM = None
        pipeline_no_llm.texts = [Document(page_content="doc")]
        assert pipeline_no_llm.suggest_questions() == []

    def test_llm_failure_degrades_to_empty_list(self):
        # A generation failure (network/provider down) must degrade to an
        # empty list — suggestions are a convenience, never a crash.
        class _FailingFakeModel(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("provider down")

        pipeline = _pipeline_with_corpus(_FailingFakeModel(responses=["unused"]))

        assert pipeline.suggest_questions() == []
