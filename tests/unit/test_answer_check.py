"""
Unit tests for the answer self-check pass (no network).

The contract: supported answers ship untouched, unsupported claims get a
VISIBLE caveat (never a silent rewrite), refusals and empty contexts are
skipped, and every checker failure fails safe — the answer always ships.
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.config.settings import get_config
from ragstone.rag.answer_check import CAVEAT_HEADER, check_answer

DOCS = [Document(page_content="The Helios MK-3 warranty is 28 years.")]


class _BoomModel(FakeListChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("checker down")


class TestCheckAnswer:
    """The pure check: verdict parsing and fail-safe behavior."""

    def test_supported_answer_gets_no_caveat(self):
        llm = FakeListChatModel(responses=["supported"])
        assert check_answer("The warranty is 28 years.", DOCS, llm) is None

    def test_unsupported_claims_become_a_visible_caveat(self):
        llm = FakeListChatModel(
            responses=["- the panel costs $400\n- installation takes two days"]
        )
        caveat = check_answer("It costs $400 and installs in two days.", DOCS, llm)
        assert caveat.startswith(CAVEAT_HEADER)
        assert "- the panel costs $400" in caveat
        assert "- installation takes two days" in caveat

    def test_refusals_are_not_checked(self):
        llm = _BoomModel(responses=["unused"])  # would raise if invoked
        assert check_answer("I don't know based on the documents.", DOCS, llm) is None

    def test_empty_context_is_skipped(self):
        llm = _BoomModel(responses=["unused"])
        assert check_answer("Some answer.", [], llm) is None

    def test_checker_exception_fails_safe(self):
        assert check_answer("Some answer.", DOCS, _BoomModel(responses=["x"])) is None

    def test_unparseable_verdict_fails_safe(self):
        llm = FakeListChatModel(responses=["hmm, it's complicated"])
        assert check_answer("Some answer.", DOCS, llm) is None


class TestPipelineIntegration:
    """The wiring: caveat appended, streamed, cached; off by default."""

    def _pipeline(self, monkeypatch, verdict, enabled=True):
        from ragstone.rag.pipeline import OpenAIPipeline

        monkeypatch.setattr(get_config().llm, "answer_check_enabled", enabled)
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        pipeline._chain_type = "simple"

        checker = FakeListChatModel(responses=[verdict])
        monkeypatch.setattr(
            "ragstone.rag.memory._make_rephrase_llm", lambda llm, name: checker
        )

        class _Chain:
            def ask_question(self, query, session_id):
                return "The panel costs $400."

            def stream_question(self, query, session_id):
                yield "The panel "
                yield "costs $400."

            def has_history(self, session_id):
                return False

        pipeline._chain = _Chain()
        pipeline._retriever = None  # get_last_retrieved_documents fallback
        monkeypatch.setattr(pipeline, "get_last_retrieved_documents", lambda: DOCS)
        return pipeline

    def test_caveat_is_appended_to_the_answer(self, monkeypatch):
        pipeline = self._pipeline(monkeypatch, "- the panel costs $400")
        answer = pipeline.ask_question("price?")
        assert answer.startswith("The panel costs $400.")
        assert CAVEAT_HEADER.strip() in answer

    def test_streaming_yields_the_caveat_as_a_final_chunk(self, monkeypatch):
        pipeline = self._pipeline(monkeypatch, "- the panel costs $400")
        chunks = list(pipeline.ask_question_stream("price?"))
        assert chunks[-1].startswith(CAVEAT_HEADER)

    def test_supported_answers_are_untouched(self, monkeypatch):
        pipeline = self._pipeline(monkeypatch, "supported")
        assert pipeline.ask_question("price?") == "The panel costs $400."

    def test_disabled_by_default_means_no_checker_call(self, monkeypatch):
        pipeline = self._pipeline(monkeypatch, "unused", enabled=False)
        assert pipeline.ask_question("price?") == "The panel costs $400."
