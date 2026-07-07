"""Corrective RAG: retrieval that grades itself and retries.

The fixed chains trust the first retrieval unconditionally. This chain
adds a self-correction loop as a LangGraph with conditional edges and a
bounded cycle:

    retrieve -> grade --(relevant)---------------------> answer
                  |(irrelevant, attempts left)
                  v
               rewrite -> retrieve   (cycle, bounded)
                  |(attempts exhausted)
                  v
               refuse  ("I don't know" — by evidence, not vibes)

Where it should pay: questions whose phrasing misses the corpus wording
(the rewrite recovers), and unanswerable questions (the refusal is
grounded in N failed searches rather than the model's mood). The cost is
one cheap grading call per answer and a rewrite+retrieval per retry — the
eval harness decides whether that trade is worth it (EXPERIMENTS.md).

Grading and rewriting run on the utility model (RAGSTONE_REPHRASE_MODEL
when configured, else the main model): both are short classification /
transformation tasks that do not need the answer model's quality.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..config.settings import get_config
from .memory import _make_rephrase_llm
from .rag import DEFAULT_RAG_PROMPT_TEMPLATE, extract_question, format_docs

logger = logging.getLogger(__name__)

# Initial retrieval + up to this many rewrite->retrieve cycles. Two is the
# knee: the first rewrite fixes phrasing mismatches; further attempts
# mostly re-find the same chunks while tripling latency.
MAX_CORRECTIVE_RETRIES = 2

GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading retrieved passages for a question-answering system.\n"
    "Question: {question}\n\n"
    "Passages:\n{context}\n\n"
    "Do the passages contain the information needed to answer the "
    "question? Reply with exactly one word: yes or no."
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "A document search for the question below returned irrelevant "
    "passages. Rewrite the question as a better search query: use "
    "alternative phrasing and the domain terms a document would use. "
    "Return ONLY the rewritten query.\n\n"
    "Question: {question}\n"
    "Previously tried queries: {tried}"
)


class CorrectiveState(TypedDict, total=False):
    """State for one corrective-RAG run (not checkpointed — the outer
    memory graph owns conversation state)."""

    question: str  # what the user asked (never mutated)
    query: str  # what we retrieve with (rewritten on retries)
    tried_queries: List[str]
    docs: List[Document]
    relevant: bool
    attempts: int  # rewrite cycles consumed
    answer: str


class CorrectiveRagChain(Runnable[Any, str]):
    """Self-correcting RAG conforming to the base-chain contract.

    invoke() returns the answer string; stream() yields answer tokens;
    stream_with_events() additionally yields progress dicts
    ({"event": "grade"|"rewrite"|...}) so UIs can show the correction
    loop working — the same event contract the agent chain uses.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        utility_llm: Optional[BaseChatModel] = None,
        max_retries: int = MAX_CORRECTIVE_RETRIES,
    ) -> None:
        """
        Args:
            llm: The answer model.
            retriever: Document retriever.
            utility_llm: Model for grading and query rewriting. Defaults
                to the configured rephrase model, falling back to `llm`.
            max_retries: Maximum rewrite->retrieve cycles.
        """
        if utility_llm is None:
            utility_llm = _make_rephrase_llm(llm, get_config().llm.rephrase_model)
        self._max_retries = max_retries
        grade_chain = GRADE_PROMPT | utility_llm | StrOutputParser()
        rewrite_chain = REWRITE_PROMPT | utility_llm | StrOutputParser()
        answer_prompt = ChatPromptTemplate.from_template(DEFAULT_RAG_PROMPT_TEMPLATE)
        answer_chain = answer_prompt | llm | StrOutputParser()

        def retrieve(state: CorrectiveState) -> CorrectiveState:
            query = state["query"]
            get_stream_writer()({"event": "retrieve", "query": query})
            docs = retriever.invoke(query)
            tried = state.get("tried_queries", []) + [query]
            return {"docs": docs, "tried_queries": tried}

        def grade(state: CorrectiveState) -> CorrectiveState:
            verdict = grade_chain.invoke(
                {
                    "question": state["question"],
                    "context": format_docs(state["docs"]) or "(nothing retrieved)",
                }
            )
            relevant = verdict.strip().lower().startswith("yes")
            get_stream_writer()(
                {"event": "grade", "relevant": relevant, "query": state["query"]}
            )
            return {"relevant": relevant}

        def rewrite(state: CorrectiveState) -> CorrectiveState:
            new_query = rewrite_chain.invoke(
                {
                    "question": state["question"],
                    "tried": "; ".join(state.get("tried_queries", [])),
                }
            ).strip()
            get_stream_writer()({"event": "rewrite", "query": new_query})
            return {"query": new_query, "attempts": state.get("attempts", 0) + 1}

        def answer(state: CorrectiveState) -> CorrectiveState:
            writer = get_stream_writer()
            parts: List[str] = []
            for chunk in answer_chain.stream(
                {
                    "question": state["question"],
                    "context": format_docs(state["docs"]),
                }
            ):
                if chunk:
                    writer(chunk)
                    parts.append(chunk)
            return {"answer": "".join(parts)}

        def refuse(state: CorrectiveState) -> CorrectiveState:
            # Refusal grounded in evidence: N searches, none graded
            # relevant. No answer-model call is spent on it.
            tried = ", ".join(f'"{q}"' for q in state.get("tried_queries", []))
            text = (
                "I couldn't find this in the documents. "
                f"I searched for: {tried} — none of the results contained "
                "the answer."
            )
            get_stream_writer()(text)
            return {"answer": text}

        def route_after_grade(state: CorrectiveState) -> str:
            if state.get("relevant"):
                return "answer"
            if state.get("attempts", 0) < self._max_retries:
                return "rewrite"
            return "refuse"

        graph = StateGraph(CorrectiveState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("grade", grade)
        graph.add_node("rewrite", rewrite)
        graph.add_node("answer", answer)
        graph.add_node("refuse", refuse)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            route_after_grade,
            {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"},
        )
        graph.add_edge("rewrite", "retrieve")  # the correction cycle
        graph.add_edge("answer", END)
        graph.add_edge("refuse", END)
        self._graph = graph.compile()

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Run the correction loop to completion and return the answer."""
        question = extract_question(input)
        result = self._graph.invoke(
            {"question": question, "query": question, "attempts": 0}, config=config
        )
        return result["answer"]

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Iterator[str]:
        """Yield answer text only (Runnable[..., str] contract)."""
        for item in self.stream_with_events(input, config=config):
            if isinstance(item, str):
                yield item

    def stream_with_events(
        self, input: Any, config: Optional[RunnableConfig] = None
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Yield answer tokens interleaved with correction-loop events."""
        question = extract_question(input)
        for chunk in self._graph.stream(
            {"question": question, "query": question, "attempts": 0},
            config=config,
            stream_mode="custom",
        ):
            if isinstance(chunk, (str, dict)):
                yield chunk
