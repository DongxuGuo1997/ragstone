import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Default RAG prompt, equivalent to "rlm/rag-prompt" from LangChain Hub.
# Bundled locally so Ollama-only setups work fully offline.
DEFAULT_RAG_PROMPT_TEMPLATE = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer:"
)


def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_question(inputs: Any) -> str:
    """Extract the question from supported chain input shapes.

    Accepts a plain string, a {"question": ...} dict, or a message.

    Raises:
        ValidationError: If the input is empty or of an unsupported type
            — failing here gives a clear error instead of an opaque
            failure deep inside the retriever.
    """
    question: Any
    if isinstance(inputs, str):
        question = inputs
    elif isinstance(inputs, dict) and "question" in inputs:
        question = inputs["question"]
    elif isinstance(inputs, BaseMessage):
        question = inputs.content
    else:
        raise ValidationError(
            "Chain input must be a question string, a {'question': ...} "
            f"dict, or a message; got {type(inputs).__name__}."
        )
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("Question must be a non-empty string.")
    return question


def parse_generated_queries(text: str) -> List[str]:
    """Split LLM-generated search queries into a clean list.

    Drops blank lines and strips list markers ("1.", "2)", "-", "*") that
    models often prepend. Blank entries would otherwise reach the retriever
    and crash embedding APIs that reject empty input.
    """
    queries = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s+)", "", line).strip()
        if line:
            queries.append(line)
    return queries


def _docs_from_fusion(ranked: List[Tuple[Document, float]]) -> List[Document]:
    """Extract the documents from reciprocal-rank-fusion (doc, score) pairs."""
    return [doc for doc, _score in ranked]


class RagProxy:
    """
    A proxy class for creating and managing RAG (Retrieval Augmented Generation) chains.
    """

    MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""

    FUSION_QUERY_TEMPLATE = """You are a helpful assistant that generates multiple search queries based on a single input query.
    Generate multiple search queries related to: {question}
    Output (4 queries):"""

    def __init__(
        self,
        model: BaseChatModel,
        retriever: VectorStoreRetriever,
        rag_prompt: Optional[BasePromptTemplate] = None,
    ):
        """
        Initializes the RagProxy.

        Args:
            model: The base chat model to use.
            retriever: The vector store retriever.
            rag_prompt: Optional custom RAG prompt. If None, a bundled default
                (equivalent to "rlm/rag-prompt") is used — no network required.
        """
        self._llm = model
        self._retriever = retriever
        if rag_prompt is None:
            self._rag_prompt: BasePromptTemplate = ChatPromptTemplate.from_template(
                DEFAULT_RAG_PROMPT_TEMPLATE
            )
            logger.info("Using bundled default RAG prompt.")
        else:
            self._rag_prompt = rag_prompt

    def get_retriever(self) -> VectorStoreRetriever:
        """Returns the base retriever."""
        return self._retriever

    def _get_question(self, inputs: Any) -> str:
        """Extract the question from supported input shapes.

        See :func:`extract_question` for the accepted shapes and errors.
        """
        return extract_question(inputs)

    def make_chain(self) -> Runnable:
        """
        Creates a basic RAG chain.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        rag_chain = (
            {
                "context": RunnableLambda(self._get_question)
                | self._retriever
                | format_docs,
                # Extract here too: a {"question": ...} dict input must not
                # be rendered verbatim into the prompt.
                "question": RunnableLambda(self._get_question),
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("Basic RAG chain created.")
        return rag_chain

    def make_multi_query_chain(self) -> Runnable:
        """
        Creates a RAG chain that uses multi-query retrieval.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        prompt_perspectives = ChatPromptTemplate.from_template(
            self.MULTI_QUERY_TEMPLATE
        )
        generate_queries_runnable = (
            prompt_perspectives
            | self._llm
            | StrOutputParser()
            | parse_generated_queries
        )

        # The input to retrieval_chain will be the original question,
        # which is then transformed by generate_queries_runnable.
        retrieval_chain: Runnable = (
            generate_queries_runnable | self._retriever.map() | self._get_unique_union
        )

        final_rag_chain = (
            {
                "context": RunnableLambda(self._get_question)
                | retrieval_chain
                | format_docs,  # Apply retrieval_chain to the extracted question
                "question": RunnableLambda(self._get_question),
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("Multi-query RAG chain created.")
        return final_rag_chain

    def make_fusion_chain(self) -> Runnable:
        """
        Creates a RAG chain that uses reciprocal rank fusion for retrieved documents.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        prompt_rag_fusion = ChatPromptTemplate.from_template(self.FUSION_QUERY_TEMPLATE)
        generate_queries_runnable = (
            prompt_rag_fusion | self._llm | StrOutputParser() | parse_generated_queries
        )

        retrieval_chain: Runnable = (
            generate_queries_runnable
            | self._retriever.map()
            | self._reciprocal_rank_fusion
            | RunnableLambda(_docs_from_fusion)
        )

        final_rag_chain = (
            {
                "context": RunnableLambda(self._get_question)
                | retrieval_chain
                | format_docs,
                "question": RunnableLambda(self._get_question),
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("RAG chain with Reciprocal Rank Fusion created.")
        return final_rag_chain

    def make_agent_chain(self) -> Runnable:
        """
        Creates an agentic RAG chain: the LLM drives retrieval via a search
        tool, deciding when and what to retrieve (and whether to search again
        with a refined query) instead of following the fixed retrieve-once
        pipeline. Costs extra LLM calls and latency; compare it against the
        fixed chains with the eval harness (see evals/run_eval.py).

        The input contract matches the other chains: a question string or a
        dict {"question": "..."}.
        """
        # Imported lazily: agent.py imports helpers from this module.
        from .agent import AgentRagChain

        chain = AgentRagChain(self._llm, self._retriever)
        logger.info("Agentic RAG chain created.")
        return chain

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, str]:
        """Hashable identity for a document (content + metadata)."""
        return doc.page_content, json.dumps(doc.metadata, sort_keys=True, default=str)

    def _get_unique_union(self, documents: List[List[Document]]) -> List[Document]:
        """
        Takes a list of lists of documents and returns a single list of unique documents.
        """
        unique: Dict[Tuple[str, str], Document] = {}
        for sublist in documents:
            for doc in sublist:
                unique.setdefault(self._doc_key(doc), doc)
        return list(unique.values())

    def _reciprocal_rank_fusion(
        self, results: List[List[Document]], k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Applies Reciprocal Rank Fusion to multiple lists of ranked documents.

        Args:
            results: A list of lists of Document objects, where each inner list is ranked.
            k: The constant k in the RRF formula, defaults to 60.

        Returns:
            A list of (Document, score) tuples, sorted by fused score in descending order.
        """
        fused_scores: Dict[Tuple[str, str], float] = {}
        doc_by_key: Dict[Tuple[str, str], Document] = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                key = self._doc_key(doc)
                doc_by_key.setdefault(key, doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (rank + k)

        return sorted(
            ((doc_by_key[key], score) for key, score in fused_scores.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
