import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain import hub
from langchain.chains.base import Chain
from langchain.load import dumps, loads
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, BasePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Import utilities from centralized location
from ..utils import safe_execute

# Local exception definitions
class RagError(Exception):
    """Base class for RAG errors."""
    pass

def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


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

    def __init__(self, model: BaseChatModel, retriever: VectorStoreRetriever, rag_prompt: Optional[BasePromptTemplate] = None):
        """
        Initializes the RagProxy.

        Args:
            model: The base chat model to use.
            retriever: The vector store retriever.
            rag_prompt: Optional custom RAG prompt. If None, "rlm/rag-prompt" from Langchain Hub is used.
        """
        self._llm = model
        self._retriever = retriever
        if rag_prompt is None:
            self._rag_prompt = hub.pull("rlm/rag-prompt")
            logger.info("Pulled default RAG prompt 'rlm/rag-prompt' from Langchain Hub.")
        else:
            self._rag_prompt = rag_prompt
        self._history_aware_retriever: Optional[VectorStoreRetriever] = None

    def get_retriever(self) -> VectorStoreRetriever:
        """Returns the base retriever."""
        return self._retriever

    def set_history_aware_retriever(self, retriever: VectorStoreRetriever) -> None:
        """Sets a history-aware retriever (currently not used in default chain methods)."""
        self._history_aware_retriever = retriever
        logger.info("History-aware retriever has been set.")

    def _get_question(self, inputs: Any) -> Optional[str]:
        """Extracts the question from various input types."""
        if not inputs:
            return None
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, dict) and "question" in inputs:
            return inputs["question"]
        if isinstance(inputs, BaseMessage): # Handle chat messages if passed directly
            return inputs.content
        logger.warning(f"Unexpected input type for question extraction: {type(inputs)}. Returning as is or None.")
        # Fallback or raise error depending on strictness needed
        return str(inputs) if inputs else None


    def make_chain(self) -> RunnableSequence:
        """
        Creates a basic RAG chain.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        rag_chain = (
            {
                "context": RunnableLambda(self._get_question) | self._retriever | format_docs,
                "question": RunnablePassthrough(), # Passes the original input (question) through
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("Basic RAG chain created.")
        return rag_chain

    def make_multi_query_chain(self) -> RunnableSequence:
        """
        Creates a RAG chain that uses multi-query retrieval.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        prompt_perspectives = ChatPromptTemplate.from_template(self.MULTI_QUERY_TEMPLATE)
        generate_queries_runnable = (
            prompt_perspectives
            | self._llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        
        # The input to retrieval_chain will be the original question,
        # which is then transformed by generate_queries_runnable.
        retrieval_chain = generate_queries_runnable | self._retriever.map() | self._get_unique_union
        
        final_rag_chain = (
            {
                "context": RunnableLambda(self._get_question) | retrieval_chain | format_docs, # Apply retrieval_chain to the extracted question
                "question": RunnablePassthrough(), # Pass the original question through
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("Multi-query RAG chain created.")
        return final_rag_chain

    def make_fusion_chain(self) -> RunnableSequence:
        """
        Creates a RAG chain that uses reciprocal rank fusion for retrieved documents.
        The input to this chain is expected to be the question string or a dict {"question": "..."}.
        """
        prompt_rag_fusion = ChatPromptTemplate.from_template(self.FUSION_QUERY_TEMPLATE)
        generate_queries_runnable = (
            prompt_rag_fusion
            | self._llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain = generate_queries_runnable | self._retriever.map() | self._reciprocal_rank_fusion | RunnableLambda(lambda reranked_results: [doc for doc, score in reranked_results])


        final_rag_chain = (
            {
                "context": RunnableLambda(self._get_question) | retrieval_chain | format_docs,
                "question": RunnablePassthrough(),
            }
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
        logger.info("RAG chain with Reciprocal Rank Fusion created.")
        return final_rag_chain

    def _get_unique_union(self, documents: List[List[Document]]) -> List[Document]:
        """
        Takes a list of lists of documents and returns a single list of unique documents.
        """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs_str = list(set(flattened_docs))
        return [loads(doc_str) for doc_str in unique_docs_str]

    def _reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
        """
        Applies Reciprocal Rank Fusion to multiple lists of ranked documents.

        Args:
            results: A list of lists of Document objects, where each inner list is ranked.
            k: The constant k in the RRF formula, defaults to 60.

        Returns:
            A list of (Document, score) tuples, sorted by fused score in descending order.
        """
        fused_scores: Dict[str, float] = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc) # Serialize document to use as a dictionary key
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0.0
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked_results_tuples: List[Tuple[str, float]] = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Deserialize documents
        final_reranked_results: List[Tuple[Document, float]] = [
            (loads(doc_str), score) for doc_str, score in reranked_results_tuples
        ]
        return final_reranked_results


def main():
    # Note: load_dotenv() is already called in config/settings.py
    pass


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()