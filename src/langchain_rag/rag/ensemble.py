import logging
from typing import List, Optional
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from .vector_db import FaissProxy
from .splitter import split_documents

logger = logging.getLogger(__name__)

def ensemble_retriever_from_docs(
    docs: List[Document], 
    weights: Optional[List[float]] = None
) -> Optional[BaseRetriever]: 
    """
    Create an ensemble retriever from a list of documents.

    Args:
        docs (List[Document]): List of Langchain Document objects to create the retriever from.
        weights (Optional[List[float]]): A list of two floats representing the weights
                                         for BM25 and vector store retrievers respectively.
                                         Defaults to [0.5, 0.5].

    Returns:
        Optional[BaseRetriever]: An ensemble retriever combining BM25 and FAISS vector
                                 space retrievers, or None if creation fails or inputs are invalid.
                                 Can also return just the FAISS retriever if BM25 fails or texts are empty.
    """
    if not docs:
        logger.warning("No documents provided to create ensemble retriever.")
        return None

    if weights is None:
        weights = [0.5, 0.5]
    if len(weights) != 2:
        logger.warning("Weights list must contain two floats. Using default [0.5, 0.5].")
        weights = [0.5, 0.5]

    logger.info(f"Starting ensemble retriever creation with {len(docs)} initial documents.")

    texts = split_documents(docs)
    if not texts:
        logger.warning("Document splitting resulted in no text chunks. Cannot create retriever.")
        return None
    
    logger.info(f"Documents split into {len(texts)} chunks.")

    # Initialize and create FAISS vector store
    faiss_proxy = FaissProxy()
    try:
        # FaissProxy.create_db uses standard LangChain OpenAI embeddings by default if no embeddings are passed.
        faiss_proxy.create_db(docs=texts) 
    except Exception as e:
        logger.error(f"Failed to create FAISS database: {e}", exc_info=True)
        return None

    vector_store = faiss_proxy.db
    if not vector_store:
        logger.error("FAISS vector store was not created successfully.")
        return None
    
    vector_store_retriever = vector_store.as_retriever()
    logger.info("FAISS vector store retriever created.")

    # Prepare texts for BM25Retriever
    page_contents = [t.page_content for t in texts if t.page_content]
    if not page_contents:
        logger.warning("No page content found in split texts for BM25Retriever. Ensemble will only use FAISS retriever.")
        # Optionally, could return just vector_store_retriever here if BM25 is critical for the ensemble
        # For now, we'll proceed and the ensemble will effectively have one retriever if BM25_retriever is None
        # However, EnsembleRetriever expects a list of retrievers.
        # A better approach is to return only the vector_store_retriever if BM25 cannot be formed.
        return vector_store_retriever


    try:
        bm25_retriever = BM25Retriever.from_texts(page_contents)
        logger.info("BM25 retriever created.")
    except Exception as e:
        logger.error(f"Failed to create BM25Retriever: {e}. Returning only FAISS retriever.", exc_info=True)
        return vector_store_retriever


    # Create Ensemble Retriever
    try:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_store_retriever],
            weights=weights
        )
        logger.info(f"Ensemble retriever created successfully with weights: {weights}.")
        return ensemble_retriever
    except Exception as e:
        logger.error(f"Failed to create EnsembleRetriever: {e}. Defaulting to FAISS retriever.", exc_info=True)
        # Fallback to the vector store retriever if ensemble creation fails for some reason
        return vector_store_retriever