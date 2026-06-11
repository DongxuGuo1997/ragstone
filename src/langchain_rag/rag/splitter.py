import logging
from typing import List, Optional, Union

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config.settings import get_config

logger = logging.getLogger(__name__)


def split_documents(
    docs: Union[List[Document], List[str]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Splits a list of Langchain Documents or a list of strings into smaller chunks.

    Args:
        docs: A list of Langchain Document objects or a list of strings to be split.
        chunk_size: Maximum chunk size. Defaults to the configured value.
        chunk_overlap: Overlap between chunks. Defaults to the configured value.

    Returns:
        A list of Langchain Document objects, where each document is a chunk
        of the original input. Returns an empty list if the input is None or empty.
    """
    if not docs:
        logger.warning("Received empty or None input for document splitting.")
        return []

    config = get_config()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else config.loader.chunk_size,
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else config.loader.chunk_overlap
        ),
        length_function=len,
        is_separator_regex=False,
    )

    if isinstance(docs[0], Document):
        # split_documents (rather than create_documents) preserves each
        # document's metadata — e.g. its source file — on every chunk,
        # which is what makes citations possible downstream.
        non_empty = [doc for doc in docs if doc.page_content]
        if not non_empty:
            logger.warning("No valid content found in the input documents to split.")
            return []
        split_texts_as_docs = text_splitter.split_documents(non_empty)
    elif isinstance(docs[0], str):
        contents_to_split = [text for text in docs if text]
        if not contents_to_split:
            logger.warning("No valid content found in the input documents to split.")
            return []
        split_texts_as_docs = text_splitter.create_documents(contents_to_split)
    else:
        logger.error(
            f"Unsupported document type: {type(docs[0])}. Expected List[Document] or List[str]."
        )
        return []

    n_chunks = len(split_texts_as_docs)
    logger.info(f"Successfully split content into {n_chunks} chunks.")

    return split_texts_as_docs
