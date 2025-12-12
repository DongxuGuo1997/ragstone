import logging
from typing import List, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

def split_documents(docs: Union[List[Document], List[str]]) -> List[Document]:
    """
    Splits a list of Langchain Documents or a list of strings into smaller chunks.

    Args:
        docs: A list of Langchain Document objects or a list of strings to be split.

    Returns:
        A list of Langchain Document objects, where each document is a chunk
        of the original input. Returns an empty list if the input is None or empty.
    """
    if not docs:
        logger.warning("Received empty or None input for document splitting.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50, 
        length_function=len,
        is_separator_regex=False
    )

    # Prepare contents for splitting
    if isinstance(docs[0], Document):
        contents_to_split = [doc.page_content for doc in docs if doc.page_content]
    elif isinstance(docs[0], str):
        contents_to_split = [text for text in docs if text]
    else:
        logger.error(f"Unsupported document type: {type(docs[0])}. Expected List[Document] or List[str].")
        return []
    
    if not contents_to_split:
        logger.warning("No valid content found in the input documents to split.")
        return []
    # Split the contents into smaller chunks
    split_texts_as_docs = text_splitter.create_documents(contents_to_split)
    
    n_chunks = len(split_texts_as_docs)
    logger.info(f"Successfully split content into {n_chunks} chunks.")

    return split_texts_as_docs
