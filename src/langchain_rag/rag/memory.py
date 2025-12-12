from typing import List, Iterable, Any, Dict, Union

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.base import Chain


class MemoryProxy:
    """
    A proxy class for managing memory in conversation chains.
    
    This class provides functionality to create memory-enabled chains that can
    maintain conversation context across multiple interactions.
    """
    
    def __init__(self, type: str = "InMemory") -> None:
        """
        Initialize the MemoryProxy.
        
        Args:
            type: The type of memory to use. Defaults to "InMemory".
        """
        if not isinstance(type, str):
            raise TypeError("Memory type must be a string")
        self._type = type
    
    def create_memory_chain(
        self, 
        llm: BaseChatModel, 
        base_chain: Union[Chain, Runnable]
    ) -> RunnableWithMessageHistory:
        """
        Create a memory-enabled chain that can maintain conversation history.
        
        Args:
            llm: The language model to use for contextualizing questions.
            base_chain: The base chain to wrap with memory functionality.
                        Can be either a Chain or Runnable (including RunnableSequence).
            
        Returns:
            A RunnableWithMessageHistory that includes conversation memory.
            
        Raises:
            TypeError: If llm or base_chain are not of the expected types.
        """
        if not isinstance(llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel")
        
        # Accept both Chain and Runnable objects (including RunnableSequence)
        if not isinstance(base_chain, (Chain, Runnable)):
            raise TypeError("base_chain must be an instance of Chain or Runnable")
            
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        runnable = contextualize_q_prompt | llm | base_chain

        # Store for session histories
        session_store: Dict[str, BaseChatMessageHistory] = {}
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            """
            Retrieve or create a chat message history for the given session.
            
            Args:
                session_id: Unique identifier for the session.
                
            Returns:
                BaseChatMessageHistory for the session.
            """
            if not isinstance(session_id, str):
                raise TypeError("session_id must be a string")
                
            if session_id not in session_store:
                session_store[session_id] = ChatMessageHistory()
            return session_store[session_id]

        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        # Update the memory type to reflect successful creation
        self._type = "InMemory"
        return with_message_history


class SimpleTextRetriever(BaseRetriever):
    """
    A simple text retriever that returns all stored documents for any query.
    
    This retriever stores a list of documents and returns all of them
    regardless of the query content. Useful for simple use cases where
    all documents should be considered relevant.
    """
    
    docs: List[Document]
    """List of documents to retrieve."""

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> "SimpleTextRetriever":
        """
        Create a SimpleTextRetriever from an iterable of text strings.
        
        Args:
            texts: An iterable of text strings to convert to documents.
            **kwargs: Additional keyword arguments passed to the constructor.
            
        Returns:
            A new SimpleTextRetriever instance.
            
        Raises:
            TypeError: If texts is not iterable.
            ValueError: If texts is empty.
        """
        try:
            text_list = list(texts)
        except TypeError:
            raise TypeError("texts must be iterable")
            
        if not text_list:
            raise ValueError("texts cannot be empty")
            
        docs = [Document(page_content=str(text)) for text in text_list]
        return cls(docs=docs, **kwargs)

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents for the given query.
        
        Note: This implementation returns all stored documents regardless
        of the query content.
        
        Args:
            query: The search query (unused in this implementation).
            run_manager: Callback manager for the retriever run.
            
        Returns:
            List of all stored documents.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
            
        return self.docs.copy()  # Return a copy to prevent external modification



