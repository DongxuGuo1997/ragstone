from typing import Annotated, Any, Iterable, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict


class MemoryState(TypedDict, total=False):
    """Per-session conversation state, checkpointed by thread_id."""

    messages: Annotated[List[AnyMessage], add_messages]
    question: str  # original user question for the current turn
    standalone_question: str  # history-rephrased question for the RAG chain
    answer: str


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
        self, llm: BaseChatModel, base_chain: Runnable
    ) -> CompiledStateGraph:
        """
        Create a memory-enabled chain that can maintain conversation history.

        Builds a small LangGraph state graph with an in-memory checkpointer:
        on follow-up turns the question is first rephrased against the chat
        history into a standalone question; the first turn skips that LLM
        call entirely. History is kept per thread_id (the caller's
        session_id).

        Args:
            llm: The language model to use for contextualizing questions.
            base_chain: The base RAG chain (any Runnable taking a question
                        string and producing an answer string).

        Returns:
            A compiled LangGraph. Invoke with {"question": ...} and
            config={"configurable": {"thread_id": ...}}; the answer is in
            the result's "answer" key. Streaming the graph with
            stream_mode="custom" yields answer text chunks only.

        Raises:
            TypeError: If llm or base_chain are not of the expected types.
        """
        if not isinstance(llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel")

        if not isinstance(base_chain, Runnable):
            raise TypeError("base_chain must be a Runnable")

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        rephrase_chain = contextualize_q_prompt | llm | StrOutputParser()

        def rephrase(state: MemoryState) -> MemoryState:
            return {
                "standalone_question": rephrase_chain.invoke(
                    {
                        "chat_history": state["messages"],
                        "question": state["question"],
                    }
                )
            }

        def answer(state: MemoryState) -> MemoryState:
            # get_stream_writer() is a no-op under .invoke(), so this one
            # implementation serves both invoke and custom-mode streaming.
            writer = get_stream_writer()
            question = state.get("standalone_question") or state["question"]
            parts: List[str] = []
            for chunk in base_chain.stream(question):
                if chunk:
                    writer(chunk)
                    parts.append(chunk)
            text = "".join(parts)
            # History records the ORIGINAL question, not the rephrased one,
            # matching the previous RunnableWithMessageHistory behavior.
            return {
                "messages": [HumanMessage(state["question"]), AIMessage(text)],
                "answer": text,
                "standalone_question": "",
            }

        def route(state: MemoryState) -> str:
            return "rephrase" if state.get("messages") else "answer"

        graph = StateGraph(MemoryState)
        graph.add_node("rephrase", rephrase)
        graph.add_node("answer", answer)
        graph.add_conditional_edges(
            START, route, {"rephrase": "rephrase", "answer": "answer"}
        )
        graph.add_edge("rephrase", "answer")
        graph.add_edge("answer", END)

        self._type = "InMemory"
        return graph.compile(checkpointer=InMemorySaver())


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
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
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
