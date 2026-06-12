from ..models.base_model import LLMProxy
from ..rag.memory import MemoryProxy
from ..rag.rag import RagProxy

# Import utilities from centralized location


class FullChain:
    """
    A class to create and manage a full RAG chain
    """

    def __init__(
        self, llm_proxy: LLMProxy, rag_proxy: RagProxy, memory_proxy: MemoryProxy
    ):
        """
        Initialize the FullChain with LLM, RAG, and Memory proxies.

        Args:
            llm_proxy (LLMProxy): The LLM proxy.
            rag_proxy (RagProxy): The RAG proxy.
            memory_proxy (MemoryProxy): The Memory proxy.
        """
        self._llm = llm_proxy
        self._rag = rag_proxy
        self._memory = memory_proxy
        self._chain = None

    def create_full_chain(self, chain_type: str = "simple"):
        """
        Create a full chain based on the specified chain type.

        Args:
            chain_type (str): The type of chain to create. Defaults to "simple".
        """
        llm = self._llm.get_llm()
        match chain_type:
            case "simple":
                rag_chain = self._rag.make_chain()
            case "multi_query":
                rag_chain = self._rag.make_multi_query_chain()
            case "fusion":
                rag_chain = self._rag.make_fusion_chain()
            case _:
                rag_chain = self._rag.make_chain()
        self._chain = self._memory.create_memory_chain(llm, rag_chain)

    def get_chain(self):
        """
        Get the created chain.

        Returns:
            The created chain.
        """
        return self._chain

    def ask_question(self, query: str, session_id: str):
        """
        Ask a question using the created chain.

        Args:
            query (str): The question to ask.
            session_id (str): The session ID.

        Returns:
            The response from the chain.
        """
        result = self._chain.invoke(
            {"question": query},
            config={"configurable": {"thread_id": session_id}},
        )
        return result["answer"]

    def stream_question(self, query: str, session_id: str):
        """
        Ask a question and yield the answer incrementally as text chunks.

        Args:
            query (str): The question to ask.
            session_id (str): The session ID.

        Yields:
            str: Successive chunks of the answer.
        """
        for chunk in self._chain.stream(
            {"question": query},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="custom",
        ):
            if chunk:
                yield chunk
