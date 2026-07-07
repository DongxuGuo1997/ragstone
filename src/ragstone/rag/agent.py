"""Agentic RAG: the model decides when and what to retrieve.

The fixed chains (simple / multi_query / fusion) retrieve exactly once,
before the LLM sees any context — one retrieval, one answer call,
predictable latency and cost. In agent mode the LLM is instead given the
retriever as a tool and drives the loop itself: search, read the results,
optionally search again with a refined query, then answer.

That flexibility can recover from a bad first retrieval, but it costs
extra LLM calls, tokens, and latency. Whether it is worth it is an
empirical question — measure it with the eval harness:

    python evals/run_eval.py --chain-type simple
    python evals/run_eval.py --chain-type agent

and compare quality and efficiency in evals/report.md.
"""

import logging
from typing import Any, Dict, Iterator, Optional, Union

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from .rag import extract_question, format_docs

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks over a document "
    "collection. Use the search_documents tool to find relevant context "
    "before answering. If the first results do not answer the question, "
    "refine your query and search again — but use at most three searches. "
    "If the documents do not contain the answer, just say that you don't "
    "know. Use three sentences maximum and keep the answer concise."
)


def _content_text(content: Any) -> str:
    """Plain text from message content (a string or a content-block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


class AgentRagChain(Runnable[Any, str]):
    """A RAG chain where the LLM drives retrieval through a search tool.

    Conforms to the same contract as the fixed chains — input is a question
    string or {"question": ...} dict, invoke() returns the answer string,
    stream() yields answer-text chunks — so memory, caching, the UI, and
    the MCP server all work with it unchanged.
    """

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever) -> None:
        """
        Args:
            llm: A chat model that supports tool calling.
            retriever: The retriever exposed to the model as a search tool.
        """

        @tool
        def search_documents(query: str) -> str:
            """Search the document collection and return relevant passages."""
            # Surface the search live so UIs can show the agent thinking.
            # get_stream_writer() is a no-op under .invoke(), so this costs
            # nothing on the non-streaming path.
            get_stream_writer()({"event": "search", "query": query})
            docs = retriever.invoke(query)
            return format_docs(docs) or "No matching passages found."

        self._agent = create_agent(
            llm, [search_documents], system_prompt=AGENT_SYSTEM_PROMPT
        )
        # "At most three searches" in the system prompt is a soft bound a
        # misbehaving model can ignore; this is the hard one. Each search
        # is a model step + a tool step, so 3 searches + the final answer
        # fits well inside 10 graph steps.
        self._config: RunnableConfig = {"recursion_limit": 10}

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Run the agent loop to completion and return the final answer."""
        question = extract_question(input)
        result = self._agent.invoke(
            {"messages": [HumanMessage(question)]}, config=config or self._config
        )
        return _content_text(result["messages"][-1].content)

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Iterator[str]:
        """Yield answer-text tokens as the agent produces them.

        Tool activity is filtered out — only the model's answer text
        reaches the caller, honoring the Runnable[..., str] contract.
        Consumers that want live search events too should use
        :meth:`stream_with_events`.
        """
        for item in self.stream_with_events(input, config=config):
            if isinstance(item, str):
                yield item

    def stream_with_events(
        self, input: Any, config: Optional[RunnableConfig] = None
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Yield answer-text tokens interleaved with progress events.

        Events are dicts (e.g. {"event": "search", "query": ...}) emitted
        by the search tool as the agent works; text chunks are the final
        answer. The memory graph prefers this method when present, so
        events reach the UI/API stream while only text becomes the answer.
        """
        question = extract_question(input)
        for item in self._agent.stream(
            {"messages": [HumanMessage(question)]},
            config=config or self._config,
            stream_mode=["custom", "messages"],
        ):
            # With a list of stream modes, items are (mode, chunk) pairs.
            if not (isinstance(item, tuple) and len(item) == 2):
                continue
            mode, chunk = item
            if mode == "custom":
                if isinstance(chunk, dict):
                    yield chunk
                continue
            msg, _meta = chunk
            if (
                isinstance(msg, AIMessageChunk)
                and msg.content
                and not msg.tool_call_chunks
            ):
                text = _content_text(msg.content)
                if text:
                    yield text
