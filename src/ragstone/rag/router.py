"""Query routing: spend the expensive chain only where it can pay.

This chain is what turns three measured negative results into product
value. Experiments 4, 8, and 9 showed that query expansion and
self-correction do not improve *average* quality — but Experiment 12's
re-measurement showed corrective RAG holds a consistent faithfulness edge
(0.986 vs 0.967 at n=224) on exactly the questions this corpus was
engineered to make hard: entity confusions, comparisons, and
unanswerables. Routing sends those to the corrective chain and everything
else to the simple chain, so the 2x token cost is paid only where the
grade-and-retry loop has something to catch.

The router itself is one utility-model call (~a few hundred ms on the
cheap model, timed as the "route" stage). Classification failures fall
back to "simple" — a broken router must degrade to the default chain,
never break answering.

Same contract as every other chain (Runnable[..., str]); the chosen
strategy is surfaced as a {"event": "route", ...} progress event.
"""

import logging
from typing import Any, Iterator, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from ..utils.observability import time_stage
from .rag import extract_question

logger = logging.getLogger(__name__)

ROUTE_PROMPT = """You are routing questions for a document QA system.

Classify the question into exactly one strategy:

- simple: a factual lookup — a specification, a date, a schedule, a
  person, a number, an instruction. This is the default.
- careful: ONLY when the question clearly compares two or more named
  entities or versions, or clearly asks for something product
  documentation rarely contains (prices, opinions, future plans,
  personal details).

When unsure, choose simple.

Question: {question}

Reply with exactly one word: simple or careful."""

STRATEGIES = ("simple", "careful")


class RouterRagChain(Runnable[Any, str]):
    """Classify each question, then delegate to the matching chain.

    Constructed by RagProxy.make_router_chain(), which supplies a simple
    chain, a corrective chain, and the cheap utility model (the same one
    that grades and rewrites). Honors the base-chain contract, so memory,
    caching, the servers, and both UIs work unchanged under
    chain_type="auto".
    """

    def __init__(
        self,
        simple_chain: Runnable,
        careful_chain: Runnable,
        utility_llm: BaseChatModel,
    ) -> None:
        self._chains = {"simple": simple_chain, "careful": careful_chain}
        self._route_chain = (
            ChatPromptTemplate.from_template(ROUTE_PROMPT)
            | utility_llm
            | StrOutputParser()
        )

    def _classify(self, question: str) -> str:
        """One utility-model call; anything unexpected routes to simple."""
        try:
            with time_stage("route"):
                verdict = self._route_chain.invoke({"question": question})
            strategy = verdict.strip().lower()
        except Exception as exc:
            logger.warning(f"Router classification failed ({exc}); using simple")
            return "simple"
        if strategy not in STRATEGIES:
            logger.warning(f"Router returned {verdict!r}; using simple")
            return "simple"
        return strategy

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        question = extract_question(input)
        strategy = self._classify(question)
        return self._chains[strategy].invoke(question)

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Answer text only — the Runnable[..., str] contract."""
        for chunk in self.stream_with_events(input):
            if isinstance(chunk, str):
                yield chunk

    def stream_with_events(self, input: Any) -> Iterator[Union[str, dict]]:
        """Route event first, then the chosen chain's stream (with its
        own events, when it has them)."""
        question = extract_question(input)
        strategy = self._classify(question)
        yield {"event": "route", "strategy": strategy}
        chain = self._chains[strategy]
        stream_fn = getattr(chain, "stream_with_events", chain.stream)
        yield from stream_fn(question)
