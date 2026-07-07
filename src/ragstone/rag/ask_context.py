"""Request-scoped ask state: the end of the single-writer caveat.

Every ask used to write its introspection state (retrieved documents,
question, metrics) onto shared pipeline attributes — correct for the
same-thread consumers this project ships (Streamlit, CLI, the eval
harness), but two CONCURRENT asks on one pipeline would interleave their
records. This module scopes that state to the ask itself:

- `begin_ask` creates an AskContext and publishes it on a ContextVar —
  the same propagation mechanism the metrics system already relies on
  (LangGraph nodes and LangChain's executors run under copied contexts,
  so the retrieval recorder finds ITS ask's context, not a neighbor's).
- Readers prefer the current context (same logical call flow), then the
  pipeline's last-completed ask — so cross-thread introspection keeps
  the old "most recent ask" semantics instead of breaking.
- Contexts are never reset, only superseded by the next begin_ask: a
  ContextVar reset inside a generator frame can fire in a different
  context than its set (the SSE lesson, Experiment 16 era) — overwrite
  semantics need no reset and cannot raise.

The recorder keeps a bounded fallback for retrievals that happen OUTSIDE
any ask (the compare view drives chain variants directly): the most
recent retrieval's documents, replaced per invoke — approximately the
old behavior, without unbounded growth.
"""

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..utils.observability import RequestMetrics, time_stage

logger = logging.getLogger(__name__)


@dataclass
class AskContext:
    """Everything one ask records about itself, owned by that ask.

    owner_id ties the context to the pipeline that opened it (its id()):
    the ContextVar is process-global, so without ownership a stale
    context from pipeline A's ask would leak into pipeline B's reads in
    the same thread. Comparing against id(self) is safe because the
    reading pipeline is alive for the duration of the comparison.
    """

    question: str
    owner_id: int
    record: List[Document] = field(default_factory=list)
    metrics: Optional[RequestMetrics] = None


_current_ask: ContextVar[Optional[AskContext]] = ContextVar(
    "ragstone_current_ask", default=None
)


def begin_ask(question: str, owner_id: int) -> AskContext:
    """Open a new ask context in the current logical call flow."""
    context = AskContext(question=question, owner_id=owner_id)
    _current_ask.set(context)
    return context


def current_ask() -> Optional[AskContext]:
    """The ask context of the current call flow, if one is open."""
    return _current_ask.get()


class SourceRecordingRetriever(BaseRetriever):
    """Wraps the final retriever and records documents into the ask.

    Lets get_sources() reuse the documents retrieved while answering
    instead of paying for a second retrieval (query embedding + ensemble
    + reranker) per question. Documents land on the CURRENT AskContext;
    retrievals outside any ask (compare-view chain variants) land on a
    per-invoke fallback so direct chain use still has sources, bounded
    to the most recent retrieval.
    """

    wrapped: BaseRetriever
    owner_id: int = 0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Timed as the "retrieval" stage: this wraps the FINAL retriever,
        # so query embedding, ensemble merge, and any reranking are all
        # included. Multiple invocations (agent/corrective) accumulate.
        with time_stage("retrieval"):
            docs = self.wrapped.invoke(query)
        context = _current_ask.get()
        if context is not None and context.owner_id == self.owner_id:
            context.record.extend(docs)
        else:
            # Outside any ask: keep only this retrieval (replace, don't
            # grow) — see the module docstring.
            self._fallback_record = list(docs)
        return docs

    @property
    def fallback_record(self) -> List[Document]:
        """Documents from the most recent context-less retrieval."""
        return list(getattr(self, "_fallback_record", []))
