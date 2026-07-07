"""Answer self-check: corrective RAG applied to the OUTPUT side.

Corrective RAG verifies retrieval before answering; this verifies the
ANSWER after generation. One cheap utility-model call compares the
answer's factual claims against the context the pipeline actually used
(the recorded retrieval), and when claims are unsupported it appends a
visible caveat naming them — transparent, in the glass-box spirit, and
deliberately NOT a silent rewrite: editing the model's answer behind the
reader's back is how trust dies.

Chain-agnostic by construction: the check runs at the pipeline level on
(answer, recorded documents), so simple/agent/corrective/auto all get it
from one place. Fail-safe like every utility step in this codebase — a
broken checker returns the answer unchanged.

Off by default pending its experiment gate (RAGSTONE_ANSWER_CHECK=on).
"""

import logging
from typing import Any, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

CHECK_PROMPT = """You are verifying a question-answering system's output.

Context the system was allowed to use (delimited by ---):
---
{context}
---

Answer to verify:
{answer}

Is every factual claim in the answer supported by the context? Reply
with exactly "supported" if so. Otherwise reply with one short line per
unsupported claim, each starting with "- ". Do not comment on style,
completeness, or claims of not knowing."""

CAVEAT_HEADER = "\n\nNote — not found in the retrieved documents:"

# Answers that are refusals/unknowns need no verification (saying "I
# don't know" is always faithful) and would only confuse the checker.
_REFUSAL_MARKERS = ("don't know", "do not know", "couldn't find", "could not find")


def check_answer(
    answer: str,
    documents: List[Any],
    utility_llm: Any,
) -> Optional[str]:
    """Return a caveat block for unsupported claims, or None if clean.

    Fail-safe: any checker error returns None (answer ships unchanged).
    """
    if not answer or not documents:
        return None
    lowered = answer.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return None

    context = "\n\n".join(doc.page_content for doc in documents)
    chain = (
        ChatPromptTemplate.from_template(CHECK_PROMPT)
        | utility_llm
        | (StrOutputParser())
    )
    try:
        verdict = chain.invoke({"context": context, "answer": answer}).strip()
    except Exception as exc:
        logger.warning(f"Answer check failed ({exc}); shipping answer unchanged")
        return None

    if verdict.lower().startswith("supported"):
        return None
    claims = [
        line.strip() for line in verdict.splitlines() if line.strip().startswith("- ")
    ]
    if not claims:
        # Unparseable verdict: fail safe rather than fabricate a caveat.
        logger.warning(f"Answer check verdict unparseable: {verdict[:120]!r}")
        return None
    return CAVEAT_HEADER + "\n" + "\n".join(claims)
