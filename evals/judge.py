"""LLM-as-judge for the evaluation harness.

The judge prompts live here, in the open, so that anyone reading the eval
scores can see exactly what "correct" and "faithful" mean. Each judge call
returns ``{"verdict": "pass" | "fail", "reason": "..."}``.
"""

import json

CORRECTNESS_PROMPT = """\
You are grading the answer of a question-answering system.

Question: {question}
Reference answer (ground truth): {gold_answer}
System answer: {answer}

Does the system answer convey the same essential fact(s) as the reference
answer? Extra correct detail is fine; wrong, missing, or contradicting facts
are not. Respond with ONLY a JSON object:
{{"verdict": "pass" or "fail", "reason": "<one short sentence>"}}
"""

DECLINE_PROMPT = """\
You are grading the answer of a question-answering system. The question
CANNOT be answered from the system's documents, so the correct behavior is
to say it doesn't know (or equivalent), without inventing an answer.

Question: {question}
System answer: {answer}

Did the system appropriately decline or state the information is not
available, rather than fabricating a specific answer? Respond with ONLY a
JSON object:
{{"verdict": "pass" or "fail", "reason": "<one short sentence>"}}
"""

FAITHFULNESS_PROMPT = """\
You are checking a question-answering system for hallucination.

Question: {question}
Retrieved context (the ONLY information the system was allowed to use):
---
{context}
---
System answer: {answer}

Is every factual claim in the system answer supported by the retrieved
context? Saying "I don't know" is always faithful. Respond with ONLY a JSON
object:
{{"verdict": "pass" or "fail", "reason": "<one short sentence>"}}
"""


def _get_judge_llm(model: str, provider: str):
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)


def _parse_verdict(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge output: {text!r}")
    result = json.loads(cleaned[start : end + 1])
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        raise ValueError(f"invalid verdict in judge output: {text!r}")
    return {"verdict": verdict, "reason": str(result.get("reason", ""))}


def _judge(prompt: str, model: str, provider: str) -> dict:
    llm = _get_judge_llm(model, provider)
    response = llm.invoke(prompt).content
    try:
        return _parse_verdict(response)
    except (ValueError, json.JSONDecodeError):
        # One retry with an explicit format reminder.
        response = llm.invoke(
            prompt + "\nRespond with ONLY the JSON object, nothing else."
        ).content
        try:
            return _parse_verdict(response)
        except (ValueError, json.JSONDecodeError):
            return {
                "verdict": "fail",
                "reason": f"unparseable judge output: {response[:100]}",
            }


def judge_correctness(
    question: str, gold_answer, answer: str, model: str, provider: str = "openai"
) -> dict:
    """Grade an answer against the gold answer (or against declining, if gold is None)."""
    if gold_answer is None:
        prompt = DECLINE_PROMPT.format(question=question, answer=answer)
    else:
        prompt = CORRECTNESS_PROMPT.format(
            question=question, gold_answer=gold_answer, answer=answer
        )
    return _judge(prompt, model, provider)


def judge_faithfulness(
    question: str, context: str, answer: str, model: str, provider: str = "openai"
) -> dict:
    """Grade whether every claim in the answer is grounded in the retrieved context."""
    prompt = FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer
    )
    return _judge(prompt, model, provider)
