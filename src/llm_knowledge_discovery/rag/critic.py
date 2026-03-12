import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_MOD = "[critic.py]"


class CritiqueResult(BaseModel):
    """Structured output from the critic LLM."""

    passed: bool
    issues: list[str]
    feedback: str = ""  # optional — may be absent if response is truncated


# Critic prompt checks three things:
# 1. Every [N] inline citation has a matching References entry and vice versa
# 2. Citations and References are numbered sequentially from [1] with no gaps
# 3. Every References title exactly matches the source document title
CRITIC_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a citation auditor. Your job is to check a RAG-generated "
            "answer for three things ONLY:\n\n"
            "1. Citation integrity: every inline [N] citation must have a "
            "matching entry in the References section, and every References "
            "entry must be cited inline. Flag any mismatch.\n\n"
            "2. Sequential numbering: the References section must be numbered "
            "sequentially starting from [1] with no gaps. For example, a "
            "References section with [1] and [3] but no [2] is a violation.\n\n"
            "3. Title accuracy: every title in the References section must "
            "exactly match one of the provided source document titles — no "
            "paraphrasing, no summarizing, no rewording. Note that the answer "
            "renumbers its references sequentially from [1] regardless of "
            "their order in the source list, so do NOT check that answer "
            "reference [N] maps to source document [N]. Only check that each "
            "title in the answer's References section appears verbatim "
            "somewhere in the source document list.\n\n"
            "If the answer states it cannot answer the question (no citations, "
            "no References section), return passed=True with no issues.\n\n"
            "Do not evaluate the factual correctness of the answer itself — "
            "only check citations, numbering, and titles.",
        ),
        (
            "human",
            "Source document titles (numbered):\n{source_titles}\n\n"
            "---\n\n"
            "Query: {query}\n\n"
            "Answer:\n{answer}",
        ),
    ]
)


def critique_response(
    query: str,
    answer: str,
    source_docs: list[Document],
    model: str,
) -> CritiqueResult:
    """
    Run a critic LLM pass over a RAG-generated answer to check citation
    integrity and title accuracy.

    Args:
        query: The original user query
        answer: The RAG-generated answer to critique
        source_docs: The source documents used to generate the answer
        model: Anthropic model ID to use for the critic

    Returns:
        CritiqueResult with passed flag, list of issues, and feedback
    """
    # Build a numbered list of source titles — keep the prompt tight
    title_lines = []
    for i, doc in enumerate(source_docs, start=1):
        title = doc.metadata.get("title", "Unknown")
        title_lines.append(f"[{i}] {title}")
    source_titles = "\n".join(title_lines)

    critic_llm = ChatAnthropic(
        model=model, max_tokens=2048
    ).with_structured_output(CritiqueResult)
    chain = CRITIC_PROMPT_TEMPLATE | critic_llm

    logger.info(
        f"{_MOD} Running critic on answer with {len(source_docs)} source docs"
    )

    result = chain.invoke(
        {
            "source_titles": source_titles,
            "query": query,
            "answer": answer,
        }
    )

    logger.info(
        f"{_MOD} Critique result: passed={result.passed}, "
        f"issues={len(result.issues)}"
    )

    return result
