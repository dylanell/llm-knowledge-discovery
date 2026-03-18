import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from llm_knowledge_discovery.eval.models import (
    DocumentRelevance,
    ContextPrecisionResult,
)

logger = logging.getLogger(__name__)
_MOD = "[context_precision.py]"

DEFAULT_RAG_MODEL = "claude-sonnet-4-6"

# Determine whether a single retrieved document was useful for answering the
# query. One LLM call per document keeps parsing simple and failures isolated.
_DOCUMENT_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluator for a retrieval-augmented generation system. "
            "Given a user query, a generated answer, and a single retrieved "
            "context document, determine whether the document is relevant — "
            "i.e., whether it contains information that is useful for "
            "answering the query. A document is relevant if it directly "
            "supports or "
            "provides evidence for the answer, or contains information clearly "
            "pertinent to the query. A document is NOT relevant if it is only "
            "tangentially related or contains no information useful for "
            "answering the query.",
        ),
        (
            "human",
            "Query: {query}\n\nAnswer:\n{answer}\n\n---\n\n"
            "Context Document:\n{document}",
        ),
    ]
)


def _check_document_relevance(
    query: str, answer: str, doc: Document, llm: ChatAnthropic
) -> DocumentRelevance:
    chain = _DOCUMENT_RELEVANCE_PROMPT | llm.with_structured_output(
        DocumentRelevance
    )
    title = doc.metadata.get("title", "Unknown")
    document_text = f"Title: {title}\n\n{doc.page_content}"
    return chain.invoke(
        {"query": query, "answer": answer, "document": document_text}
    )


def score_context_precision(
    query: str,
    answer: str,
    context_docs: list[Document],
    rag_model: str = DEFAULT_RAG_MODEL,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> ContextPrecisionResult:
    """
    Score the context precision of the documents retrieved for a RAG query.

    Context precision measures whether the retrieved documents are relevant to
    the query and answer. Follows the RAGAS approach: for each retrieved
    document, determine whether it is relevant, then compute a weighted
    precision (MAP-style) that rewards systems which rank relevant documents
    higher in the retrieval order.

    Score = sum(precision@k * relevance_k) / num_relevant, where precision@k
    is the fraction of relevant docs in the top-k results and relevance_k is
    1 if the doc at position k is relevant, 0 otherwise. Score is 0.0 if no
    documents are relevant.

    Args:
        query: The original query posed to the RAG system
        answer: The RAG answer body only — do not include the References
            section. Pass `rag_result.answer`, not the output of
            `_format_answer`.
        context_docs: The retrieved context documents, in retrieval rank order
        rag_model: Anthropic model ID to use for relevance judgements
        temperature: LLM sampling temperature; default 0 for maximum
            determinism across benchmark runs
        max_tokens: Maximum tokens for LLM responses

    Returns:
        ContextPrecisionResult with per-document verdicts and an overall score
    """
    llm = ChatAnthropic(
        model=rag_model, max_tokens=max_tokens, temperature=temperature
    )

    # Check relevance of each document (one LLM call per doc)
    verdicts = []
    for doc in context_docs:
        verdict = _check_document_relevance(query, answer, doc, llm)
        verdicts.append(verdict)

    num_relevant = sum(1 for v in verdicts if v.relevant)
    logger.info(
        f"{_MOD} {num_relevant}/{len(context_docs)} context documents "
        f"are relevant"
    )

    # No relevant documents → score 0.0
    if num_relevant == 0:
        return ContextPrecisionResult(
            query=query,
            answer=answer,
            score=0.0,
            num_docs=len(context_docs),
            num_relevant=0,
            verdicts=verdicts,
        )

    # Weighted precision (MAP-style): rewards systems that rank relevant docs
    # higher. For each position k, if the doc at k is relevant, compute
    # precision@k and add it to the running sum. Normalize by num_relevant.
    precision_sum = 0.0
    running_relevant = 0
    for k, verdict in enumerate(verdicts, start=1):
        if verdict.relevant:
            running_relevant += 1
            precision_sum += running_relevant / k
    score = precision_sum / num_relevant

    logger.info(f"{_MOD} Context Precision score: {score:.2f}")

    return ContextPrecisionResult(
        query=query,
        answer=answer,
        score=score,
        num_docs=len(context_docs),
        num_relevant=num_relevant,
        verdicts=verdicts,
    )
