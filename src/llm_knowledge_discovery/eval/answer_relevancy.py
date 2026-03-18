import logging
import numpy as np

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from llm_knowledge_discovery.eval.models import (
    RelevantQueries,
    AnswerRelevancyResult,
)

logger = logging.getLogger(__name__)
_MOD = "[answer_relevancy.py]"

DEFAULT_RAG_MODEL = "claude-sonnet-4-6"
DEFAULT_EMB_MODEL = "all-MiniLM-L6-v2"

_RELEVANT_QUERIES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluator for a question-answering system. Given an "
            "answer, generate {k} diverse, standalone questions that this "
            "answer most directly addresses. Each question should be phrased "
            "as a natural language question a user might ask. Focus on the "
            "primary intent of the answer — do not generate questions about "
            "peripheral details. Do not generate near-duplicate paraphrases; "
            "each question should represent a meaningfully different "
            "formulation of the same underlying information need.",
        ),
        (
            "human",
            "Answer:\n{answer}",
        ),
    ]
)


def _get_relevant_queries(
    answer: str,
    llm: ChatAnthropic,
    k: int = 3,
) -> RelevantQueries:
    chain = _RELEVANT_QUERIES_PROMPT | llm.with_structured_output(
        RelevantQueries
    )
    result = chain.invoke({"k": k, "answer": answer})

    # Enforce exactly k queries — truncate if over, warn if under
    if len(result.queries) != k:
        logger.warning(
            f"{_MOD} Expected {k} queries, got {len(result.queries)}"
        )
    return RelevantQueries(queries=result.queries[:k])


def score_answer_relevancy(
    query: str,
    answer: str,
    k: int = 3,
    rag_model: str = DEFAULT_RAG_MODEL,
    emb_model: str = DEFAULT_EMB_MODEL,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> AnswerRelevancyResult:
    """
    Score the answer relevancy of a RAG-generated answer against the original query.

    Answer relevancy measures whether the answer actually addresses the question
    asked. Follows the RAGAS approach: (1) generate k synthetic queries that the
    answer appears to address, (2) embed those queries and the original query,
    (3) compute mean cosine similarity between the original and synthetic queries.

    A high score means the answer is tightly focused on the original question.
    A low score suggests the answer drifted — addressing a different question or
    providing tangential information.

    Sentence-transformer embeddings are L2-normalized, so dot product equals
    cosine similarity and no normalization step is needed.

    Args:
        query: The original query posed to the RAG system
        answer: The RAG answer body only — do not include the References
            section. Pass `rag_result.answer`, not the output of
            `_format_answer`. The References section could distort the
            synthetic query generation.
        k: Number of synthetic queries to generate from the answer
        rag_model: Anthropic model ID to use for synthetic query generation
        emb_model: HuggingFace sentence-transformer model ID for embeddings
        temperature: LLM sampling temperature; default 0 for maximum
            determinism across benchmark runs
        max_tokens: Maximum tokens for LLM responses

    Returns:
        AnswerRelevancyResult with the generated queries and overall score
    """
    llm = ChatAnthropic(
        model=rag_model, max_tokens=max_tokens, temperature=temperature
    )
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    # Generate k synthetic queries that the answer appears to address
    relevant_queries_res = _get_relevant_queries(answer=answer, llm=llm, k=k)

    logger.info(f"{_MOD} Generated {k} relevant queries from answer")
    for i, relevant_query in enumerate(relevant_queries_res.queries):
        logger.info(f"{_MOD} Generated query [{i + 1}]: {relevant_query}")

    # Vectorize original query and relevant queries. Embeddings are L2-
    # normalized so dot product == cosine similarity — no norm term needed.
    query_vec = np.expand_dims(np.array(embeddings.embed_query(query)), axis=0)
    relevant_query_vecs = np.array(
        embeddings.embed_documents(relevant_queries_res.queries)
    )
    score = float(np.mean(np.matmul(query_vec, relevant_query_vecs.T)))

    logger.info(f"{_MOD} Answer Relevancy score: {score:.2f}")

    return AnswerRelevancyResult(
        query=query,
        answer=answer,
        relevant_queries=relevant_queries_res,
        score=score,
    )
