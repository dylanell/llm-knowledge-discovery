import logging

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
_MOD = "[reranking.py]"

# Fast cross-encoder fine-tuned on MS MARCO passage ranking
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANK_MODEL,
) -> list[Document]:
    """
    Re-rank a list of Documents by relevance to the query using a cross-encoder.

    A cross-encoder scores each (query, document) pair jointly — unlike the
    bi-encoder used for initial retrieval, which embeds query and document
    independently. This produces more accurate relevance scores at the cost of
    being slower, so it's best applied to a small candidate set (e.g. top-10
    from similarity search) rather than the full corpus.

    Args:
        query: The search query string
        documents: Candidate documents from similarity search
        top_k: Number of top-ranked documents to return
        model_name: HuggingFace cross-encoder model name

    Returns:
        Top-k documents sorted by cross-encoder score (highest first)
    """
    logger.info(
        f"{_MOD} Reranking {len(documents)} documents with model="
        f"'{model_name}', top_k={top_k}"
    )

    model = CrossEncoder(model_name)

    # Score each (query, abstract) pair
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    # Sort documents by score descending and return top_k
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored[:top_k]]

    logger.info(f"{_MOD} Reranking complete, returning top {top_k} documents")
    return reranked
