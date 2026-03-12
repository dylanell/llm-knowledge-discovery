import logging

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from llm_knowledge_discovery.rag.critic import CritiqueResult, critique_response
from llm_knowledge_discovery.vectorstore.reranking import rerank_documents

logger = logging.getLogger(__name__)
_MOD = "[chain.py]"

# Default model and retrieval settings
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_RETRIEVAL_K = 10  # broad candidate pool for the reranker
DEFAULT_RERANK_K = 5  # top-k abstracts passed to the prompt

# System prompt instructs the model to ground its answer in the provided context
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a plant biology research assistant. Answer the user's "
            "question using ONLY the provided abstracts. If the abstracts do "
            "not contain enough information to answer the question, say so "
            "and do not include a references section. If you can answer the "
            "question, select only the abstracts you actually use, renumber "
            "them sequentially starting from [1], cite them inline by their "
            "new number (e.g. [1], [2]), and include a References section at "
            "the end listing only the abstracts you cited in that same "
            "sequential order, using their exact titles as they appear in the "
            "context — do not paraphrase or summarize titles.",
        ),
        (
            "human",
            "Abstracts:\n\n{context}\n\n---\n\nQuestion: {question}",
        ),
    ]
)


class RagResult(BaseModel):
    """Structured output from the initial RAG generation step."""

    answer: str = Field(
        description=(
            "The answer body with inline citations. When citing multiple "
            "sources together, use comma-separated format: [1, 2] not [1][2]."
            " Do not include a References section in this field."
        )
    )
    references: list[str] = Field(
        description=(
            "Exact titles of the cited sources in citation order — "
            "index 0 corresponds to [1], index 1 to [2], etc. "
            "Use the exact title as it appears in the context."
        )
    )


def _format_answer(body: str, references: list[str]) -> str:
    # Reconstruct a display-ready answer string from structured fields
    if not references:
        return body
    ref_lines = "\n".join(
        f"[{i + 1}] {title}" for i, title in enumerate(references)
    )
    return f"{body}\n\n---\n\n**References:**\n{ref_lines}"


def _format_context(docs: list[Document]) -> str:
    # Number each abstract so Claude can cite them by number accurately
    sections = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Unknown")
        sections.append(f"[{i}] Title: {title}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def retrieve_and_rerank(
    query: str,
    vectorstore: Chroma,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    rerank_k: int = DEFAULT_RERANK_K,
) -> list[Document]:
    """
    Retrieve candidate documents from the vectorstore and rerank them.

    Args:
        query: The search query
        vectorstore: A loaded Chroma vector store
        retrieval_k: Number of candidates to pull via similarity search
        rerank_k: Number of top docs to keep after reranking

    Returns:
        Reranked list of top-k documents
    """
    # Broad similarity search to get candidates for the cross-encoder
    candidates = vectorstore.similarity_search(query, k=retrieval_k)
    logger.info(
        f"{_MOD} Retrieved {len(candidates)} candidates, "
        f"reranking to top {rerank_k}"
    )
    # Cross-encoder reranks by joint (query, doc) relevance
    return rerank_documents(query, candidates, top_k=rerank_k)


def build_rag_chain(
    vectorstore: Chroma,
    model: str = DEFAULT_MODEL,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    rerank_k: int = DEFAULT_RERANK_K,
):
    """
    Build an LCEL RAG chain that retrieves + reranks abstracts from the
    vector store and passes them as context to Claude to answer a question.

    Args:
        vectorstore: A loaded Chroma vector store
        model: Anthropic model ID to use for generation
        retrieval_k: Number of candidate docs to pull via similarity search
        rerank_k: Number of top docs (after reranking) to include in the prompt

    Returns:
        A Runnable[str, str] — takes a query string, returns an answer string
    """
    llm = ChatAnthropic(model=model)

    chain = (
        {
            # Retrieve + rerank then format into a single context string
            "context": (
                RunnableLambda(
                    lambda q: retrieve_and_rerank(
                        q, vectorstore, retrieval_k, rerank_k
                    )
                )
                | RunnableLambda(_format_context)
            ),
            # Pass the original query through unchanged as the question
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    logger.info(
        f"{_MOD} RAG chain built: model='{model}', "
        f"retrieval_k={retrieval_k}, rerank_k={rerank_k}"
    )
    return chain


def invoke_and_review(
    query: str,
    vectorstore: Chroma,
    model: str = DEFAULT_MODEL,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    rerank_k: int = DEFAULT_RERANK_K,
    review_steps: int = 1,
) -> tuple[str, list[CritiqueResult]]:
    """
    Run the full RAG pipeline with an optional self-correction loop.

    Retrieves and reranks docs once, generates an answer via structured output
    (so cited references are available without parsing), then critiques it.
    If the critique fails and review_steps > 1, feeds the issues back to the
    model as a follow-up turn and regenerates. Repeats up to review_steps times.

    Args:
        query: The user's question
        vectorstore: A loaded Chroma vector store
        model: Anthropic model ID for both generation and critique
        retrieval_k: Number of candidate docs to pull via similarity search
        rerank_k: Number of top docs (after reranking) to include in the prompt
        review_steps: Max number of critique + regenerate cycles
            (1 = generate once, critique once)

    Returns:
        Tuple of (final_answer, list of CritiqueResult from each step)
    """
    llm = ChatAnthropic(model=model)

    # Retrieve and rerank once — same docs for generation and critic checks
    docs = retrieve_and_rerank(query, vectorstore, retrieval_k, rerank_k)

    # Format context and build the initial prompt messages manually
    context = _format_context(docs)
    initial_messages = RAG_PROMPT_TEMPLATE.format_messages(
        context=context, question=query
    )

    # Generate the initial answer with structured output so references are
    # available directly as a list — no regex parsing needed.
    structured_chain = RAG_PROMPT_TEMPLATE | llm.with_structured_output(
        RagResult
    )
    rag_result = structured_chain.invoke(
        {"context": context, "question": query}
    )

    # Lock in the cited sources from the initial answer. Refinement fixes
    # formatting only — the set of cited sources should not change.
    original_references = rag_result.references
    answer = _format_answer(rag_result.answer, original_references)
    logger.info(f"{_MOD} Initial answer generated ({len(answer)} chars)")

    # pre_filled_chain is used for refinement turns — it receives
    # already-formatted messages rather than a raw query string.
    pre_filled_chain = llm | StrOutputParser()

    critiques = []

    for step in range(review_steps):
        logger.info(f"{_MOD} Critique step {step + 1}/{review_steps}")

        critique = critique_response(query, answer, docs, model)
        critiques.append(critique)

        # Stop if the answer passed or this is the last allowed step
        if critique.passed or step == review_steps - 1:
            break

        # Build refinement issues as a bulleted list for clarity
        issue_lines = "\n".join(f"- {issue}" for issue in critique.issues)
        ref_lines = "\n".join(
            f"[{i + 1}] {title}"
            for i, title in enumerate(original_references)
        )
        refinement_prompt = (
            f"Your answer has the following citation/title issues:"
            f"\n{issue_lines}\n\n"
            f"Additional feedback: {critique.feedback}\n\n"
            f"You must preserve all of these cited sources in your revised "
            f"answer (do not drop any):\n{ref_lines}\n\n"
            "Please correct these issues and provide a revised answer."
        )

        # Multi-turn: original messages + AI answer + human correction request
        refinement_messages = initial_messages + [
            AIMessage(content=answer),
            HumanMessage(content=refinement_prompt),
        ]

        answer = pre_filled_chain.invoke(refinement_messages)
        logger.info(f"{_MOD} Refined answer generated (step {step + 1})")

    return answer, critiques
