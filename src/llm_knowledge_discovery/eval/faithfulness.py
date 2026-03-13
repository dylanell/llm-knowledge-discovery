import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from llm_knowledge_discovery.eval.models import (
    Claims,
    ClaimVerification,
    FaithfulnessResult,
)

logger = logging.getLogger(__name__)
_MOD = "[faithfulness.py]"

DEFAULT_MODEL = "claude-sonnet-4-6"

# Step 1: decompose the answer into atomic factual claims.
# Context is deliberately excluded — extraction is independent of the source.
_CLAIM_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise fact extractor. Given an answer text, decompose "
            "it into atomic factual claims — one discrete assertion per claim. "
            "Each claim must be a single, standalone fact. If two facts appear "
            "in the same sentence (e.g. 'FLC represses FT and SOC1'), split "
            "them into separate claims. Do not include meta-statements like "
            "'the answer says' or 'according to the abstracts' — extract only "
            "the underlying factual assertions. If the answer states it cannot "
            "answer the query, return an empty claims list.",
        ),
        (
            "human",
            "Answer:\n{answer}",
        ),
    ]
)

# Step 2: verify whether a single claim is supported by the retrieved context.
# One LLM call per claim keeps parsing simple and failures isolated.
_CLAIM_VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fact verifier. Given a context (a set of scientific "
            "abstracts) and a single factual claim, determine whether the "
            "claim is directly supported by the context. "
            "A claim is supported if the context explicitly states it or if "
            "it is a trivial direct inference (e.g. a restatement in "
            "different words). A claim is NOT supported if it requires "
            "multi-hop reasoning beyond what the context says, or if it "
            "introduces information not present in the context.",
        ),
        (
            "human",
            "Context:\n{context}\n\n---\n\nClaim: {claim}",
        ),
    ]
)


def _format_context(docs: list[Document]) -> str:
    # Numbered abstract blocks — matches the format used in chain.py
    sections = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Unknown")
        sections.append(f"[{i}] Title: {title}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def _extract_claims(answer: str, llm: ChatAnthropic) -> Claims:
    chain = _CLAIM_EXTRACTION_PROMPT | llm.with_structured_output(Claims)
    return chain.invoke({"answer": answer})


def _verify_claim(
    claim: str, context: str, llm: ChatAnthropic
) -> ClaimVerification:
    chain = _CLAIM_VERIFICATION_PROMPT | llm.with_structured_output(
        ClaimVerification
    )
    return chain.invoke({"context": context, "claim": claim})


def score_faithfulness(
    query: str,
    answer: str,
    context_docs: list[Document],
    model: str = DEFAULT_MODEL,
) -> FaithfulnessResult:
    """
    Score the faithfulness of a RAG-generated answer against its source context.

    Faithfulness measures whether the answer only asserts claims that are
    directly supported by the retrieved context docs. Follows the two-step
    RAGAS approach: (1) extract atomic claims from the answer, (2) verify each
    claim against the context.

    Score = supported_claims / total_claims. Answers with no claims (e.g. the
    model correctly said it couldn't answer) score 1.0 by convention.

    Args:
        query: The original query posed to the RAG system
        answer: The RAG answer body only — do not include the References
            section. Pass `rag_result.answer`, not the output of
            `_format_answer`. References are for the critic and human review,
            not faithfulness scoring.
        context_docs: The retrieved context documents used to generate the
            answer
        model: Anthropic model ID to use for extraction and verification

    Returns:
        FaithfulnessResult with per-claim detail and an overall score
    """
    llm = ChatAnthropic(model=model, max_tokens=1024)
    context = _format_context(context_docs)

    # Step 1: extract atomic claims from the answer
    claims_result = _extract_claims(answer, llm)
    claims = claims_result.claims
    logger.info(f"{_MOD} Extracted {len(claims)} claims from answer")

    # An answer with no claims cannot be unfaithful — score 1.0 by convention
    if not claims:
        return FaithfulnessResult(
            query=query,
            answer=answer,
            score=1.0,
            total_claims=0,
            supported_claims=0,
            verifications=[],
        )

    # Step 2: verify each claim against the context (one LLM call per claim)
    verifications = []
    for claim in claims:
        verification = _verify_claim(claim, context, llm)
        verifications.append(verification)

    num_supported = sum(1 for v in verifications if v.supported)
    score = num_supported / len(claims)

    logger.info(
        f"{_MOD} Faithfulness score: {score:.2f} "
        f"({num_supported}/{len(claims)} claims supported)"
    )

    return FaithfulnessResult(
        query=query,
        answer=answer,
        score=score,
        total_claims=len(claims),
        supported_claims=num_supported,
        verifications=verifications,
    )
