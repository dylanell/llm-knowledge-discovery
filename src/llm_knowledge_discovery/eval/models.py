from pydantic import BaseModel

# These models are used as structured output targets via with_structured_output.
# Unlike RagResult, none of them need Field(description=...) because their field
# names are self-explanatory and require no special constraints to be passed
# to the model as tool schema instructions.


class Claims(BaseModel):
    """Atomic factual claims extracted from an LLM-generated answer."""

    claims: list[str]


class ClaimVerification(BaseModel):
    """Verdict for a single claim checked against retrieved context."""

    claim: str
    supported: bool
    reason: str  # brief justification — useful for debugging


class FaithfulnessResult(BaseModel):
    """Faithfulness score and per-claim detail for a single query."""

    query: str
    answer: str
    score: float  # supported_claims / total_claims; 1.0 if no claims
    total_claims: int
    supported_claims: int
    verifications: list[ClaimVerification]
