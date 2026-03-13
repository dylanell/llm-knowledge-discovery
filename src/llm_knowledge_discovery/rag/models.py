from pydantic import BaseModel, Field


class RagResult(BaseModel):
    """Structured output from a RAG generation step."""

    # Field descriptions are passed to Claude as part of the tool schema when
    # using with_structured_output — they act as model instructions, not just
    # documentation. RagResult needs them to enforce constraints that aren't
    # obvious from field names alone (no References section in answer body,
    # exact title matching in references).
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
