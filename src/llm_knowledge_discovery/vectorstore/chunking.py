import logging

from langchain_core.documents import Document

from llm_knowledge_discovery.corpus.models import PaperRecord

logger = logging.getLogger(__name__)
_MOD = "[chunking.py]"

# All supported chunking strategies
CHUNK_STRATEGIES = ["whole_abstract"]


def chunk_records(
    records: list[PaperRecord], strategy: str = "whole_abstract"
) -> list[Document]:
    """
    Convert a list of PaperRecords into LangChain Documents using the
    specified chunking strategy.

    Args:
        records: List of PaperRecord objects to convert
        strategy: Chunking strategy to use. Options: "whole_abstract"

    Returns:
        List of LangChain Document objects ready for embedding
    """
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(
            f"{_MOD} Unknown strategy '{strategy}'. "
            f"Valid options: {CHUNK_STRATEGIES}"
        )

    if strategy == "whole_abstract":
        docs = _chunk_whole_abstract(records)

    logger.info(
        f"{_MOD} Chunked {len(records)} records into {len(docs)} documents "
        f"using strategy='{strategy}'"
    )
    return docs


def _chunk_whole_abstract(records: list[PaperRecord]) -> list[Document]:
    """
    One Document per record: the full abstract as page_content, with
    key paper fields stored in metadata for filtering/display later.
    """
    docs = []
    for record in records:
        doc = Document(
            page_content=record.abstract,
            metadata={
                "pubmed_id": record.pubmed_id,
                "title": record.title,
                "year": record.year,
                "journal": record.journal,
                "collection_name": record.collection_name,
            },
        )
        docs.append(doc)
    return docs
