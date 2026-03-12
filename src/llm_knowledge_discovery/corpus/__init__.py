from .models import PaperRecord
from .pubmed import fetch_records
from .storage import upsert_papers, load_papers


def fetch_and_store(query: str, max_results: int, collection_name: str) -> dict:
    """
    Fetch records from PubMed and store them in MongoDB in one call.

    Args:
        query: Free-text PubMed search query
        max_results: Maximum number of results to fetch
        collection_name: Label for this corpus (e.g. "arabidopsis")

    Returns:
        {"inserted": N, "skipped": M, "errors": K}
    """
    records = fetch_records(query, max_results, collection_name)
    return upsert_papers(records, collection_name)


__all__ = [
    "PaperRecord",
    "fetch_records",
    "upsert_papers",
    "load_papers",
    "fetch_and_store",
]
