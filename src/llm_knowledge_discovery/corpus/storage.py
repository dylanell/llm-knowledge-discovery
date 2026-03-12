import logging
import os

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import UpdateOne

from .models import PaperRecord

logger = logging.getLogger(__name__)
_MOD = "[storage.py]"


def _get_collection(corpus_tag: str) -> Collection:
    """
    Connect to MongoDB and return the collection for the given corpus_tag.
    Collection name: papers_<corpus_tag>  (e.g. papers_arabidopsis).
    Reads MONGO_URI and MONGO_DB_NAME from environment.
    Called lazily — not at import time.
    """
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "llm_knowledge_discovery")

    if not mongo_uri:
        raise ValueError(f"{_MOD} MONGO_URI environment variable is required")

    client = MongoClient(mongo_uri)
    return client[db_name][f"papers_{corpus_tag}"]


def upsert_papers(records: list[PaperRecord], corpus_tag: str) -> dict:
    """
    Upsert PaperRecord objects into MongoDB using $setOnInsert semantics.
    Existing records are never overwritten — pure deduplication.

    Returns a summary dict: {"inserted": N, "skipped": M, "errors": K}
    """
    collection = _get_collection(corpus_tag)

    # Build all upsert operations and send in a single bulk request
    # $setOnInsert is a no-op if the document already exists
    operations = []
    for record in records:
        operations.append(
            UpdateOne(
                {"_id": record.pubmed_id},
                {
                    "$setOnInsert": {
                        "_id": record.pubmed_id,
                        **record.model_dump(exclude={"pubmed_id"}),
                    }
                },
                upsert=True,
            )
        )

    try:
        result = collection.bulk_write(operations, ordered=False)
        inserted = result.upserted_count
        skipped = len(records) - inserted
        errors = 0
    except Exception as e:
        logger.error(f"{_MOD} Bulk write failed: {e}")
        inserted = 0
        skipped = 0
        errors = len(records)

    logger.info(
        f"{_MOD} Upsert complete: "
        f"inserted={inserted}, skipped={skipped}, errors={errors}"
    )
    return {"inserted": inserted, "skipped": skipped, "errors": errors}
