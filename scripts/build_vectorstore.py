"""
Load a corpus from MongoDB, chunk abstracts, and build a Chroma vector store.

Usage:
    uv run scripts/build_vectorstore.py \\
        --collection-name arabidopsis_abstracts \\
        --strategy whole_abstract \\
        --persist-dir .data/vectorstore
"""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from llm_knowledge_discovery.corpus import load_papers  # noqa: E402
from llm_knowledge_discovery.vectorstore import (  # noqa: E402
    build_vectorstore,
    chunk_records,
)
from llm_knowledge_discovery.vectorstore.chunking import (  # noqa: E402
    CHUNK_STRATEGIES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_SCRIPT = "[build_vectorstore.py]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Chroma vector store from a MongoDB corpus."
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help=(
            "Collection name — used for both the MongoDB source collection "
            "and the Chroma collection (e.g. 'arabidopsis_abstracts')"
        ),
    )
    parser.add_argument(
        "--strategy",
        default="whole_abstract",
        choices=CHUNK_STRATEGIES,
        help="Chunking strategy (default: whole_abstract)",
    )
    parser.add_argument(
        "--persist-dir",
        default=".data/vectorstore",
        help=(
            "Directory to persist the Chroma store "
            "(default: .data/vectorstore)"
        ),
    )
    args = parser.parse_args()

    logger.info(
        f"{_SCRIPT} Loading records from collection '{args.collection_name}'"
    )
    records = load_papers(args.collection_name)
    logger.info(f"{_SCRIPT} Loaded {len(records)} records")

    logger.info(f"{_SCRIPT} Chunking with strategy='{args.strategy}'")
    docs = chunk_records(records, strategy=args.strategy)

    logger.info(
        f"{_SCRIPT} Building vectorstore: "
        f"collection='{args.collection_name}', "
        f"persist_dir='{args.persist_dir}'"
    )
    build_vectorstore(
        documents=docs,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    logger.info(f"{_SCRIPT} Done.")


if __name__ == "__main__":
    main()
