"""
Fetch PubMed abstracts and store them in MongoDB for one or more queries.

Usage:
    uv run scripts/onboard_corpus.py \\
        --corpus-tag arabidopsis \\
        --max-results-per-query 500 \\
        --queries "Arabidopsis thaliana transcription factor" \\
                  "Arabidopsis gene regulation chromatin" \\
                  "Arabidopsis thaliana RNA sequencing"
"""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from llm_knowledge_discovery.corpus import fetch_and_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_SCRIPT = "[onboard_corpus.py]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Onboard a PubMed corpus into MongoDB."
    )
    parser.add_argument(
        "--corpus-tag",
        required=True,
        help="Label for this corpus (e.g. 'arabidopsis')",
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        required=True,
        help="Max results to fetch per query",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="One or more PubMed search queries",
    )
    args = parser.parse_args()

    totals = {"inserted": 0, "skipped": 0, "errors": 0}

    for i, query in enumerate(args.queries, start=1):
        logger.info(f"{_SCRIPT} Query {i}/{len(args.queries)}: '{query}'")
        result = fetch_and_store(
            query, args.max_results_per_query, args.corpus_tag
        )
        ins = result["inserted"]
        skp = result["skipped"]
        err = result["errors"]
        logger.info(f"{_SCRIPT} inserted={ins}, skipped={skp}, errors={err}")
        for key in totals:
            totals[key] += result[key]

    ins = totals["inserted"]
    skp = totals["skipped"]
    err = totals["errors"]
    logger.info(
        f"{_SCRIPT} Done. Total: inserted={ins}, skipped={skp}, errors={err}"
    )


if __name__ == "__main__":
    main()
