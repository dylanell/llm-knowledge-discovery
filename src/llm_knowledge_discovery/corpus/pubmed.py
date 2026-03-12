import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import BytesIO

from Bio import Entrez

from .models import PaperRecord

logger = logging.getLogger(__name__)
_MOD = "[pubmed.py]"


def _configure_entrez() -> None:
    """
    Read ENTREZ_EMAIL and ENTREZ_API_KEY from env and configure Biopython's
    Entrez globals. Raises ValueError if ENTREZ_EMAIL is missing.
    """
    email = os.getenv("ENTREZ_EMAIL")
    if not email:
        raise ValueError(
            f"{_MOD} ENTREZ_EMAIL environment variable is required by NCBI"
        )

    Entrez.email = email
    api_key = os.getenv("ENTREZ_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _search_pubmed(query: str, max_results: int) -> list[str] | None:
    """
    Search PubMed and return a list of PubMed ID strings.
    Returns None on error (logged) so callers can distinguish failure from a
    legitimate empty result set. Assumes _configure_entrez() has been called.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    except Exception as e:
        logger.error(f"{_MOD} PubMed search failed for query '{query}': {e}")
        return None
    try:
        record = Entrez.read(handle)
        return list(record["IdList"])
    except Exception as e:
        logger.error(f"{_MOD} PubMed search failed for query '{query}': {e}")
        return None
    finally:
        handle.close()


def _parse_batch_xml(raw_xml: bytes, collection_name: str) -> list[PaperRecord]:
    """
    Parse a batch of PubMed XML records. Returns a list of PaperRecord objects,
    skipping any records that fail to parse.
    """
    try:
        parsed = Entrez.read(BytesIO(raw_xml))
    except Exception as e:
        logger.error(f"{_MOD} Failed to parse XML batch: {e}")
        return []

    articles = parsed.get("PubmedArticle", [])

    # Extract per-article XML strings so each record stores only its own XML.
    # Falls back to empty strings if ET parsing fails — valid parsed data is
    # never discarded just because raw XML extraction failed.
    try:
        tree = ET.fromstring(raw_xml)
        article_xml_strings = [
            ET.tostring(el, encoding="unicode")
            for el in tree.findall("PubmedArticle")
        ]
    except ET.ParseError as e:
        logger.warning(
            f"{_MOD} Failed to extract per-article XML, "
            f"storing empty raw_xml: {e}"
        )
        article_xml_strings = [""] * len(articles)

    # Order of Entrez.read articles and ET elements is guaranteed to match
    results = []
    for article, article_xml in zip(articles, article_xml_strings):
        record = _parse_record(article, article_xml, collection_name)
        if record is not None:
            results.append(record)

    return results


def _parse_record(
    article: dict, raw_xml: str, collection_name: str
) -> PaperRecord | None:
    """
    Extract fields from a biopython-parsed PubmedArticle dict and return a
    formatted PaperRecord object. Returns None on failure (logs the error).
    """
    try:
        citation = article["MedlineCitation"]
        article_data = citation["Article"]

        # Parse PubMed ID
        pubmed_id = str(citation["PMID"])

        # Parse title, removing XML tags
        title = re.sub(
            r"<[^>]+>", "", str(article_data.get("ArticleTitle", ""))
        )

        # Parse abstract, removing XML tags
        # AbstractText can be a list (structured abstract) or a plain string.
        # Strip HTML/XML tags (e.g. <i>, <sub>) — we only want plain text.
        abstract_raw = article_data.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abstract_raw, list):
            abstract = " ".join(str(part) for part in abstract_raw)
        else:
            abstract = str(abstract_raw)
        abstract = re.sub(r"<[^>]+>", "", abstract)

        # Parse list of author objects to list of author strings
        author_objs = article_data.get("AuthorList", [])
        authors = []
        for author_obj in author_objs:
            # Some entries are CollectiveName (org), not individual authors
            if "CollectiveName" in author_obj:
                authors.append(str(author_obj["CollectiveName"]))
            elif "LastName" in author_obj:
                last = author_obj["LastName"]
                initials = author_obj.get("Initials", "")
                authors.append(f"{last} {initials}".strip())

        # Parse year
        year = None
        pub_date = (
            article_data.get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
        )
        if "Year" in pub_date:
            try:
                year = int(pub_date["Year"])
            except (ValueError, TypeError):
                pass
        elif "MedlineDate" in pub_date:
            # MedlineDate looks like "2021 Jan-Feb" — grab first 4 chars
            try:
                year = int(str(pub_date["MedlineDate"])[:4])
            except (ValueError, TypeError):
                pass

        # Parse journal
        journal_data = article_data.get("Journal", {})
        journal = None
        if "Title" in journal_data:
            journal = str(journal_data["Title"])

        return PaperRecord(
            pubmed_id=pubmed_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            journal=journal,
            collection_name=collection_name,
            fetched_at=datetime.now(timezone.utc),
            raw_xml=raw_xml,
        )

    except Exception as e:
        logger.warning(f"{_MOD} Failed to parse record: {e}")
        return None


def fetch_records(
    query: str, max_results: int, collection_name: str, batch_size: int = 100
) -> list[PaperRecord]:
    """
    Search PubMed and fetch parsed records. No MongoDB interaction —
    independently testable.

    Args:
        query: Free-text PubMed search query (e.g. "BRCA1 breast cancer")
        max_results: Maximum number of results to fetch
        collection_name: MongoDB collection name (e.g. "arabidopsis_abstracts")
        batch_size: Number of records to fetch per Entrez request

    Returns:
        List of PaperRecord objects
    """
    _configure_entrez()
    rate = 10 if os.getenv("ENTREZ_API_KEY") else 3
    sleep_interval = 1.0 / rate

    logger.info(
        f"{_MOD} Searching PubMed: '{query}' "
        f"(max={max_results}, collection={collection_name})"
    )
    pubmed_ids = _search_pubmed(query, max_results)

    if pubmed_ids is None:
        return []  # error already logged in _search_pubmed
    if not pubmed_ids:
        logger.warning(f"{_MOD} Search returned 0 results for query.")
        return []

    logger.info(f"{_MOD} Found {len(pubmed_ids)} IDs, fetching records...")
    records = []
    for i in range(0, len(pubmed_ids), batch_size):
        batch = pubmed_ids[i : i + batch_size]
        ids_str = ",".join(batch)

        try:
            handle = Entrez.efetch(
                db="pubmed", id=ids_str, rettype="xml", retmode="xml"
            )
            try:
                raw_xml = handle.read()
            finally:
                handle.close()

            xml_records = _parse_batch_xml(raw_xml, collection_name)
            records.extend(xml_records)

        except Exception as e:
            logger.error(
                f"{_MOD} Failed to fetch batch starting at index {i}: {e}"
            )

        # Sleep between batches to stay within rate limit
        if i + batch_size < len(pubmed_ids):
            time.sleep(sleep_interval)

    logger.info(f"{_MOD} Fetched and parsed {len(records)} records.")
    return records
