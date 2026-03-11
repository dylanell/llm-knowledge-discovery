from datetime import datetime
from pydantic import BaseModel


class PaperRecord(BaseModel):
    pubmed_id: str  # PubMed UID as string (used as MongoDB _id)
    title: str
    abstract: str
    authors: list[str]  # "LastName Initials" format
    year: int | None  # None if unparseable
    journal: str | None  # None if missing
    corpus_tag: str  # caller-supplied label, e.g. "arabidopsis"
    fetched_at: datetime  # UTC timestamp
    raw_xml: str  # raw Entrez XML — preserved for future re-parsing
