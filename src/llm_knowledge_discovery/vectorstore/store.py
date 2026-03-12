import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
_MOD = "[store.py]"

# Default embedding model — fast, good quality, standard in LangChain examples
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vectorstore(
    documents: list[Document],
    persist_dir: str,
    collection_name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Embed a list of LangChain Documents and store them in a persistent
    Chroma vector store on disk.

    Args:
        documents: LangChain Documents to embed and index
        persist_dir: Directory path where Chroma will persist its files
        collection_name: Logical name for this collection inside Chroma
        embedding_model: HuggingFace model name for sentence embeddings

    Returns:
        The populated Chroma vector store instance
    """
    logger.info(
        f"{_MOD} Building vectorstore: collection='{collection_name}', "
        f"model='{embedding_model}', docs={len(documents)}, "
        f"persist_dir='{persist_dir}'"
    )

    # Delete the collection if it already exists so re-runs don't accumulate
    # duplicate documents. Chroma.from_documents appends to existing
    # collections rather than replacing them.
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info(
            f"{_MOD} Deleted existing collection '{collection_name}'"
        )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # from_documents embeds all documents and indexes them in one call
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    logger.info(f"{_MOD} Vectorstore built and persisted to '{persist_dir}'")
    return vectorstore


def load_vectorstore(
    persist_dir: str,
    collection_name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Load an existing Chroma vector store from disk without re-embedding.

    Args:
        persist_dir: Directory where Chroma files are stored
        collection_name: Logical name of the collection to load
        embedding_model: HuggingFace model name (must match build-time model)

    Returns:
        The loaded Chroma vector store instance, ready for similarity search
    """
    logger.info(
        f"{_MOD} Loading vectorstore: collection='{collection_name}', "
        f"persist_dir='{persist_dir}'"
    )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    logger.info(f"{_MOD} Vectorstore loaded from '{persist_dir}'")
    return vectorstore
