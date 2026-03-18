# LLM Knowledge Discovery

Exploring knowledge extraction and discovery with LLMs. 

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Services](#services)
- [Datasets](#datasets)
- [Vector Store](#vector-store)
- [RAG Workflow](#rag-workflow)
- [RAGAS Evaluation](#ragas-evaluation)
- [Testing](#testing)
- [Todo](#todo)

## Project Structure

```
llm-knowledge-discovery/
├── doc/                        # Technical specs and design docs
├── notebooks/                  # Exploratory notebooks (numbered by phase)
│   ├── 0.corpus.ipynb          # Corpus retrieval walkthrough
│   ├── 1.vectorstore.ipynb     # Vector store setup and embedding walkthrough
│   ├── 2.rag.ipynb             # RAG chain, critic, and refinement loop walkthrough
│   └── 3.ragas_metrics.ipynb   # RAGAS evaluation metrics walkthrough
├── scripts/                    # Runnable entry points
│   ├── onboard_corpus.py       # Fetch PubMed abstracts → MongoDB
│   └── build_vectorstore.py    # Chunk abstracts → Chroma vector store
├── src/llm_knowledge_discovery/
│   ├── corpus/                 # PubMed fetching, MongoDB storage, data models
│   │   ├── models.py           # PaperRecord (Pydantic)
│   │   ├── pubmed.py           # Entrez API fetch
│   │   └── storage.py          # MongoDB upsert and load
│   ├── vectorstore/            # Chunking, embedding, and Chroma vector store
│   │   ├── chunking.py         # PaperRecord → LangChain Document
│   │   ├── reranking.py        # Cross-encoder reranking (ms-marco-MiniLM)
│   │   └── store.py            # Build and load Chroma stores
│   ├── rag/                    # RAG chain and critic
│   │   ├── models.py           # RagResult (Pydantic)
│   │   ├── chain.py            # build_rag_chain, invoke_and_review
│   │   └── critic.py           # critique_response, CritiqueResult
│   └── eval/                   # RAGAS evaluation metrics
│       ├── models.py           # Pydantic result models for all eval metrics
│       ├── faithfulness.py     # score_faithfulness (claim extraction + verification)
│       ├── answer_relevancy.py # score_answer_relevancy (synthetic query cosine similarity)
│       └── context_precision.py # score_context_precision (MAP-style weighted precision)
├── .github/workflows/          # GitHub Actions CI (ruff lint on PRs)
├── .claude/                    # Gitignored: Claude session notes and lessons
│   ├── lessons.md              # Accumulated corrections — Claude reads this on startup
│   ├── session_notes_<N>.md    # Per-session summaries of work done and TODOs
│   └── settings.json           # Claude hooks config (e.g. auto-load lessons on start)
├── .vscode/                    # Gitignored: VSCode workspace settings
│   └── settings.json           # Notebook root, ruler at col 80, ruff formatter
├── .venv/                      # Gitignored: Python virtual environment (managed by uv)
├── .data/                      # Gitignored: datasets and vector store files
├── .env.example                # Environment variable template (copy to .env and fill in)
├── docker-compose.yml          # MongoDB, Mongo Express, Neo4j
└── pyproject.toml              # Dependencies and tooling config
```

## Setup

### Claude

Create a `.claude/` directory to hold the following files:

- `lessons.md`: This is updated when you correct Claude about a mistake to note gotchas Claude should look out for. 

- `session_notes_<N>.md`: This is populated at the end of each session to summarize the additional work done in each session. 

- `settings.json`: Claude settings file that is read by default when claude starts up. Put the following in this file to ensure Claude reviews lessions and the session notes at start up automatically.

  ```json
  {
    "hooks": {
      "SessionStart": [
        {
          "matcher": "startup",
          "hooks": [
            {
              "type": "command",
              "command": "echo '## .claude/lessons.md' && cat .claude/lessons.md 2>/dev/null || echo 'none'; echo '## Latest Session Notes' && ls -t .claude/session_notes_*.md 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo 'none'"
            }
          ]
        }
      ]
    }
  }
  ```

### Python Project & Dependencies

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize as Python package (first time project setup)
# from directory containing this project directory (e.g. cd ../)
uv init --package llm-knowledge-discovery

# Create virtual environment manually (above does this as well)
uv venv
source .venv/bin/activate

# Sync dependencies and create the lock file
uv sync
```

### VSCode

`.vscode/` is gitignored. Create `.vscode/settings.json` with the following:

```json
{
  "jupyter.notebookFileRoot": "${workspaceFolder}",

  // PEP8: show vertical ruler at column 80
  "editor.rulers": [80],

  // Wrap at 80 characters
  "editor.wordWrapColumn": 80,

  // Auto-format on save using ruff (PEP8-compliant, fast)
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

Also install the [Ruff VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

## Services

All services run via Docker Compose. Copy `.env.example` to `.env` and fill in passwords before starting.

```bash
docker-compose up -d
```

| Service | Port | Purpose |
|---|---|---|
| MongoDB | `27017` | Stores the paper corpus (`papers_<corpus_tag>` collections) |
| Mongo Express | `8081` | Web UI for browsing MongoDB — `http://localhost:8081` |
| Neo4j | `7474` / `7687` | Graph DB for future knowledge graph phases (not yet used) |

**Chroma** (vector store) runs embedded in-process — no service needed. Data persists to `.data/vectorstore/`.

## Datasets

### Arabidopsis Corpus

PubMed abstracts related to *Arabidopsis thaliana* gene regulation, fetched via the Entrez API and stored in MongoDB under the `papers_arabidopsis` collection.

Before running, set `ENTREZ_EMAIL` in your `.env` file (no account needed — just a valid email for NCBI):

```
ENTREZ_EMAIL=you@example.com
```

Onboard the corpus by running the following script. Each query fetches up to 500 results; duplicate records across queries are automatically skipped.

```bash
uv run scripts/onboard_corpus.py \
    --collection-name arabidopsis_abstracts \
    --max-results-per-query 500 \
    --queries \
        "Arabidopsis thaliana transcription factor gene regulation" \
        "Arabidopsis thaliana chromatin remodeling epigenetics" \
        "Arabidopsis thaliana RNA sequencing transcriptome" \
        "Arabidopsis thaliana promoter binding gene expression" \
        "Arabidopsis thaliana stress response signaling pathway"
```

## Vector Store

Once the corpus is in MongoDB, build the Chroma vector store for semantic search:

```bash
uv run scripts/build_vectorstore.py \
    --collection-name arabidopsis_abstracts \
    --strategy whole_abstract \
    --persist-dir .data/vectorstore
```

This embeds all abstracts using `all-MiniLM-L6-v2` and persists the index to `.data/vectorstore/`. Only needs to be run once; subsequent RAG workflows load from disk via `load_vectorstore()`.

```python
from llm_knowledge_discovery.vectorstore import load_vectorstore

vectorstore = load_vectorstore(
    persist_dir=".data/vectorstore",
    collection_name="arabidopsis_abstracts",
)
```

## RAG Workflow

The RAG pipeline is implemented in `src/llm_knowledge_discovery/rag/` and demonstrated in `notebooks/2.rag.ipynb`.

### Basic chain

`build_rag_chain` returns an LCEL chain (`str → RagResult`) that handles the full pipeline in one call:

```
query
  → similarity search (retrieval_k=10 candidates)
  → cross-encoder rerank (rerank_k=5 kept)
  → ChatPromptTemplate (abstracts + question)
  → Claude (structured output)
  → RagResult(answer, references)
```

```python
from llm_knowledge_discovery.rag import build_rag_chain

chain = build_rag_chain(vectorstore)
rag_result = chain.invoke("What genes regulate flowering time in Arabidopsis?")
print(rag_result.answer)
print(rag_result.references)
```

### Critic + refinement loop

`invoke_and_review` wraps the pipeline with an optional self-correction loop. The initial answer is generated via structured output (`RagResult`) so cited references are available as a typed list without parsing. A critic LLM then audits the answer for three things: citation integrity, sequential numbering, and exact title accuracy. If the critique fails, the issues are fed back in a follow-up turn and the answer is regenerated — up to `review_steps` times.

```python
from llm_knowledge_discovery.rag import invoke_and_review

rag_result, critiques = invoke_and_review(
    query="What genes regulate flowering time in Arabidopsis?",
    vectorstore=vectorstore,
    review_steps=2,
)
print(rag_result.answer)
print(rag_result.references)
```

### Standalone retrieval

`retrieve_and_rerank` is also available directly — it's the same function the chain calls internally:

```python
from llm_knowledge_discovery.rag import retrieve_and_rerank

docs = retrieve_and_rerank(query, vectorstore)
```

## RAGAS Evaluation

The `eval/` subpackage implements three RAGAS metrics for evaluating RAG quality, demonstrated in `notebooks/3.ragas_metrics.ipynb`.

| Metric | Function | What it measures |
|---|---|---|
| Faithfulness | `score_faithfulness` | Are the answer's claims supported by the retrieved context? Detects hallucination. |
| Answer Relevancy | `score_answer_relevancy` | Does the answer actually address the question asked? |
| Context Precision | `score_context_precision` | Were the retrieved documents relevant? Rewards ranking relevant docs higher (MAP-style). |

```python
from llm_knowledge_discovery.rag import retrieve_and_rerank
from llm_knowledge_discovery.eval import (
    score_faithfulness,
    score_answer_relevancy,
    score_context_precision,
)

context_docs = retrieve_and_rerank(query, vectorstore)
rag_result = chain.invoke(query)

faithfulness = score_faithfulness(
    query=query, answer=rag_result.answer, context_docs=context_docs
)
relevancy = score_answer_relevancy(
    query=query, answer=rag_result.answer
)
precision = score_context_precision(
    query=query, answer=rag_result.answer, context_docs=context_docs
)

print(faithfulness.score, relevancy.score, precision.score)
```

All three scorers default to `temperature=0` for maximum determinism across benchmark runs.

## Testing

TODO

## Todo

- [ ] Phase 2.4 — Context recall evaluation (requires synthetic QA dataset)
- [ ] Phase 2.0 — Synthetic QA test set generation
- [ ] Phase 3 — RAG Delivery: eval bench runner, GitHub Actions gate, FastAPI endpoints, containerization
- [ ] Add test suite for corpus retrieval module (`src/llm_knowledge_discovery/corpus/`)