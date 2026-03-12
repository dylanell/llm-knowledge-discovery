# LLM Knowledge Discovery

Exploring knowledge extraction and discovery with LLMs. 

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Services](#services)
- [Datasets](#datasets)
- [Testing](#testing)
- [Todo](#todo)

## Project Structure

```
llm-knowledge-discovery/
├── doc/                        # Technical specs and design docs
├── notebooks/                  # Exploratory notebooks (numbered by phase)
│   ├── 0.corpus.ipynb          # Corpus retrieval walkthrough
│   └── 1.vectorstore.ipynb     # Vector store setup and embedding walkthrough
├── scripts/                    # Runnable entry points
│   ├── onboard_corpus.py       # Fetch PubMed abstracts → MongoDB
│   └── build_vectorstore.py    # Chunk abstracts → Chroma vector store
├── src/llm_knowledge_discovery/
│   ├── corpus/                 # PubMed fetching, MongoDB storage, data models
│   │   ├── models.py           # PaperRecord (Pydantic)
│   │   ├── pubmed.py           # Entrez API fetch
│   │   └── storage.py          # MongoDB upsert and load
│   └── vectorstore/            # Chunking and Chroma vector store
│       ├── chunking.py         # PaperRecord → LangChain Document
│       └── store.py            # Build and load Chroma stores
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

### Vector Store

Once the corpus is in MongoDB, build the Chroma vector store for semantic search:

```bash
uv run scripts/build_vectorstore.py \
    --collection-name arabidopsis_abstracts \
    --strategy whole_abstract \
    --persist-dir .data/vectorstore
```

This embeds all abstracts using `all-MiniLM-L6-v2` and persists the index to `.data/vectorstore/`. Only needs to be run once; subsequent RAG workflows load from disk via `load_vectorstore()`.

## Testing

TODO

## Todo

- [ ] Add test suite for corpus retrieval module (`src/llm_knowledge_discovery/corpus/`)