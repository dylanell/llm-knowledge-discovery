# LLM Knowledge Discovery

Exploring knowledge extraction and discovery with LLMs. 

## Table of Contents

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

## Project Structure

TODO

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
    --corpus-tag arabidopsis \
    --max-results-per-query 500 \
    --queries \
        "Arabidopsis thaliana transcription factor gene regulation" \
        "Arabidopsis thaliana chromatin remodeling epigenetics" \
        "Arabidopsis thaliana RNA sequencing transcriptome" \
        "Arabidopsis thaliana promoter binding gene expression" \
        "Arabidopsis thaliana stress response signaling pathway"
```

## Testing

TODO

## Todo

- [ ] Add test suite for corpus retrieval module (`src/llm_knowledge_discovery/corpus/`)