# LLM Knowledge Discovery: Plant Gene Regulation
## Technical Specification v1.3
**Date:** 2026-03-12
**Status:** Active — Corpus Retrieval Complete, Beginning Text RAG (Phase 1)

---

## 1. Executive Summary

This system extracts, structures, and reasons over plant gene regulation knowledge from scientific literature. The initial scope is *Arabidopsis thaliana* with gene names normalized to TAIR identifiers. The architecture is parameterized by species to allow future expansion.

The MVP corpus is built from **PubMed abstracts only**. A key study goal is to quantify how much knowledge can be extracted from abstracts alone, then measure the cost/benefit of upgrading to full-text retrieval — using the same evaluation metrics to make the comparison rigorous.

The system pursues two distinct goals:

1. **Knowledge Extraction** — faithfully surface facts already present in the literature corpus, evaluated via RAG + RAGAS metrics.
2. **Knowledge Discovery** — infer regulatory relationships not explicitly stated in any single paper, evaluated via a masked-prediction benchmark on the knowledge graph.

---

## 2. Domain & Scope

### 2.1 Target Domain

Plant gene regulation encompasses:

- **Transcriptional regulation** — TF X activates/represses gene Y in condition Z
- **Differential expression** — gene W is up/down-regulated under treatment T or in tissue/developmental stage S
- **Knockout / loss-of-function experiments** — KO of gene G produces phenotype P
- **Protein-DNA binding** — TF X binds the promoter of gene Y
- **Signaling cascades** — upstream kinase/pathway activates downstream TF

All of these produce structured triples of the form `(entity_A, relationship, entity_B, context)` that populate the knowledge graph.

### 2.2 Species Parameterization

The corpus retrieval and ontology mapping are parameterized by `SPECIES`. Phase 1 targets *Arabidopsis thaliana* exclusively.

| Value | Organism | Gene Name Authority |
|---|---|---|
| `arabidopsis` | *Arabidopsis thaliana* | **TAIR** (e.g., `AT1G65480`, alias `FT`) ← **current** |
| `maize` | *Zea mays* | MaizeGDB |
| `rice` | *Oryza sativa* | RAP-DB |
| `tomato` | *Solanum lycopersicum* | SGN |

All gene names extracted from abstracts are normalized to their canonical TAIR locus ID (e.g., `FLOWERING LOCUS T`, `FT`, `ft-1` all resolve to `AT1G65480`). Common aliases are maintained as a `aliases[]` property on the node.

### 2.3 Abstract vs. Full Text (Planned Study)

The MVP uses **PubMed abstracts only**. After evaluation baselines are established, full-text retrieval will be added (via PubMed Central Open Access) and the same metrics recomputed. This produces a direct cost/benefit comparison:

- **Abstracts:** free, fast, widely available, lower information density
- **Full text:** higher cost (API access or PMC scraping), slower, richer context

The delta in RAGAS scores and Hits@K between the two corpus types is a primary research output of this project.

### 2.4 Out of Scope (for now)

- Post-translational regulation (phosphorylation, ubiquitination)
- Epigenetic mechanisms (methylation, histone modification)
- Multi-species comparative genomics
- Real-time inference serving
- Non-PubMed corpus sources — Google Scholar (no API, ToS issues) and arXiv (wrong preprint server for plant biology) are not worth pursuing. **bioRxiv** is the relevant biology preprint server and a reasonable future corpus expansion once PubMed baselines are established; preprints are excluded from the MVP to keep ground truth clean.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Literature Corpus                           │
│   PubMed (biopython/Entrez) → MongoDB (raw abstracts/papers)    │
└──────────────────┬──────────────────────────┬───────────────────┘
                   │                          │
                   ▼                          ▼
┌─────────────────────────────┐  ┌────────────────────────────────┐
│       RAG Pipeline           │  │      Extraction Pipeline        │
│  Chunking → Vector DB        │  │  LCEL: Prompt | Claude          │
│  KNN Retrieval + Reranking   │  │  | PydanticOutputParser         │
└──────────────┬──────────────┘  │  → ExtractionResult (triples)   │
               │                 └──────────────┬───────────────────┘
               ▼                                │
┌─────────────────────────────┐                 ▼
│   Text RAG Evaluation        │  ┌────────────────────────────────┐
│   RAGAS: faithfulness,       │  │     Knowledge Graph (Neo4j)     │
│   answer relevancy,          │  │  Nodes: Gene, TF, Pathway, …    │
│   context precision,         │  │  Edges: ACTIVATES, REPRESSES, … │
│   context recall             │  │  Provenance on each edge         │
└─────────────────────────────┘  └──────────────┬───────────────────┘
                                                 │
                          ┌──────────────────────┼─────────────────────┐
                          │                      │                     │
                          ▼                      ▼                     ▼
               ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
               │   KG Query      │  │  Extraction Eval  │  │   Discovery Eval      │
               │   Engine        │  │  (known facts,    │  │   (masked facts,      │
               │   (NL → Cypher) │  │   RAGAS metrics)  │  │   Hits@K, MRR)        │
               └─────────────────┘  └──────────────────┘  └──────────────────────┘
```

---

## 4. Knowledge Graph Schema

### 4.1 Node Types

| Label | Description | Key Properties |
|---|---|---|
| `Gene` | Any gene locus | `name`, `species`, `locus_id`, `aliases[]` |
| `TranscriptionFactor` | TF (subtype of Gene) | inherits Gene + `tf_family` |
| `Protein` | Protein product of a gene | `name`, `uniprot_id` |
| `Pathway` | Biological pathway | `name`, `pathway_db_id` |
| `Phenotype` | Observable trait or outcome | `name`, `description`, `measurement_type` (`quantitative`, `qualitative`, `binary`) |
| `Condition` | Experimental condition or tissue/stage | `name`, `condition_type` (`tissue`, `treatment`, `developmental_stage`, `stress`) |
| `Experiment` | Source experiment | `type` (`KO`, `overexpression`, `RNA-seq`, `ChIP-seq`, `Y1H`, etc.), `paper_id` |
| `Paper` | Source publication | `pubmed_id`, `title`, `year`, `abstract` |

### 4.2 Relationship Types

Regulatory direction is encoded in the relationship type itself (not a property) for clean Cypher querying. Provenance is stored as `paper_ids[]` / `paper_count` directly on each edge — Neo4j does not support edges pointing to other edges, so there is no separate provenance relationship.

| Type | From → To | Description |
|---|---|---|
| `ACTIVATES` | Gene/TF → Gene | Explicit transcriptional activation |
| `REPRESSES` | Gene/TF → Gene | Explicit transcriptional repression |
| `REGULATES` | Gene/TF → Gene | Direction unknown or unclear from evidence |
| `BINDS_PROMOTER_OF` | TF/Protein → Gene | Physical DNA binding evidence |
| `DIFFERENTIALLY_EXPRESSED_IN` | Gene → Condition | Up/down-regulation in a context |
| `PART_OF` | Gene → Pathway | Pathway membership |
| `PRODUCES_PHENOTYPE` | Experiment → Phenotype | Experimental outcome — carries direction and magnitude (see §4.3) |
| `TARGETS` | Experiment → Gene | The gene under study |

### 4.3 Edge Properties

**All edges** carry document support metadata:
- `paper_ids[]`: list of supporting PubMed IDs
- `paper_count`: integer count of supporting papers (derived from `paper_ids[]`; used for confidence thresholding and masked prediction stratification)

**Regulatory edges** (`ACTIVATES`, `REPRESSES`, `REGULATES`, `BINDS_PROMOTER_OF`, `DIFFERENTIALLY_EXPRESSED_IN`) additionally carry:
- `evidence_type`: `direct` / `indirect` / `inferred`
- `confidence`: float 0–1 (LLM-assigned per extraction)
- `condition_id`: FK to Condition node

Note: direction is encoded in the relationship type itself (`ACTIVATES` vs `REPRESSES` vs `REGULATES`), so no separate `direction` property is needed.

**`PRODUCES_PHENOTYPE` edges** carry explicit directionality for the phenotypic outcome:
- `perturbation_type`: `KO` / `overexpression` / `RNAi` / `natural_variant` / `other`
- `phenotype_direction`: `increase` / `decrease` / `no_change` / `ectopic` / `loss` / `gain` / `unknown`
  - e.g., KO of FT → `decrease` in flowering time trait
- `phenotype_magnitude`: `strong` / `moderate` / `weak` / `unknown` (qualitative scale; quantitative values go in `phenotype_value`)
- `phenotype_value`: optional float — the actual measured value if reported (e.g., days-to-flowering)
- `phenotype_unit`: optional string — unit for `phenotype_value` (e.g., `days`, `cm`, `fold-change`)
- `confidence`: float 0–1 (LLM-assigned)
- `condition_id`: FK to Condition node

This allows queries like: *"Which genes, when knocked out, strongly decrease hypocotyl length?"*

---

## 5. Pipeline Design

All LLM stages use LangChain LCEL. Each stage is independently runnable and testable.

### 5.1 Corpus Retrieval

```python
# Parameterized PubMed query builder
query = build_pubmed_query(species="arabidopsis", topics=["gene regulation", "transcription factor", "differential expression", "knockout"])
# → fetches abstracts + metadata via biopython Entrez
# → stores raw documents in MongoDB collection: papers_{species}
```

Query construction targets papers that include:
- TF-target relationships
- Differential expression studies (RNA-seq, microarray)
- KO / T-DNA insertion phenotype studies
- ChIP-seq / Y1H binding data

Initial target: **500–1000 abstracts** per species (sufficient ground truth for evaluation).

**Status: Complete.** Arabidopsis corpus loaded in MongoDB (`papers_arabidopsis`).

### 5.2 RAG Pipeline

```
abstract_text (from MongoDB)
    → ChunkingStrategy              # whole-abstract or sentence-level (see §9)
    → EmbeddingModel                # e.g., text-embedding-3-small or equivalent
    → VectorDB (Chroma)             # indexed for KNN retrieval
    ↓
user_query
    → KNN retrieval (top-K chunks)
    → CrossEncoderReranker          # rerank to top-N most relevant
    → RAGPromptTemplate             # assembled context + question
    | ClaudeModel                   # claude-sonnet-4-6
    → answer
```

The retrieval step uses approximate KNN for candidate recall, then a cross-encoder reranker to refine the ranking before assembling the final context window. This two-stage approach is standard for production RAG and catches cases where the embedding model's coarse similarity ranking misorders semantically close chunks.

### 5.3 Extraction Chain

```
abstract_text
    → ExtractionPromptTemplate      # few-shot examples for plant gene regulation
    | ClaudeModel                   # claude-sonnet-4-6 or claude-opus-4-6
    | PydanticOutputParser          # → ExtractionResult(entities[], relationships[])
    → self-review step              # LLM checks its own output for consistency/hallucination
    → MongoDB: extractions_{species}
```

The extraction prompt instructs Claude to:
1. Identify all gene names, TF names, pathways, phenotypes, conditions
2. Extract all regulatory relationships with direction and evidence type
3. Note the experimental method used
4. Flag uncertainty explicitly (do not hallucinate relationships)

The self-review step passes the extraction result back to the LLM with a critique prompt, which returns a corrected or confirmed version before persisting to MongoDB.

### 5.4 Graph Loader

```
ExtractionResult
    → entity deduplication (normalize gene names via species DB lookup)
    → Neo4j MERGE on Gene/TF/Condition nodes
    → Neo4j MERGE on relationship edges with provenance
```

Deduplication is critical: `FT`, `FLOWERING LOCUS T`, and `AT1G65480` must resolve to the same node. The TAIR API is used to look up canonical locus IDs and populate the `aliases[]` list at load time. When the TAIR API cannot resolve a name, the raw name is retained with a `tair_unresolved=True` flag for manual review.

### 5.5 KG Query Engine

Natural language → Cypher query via LangChain Neo4j integration. Implemented in Phase 5.

---

## 6. Evaluation Framework

### 6.1 Track 1: Knowledge Extraction (Text RAG + RAGAS)

**Goal:** Measure how faithfully the system can answer questions about gene regulation that are directly answerable from the corpus.

**Setup:**
1. Build the RAG pipeline over the raw abstract corpus (Phase 1)
2. Construct a QA benchmark (see below)
3. Run the RAG pipeline and evaluate with **RAGAS** (Phase 2)

**RAGAS Metrics:**
| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer only state things the context supports? |
| Answer Relevancy | Is the answer relevant to the question? |
| Context Precision | Is the retrieved context actually useful? |
| Context Recall | Does the retrieval capture all relevant evidence? |

**Benchmark Construction:**

*Phase 2 — Synthetic QA pairs (initial baseline)*
Use an LLM to generate Q&A pairs directly from the abstract corpus. This provides immediate coverage for RAGAS evaluation before the KG is built. Questions target factual regulatory claims stated in the abstracts (e.g., "What does FT regulate in Arabidopsis?").

*Phase 2 — PlantRegMap gold-standard anchor (external ground truth)*
Download the curated *Arabidopsis* TF-target dataset from **PlantRegMap** (plantregmap.gao-lab.org), which provides experimentally validated regulatory relationships with evidence codes. Generate natural-language questions for each curated relationship (e.g., "Does FT activate SOC1 in Arabidopsis?") with expected answers derived from the curated record — not from our extraction. This anchors evaluation to an independent ground truth and catches systematic errors that self-consistency metrics would miss.

*Phase 5 — KG-derived pairs (extended benchmark)*
For each extracted triple `(gene_A, ACTIVATES/REPRESSES/REGULATES, gene_B)` from the KG, generate a QA pair grounded in the source abstract. Added after Phase 3 (extraction) is complete. Tests breadth of coverage beyond the curated set and feeds directly into the text RAG vs. KG RAG comparison.

RAGAS is run over all parts. PlantRegMap scores reflect accuracy against known ground truth; synthetic and KG-derived scores reflect internal consistency of extraction.

### 6.2 Track 2: Knowledge Discovery (KG RAG + Masked Prediction)

**Goal:** Measure how well the system answers questions and infers regulatory edges using only the knowledge graph — no access to raw text.

**Setup:**
1. Build the full KG from the entire corpus (Phase 4)
2. Build evaluation dataset: known high-confidence facts + held-out masked edges (Phase 5)
3. Run the KG query engine on known facts; evaluate with RAGAS metrics (Phase 5)
4. Randomly mask a held-out set of edges (10–20% of `ACTIVATES`/`REPRESSES` edges); run discovery system on incomplete KG; evaluate against held-out ground truth (Phase 5)

**Critical constraint: KG-only access**
The discovery system must operate solely on the knowledge graph — no access to raw abstracts or the vector store. This is essential to ensure the task is genuinely about inferring missing structure from graph topology and metadata, not re-reading the evidence that was used to build the graph.

**Configurable Reference Mode**

| Mode | Reference Source | Use Case |
|---|---|---|
| `corpus` | Raw abstract corpus (RAG only) | Baseline: what can we get from text alone? |
| `kg` | Knowledge graph only | Graph-based reasoning; used for discovery eval |
| `hybrid` | Both corpus + KG | Full system; expected best performance |

The masked prediction benchmark always uses `kg` mode to maintain eval integrity.

**Masking Strategy:**
- Mask edges, not nodes (retain node existence)
- Stratify by `paper_count`: high-support edges are candidates for masking; low-support edges excluded from test set (likely noisy)
- Separate validation and test splits to avoid overfitting discovery heuristics

**Discovery Approaches (progressive):**
1. **Graph topology heuristics** — common neighbors, Jaccard similarity, Adamic-Adar (interpretable baseline)
2. **LLM-based reasoning** — prompt Claude with the partial KG neighborhood and node/edge metadata only; no raw text
3. **Embedding-based methods** — node2vec embeddings + cosine similarity; later graph neural network approaches (e.g., GraphSAGE, TransE)

**Metrics:**
| Metric | Description |
|---|---|
| Hits@K | Is the true edge in the top-K predictions? |
| MRR | Mean reciprocal rank of correct predictions |
| Precision@K | Fraction of top-K predictions that are correct |
| AUC-ROC | Overall discrimination ability |

**Head-to-head comparison (Phase 5)**
Text RAG (Phase 1–2) and KG RAG (Phase 5) are evaluated on the same RAGAS metrics over the same question set. This direct comparison is a primary research output of the project.

---

## 7. Implementation Phases

### Completed: Corpus Retrieval

- [x] Set up local services (MongoDB, Neo4j via Docker Compose)
- [x] Implement `src/corpus/` — PubMed retriever, parameterized by species
- [x] Populate Arabidopsis corpus: `papers_arabidopsis` (MongoDB)

### Phase 1: Text RAG

- [x] 1.0 Decide vector DB and chunking strategy (whole-abstract vs. sentence-level)
- [x] 1.1 Chunk abstracts and populate vector DB
- [x] 1.2 KNN retrieval + cross-encoder reranking
- [x] 1.3 Wire into RAG prompt (Anthropic API via LangChain)

### Phase 2: RAGAS Evaluation of Text RAG

- [ ] 2.0 Synthetic QA test set generation (LLM-generated Q&A from abstracts)
- [x] 2.1 Faithfulness evaluation
- [ ] 2.2 Answer relevancy evaluation
- [ ] 2.3 Context precision evaluation
- [ ] 2.4 Context recall evaluation

### Phase 3: RAG Delivery (after Phase 2)

Gate delivery of a RAG version behind an automated eval bench, then expose the
passing version as a REST API.

**Eval bench**
- [ ] 3.1 Implement an eval runner script that runs the full Phase 2 RAGAS workflow
      (faithfulness, answer relevancy, context precision, context recall) over a
      fixed question set and writes scores to a results artifact
- [ ] 3.2 Define per-metric delivery thresholds (e.g. faithfulness ≥ 0.85,
      answer relevancy ≥ 0.80) in a config file; the runner fails with a non-zero
      exit code if any threshold is not met
- [ ] 3.3 Wire the eval runner into a GitHub Actions workflow that triggers on PRs
      to `main`; a failing eval blocks merge

**API**
- [ ] 3.4 FastAPI app with a `/query` endpoint accepting a question, returning `RagResult`
- [ ] 3.5 `/query/review` endpoint using `invoke_and_review` with configurable `review_steps`
- [ ] 3.6 Containerize alongside existing services in `docker-compose.yml`

**Versioned delivery**
- [ ] 3.7 Tag a RAG version (model, retrieval_k, rerank_k, prompt hash) in a config
      file; the API reads this config at startup so a specific eval-passing version
      is always what gets served

> **Future:** extend the GitHub Actions workflow to compare new eval scores against
> the currently deployed version's scores. If all metrics improve (or meet a
> configurable "better-than-baseline" threshold), automatically upsert the deployment
> config to point to the new version — making promotion fully automatic for
> improvements and still blocking regressions.

### Phase 4: Extraction Pipeline (can run in parallel with Phase 3)

- [ ] 4.1 Abstract → entity/relationship extraction (LCEL chain, Pydantic models)
- [ ] 4.2 Self-review feedback loop on extraction output
- [ ] 4.3 Publish entities/relationships to Neo4j KG (with gene name deduplication via TAIR)

### Phase 5: KG RAG + Evaluation

- [ ] 5.0 Build evaluation dataset (known high-confidence facts + masked facts)
- [ ] 5.1 KG query engine (natural language → Cypher via LangChain Neo4j integration)
- [ ] 5.2 Knowledge extraction evaluation (RAGAS metrics on KG-grounded questions)
- [ ] 5.3 Knowledge discovery evaluation (masked facts, Hits@K, MRR)
- [ ] 5.4 Head-to-head comparison: text RAG (Phase 1–2) vs. KG RAG (Phase 5)

### Phase 6: Full Text Upgrade Study

- [ ] Add PMC Open Access full-text retrieval
- [ ] Re-run extraction pipeline on full text corpus
- [ ] Re-run both evaluation tracks (Track 1 + Track 2)
- [ ] Report delta in RAGAS scores and Hits@K vs. abstract-only baseline
- [ ] Cost/benefit analysis: additional compute + API cost vs. metric gains

### Phase 7: Iteration & Improvement

- [ ] Improve extraction prompts based on Phase 5 failures
- [ ] Add embedding-based link prediction (node2vec, GraphSAGE)
- [ ] Add second species for cross-species evaluation
- [ ] MLflow experiment tracking throughout

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Package management | uv |
| LLM | Anthropic Claude (claude-sonnet-4-6 for extraction/RAG, claude-opus-4-6 for reasoning) |
| LLM Framework | LangChain LCEL |
| Graph DB | Neo4j 5.14+ |
| Document Store | MongoDB 7.0 |
| Vector Store | Chroma (local dev) → Pinecone (if scale needed) |
| Reranker | Cross-encoder (HuggingFace `cross-encoder/ms-marco-*`) |
| RAG Evaluation | RAGAS |
| Literature Retrieval | biopython (Entrez/PubMed) |
| Data Validation | Pydantic v2 |
| Experiment Tracking | MLflow |
| Containerization | Docker Compose |
| Testing | pytest |

---

## 9. Resolved Decisions

| Decision | Resolution |
|---|---|
| Species for Phase 1 | *Arabidopsis thaliana* |
| Gene name authority | TAIR (canonical locus IDs); aliases retained on node |
| Corpus MVP | PubMed abstracts only |
| Full text | Added in Phase 5 as a controlled study; same metrics used for direct comparison |
| Ground truth confidence threshold | Determined empirically from the `paper_count` distribution after KG is built; not hardcoded upfront |
| Regulatory direction encoding | Typed relationships (`ACTIVATES`, `REPRESSES`, `REGULATES`); no separate `direction` property |
| Edge provenance | `paper_ids[]` + `paper_count` properties on each edge; no `SUPPORTED_BY` relationship (not possible in Neo4j) |
| Track 1 benchmark | Two-part: synthetic QA pairs (Phase 2 baseline) + PlantRegMap gold-standard anchor (Phase 2) + KG-derived pairs (Phase 5 extended benchmark) |
| Track 2 discovery access | KG-only; corpus access would allow trivial recovery of masked edges from source text |
| Reference mode | Configurable: `corpus`, `kg`, or `hybrid`; masked prediction eval always uses `kg` |
| Discovery methods | Progressive: topology baselines → LLM (KG-only) → node2vec → GNNs; each evaluated independently |
| Chunking strategy | Whole-abstract as default chunk unit (abstracts are 150–300 words; sub-abstract chunking may be added if retrieval quality is poor) |
| Vector store | Chroma for local dev; MongoDB Atlas Vector Search considered to avoid a new service but Chroma is simpler for local iteration |
| Reranker | HuggingFace cross-encoder (`cross-encoder/ms-marco-*`); avoids external API dependency |
