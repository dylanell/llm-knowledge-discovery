# LLM Knowledge Discovery: Plant Gene Regulation
## Technical Specification v1.2
**Date:** 2026-03-11
**Status:** Draft — Phase 1 Planning

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
| `arabidopsis` | *Arabidopsis thaliana* | **TAIR** (e.g., `AT1G65480`, alias `FT`) ← **Phase 1** |
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
- Non-PubMed corpus sources — Google Scholar (no API, ToS issues) and arXiv (wrong preprint server for plant biology) are not worth pursuing. **bioRxiv** is the relevant biology preprint server and a reasonable Phase 2 corpus expansion once PubMed baselines are established; preprints are excluded from the MVP to keep ground truth clean.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Literature Corpus                         │
│  PubMed (biopython/Entrez) → MongoDB (raw abstracts/papers) │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Extraction Pipeline                          │
│  LangChain LCEL: Prompt | Claude | PydanticOutputParser      │
│  Output: structured triples (entity, relation, entity, ctx) │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Knowledge Graph (Neo4j)                     │
│  Nodes: Gene, TF, Pathway, Phenotype, Condition, Experiment  │
│  Edges: REGULATES, EXPRESSED_IN, BINDS_PROMOTER_OF, etc.    │
│  Provenance: each edge links back to source paper(s)         │
└──────────────┬────────────────────────┬─────────────────────┘
               │                        │
               ▼                        ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│  Eval Track 1: RAG   │   │  Eval Track 2: Masked Prediction  │
│  (Knowledge          │   │  (Knowledge Discovery)            │
│   Extraction)        │   │                                   │
│  RAG + RAGAS         │   │  Edge masking → link prediction   │
└──────────────────────┘   └──────────────────────────────────┘
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

All stages use LangChain LCEL. Each stage is independently runnable and testable.

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

### 5.2 Extraction Chain

```
abstract_text
    → ExtractionPromptTemplate      # few-shot examples for plant gene regulation
    | ClaudeModel                   # claude-sonnet-4-6 or claude-opus-4-6
    | PydanticOutputParser          # → ExtractionResult(entities[], relationships[])
    → MongoDB: extractions_{species}
```

The extraction prompt instructs Claude to:
1. Identify all gene names, TF names, pathways, phenotypes, conditions
2. Extract all regulatory relationships with direction and evidence type
3. Note the experimental method used
4. Flag uncertainty explicitly (do not hallucinate relationships)

### 5.3 Graph Loader

```
ExtractionResult
    → entity deduplication (normalize gene names via species DB lookup)
    → Neo4j MERGE on Gene/TF/Condition nodes
    → Neo4j MERGE on relationship edges with provenance
```

Deduplication is critical: `FT`, `FLOWERING LOCUS T`, and `AT1G65480` must resolve to the same node. The TAIR API is used to look up canonical locus IDs and populate the `aliases[]` list at load time. When the TAIR API cannot resolve a name, the raw name is retained with a `tair_unresolved=True` flag for manual review.

### 5.4 Query Interface (later phase)

Natural language → Cypher query via LangChain Neo4j integration. Out of scope for Phase 1.

---

## 6. Evaluation Framework

### 6.1 Track 1: Knowledge Extraction (RAG + RAGAS)

**Goal:** Measure how faithfully the system can answer questions about gene regulation that are directly answerable from the corpus.

**Setup:**
1. Build a RAG retriever over the raw abstract corpus (vector embeddings in Chroma or Pinecone)
2. Construct a two-part QA benchmark (see below)
3. Run the RAG pipeline and evaluate with **RAGAS**

**RAGAS Metrics:**
| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer only state things the context supports? |
| Answer Relevancy | Is the answer relevant to the question? |
| Context Precision | Is the retrieved context actually useful? |
| Context Recall | Does the retrieval capture all relevant evidence? |

**Benchmark Construction — Two Parts:**

*Part A: Gold-standard anchor (external ground truth)*
Download the curated *Arabidopsis* TF-target dataset from **PlantRegMap** (plantregmap.gao-lab.org), which provides experimentally validated regulatory relationships with evidence codes. For each curated relationship, generate a natural-language question (e.g., "Does FT activate SOC1 in Arabidopsis?") with the expected answer derived from the curated record — not from our extraction. This anchors evaluation to an independent ground truth and catches systematic extraction errors that self-consistency metrics would miss.

*Part B: Corpus-derived pairs (coverage)*
For each extracted triple `(gene_A, ACTIVATES/REPRESSES/REGULATES, gene_B)` from the KG, generate a QA pair grounded in the source abstract. This tests breadth of coverage beyond the curated set.

RAGAS is run over both parts. Part A scores reflect accuracy against known ground truth; Part B scores reflect internal consistency of extraction.

**Benchmark construction:** Semi-automated. LLM generates candidate QA pairs; human spot-checks a sample, especially for Part A where answer correctness matters most.

### 6.2 Track 2: Knowledge Discovery (Masked Prediction)

**Goal:** Measure how well the system infers regulatory edges that are not explicitly stated in any retrieved context — i.e., genuine discovery from indirect signal.

**Setup:**
1. Build the full KG from the entire corpus
2. Randomly mask a held-out set of edges (e.g., 10–20% of `ACTIVATES`/`REPRESSES` edges) — removed from the KG before the discovery system sees it
3. Run the discovery system on the incomplete KG; it predicts missing edges
4. Evaluate predictions against the held-out ground truth

**Critical constraint: KG-only access**
The discovery system must operate solely on the knowledge graph — no access to raw abstracts or the vector store. This is essential to ensure the task is genuinely about inferring missing structure from graph topology and metadata, not re-reading the evidence that was used to build the graph. Allowing corpus access would let the system trivially recover masked edges by finding the original supporting text.

**Configurable Reference Mode**

The broader query/reasoning system (outside of the masked prediction eval) is configurable via a `reference_mode` parameter:

| Mode | Reference Source | Use Case |
|---|---|---|
| `corpus` | Raw abstract corpus (RAG only) | Baseline: what can we get from text alone? |
| `kg` | Knowledge graph only | Graph-based reasoning; used for discovery eval |
| `hybrid` | Both corpus + KG | Full system; expected best performance |

The masked prediction benchmark always uses `kg` mode to maintain eval integrity. The `hybrid` mode is the default for the deployed query interface.

**Masking Strategy:**
- Mask edges, not nodes (retain node existence)
- Stratify by `paper_count`: confidence threshold determined empirically from the distribution across all edges. Low-support edges excluded from test set (likely noisy); high-support edges are candidates for masking.
- Separate validation and test splits to avoid overfitting discovery heuristics

**Discovery Approaches (progressive):**
1. **Graph topology heuristics** — common neighbors, Jaccard similarity, Adamic-Adar (interpretable baseline)
2. **LLM-based reasoning** — prompt Claude with the partial KG neighborhood and node/edge metadata only; no raw text
3. **Embedding-based methods** — node2vec embeddings + cosine similarity; later graph neural network approaches (e.g., GraphSAGE, TransE for KG-specific embedding)

The progression from topology → LLM → embeddings → GNNs is deliberate: each step adds complexity and we want to understand what each buys us in terms of Hits@K before investing in more sophisticated models.

**Metrics:**
| Metric | Description |
|---|---|
| Hits@K | Is the true edge in the top-K predictions? |
| MRR | Mean reciprocal rank of correct predictions |
| Precision@K | Fraction of top-K predictions that are correct |
| AUC-ROC | Overall discrimination ability |

---

## 7. Implementation Phases

### Phase 1: Corpus + Extraction Pipeline
- [ ] Add LangChain deps (`langchain`, `langchain-anthropic`, `langchain-neo4j`, `ragas`, `chromadb`)
- [ ] Implement `src/corpus/` — PubMed retriever, parameterized by species
- [ ] Implement `src/extraction/` — LCEL extraction chain with Pydantic models
- [ ] Implement `src/graph/` — Neo4j loader with gene name deduplication
- [ ] Notebook: inspect first batch of extractions for one species

### Phase 2: RAG Evaluation (Track 1)
- [ ] Build vector index over abstract corpus
- [ ] Auto-generate QA benchmark from extracted triples
- [ ] Implement RAG pipeline
- [ ] Run RAGAS evaluation; establish baseline scores

### Phase 3: Masked Prediction Benchmark (Track 2)
- [ ] Implement edge masking framework
- [ ] Implement baseline link prediction (topology heuristics)
- [ ] Implement LLM-based edge prediction
- [ ] Evaluate against held-out masked edges; compute Hits@K, MRR

### Phase 4: Full Text Upgrade Study
- [ ] Add PMC Open Access full-text retrieval
- [ ] Re-run extraction pipeline on full text corpus
- [ ] Re-run both evaluation tracks (Track 1 + Track 2)
- [ ] Report delta in RAGAS scores and Hits@K vs. abstract-only baseline
- [ ] Cost/benefit analysis: additional compute + API cost vs. metric gains

### Phase 5: Iteration & Improvement
- [ ] Improve extraction prompts based on Track 1 failures
- [ ] Improve discovery methods based on Track 2 failures (embeddings, GNNs)
- [ ] Add second species for cross-species evaluation
- [ ] MLflow experiment tracking throughout

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Package management | uv |
| LLM | Anthropic Claude (claude-sonnet-4-6 for extraction, claude-opus-4-6 for reasoning) |
| LLM Framework | LangChain LCEL |
| Graph DB | Neo4j 5.14+ |
| Document Store | MongoDB 7.0 |
| Vector Store | Chroma (local) → Pinecone (if scale needed) |
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
| Full text | Added in Phase 4 as a controlled study; same metrics used for direct comparison |
| Ground truth confidence threshold | Determined empirically from the `paper_count` distribution after KG is built; not hardcoded upfront |
| Regulatory direction encoding | Typed relationships (`ACTIVATES`, `REPRESSES`, `REGULATES`); no separate `direction` property |
| Edge provenance | `paper_ids[]` + `paper_count` properties on each edge; no `SUPPORTED_BY` relationship (not possible in Neo4j) |
| Track 1 benchmark | Two-part: PlantRegMap gold-standard anchor (external ground truth) + corpus-derived pairs (coverage) |
| Track 2 discovery access | KG-only; corpus access would allow trivial recovery of masked edges from source text |
| Reference mode | Configurable: `corpus`, `kg`, or `hybrid`; masked prediction eval always uses `kg` |
| Discovery methods | Progressive: topology baselines → LLM (KG-only) → node2vec → GNNs; each evaluated independently |
