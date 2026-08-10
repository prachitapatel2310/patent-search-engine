# Patent Search Engine

A semantic + hybrid search engine over USPTO vehicle patent applications (2024–present).

---

## Demo

[Watch the demo video on Google Drive](https://drive.google.com/drive/u/0/folders/1M2BuWD5xYA6Wr-BcVhmFPbObn1QLvQRJ)

---

## Problem Statement

Patent examiners need to rapidly find patents related to a specific technology
(e.g., "EV battery thermal management") while filtering by classification code,
keywords, or exact title.  This engine provides:

- **Semantic search** — finds patents by meaning, not just keywords
- **Hybrid filters** — narrow by CPC/IPC classification prefix, keyword in
  title/abstract, or exact title
- **Two-phase re-ranking** — a cross-encoder improves ranking precision for
  the top candidates returned by the fast semantic search

---

## Project Structure

```
PSE/
├── search_engine.py     # Part 1 — core engine (embeddings + numpy ANN + hybrid filters)
├── reranker.py          # Part 3 — cross-encoder two-phase re-ranking
├── app.py               # Interactive CLI (Parts 1 & 3)
├── api_server.py        # Part 2 PoC — FastAPI REST server
├── Dockerfile           # Part 2 PoC — containerization
├── scale_design.md      # Part 2 — written system design
├── requirements.txt
├── patent_data/         # extracted from UTF-8patent_data_small.zip
│   └── data/patent_data_small/*.json
└── patent_index.pkl     # auto-generated embedding cache (created on first run)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.  Note: `faiss-cpu` does not build on Python 3.14,
so a numpy brute-force cosine index is used instead (see *Design Decisions*).

---

## How to Run

### Part 1 + Part 3 — Interactive CLI

```bash
# Semantic search only (Part 1)
python app.py

# With two-phase cross-encoder re-ranking (Part 3)
python app.py --rerank

# Force re-embed all patents (ignore cache)
python app.py --rebuild
```

On first run the engine embeds all 640 patents (~15 s).  Subsequent runs
load from `patent_index.pkl` in ~1 s.

**Example session:**
```
Query › electric vehicle battery thermal management
  Classification prefix: B60L
  Keyword: battery
  Number of results: 5
```

You can also enter an 11-digit doc_number as the query:
```
Query › 20240075768
```
The engine will look up that patent and use its full text as the query.

### Part 2 PoC — REST API (local)

```bash
pip install fastapi uvicorn
python api_server.py
# API docs: http://localhost:8000/docs
curl "http://localhost:8000/search?q=wheel+bearing+noise&classification=B60B&top_k=5"
```

### Part 2 PoC — Docker

```bash
# First build the index locally (creates patent_index.pkl)
python -c "from search_engine import get_engine; get_engine()"

# Build and run container
docker build -t patent-search .
docker run -p 8000:8000 \
  -v $(pwd)/patent_index.pkl:/app/patent_index.pkl \
  -v $(pwd)/patent_data:/app/patent_data \
  patent-search
```

---

## Design Decisions

### Missing Fields
Patents with empty/missing title, abstract, claims, or classification are
**included** in the index.  Missing fields are replaced with empty strings
so they are silently skipped during text concatenation.  No patents are
excluded.  This policy is documented in `search_engine.py`.

### Vector Index (FAISS → Numpy)
`faiss-cpu` does not compile on Python 3.14 (the installed version) due to
a SWIG/header incompatibility.  A numpy brute-force cosine similarity matrix
is used instead.  For 640 patents the ANN search takes < 5 ms — negligible.
At scale, swapping in FAISS or Pinecone requires changing only the
`NumpyIndex` class; the rest of the engine is unchanged.

### Embedding Model
`all-MiniLM-L6-v2` (384-dim, ~80 MB):
- Good balance of speed and quality for English text
- Encodes a patent's title + abstract + first 1 000 chars of claims
- At scale, `DAPT-patent-bert` or `PatentSBERTa` would improve recall

### Hybrid Filter Efficiency
At this scale (640 patents), the filters are applied as post-processing on the
ANN candidates.  Commentary in `search_engine.py` explains how at 10M patents
this should be replaced with a vector DB that applies metadata filters *inside*
the HNSW traversal (Pinecone/Weaviate), which is both faster and higher-recall.

### Two-Phase Re-ranking (Part 3)
`cross-encoder/ms-marco-MiniLM-L-6-v2`:
- Phase 1 retrieves 100 candidates with the bi-encoder (fast)
- Phase 2 scores each (query, patent_text) pair jointly with the
  cross-encoder (~200 ms on CPU for 100 pairs)
- Re-ranked results consistently place more relevant patents at rank 1–3

---

## Timing Benchmarks

Measured on a 2023 MacBook Pro M2, 640 patents:

| Mode | Encode (ms) | ANN (ms) | Filter (ms) | Total (ms) |
|------|-------------|----------|-------------|------------|
| Semantic only | 17 | 1 | 0 | 18 |
| + Classification filter | 17 | 1 | <1 | 18 |
| + Keyword filter | 17 | 1 | <1 | 18 |

At 640 patents the hybrid filter adds < 1 ms.  At 10M patents, the benefit
of pre-filtering (via vector DB native filters) would be a **reduction** in
ANN search time by limiting the search graph to the relevant subset.

Two-phase with cross-encoder adds ~200 ms for 100 candidates on CPU.

---

## Part 2 Scale Design

See [`scale_design.md`](scale_design.md) for:
- Full system architecture diagram
- Ingestion pipeline design
- Vector DB choice and schema
- Cost estimate (~$495/month for 10M patents)
- Error handling, observability, and major challenges
