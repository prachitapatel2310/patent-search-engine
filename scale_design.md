# Patent Search Engine — Scale Design (Part 2)

## Problem Statement

Scale the patent search engine from Part 1 to handle the full USPTO/EPO corpus
of ~10 million patents, with new patents ingested weekly and search latency
under 500 ms for the end user.

---

## System Overview

```
┌──────────────┐      ┌───────────────┐      ┌────────────────────┐
│  Data Source │─────▶│  Ingestion    │─────▶│  Vector Database   │
│  (XML feeds/ │      │  Pipeline     │      │  (Pinecone / Weaviate│
│   raw JSONs) │      │               │      │   + Postgres meta) │
└──────────────┘      └───────────────┘      └────────────┬───────┘
                                                           │
                                             ┌─────────────▼──────┐
                                             │   Search API        │
                                             │  (FastAPI + Redis)  │
                                             └────────────┬───────┘
                                                          │
                                             ┌────────────▼───────┐
                                             │  Re-ranking Layer   │
                                             │  (Cross-encoder)   │
                                             └────────────┬───────┘
                                                          │
                                             ┌────────────▼───────┐
                                             │  Frontend / API     │
                                             │  (REST / UI)        │
                                             └────────────────────┘
```

---

## Components

### 1. Data Ingestion Pipeline

**Trigger:** Weekly batch job (USPTO publishes new IPA applications every Thursday)

**Steps:**
1. **Download** — Pull new XML patent dumps from USPTO bulk data endpoint
2. **Parse** — Extract title, abstract, claims, description, classification,
   doc_number using a streaming XML parser (lxml SAX mode to avoid loading
   full files into memory)
3. **Validate** — Reject records with no doc_number; log missing fields rather
   than dropping records (same policy as Part 1)
4. **Embed** — Run `all-MiniLM-L6-v2` (or a patent-specific model like
   `DAPT-patent-bert`) in batches of 512 on GPU; output 384-dim float32 vectors
5. **Upsert** — Write vectors + metadata to the vector DB; write full patent
   text to Postgres for keyword search and display

**Tech:** Apache Airflow DAG, AWS Batch for GPU embedding workers,
S3 as intermediate storage.

**Estimated throughput:** 10M patents × ~2 s / 512 patents on 1× A10G GPU
≈ 11 hours of GPU time for initial backfill.  Weekly delta (~50k patents)
≈ 3 minutes.

---

### 2. Vector Database

**Choice:** Pinecone (managed) or Weaviate (self-hosted)

**Why:** Both support:
- Filtered ANN search (classification prefix, date range) without post-hoc
  filtering — the filter is applied inside the ANN traversal, which is the
  key efficiency win vs. the naïve approach in Part 1.
- Horizontal sharding across nodes
- Real-time upsert

**Schema / payload per vector:**
```json
{
  "id": "20240075768",
  "values": [0.12, -0.03, ...],       // 384-dim embedding
  "metadata": {
    "title": "...",
    "classification": "B60R2246FI",
    "abstract_snippet": "...",
    "filing_date": "2024-03-07"
  }
}
```

**Index type:** HNSW (Hierarchical Navigable Small World) — sub-linear ANN
search, better recall/speed trade-off than IVF at 10M scale.

---

### 3. Relational Metadata Store (Postgres)

Stores full patent text for:
- Keyword / full-text search (`tsvector` + GIN index)
- Exact title lookup
- Displaying full claims and description in search results
- Cross-encoder phase 2 (fetches full text by doc_number)

**Estimated size:** 10M patents × ~5 KB avg text = ~50 GB (fits on a single
RDS `db.m6g.2xlarge` with 100 GB storage, ~$250/month)

---

### 4. Search API

**Stack:** FastAPI + Uvicorn, containerized with Docker, deployed on ECS Fargate

**Request flow:**
```
1. Receive query + optional filters
2. Redis cache check (TTL = 1 hour) → return cached result if hit
3. Embed query (single forward pass ~20 ms on CPU)
4. Vector DB filtered ANN search → 100 candidates (~50 ms)
5. Fetch full texts from Postgres for top-100 (~20 ms)
6. Cross-encoder re-rank → top-10 (~200 ms on CPU, ~30 ms on GPU)
7. Return JSON; write to Redis cache
```

**Target p99 latency:** ~350 ms (CPU) / ~120 ms (GPU re-ranker)

---

### 5. Hybrid Filter Efficiency

**The problem with naïve post-filtering:**
Retrieve N=1000 candidates, filter by classification → might keep only 5,
wasting retrieval budget and requiring re-queries to fill top-k.

**The solution:** Vector DB native metadata pre-filter.  Pinecone and Weaviate
apply the filter inside the HNSW graph traversal, so the ANN search only
visits nodes that satisfy the filter.  This is both faster and higher-recall
than post-filtering.

---

## Cost Estimate (Monthly, 10M patents)

| Component | Config | Est. Cost/month |
|-----------|--------|-----------------|
| Pinecone (Serverless) | 10M × 384-dim | ~$70 |
| Postgres RDS | db.m6g.2xlarge, 100 GB | ~$250 |
| API (ECS Fargate) | 2 vCPU / 4 GB × 2 replicas | ~$60 |
| Re-ranker (ECS + GPU) | 1× g4dn.xlarge spot | ~$80 |
| Airflow + Batch (weekly ingestion) | Spot GPU, ~3 min/week | ~$5 |
| Redis (ElastiCache) | cache.t3.medium | ~$30 |
| **Total** | | **~$495/month** |

*Costs drop significantly with reserved instances (~30–40% savings).*

---

## Error Handling & Observability

- **Ingestion failures:** each patent is processed independently; parse errors
  are logged to CloudWatch and written to a `failed_patents` Postgres table for
  manual review.  The DAG retries 3× before alerting.
- **Embedding worker crash:** Airflow marks the task failed; the batch job
  can re-run from the last S3 checkpoint (patents are keyed by doc_number).
- **Vector DB downtime:** API falls back to Postgres full-text search (lower
  quality but always available).
- **Stale cache:** Redis TTL of 1 hour; cache invalidated on upsert for
  doc_numbers appearing in the new batch.
- **Monitoring:** Prometheus metrics (query latency, cache hit rate, ingestion
  lag) → Grafana dashboard.

---

## Major Challenges at Scale

1. **Patent XML parsing is messy** — USPTO XML schema has changed multiple
   times; a robust parser needs schema-version detection.
2. **Embedding cost for backfill** — 10M patents × ~0.004 s/patent = 11 GPU
   hours.  Use chunked checkpointing so crashes don't restart from scratch.
3. **Model drift** — If the embedding model is updated, all vectors must be
   re-computed.  Maintain a `model_version` field in the vector DB and run
   rolling re-embeds.
4. **Non-English patents** — EPA, JPO filings are in German, Japanese, etc.
   Use a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`) or
   translate at ingest time.
5. **Legal sensitivity** — Search results may affect IP decisions; audit logs
   of all queries should be retained for compliance.

---

## Proof-of-Concept: Containerized Search API (see Dockerfile)

The `Dockerfile` in this repo packages the Part 1 search engine into a
FastAPI container.  It demonstrates:
- Correct Python + dependency packaging
- Pre-baked index mount via a Docker volume
- A single `/search` endpoint matching the design above

To run:
```bash
docker build -t patent-search .
docker run -p 8000:8000 -v $(pwd)/patent_index.pkl:/app/patent_index.pkl patent-search
# Then: curl "http://localhost:8000/search?q=electric+vehicle+battery"
```
