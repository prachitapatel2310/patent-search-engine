"""
FastAPI REST API — Part 2 Proof-of-Concept
==========================================
Exposes the search engine over HTTP so it can be containerized and
called from any client.

Endpoints:
  GET /search?q=<query>&top_k=10&classification=B60B&keyword=bearing
  GET /patent/<doc_number>
  GET /health

Run locally:
  pip install fastapi uvicorn
  python api_server.py
"""

import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from search_engine import get_engine

app = FastAPI(
    title="Patent Search Engine API",
    description="Semantic + hybrid search over vehicle patent applications",
    version="1.0.0",
)

# Load engine once at startup
_engine = None

@app.on_event("startup")
def startup_event():
    global _engine
    _engine = get_engine()


@app.get("/health")
def health():
    return {"status": "ok", "patents_indexed": len(_engine.patents) if _engine else 0}


@app.get("/search")
def search(
    q: str = Query(..., description="Natural language query, patent claim, or doc_number"),
    top_k: int = Query(10, ge=1, le=50),
    classification: Optional[str] = Query(None, description="Classification prefix e.g. B60B"),
    keyword: Optional[str] = Query(None, description="Keyword in title or abstract"),
    exact_title: Optional[str] = Query(None, description="Exact title match"),
):
    results, timing = _engine.search(
        query=q,
        top_k=top_k,
        classification_prefix=classification,
        keyword=keyword,
        exact_title=exact_title,
        return_timing=True,
    )
    return JSONResponse({
        "query": q,
        "filters": {
            "classification": classification,
            "keyword": keyword,
            "exact_title": exact_title,
        },
        "timing": timing,
        "results": results,
    })


@app.get("/patent/{doc_number}")
def get_patent(doc_number: str):
    for p in _engine.patents:
        if p["doc_number"] == doc_number:
            return JSONResponse(p)
    raise HTTPException(status_code=404, detail=f"Patent {doc_number} not found")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
