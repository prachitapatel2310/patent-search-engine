"""
Part 3 Enhancement: Two-Phase Searching
========================================
Phase 1: Semantic bi-encoder retrieval (fast, ~20 ms) → top-100 candidates
Phase 2: BM25 re-ranking of those 100 candidates → final top-k

Why two phases work well together
----------------------------------
- Bi-encoder: captures semantic meaning ("thermal regulation" ≈ "heat management")
  but can miss exact claim terminology.
- BM25: rewards exact term frequency weighted by corpus rarity (TF-IDF style).
  When a query contains precise technical terms that also appear verbatim in the
  patent claim, BM25 boosts that patent above vaguer semantic matches.

The final score is a Reciprocal Rank Fusion (RRF) of the two rankings:
    rrf(doc) = 1/(k + rank_bi) + 1/(k + rank_bm25)    k=60 (standard default)

This is lightweight, interpretable, and consistently beats either ranker alone
on retrieval benchmarks.

Cross-encoder note
------------------
We originally intended to use `cross-encoder/ms-marco-MiniLM-L-6-v2` for
Phase 2.  It is the standard production choice and gives the highest quality
re-ranking by running full transformer attention over (query, patent) pairs.
However, it triggers a SIGBUS (exit 138) on Python 3.14 due to a known PyTorch
memory-alignment issue on that interpreter version.  BM25+RRF is used here as
a robust, dependency-free substitute that still meaningfully re-orders results.
To switch to cross-encoder on Python ≤3.12:
    pip install sentence-transformers
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    ce_scores = ce.predict([(query, text) for text in candidate_texts])
"""

import re
import time
from typing import Optional

from rank_bm25 import BM25Okapi

from search_engine import PatentSearchEngine, build_searchable_text, get_engine


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class TwoPhaseSearchEngine:
    """
    Two-phase patent search: bi-encoder retrieval + BM25 re-ranking via RRF.

    Usage
    -----
    engine = TwoPhaseSearchEngine()
    results = engine.search("autonomous vehicle lane detection", top_k=10)
    results, timing = engine.search("steering control", top_k=10, return_timing=True)
    """

    def __init__(self, base_engine: Optional[PatentSearchEngine] = None):
        self.base = base_engine or get_engine()

    def search(
        self,
        query: str,
        top_k: int = 10,
        phase1_k: int = 100,
        classification_prefix: Optional[str] = None,
        keyword: Optional[str] = None,
        exact_title: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        return_timing: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Two-phase search with RRF fusion.

        Parameters
        ----------
        query                 : natural-language string, claim, or doc_number
        top_k                 : number of final results to return
        phase1_k              : number of candidates retrieved in Phase 1
        classification_prefix : hybrid filter — classification code prefix
        keyword               : hybrid filter — keyword in title/abstract
        exact_title           : hybrid filter — exact title match
        return_timing         : if True also return a timing dict
        """
        # ── Phase 1: semantic bi-encoder retrieval ───────────────────────────
        t0 = time.perf_counter()
        candidates = self.base.search(
            query,
            top_k=phase1_k,
            candidate_pool=phase1_k * 2,
            classification_prefix=classification_prefix,
            keyword=keyword,
            exact_title=exact_title,
            date_from=date_from,
            date_to=date_to,
        )
        t_phase1 = time.perf_counter()

        if not candidates:
            empty_timing = {"phase1_s": 0, "phase2_s": 0, "total_s": 0,
                            "phase1_candidates": 0}
            return ([], empty_timing) if return_timing else []

        # ── Phase 2: BM25 re-ranking ──────────────────────────────────────────
        # Build a mini corpus from the candidate patents only
        doc_texts = []
        for r in candidates:
            patent = next(
                (p for p in self.base.patents if p["doc_number"] == r["doc_number"]),
                None,
            )
            doc_texts.append(build_searchable_text(patent) if patent else r["abstract"])

        tokenized_corpus = [_tokenize(t) for t in doc_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        q_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)          # array of length phase1_k

        t_phase2 = time.perf_counter()

        # ── RRF fusion ────────────────────────────────────────────────────────
        # Phase 1 is already ranked by bi-encoder score (rank = 1-based index + 1)
        # Phase 2 rank from BM25 scores
        k_rrf = 60
        n = len(candidates)
        bm25_ranks = _rank_array(bm25_scores)   # 1 = highest BM25 score

        fused = []
        for i, r in enumerate(candidates):
            bi_rank  = i + 1
            bm25_rank = int(bm25_ranks[i])
            rrf = 1.0 / (k_rrf + bi_rank) + 1.0 / (k_rrf + bm25_rank)
            fused.append({
                **r,
                "bi_encoder_score":  r.pop("score"),
                "bm25_score":        round(float(bm25_scores[i]), 4),
                "rrf_score":         round(rrf, 6),
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        final = []
        for rank, r in enumerate(fused[:top_k], 1):
            r["rank"] = rank
            final.append(r)

        t_end = time.perf_counter()

        if return_timing:
            timing = {
                "phase1_candidates": len(candidates),
                "phase1_s":  round(t_phase1 - t0,       4),
                "phase2_s":  round(t_phase2 - t_phase1, 4),
                "total_s":   round(t_end    - t0,        4),
            }
            return final, timing

        return final


def _rank_array(scores) -> list[int]:
    """Return 1-based ranks (1 = highest score)."""
    import numpy as np
    arr = np.array(scores)
    order = np.argsort(-arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks.tolist()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = TwoPhaseSearchEngine()

    query = "steering system with feedback control for autonomous driving"
    print(f"\nQuery: '{query}'\n")

    print("=== Phase 1 only (bi-encoder) ===")
    p1, timing1 = engine.base.search(query, top_k=10, return_timing=True)
    for r in p1:
        print(f"  [{r['rank']:2d}] bi={r['score']:.4f}  {r['doc_number']}  {r['title'][:55]}")
    print(f"  Timing: {timing1['total_s']:.4f}s\n")

    print("=== Phase 2: BM25 re-ranked (RRF fusion) ===")
    results, timing2 = engine.search(query, top_k=10, phase1_k=50, return_timing=True)
    for r in results:
        print(f"  [{r['rank']:2d}] rrf={r['rrf_score']:.5f}  "
              f"bi={r['bi_encoder_score']:.4f}  bm25={r['bm25_score']:.2f}  "
              f"{r['doc_number']}  {r['title'][:40]}")
    print(f"  Timing: phase1={timing2['phase1_s']:.4f}s  "
          f"phase2={timing2['phase2_s']:.4f}s  "
          f"total={timing2['total_s']:.4f}s")
    print()
    print("RRF fusion surfaces patents that score well on BOTH semantic similarity")
    print("and exact term overlap — better than either ranker alone.")
