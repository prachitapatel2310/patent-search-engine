"""
Claim-Level Mapper — Part 3 Enhancement
========================================
The overview for this project says Thinkstruct's core technology is
*claim mapping*: taking a claim from one patent and finding the most
similar claims from other patents to detect overlap or conflict.

This module indexes every individual claim separately (not whole patents).
640 patents × ~8 claims each ≈ 5,000 individual claims in the index.

Usage
-----
    from claim_mapper import ClaimMapper
    from search_engine import get_engine

    mapper = ClaimMapper(get_engine())
    results = mapper.search_claims("a brake pad with heat-resistant coating", top_k=5)
    for r in results:
        print(r['rank'], r['score'], r['doc_number'], f"Claim {r['claim_num']}")
        print("   ", r['claim_text'][:120])
"""

import glob
import json
import os
import time
from typing import Optional

import numpy as np

from search_engine import (
    DATA_DIR,
    NumpyIndex,
    PatentSearchEngine,
    get_engine,
)


class ClaimMapper:
    """
    Indexes individual patent claims and finds the most similar ones
    for a given query, enabling fine-grained claim-to-claim mapping.
    """

    def __init__(self, base_engine: Optional[PatentSearchEngine] = None):
        self.engine = base_engine or get_engine()
        self.model  = self.engine.model
        self.claims: list[dict] = []   # flat list of {claim_text, claim_num, ...}
        self.index  = NumpyIndex()
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, data_dir: str = DATA_DIR) -> None:
        """Load every individual claim from the JSON files and embed them."""
        print("Building claim-level index …")
        t0 = time.perf_counter()

        # Build a doc_number → patent metadata lookup from already-loaded patents
        meta = {p["doc_number"]: p for p in self.engine.patents}

        for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
            with open(path, encoding="utf-8") as fh:
                batch = json.load(fh)
            for p in batch:
                doc_num = p.get("doc_number") or ""
                raw_claims = p.get("claims")
                if not isinstance(raw_claims, list):
                    continue
                parent = meta.get(doc_num, {})
                for i, claim_text in enumerate(raw_claims, 1):
                    text = str(claim_text).strip()
                    if not text:
                        continue
                    self.claims.append({
                        "claim_text":     text,
                        "claim_num":      i,
                        "doc_number":     doc_num,
                        "title":          parent.get("title", ""),
                        "classification": parent.get("classification", ""),
                        "filing_date":    parent.get("filing_date", ""),
                    })

        print(f"  {len(self.claims)} individual claims loaded from {len(self.engine.patents)} patents")

        texts = [c["claim_text"] for c in self.claims]
        print("  Encoding claims …")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.index.add(embeddings)
        self._built = True
        print(f"  Claim index built in {time.perf_counter()-t0:.1f}s")

    def _ensure_built(self):
        if not self._built:
            self.build()

    # ── Search ────────────────────────────────────────────────────────────────

    def search_claims(
        self,
        query: str,
        top_k: int = 10,
        exclude_doc: Optional[str] = None,
    ) -> list[dict]:
        """
        Find the top-k most similar individual claims to `query`.

        Parameters
        ----------
        query       : natural language string or a verbatim patent claim
        top_k       : number of claim results to return
        exclude_doc : if set, skip claims from this patent (useful when the
                      query IS a claim from a specific patent)
        """
        self._ensure_built()

        q_vec = self.model.encode([query], convert_to_numpy=True)[0]
        # Retrieve more than top_k in case we need to skip some
        fetch_k = min(top_k + (100 if exclude_doc else 0), len(self.claims))
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        rank = 1
        for score, idx in zip(scores, indices):
            c = self.claims[int(idx)]
            if exclude_doc and c["doc_number"] == exclude_doc:
                continue
            results.append({
                "rank":           rank,
                "score":          round(float(score), 4),
                "claim_num":      c["claim_num"],
                "claim_text":     c["claim_text"],
                "doc_number":     c["doc_number"],
                "title":          c["title"],
                "classification": c["classification"],
                "filing_date":    c["filing_date"],
            })
            rank += 1
            if rank > top_k:
                break

        return results

    def find_overlapping_claims(
        self,
        doc_number: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Given a patent's doc_number, find the claims from OTHER patents
        that overlap most with its independent claim (claim 1).

        This is the core claim-mapping use case.
        """
        self._ensure_built()

        # Find claim 1 of the given patent
        source_claim = next(
            (c for c in self.claims
             if c["doc_number"] == doc_number and c["claim_num"] == 1),
            None,
        )
        if source_claim is None:
            return []

        print(f"\n  Source (Claim 1 of {doc_number}):")
        print(f"  {source_claim['claim_text'][:200]}…\n")

        return self.search_claims(
            source_claim["claim_text"],
            top_k=top_k,
            exclude_doc=doc_number,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = get_engine()
    mapper = ClaimMapper(engine)
    mapper.build()

    print("\n=== Claim search: 'anti-lock braking system with sensor feedback' ===")
    for r in mapper.search_claims("anti-lock braking system with sensor feedback", top_k=5):
        print(f"  [{r['rank']}] score={r['score']:.4f}  {r['doc_number']} Claim {r['claim_num']}")
        print(f"       {r['title'][:60]}")
        print(f"       \"{r['claim_text'][:120]}…\"")
        print()

    print("\n=== Find overlapping claims for first patent in corpus ===")
    first_doc = engine.patents[0]["doc_number"]
    overlaps = mapper.find_overlapping_claims(first_doc, top_k=3)
    for r in overlaps:
        print(f"  [{r['rank']}] score={r['score']:.4f}  {r['doc_number']} Claim {r['claim_num']}")
        print(f"       {r['title'][:60]}")
        print(f"       \"{r['claim_text'][:120]}…\"")
        print()
