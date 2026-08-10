"""
Patent Search Engine — Part 1
==============================
Inputs:
  - Natural language query string
  - A patent claim (free text)
  - A patent's Document Number (e.g. "20240075768")

Outputs:
  - Ranked list of patents with title, doc number, classification,
    abstract snippet, and similarity score

Missing-field policy:
  - Patents with empty/missing title, abstract, claims, or classification
    are INCLUDED but the missing field is replaced with an empty string
    so it is excluded from the searchable text.  This lets us index
    every patent while degrading gracefully.

Vector index:
  - faiss-cpu does not build on Python 3.14, so we use a numpy
    brute-force cosine similarity matrix (O(N·d)).  For 640 patents
    this is <50 ms.  At scale, swap NumpyIndex for a FAISS/Pinecone
    index — the interface is identical.
"""

import os
import json
import glob
import time
import pickle
import re
import csv
from collections import Counter
from datetime import date as _date
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__),
                          "patent_data", "data", "patent_data_small")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "patent_index.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, 384-dim, fast & good quality


# ── Data helpers ─────────────────────────────────────────────────────────────

def _parse_date(filename: str) -> str:
    """Extract ISO filing date from filename like 'US20240051333A1-20240215.XML'."""
    m = re.search(r"-(\d{8})\.", filename)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"  # e.g. "2024-02-15"
    return ""


def _claims_text(claims) -> str:
    if isinstance(claims, list):
        return " ".join(str(c) for c in claims)
    return str(claims) if claims else ""


def _desc_text(desc) -> str:
    if isinstance(desc, list):
        return " ".join(str(p) for p in desc[:5])   # first 5 paragraphs
    return str(desc) if desc else ""


def load_patents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all JSON files and return a flat list of patent dicts."""
    patents = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            batch = json.load(fh)
        for p in batch:
            # Normalise fields — never drop a patent
            raw_claims = p.get("claims")
            patents.append({
                "title":          p.get("title") or "",
                "doc_number":     p.get("doc_number") or "",
                "abstract":       p.get("abstract") or "",
                "claims":         _claims_text(raw_claims),
                "claims_list":    raw_claims if isinstance(raw_claims, list) else [],
                "description":    _desc_text(p.get("detailed_description")),
                "classification": p.get("classification") or "",
                "bibtex":         p.get("bibtex") or "",
                "filing_date":    _parse_date(p.get("filename") or ""),
            })
    return patents


def build_searchable_text(patent: dict) -> str:
    """Concatenate the most semantically rich fields for embedding."""
    parts = [
        patent["title"],
        patent["abstract"],
        patent["claims"][:1000],    # cap to avoid token limits
        patent["description"][:500],
    ]
    return " ".join(p for p in parts if p).strip()


# ── Numpy-based vector index (FAISS drop-in for small scale) ─────────────────

class NumpyIndex:
    """Flat cosine-similarity index backed by a normalised numpy matrix."""

    def __init__(self):
        self.matrix: Optional[np.ndarray] = None   # (N, d), float32, L2-normed

    def add(self, embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.matrix = (embeddings / norms).astype(np.float32)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) for top-k results, highest score first."""
        q = query_vec.astype(np.float32).flatten()
        q /= (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q                    # (N,) cosine similarities
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            # argpartition for O(N) top-k, then sort the small slice
            part = np.argpartition(-scores, k)[:k]
            idx  = part[np.argsort(-scores[part])]
        return scores[idx], idx


# ── Main engine ───────────────────────────────────────────────────────────────

class PatentSearchEngine:
    """
    Semantic + hybrid patent search engine.

    Usage
    -----
    engine = PatentSearchEngine()
    engine.build()                          # or engine.load() if cache exists
    results = engine.search("battery management system for EV")
    results = engine.search("wheel bearing noise", classification_prefix="B60B",
                             keyword="bearing", top_k=10)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model   = SentenceTransformer(model_name)
        self.patents : list[dict]  = []
        self.texts   : list[str]   = []
        self.index   = NumpyIndex()

    # ── Build / persist ──────────────────────────────────────────────────────

    def build(self, data_dir: str = DATA_DIR, save: bool = True) -> None:
        """Load patents, embed them, build the index, optionally cache."""
        print(f"Loading patents from {data_dir} …")
        self.patents = load_patents(data_dir)
        print(f"  Loaded {len(self.patents)} patents")

        self.texts = [build_searchable_text(p) for p in self.patents]

        print("Encoding patents (this takes ~10-20 s on first run) …")
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"  Encoded in {time.perf_counter()-t0:.1f}s")

        self.index.add(embeddings)

        if save:
            with open(CACHE_FILE, "wb") as fh:
                pickle.dump({"patents": self.patents,
                             "texts":   self.texts,
                             "matrix":  self.index.matrix}, fh)
            print(f"  Index saved → {CACHE_FILE}")

    def load(self, cache_file: str = CACHE_FILE) -> bool:
        """Load a previously saved index.  Returns True on success."""
        if not os.path.exists(cache_file):
            return False
        print(f"Loading index from cache {cache_file} …")
        with open(cache_file, "rb") as fh:
            data = pickle.load(fh)
        self.patents = data["patents"]
        self.texts   = data["texts"]
        self.index.matrix = data["matrix"]
        print(f"  Loaded {len(self.patents)} patents from cache")
        return True

    # ── Hybrid filters ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(patents: list[dict],
                       indices: np.ndarray,
                       classification_prefix: Optional[str] = None,
                       keyword: Optional[str] = None,
                       exact_title: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None) -> np.ndarray:
        """
        Return the subset of `indices` whose patents pass all active filters.

        classification_prefix : patent classification must start with this
                                 string (case-insensitive), e.g. "B60B"
        keyword               : must appear in title OR abstract
                                 (case-insensitive)
        exact_title           : patent title must match exactly
                                 (case-insensitive, stripped)
        date_from / date_to   : ISO date strings "YYYY-MM-DD", inclusive
        """
        keep = []
        for i in indices:
            p = patents[i]
            if classification_prefix:
                if not p["classification"].upper().startswith(
                        classification_prefix.upper()):
                    continue
            if keyword:
                kw = keyword.lower()
                if kw not in p["title"].lower() and kw not in p["abstract"].lower():
                    continue
            if exact_title:
                if p["title"].strip().lower() != exact_title.strip().lower():
                    continue
            fd = p.get("filing_date", "")
            if date_from and (not fd or fd < date_from):
                continue
            if date_to and (not fd or fd > date_to):
                continue
            keep.append(i)
        return np.array(keep, dtype=np.int64)

    # ── Search ───────────────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

    def _resolve_query(self, query: str) -> str:
        """
        If `query` looks like a doc_number, expand it to the patent's text.
        Otherwise treat it as a free-text query.
        """
        if re.fullmatch(r"\d{11}", query.strip()):
            for p in self.patents:
                if p["doc_number"] == query.strip():
                    print(f"  Query resolved to patent: {p['title']}")
                    return build_searchable_text(p)
        return query

    def get_stats(self) -> dict:
        """Return corpus statistics."""
        cls_codes = [p["classification"][:4] for p in self.patents if p["classification"]]
        dates = [p["filing_date"] for p in self.patents if p.get("filing_date")]
        total_claims = sum(len(p.get("claims_list", [])) for p in self.patents)
        return {
            "total_patents":       len(self.patents),
            "total_claims":        total_claims,
            "top_classifications": Counter(cls_codes).most_common(10),
            "filing_date_range":   (min(dates), max(dates)) if dates else ("N/A", "N/A"),
            "missing_abstract":    sum(1 for p in self.patents if not p["abstract"]),
            "missing_claims":      sum(1 for p in self.patents if not p["claims"]),
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 200,
        classification_prefix: Optional[str] = None,
        keyword: Optional[str] = None,
        exact_title: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        return_timing: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Perform a semantic search with optional hybrid filters.

        Parameters
        ----------
        query                 : natural-language string, patent claim, or doc_number
        top_k                 : number of results to return
        candidate_pool        : ANN candidates retrieved before filtering
                                (larger = higher recall, slower if filtering)
        classification_prefix : e.g. "B60B" — filter by CPC/IPC prefix
        keyword               : must appear in title or abstract
        exact_title           : exact title match (case-insensitive)
        date_from / date_to   : ISO date strings "YYYY-MM-DD" (inclusive)
        return_timing         : if True also return a timing dict

        Returns
        -------
        list of result dicts (or (results, timing) if return_timing=True)
        """
        t_start = time.perf_counter()

        expanded_query = self._resolve_query(query)
        q_vec = self._encode_query(expanded_query)
        t_encode = time.perf_counter()

        any_filter = any([classification_prefix, keyword, exact_title,
                          date_from, date_to])

        if any_filter:
            # Retrieve larger pool then filter
            pool = min(candidate_pool, len(self.patents))
            scores, indices = self.index.search(q_vec, pool)
            t_ann = time.perf_counter()

            filtered = self._apply_filters(
                self.patents, indices,
                classification_prefix, keyword, exact_title,
                date_from, date_to,
            )
            # Re-order filtered indices by original ANN score
            score_map = {int(idx): float(sc) for sc, idx in zip(scores, indices)}
            results_idx = filtered[:top_k]
        else:
            k = min(top_k, len(self.patents))
            scores, indices = self.index.search(q_vec, k)
            t_ann = time.perf_counter()
            results_idx = indices
            score_map = {int(idx): float(sc) for sc, idx in zip(scores, indices)}

        t_end = time.perf_counter()

        results = []
        for rank, i in enumerate(results_idx, 1):
            i = int(i)
            p = self.patents[i]
            results.append({
                "rank":           rank,
                "score":          round(score_map[i], 4),
                "doc_number":     p["doc_number"],
                "title":          p["title"],
                "classification": p["classification"],
                "filing_date":    p.get("filing_date", ""),
                "abstract":       p["abstract"][:300] + ("…" if len(p["abstract"]) > 300 else ""),
            })

        if return_timing:
            timing = {
                "encode_s":     round(t_encode - t_start, 4),
                "ann_s":        round(t_ann    - t_encode, 4),
                "filter_s":     round(t_end    - t_ann,    4),
                "total_s":      round(t_end    - t_start,  4),
                "hybrid_active": any_filter,
            }
            return results, timing

        return results


# ── Convenience factory ───────────────────────────────────────────────────────

def get_engine(rebuild: bool = False) -> PatentSearchEngine:
    engine = PatentSearchEngine()
    if not rebuild and engine.load():
        return engine
    engine.build()
    return engine


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = get_engine()
    print("\n=== Semantic search: 'electric vehicle battery thermal management' ===")
    for r in engine.search("electric vehicle battery thermal management", top_k=5):
        print(f"  [{r['rank']}] {r['score']:.4f}  {r['doc_number']}  {r['title'][:60]}")

    print("\n=== Hybrid search: same query + classification B60L ===")
    results, timing = engine.search(
        "electric vehicle battery thermal management",
        top_k=5,
        classification_prefix="B60L",
        return_timing=True,
    )
    for r in results:
        print(f"  [{r['rank']}] {r['score']:.4f}  {r['classification']:<14}  {r['title'][:55]}")
    print(f"  Timing: {timing}")

    print("\n=== Timing comparison ===")
    _, t_no_filter  = engine.search("wheel bearing vibration", top_k=10, return_timing=True)
    _, t_with_filter = engine.search("wheel bearing vibration", top_k=10,
                                     classification_prefix="B60B", return_timing=True)
    print(f"  Without hybrid filter : {t_no_filter['total_s']:.4f}s")
    print(f"  With    hybrid filter : {t_with_filter['total_s']:.4f}s")
    print("  (Filters add negligible overhead at this scale; at 10M patents,")
    print("   a pre-filter via an inverted index on classification codes")
    print("   drastically reduces the ANN search space, making it *faster*.)")
