"""
Interactive CLI for the Patent Search Engine
=============================================
Run:  python app.py
      python app.py --rerank        # two-phase BM25+RRF re-ranking
      python app.py --rebuild       # re-embed all patents (ignore cache)

Commands inside the CLI:
  <query>   normal search
  stats     show corpus statistics
  claims    switch to claim-level mapping mode
  patents   switch back to patent search mode
  export    export last results to CSV
  quit / q  exit
"""

import argparse
import csv
import os
import textwrap

from search_engine import get_engine
from reranker import TwoPhaseSearchEngine
from claim_mapper import ClaimMapper


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         Patent Search Engine  (Thinkstruct PSE)             ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    <query>   search by natural language / claim / doc_number ║
║    stats     show corpus statistics                          ║
║    claims    switch to claim-level mapping mode              ║
║    patents   switch back to patent search mode               ║
║    export    save last results to CSV                        ║
║    quit / q  exit                                            ║
║                                                              ║
║  Optional filters (press Enter to skip any):                 ║
║    classification prefix  e.g. B60B                          ║
║    keyword in title/abstract                                 ║
║    exact title                                               ║
║    date from / date to    e.g. 2024-03-01                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def _prompt(label: str, default: str = "") -> str:
    val = input(f"  {label}: ").strip()
    return val if val else default


def _print_results(results: list[dict], timing: dict) -> None:
    if not results:
        print("  No results found.\n")
        return

    hybrid_note = " | hybrid active" if timing.get("hybrid_active") else ""
    phase_note = ""
    if "phase1_s" in timing and "phase2_s" in timing:
        phase_note = f" | phase1={timing['phase1_s']:.3f}s phase2={timing['phase2_s']:.3f}s"

    print(f"\n  Found {len(results)} result(s)  "
          f"[{timing['total_s']:.3f}s{hybrid_note}{phase_note}]\n")

    for r in results:
        title = r["title"] or "(no title)"
        doc   = r["doc_number"]
        cls   = r["classification"]
        date  = r.get("filing_date", "")
        abstr = r["abstract"]

        if "rrf_score" in r:
            score_str = (f"RRF={r['rrf_score']:.5f} "
                         f"| bi={r['bi_encoder_score']:.4f} "
                         f"| bm25={r['bm25_score']:.2f}")
        else:
            score_str = f"score={r['score']:.4f}"

        print(f"  ── #{r['rank']} ──────────────────────────────────────────")
        print(f"     {score_str}")
        print(f"     Doc:   {doc}  |  Filed: {date or 'unknown'}")
        print(f"     Class: {cls}")
        print(f"     Title: {textwrap.fill(title, 70, subsequent_indent='            ')}")
        print(f"     Abstr: {textwrap.fill(abstr, 70, subsequent_indent='            ')}")
        print()


def _print_claim_results(results: list[dict]) -> None:
    if not results:
        print("  No matching claims found.\n")
        return
    print(f"\n  Found {len(results)} matching claim(s)\n")
    for r in results:
        print(f"  ── #{r['rank']} ──────────────────────────────────────────")
        print(f"     Score: {r['score']:.4f}")
        print(f"     Doc:   {r['doc_number']}  |  Filed: {r.get('filing_date','unknown')}")
        print(f"     Class: {r['classification']}")
        print(f"     Title: {textwrap.fill(r['title'] or '(no title)', 70, subsequent_indent='            ')}")
        print(f"     Claim {r['claim_num']}:")
        wrapped = textwrap.fill(r['claim_text'], 68, initial_indent="       ",
                                subsequent_indent="       ")
        print(wrapped[:600] + ("…" if len(r['claim_text']) > 600 else ""))
        print()


def _print_stats(engine) -> None:
    stats = engine.get_stats()
    print("\n  ── Corpus Statistics ──────────────────────────────────")
    print(f"     Total patents : {stats['total_patents']}")
    print(f"     Total claims  : {stats['total_claims']}")
    dr = stats["filing_date_range"]
    print(f"     Filing dates  : {dr[0]}  →  {dr[1]}")
    print(f"     Missing abstract : {stats['missing_abstract']}")
    print(f"     Missing claims   : {stats['missing_claims']}")
    print("\n     Top classification prefixes:")
    for code, count in stats["top_classifications"]:
        bar = "█" * (count // 2)
        print(f"       {code:6s}  {bar} {count}")
    print()


def _export_csv(results: list[dict], filename: str = "results.csv") -> None:
    if not results:
        print("  Nothing to export.\n")
        return
    keys = [k for k in results[0].keys() if k != "abstract"]
    keys.append("abstract")
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Exported {len(results)} result(s) to {path}\n")


def run_interactive(use_rerank: bool, rebuild: bool) -> None:
    print(BANNER)

    base_engine = get_engine(rebuild=rebuild)

    if use_rerank:
        search_engine = TwoPhaseSearchEngine(base_engine=base_engine)
        search_fn     = search_engine.search
        extra_kwargs  = {"phase1_k": 100}
        print("  Two-phase search active: bi-encoder retrieval + BM25 RRF re-ranking\n")
    else:
        search_fn    = base_engine.search
        extra_kwargs = {}

    # Claim mapper — lazily built on first use
    claim_mapper = ClaimMapper(base_engine=base_engine)

    mode          = "patents"   # "patents" or "claims"
    last_results  = []

    while True:
        mode_tag = "[claim-map]" if mode == "claims" else "[patent]"
        print("─" * 60)
        raw = input(f"Query {mode_tag} › ").strip()

        if not raw or raw.lower() in ("quit", "q", "exit"):
            print("Bye!")
            break

        # ── Meta-commands ─────────────────────────────────────────────────
        if raw.lower() == "stats":
            _print_stats(base_engine)
            continue

        if raw.lower() == "claims":
            mode = "claims"
            print("  Switched to claim-level mapping mode.\n")
            continue

        if raw.lower() == "patents":
            mode = "patents"
            print("  Switched to patent search mode.\n")
            continue

        if raw.lower() == "export":
            fname = _prompt("Filename (default: results.csv)", "results.csv")
            _export_csv(last_results, fname)
            continue

        # ── Claim-level mode ──────────────────────────────────────────────
        if mode == "claims":
            top_k_str = _prompt("Number of claims to return", "10")
            try:
                top_k = int(top_k_str)
            except ValueError:
                top_k = 10

            # If query looks like a doc_number, do patent-to-patent claim overlap
            import re
            if re.fullmatch(r"\d{11}", raw):
                print(f"\n  Finding claims that overlap with patent {raw} (Claim 1) …")
                results = claim_mapper.find_overlapping_claims(raw, top_k=top_k)
            else:
                results = claim_mapper.search_claims(raw, top_k=top_k)

            last_results = results
            _print_claim_results(results)

            save = input("  Export to CSV? [y/N] ").strip().lower()
            if save == "y":
                _export_csv(results, "claim_results.csv")
            continue

        # ── Patent search mode ────────────────────────────────────────────
        cls_prefix = _prompt("Classification prefix (e.g. B60B, blank=skip)")
        keyword    = _prompt("Keyword in title/abstract   (blank=skip)")
        exact_ttl  = _prompt("Exact title match           (blank=skip)")
        date_from  = _prompt("Filing date from (YYYY-MM-DD, blank=skip)")
        date_to    = _prompt("Filing date to   (YYYY-MM-DD, blank=skip)")
        top_k_str  = _prompt("Number of results", "10")
        try:
            top_k = int(top_k_str)
        except ValueError:
            top_k = 10

        results, timing = search_fn(
            raw,
            top_k=top_k,
            classification_prefix=cls_prefix or None,
            keyword=keyword or None,
            exact_title=exact_ttl or None,
            date_from=date_from or None,
            date_to=date_to or None,
            return_timing=True,
            **extra_kwargs,
        )
        last_results = results
        _print_results(results, timing)

        save = input("  Export to CSV? [y/N] ").strip().lower()
        if save == "y":
            _export_csv(results, "search_results.csv")


def main():
    parser = argparse.ArgumentParser(description="Patent Search Engine CLI")
    parser.add_argument("--rerank",  action="store_true",
                        help="Enable two-phase BM25+RRF re-ranking (Part 3)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-embed all patents and overwrite cache")
    args = parser.parse_args()
    run_interactive(use_rerank=args.rerank, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
