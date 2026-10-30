"""Command-line search: run ONE retrieval method on ONE natural-language query.

Usage
-----
    python src/search.py --method keyword  --query "breast cancer classification"
    python src/search.py --method tfidf    --query "breast cancer classification"
    python src/search.py --method semantic --query "breast cancer classification"      # MiniLM
    python src/search.py --method metadata --query "labeled binary dataset with at least 500 instances"
    python src/search.py --method hybrid   --alpha 0.7 --query "labeled lung cancer CT images, at least 500 patients"

    # offline pipeline check on the SYNTHETIC fixture with the LSA stand-in (not MiniLM):
    python src/search.py --fixture --backend lsa --method hybrid --query "ECG arrhythmia detection"

The script reads catalogue METADATA only. Scores are ranking scores, not probabilities.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, resolve  # noqa: E402
from retrieval import Recommender, make_backend  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True)
    ap.add_argument("--method", choices=["keyword", "tfidf", "semantic", "metadata", "hybrid"], default="hybrid")
    ap.add_argument("--alpha", type=float, default=None, help="hybrid weight on semantic similarity (default from config)")
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--fixture", action="store_true", help="use the SYNTHETIC fixture catalogue (demo only)")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--backend", choices=["minilm", "lsa"], default="minilm")
    args = ap.parse_args()

    path = args.catalog or (f"{cfg['paths']['fixture_dir']}/catalog_text.csv" if args.fixture
                            else f"{cfg['paths']['processed_dir']}/catalog_text.csv")
    if not resolve(path).exists():
        sys.exit(f"Catalogue not found: {path}. Run the pipeline first (see README) or use --fixture.")
    if args.fixture:
        print("NOTE: SYNTHETIC FIXTURE - fabricated records, not OpenML data.")
    if args.backend == "lsa":
        print("NOTE: LSA offline stand-in is NOT the pretrained MiniLM model.")

    df = pd.read_csv(resolve(path))
    needs_semantic = args.method in ("semantic", "hybrid")
    backend = make_backend(args.backend, cfg) if needs_semantic else None
    rec = Recommender(df, backend, cfg)

    parsed = rec.parse(args.query)
    print("Extracted requirements:", [r.label() for r in parsed.requirements] or "none")
    for r in rec.recommend(args.query, k=args.k, method=args.method, alpha=args.alpha):
        print(f"\n#{r['rank']}  {r['dataset_name']}  (id {r['dataset_id']})")
        print(f"    source : {r['source_url']}")
        print(f"    score  : {r['score']:.4f}")
        print(f"    why    : {r['explanation']}")


if __name__ == "__main__":
    main()
