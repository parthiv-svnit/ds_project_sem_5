"""Run all retrieval methods, tune alpha on VALIDATION queries only, evaluate on TEST queries.

Usage
-----
    # real catalogue, real MiniLM embeddings (needs internet once to download the model)
    python src/run_experiments.py --semantic-backend minilm

    # offline pipeline check on the synthetic fixture (LSA stand-in; NOT paper results)
    python src/run_experiments.py --fixture --semantic-backend lsa

Protocol
--------
  1. All five methods produce a score matrix (queries x datasets) once.
  2. alpha is selected on the validation split (criterion: mean NDCG@5 of the hybrid; ties go to
     the value closest to the default 0.7, then the smaller one).
  3. The test split is evaluated exactly once per configuration using the selected alpha.
     Test queries are never used to choose alpha, thresholds, weights or lexicons.
  4. The train split is not used for tuning (nothing is learned); it is reserved for development
     inspection and for a possible future learned ranker.

Outputs (in outputs/results/, or outputs/results_fixture/ for --fixture)
  run_metadata.json  per_query_metrics.csv  main_results.csv  ablation.csv  alpha_selection.json
  paired_comparisons.csv  by_query_type.csv  rankings_top10.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from common import load_config, read_json, resolve, resolve_dir, write_json
from evaluate import cluster_bootstrap_ci, evaluate_query, paired_cluster_bootstrap
from retrieval import Recommender, make_backend, rank_scores

K = 5
MRR_DEPTH = 100
METRICS = ["P@5", "R@5", "MRR", "NDCG@5"]
METHOD_LABELS = {"keyword": "A: Keyword", "tfidf": "B: TF-IDF", "semantic": "C: Semantic", "metadata": "D: Metadata-only"}


def pkg_versions(names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not installed"
    return out


def load_qrels(bench_dir: str, which: str) -> tuple[pd.DataFrame, str]:
    reviewed = resolve(f"{bench_dir}/qrels_reviewed.csv")
    if which == "reviewed":
        if not reviewed.exists():
            raise SystemExit(f"{reviewed} not found. Run build_benchmark.py merge first or use --qrels synthetic.")
        return pd.read_csv(reviewed), "reviewed"
    return pd.read_csv(resolve(f"{bench_dir}/qrels_synthetic.csv")), "synthetic"


def qrels_dict(qrels: pd.DataFrame) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for qid, did, rel in zip(qrels["query_id"], qrels["candidate_dataset_id"], qrels["relevance"]):
        out.setdefault(qid, {})[int(did)] = int(rel)
    return out


def per_query_metrics(scores: np.ndarray, ids: np.ndarray, qids: list[str], rel: dict[str, dict[int, int]]) -> pd.DataFrame:
    rows = []
    for i, qid in enumerate(qids):
        order = rank_scores(scores[i], ids, MRR_DEPTH)
        ranked = [int(ids[j]) for j in order]
        m = evaluate_query(ranked, rel.get(qid, {}), K, threshold=2)
        rows.append({"query_id": qid, **m})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", action="store_true", help="use the synthetic fixture (pipeline check only)")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--qrels", choices=["synthetic", "reviewed"], default="synthetic")
    ap.add_argument("--out", default=None)
    ap.add_argument("--semantic-backend", choices=["minilm", "lsa"], default="minilm")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    p = cfg["paths"]
    if args.fixture:
        catalog_path = args.catalog or f"{p['fixture_dir']}/catalog_text.csv"
        bench_dir = args.bench or f"{p['fixture_dir']}/benchmark"
        out_dir = args.out or "outputs/results_fixture"
    else:
        catalog_path = args.catalog or f"{p['processed_dir']}/catalog_text.csv"
        bench_dir = args.bench or p["benchmark_dir"]
        out_dir = args.out or p["results_dir"]

    cat = pd.read_csv(resolve(catalog_path))
    is_fixture = bool(cat["is_synthetic"].astype(str).str.lower().eq("true").all()) if "is_synthetic" in cat.columns else False
    mixed = ("is_synthetic" in cat.columns) and cat["is_synthetic"].astype(str).str.lower().eq("true").any() and not is_fixture
    if mixed:
        raise SystemExit("Catalogue mixes synthetic fixture and real records - refusing to run.")
    if is_fixture and out_dir.rstrip("/") == p["results_dir"].rstrip("/"):
        raise SystemExit("Refusing to write synthetic-fixture results into the main results directory.")
    out = resolve_dir(out_dir)

    queries = pd.read_csv(resolve(f"{bench_dir}/queries.csv"))
    qrels, qrels_kind = load_qrels(bench_dir, args.qrels)
    rel = qrels_dict(qrels)
    qids = queries["query_id"].tolist()
    ids = cat["dataset_id"].to_numpy()
    print(f"[run] catalogue={len(cat)} records (fixture={is_fixture}); queries={len(queries)}; qrels={qrels_kind} ({len(qrels)} listed)")

    # ---------- build recommender (+ embeddings), timing it
    t0 = time.perf_counter()
    backend = make_backend(args.semantic_backend, cfg)
    try:
        rec = Recommender(cat, backend, cfg)
    except Exception as exc:  # e.g. model cannot be downloaded
        raise SystemExit(f"[run] could not initialise the '{args.semantic_backend}' backend: {type(exc).__name__}: {exc}\n"
                         "For MiniLM you need internet access once (to download the model) and sentence-transformers installed.")
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    comp = rec.components(queries["query_text"].tolist())
    t_scores = time.perf_counter() - t0

    split = queries["split"].to_numpy()
    groups = queries["topic_group"].to_numpy()
    alpha_grid = cfg["retrieval"]["alpha_grid"]
    default_alpha = cfg["retrieval"]["alpha_default"]

    # ---------- per-query metrics for every configuration
    configs: list[dict[str, Any]] = []
    for m in ("keyword", "tfidf", "semantic", "metadata"):
        configs.append({"name": m, "method": m, "alpha": None, "constraints": None})
    for a in alpha_grid:
        configs.append({"name": f"hybrid_a{a:g}_constrained", "method": "hybrid", "alpha": a, "constraints": True})
        configs.append({"name": f"hybrid_a{a:g}_unconstrained", "method": "hybrid", "alpha": a, "constraints": False})

    pq_frames, score_cache = [], {}
    for c in configs:
        s = rec.scores_from_components(comp, c["method"], c["alpha"], bool(c["constraints"]) if c["constraints"] is not None else True)
        score_cache[c["name"]] = s
        df = per_query_metrics(s, ids, qids, rel)
        df["config"], df["method"], df["alpha"], df["constraints"] = c["name"], c["method"], c["alpha"], c["constraints"]
        df["split"], df["topic_group"] = split, groups
        df["has_constraints"] = queries["has_constraints"].to_numpy()
        pq_frames.append(df)
    pq = pd.concat(pq_frames, ignore_index=True)
    pq.to_csv(out / "per_query_metrics.csv", index=False)

    # ---------- alpha selection on VALIDATION only
    val = pq[(pq.split == "validation") & (pq.method == "hybrid") & (pq.constraints == True)]  # noqa: E712
    val_means = val.groupby("alpha")["NDCG@5"].mean()
    ordered = sorted(val_means.index, key=lambda a: (-round(val_means[a], 12), abs(a - default_alpha), a))
    selected_alpha = float(ordered[0]) if len(ordered) else default_alpha
    write_json({"criterion": "mean NDCG@5 on validation queries, constrained hybrid", "validation_ndcg5_by_alpha": {str(k): float(v) for k, v in val_means.items()},
                "selected_alpha": selected_alpha, "default_alpha": default_alpha, "n_validation_queries": int((split == "validation").sum()),
                "tie_break": "closest to default alpha, then smaller alpha"}, out / "alpha_selection.json")
    print(f"[run] selected alpha (validation) = {selected_alpha}")

    # ---------- main results table (test split)
    def summarize(frame: pd.DataFrame, label: str) -> dict[str, Any]:
        row: dict[str, Any] = {"method": label, "n_queries": int(frame["query_id"].nunique())}
        for m in METRICS:
            mean, lo, hi = cluster_bootstrap_ci(frame[m].to_numpy(dtype=float), frame["topic_group"].to_numpy(), args.n_boot, cfg["seed"])
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        return row

    sel_name = f"hybrid_a{selected_alpha:g}_constrained"
    def_name = f"hybrid_a{default_alpha:g}_constrained"
    main_rows = []
    test_pq = pq[pq.split == "test"]
    for key in ("keyword", "tfidf", "semantic", "metadata"):
        main_rows.append(summarize(test_pq[test_pq.config == key], METHOD_LABELS[key]))
    main_rows.append(summarize(test_pq[test_pq.config == def_name], f"E: Hybrid (alpha={default_alpha:g}, default)"))
    main_rows.append(summarize(test_pq[test_pq.config == sel_name], f"E: Hybrid (alpha={selected_alpha:g}, tuned on validation)"))
    main = pd.DataFrame(main_rows)
    main.to_csv(out / "main_results.csv", index=False)

    # ---------- ablation (validation + test, all configurations)
    abl_rows = []
    for sp in ("validation", "test"):
        sp_pq = pq[pq.split == sp]
        for c in configs:
            frame = sp_pq[sp_pq.config == c["name"]]
            if frame.empty:
                continue
            r = {"split": sp, "config": c["name"], "method": c["method"], "alpha": c["alpha"], "constraint_handling": c["constraints"],
                 "n_queries": int(frame["query_id"].nunique())}
            for m in METRICS:
                r[m] = float(frame[m].mean())
            abl_rows.append(r)
    pd.DataFrame(abl_rows).to_csv(out / "ablation.csv", index=False)

    # ---------- paired comparisons on test (selected hybrid minus each baseline)
    comp_rows = []
    sel = test_pq[test_pq.config == sel_name].set_index("query_id")
    for key in ("keyword", "tfidf", "semantic", "metadata", def_name):
        base = test_pq[test_pq.config == key].set_index("query_id").loc[sel.index]
        for m in METRICS:
            d, lo, hi = paired_cluster_bootstrap(sel[m].to_numpy(float), base[m].to_numpy(float), sel["topic_group"].to_numpy(), args.n_boot, cfg["seed"])
            comp_rows.append({"comparison": f"hybrid(alpha={selected_alpha:g}) - {key}", "metric": m, "mean_difference": d, "ci_lo": lo, "ci_hi": hi,
                              "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(comp_rows).to_csv(out / "paired_comparisons.csv", index=False)

    # ---------- by query type (constrained vs topic-only), test split
    bq_rows = []
    for key, label in [("keyword", "A: Keyword"), ("tfidf", "B: TF-IDF"), ("semantic", "C: Semantic"), ("metadata", "D: Metadata-only"),
                       (sel_name, "E: Hybrid (tuned)"), (f"hybrid_a{selected_alpha:g}_unconstrained", "E: Hybrid (tuned, no constraint handling)")]:
        for flag, name in [(False, "topic-only queries"), (True, "queries with explicit constraints")]:
            fr = test_pq[(test_pq.config == key) & (test_pq.has_constraints == flag)]
            if fr.empty:
                continue
            bq_rows.append({"method": label, "query_type": name, "n_queries": int(len(fr)), **{m: float(fr[m].mean()) for m in METRICS}})
    pd.DataFrame(bq_rows).to_csv(out / "by_query_type.csv", index=False)

    # ---------- rankings for annotation pooling (validation + test queries; top 10)
    rank_rows = []
    pool_cfgs = {"keyword": "keyword", "tfidf": "tfidf", "semantic": "semantic", "metadata": "metadata", "hybrid": sel_name}
    for i, qid in enumerate(qids):
        if split[i] == "train":
            continue
        for mname, cname in pool_cfgs.items():
            order = rank_scores(score_cache[cname][i], ids, 10)
            for r, j in enumerate(order, start=1):
                rank_rows.append({"query_id": qid, "method": mname, "rank": r, "dataset_id": int(ids[j]), "score": float(score_cache[cname][i][j])})
    pd.DataFrame(rank_rows).to_csv(out / "rankings_top10.csv", index=False)

    # ---------- latency of an end-to-end explained query
    sample_q = queries["query_text"].head(20).tolist()
    t0 = time.perf_counter()
    for q in sample_q:
        rec.recommend(q, k=5, alpha=selected_alpha)
    latency = (time.perf_counter() - t0) / max(len(sample_q), 1)

    # ---------- metadata about the run
    clean_report_path = resolve(catalog_path).with_name("cleaning_report.json")
    clean_report = read_json(clean_report_path) if clean_report_path.exists() else None
    meta = {
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "is_fixture_run": is_fixture,
        "semantic_backend": backend.name,
        "semantic_is_pretrained_minilm": args.semantic_backend == "minilm",
        "qrels_kind": qrels_kind,
        "human_verified_judgments": int((qrels["annotation_source"] == "human_reviewed").sum()) if "annotation_source" in qrels.columns else 0,
        "catalogue_size": int(len(cat)),
        "n_queries": int(len(queries)),
        "queries_per_split": queries["split"].value_counts().to_dict(),
        "n_relevance_judgments_listed": int(len(qrels)),
        "n_topic_groups": int(queries["topic_group"].nunique()),
        "selected_alpha": selected_alpha, "default_alpha": default_alpha, "alpha_grid": alpha_grid,
        "seed": cfg["seed"], "k": K, "relevance_threshold_for_binary_metrics": 2, "mrr_depth": MRR_DEPTH,
        "bootstrap": {"type": "cluster (topic group) percentile", "n_boot": args.n_boot, "level": 0.95},
        "timing_seconds": {"index_and_embedding_build": round(t_index, 3), "score_matrices_all_queries": round(t_scores, 3),
                           "mean_end_to_end_explained_query": round(latency, 4)},
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
                     "note": "CPU only; no GPU used"},
        "packages": pkg_versions(["numpy", "pandas", "scikit-learn", "scipy", "sentence-transformers", "torch", "openml", "streamlit"]),
        "cleaning_report": clean_report,
    }
    write_json(meta, out / "run_metadata.json")
    print(main[["method", "n_queries"] + METRICS].round(4).to_string(index=False))
    print(f"[run] results written to {out}")
    if is_fixture or not meta["semantic_is_pretrained_minilm"]:
        print("[run] NOTE: this run used the synthetic fixture and/or the LSA stand-in. These numbers are NOT paper results.")


if __name__ == "__main__":
    main()
