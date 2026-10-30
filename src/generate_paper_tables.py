"""Generate LaTeX tables, macros and figures for the paper FROM RESULT FILES ONLY.

Usage
-----
    python src/generate_paper_tables.py                       # reads outputs/results/
    python src/generate_paper_tables.py --results outputs/results

Behaviour
---------
  * Always writes the architecture figure (it describes the implementation, not results).
  * If outputs/results/ has no valid run, every results table/figure is replaced by a clearly
    marked PENDING placeholder, and \\resultsstatus is 'pending'.
  * A run is only accepted as final if it was NOT the synthetic fixture, used the pretrained
    MiniLM backend (not the LSA stand-in), and has the expected files. Otherwise the paper stays
    'pending' - fixture or stand-in numbers can never leak into the manuscript.
  * paper/tables/macros.tex defines the numbers quoted in the prose (catalogue size, query
    counts, ...) so text and tables cannot disagree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import load_config, resolve, resolve_dir

PENDING = r"\textbf{[PENDING: run \texttt{src/run\_experiments.py}]}"
METRICS = ["P@5", "R@5", "MRR", "NDCG@5"]


def tex(s: object) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$"))


def f3(x: float) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# ----------------------------------------------------------------------------- figures

def architecture_figure(path_pdf: Path, path_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8EEF7", ec="#2B4C7E"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#333333"))

    # top row: offline catalogue construction
    box(1, 32, 15, 10, "OpenML API\n(metadata only,\nno data files)")
    box(21, 32, 15, 10, "Raw metadata\nJSONL +\ntimestamp")
    box(41, 32, 15, 10, "Cleaning,\ndedup, quality\nfilter, provenance")
    box(61, 32, 15, 10, "Catalogue\n(explicit vs\ninferred fields)")
    box(81, 32, 17, 10, "Text representation\n(explicit fields\nonly)")
    for a, b in [(16, 21), (36, 41), (56, 61), (76, 81)]:
        arrow(a + 0.3, 37, b - 0.3, 37)
    # middle: indices
    box(61, 17, 15, 9, "Metadata index\n(counts, labels,\ntask, modality)", fc="#EAF4EA", ec="#2E7D32")
    box(81, 17, 17, 9, "TF-IDF index  +\nMiniLM embeddings\n(cached)", fc="#EAF4EA", ec="#2E7D32")
    arrow(69, 32, 69, 26.5)
    arrow(89, 32, 89, 26.5)
    # bottom row: online query
    box(1, 3, 15, 10, "User query\n(natural\nlanguage)", fc="#FFF4E0", ec="#B26A00")
    box(21, 3, 18, 10, "Requirement parser\n(rules: size, labels,\nmodality, condition)", fc="#FFF4E0", ec="#B26A00")
    box(44, 3, 24, 10, "Hybrid ranker\nalpha*Semantic +\n(1-alpha)*MetadataMatch\n+ hard-constraint handling", fc="#FDE8E8", ec="#B71C1C")
    box(74, 3, 24, 10, "Top-5 datasets\n+ source URL, scores,\nmatched / unmet / unknown\nrequirements, explanation", fc="#FFF4E0", ec="#B26A00")
    arrow(16.3, 8, 20.7, 8)
    arrow(39.3, 8, 43.7, 8)
    arrow(68.3, 8, 73.7, 8)
    arrow(69, 17, 60, 13.3)
    arrow(89, 17, 64, 13.3)
    ax.text(50, 44.5, "Offline: catalogue construction", ha="center", fontsize=7, style="italic", color="#2B4C7E")
    ax.text(30, 15.2, "Online: query processing", ha="center", fontsize=7, style="italic", color="#B26A00")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def alpha_figure(abl: pd.DataFrame, selected: float, path_pdf: Path, path_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for sp, style in (("validation", "--o"), ("test", "-s")):
        d = abl[(abl.split == sp) & (abl.method == "hybrid") & (abl.constraint_handling == True)].sort_values("alpha")  # noqa: E712
        ax.plot(d["alpha"], d["NDCG@5"], style, ms=3.5, lw=1.1, label=f"{sp}")
    ax.axvline(selected, color="gray", lw=0.7, ls=":")
    ax.set_xlabel(r"$\alpha$ (weight on semantic similarity)", fontsize=7)
    ax.set_ylabel("NDCG@5", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def methods_figure(main: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    labels = [m.split(" (")[0] for m in main["method"]]
    x = range(len(main))
    ax.bar([i - 0.2 for i in x], main["NDCG@5"], width=0.4, label="NDCG@5")
    ax.bar([i + 0.2 for i in x], main["P@5"], width=0.4, label="P@5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

def table_main(main: pd.DataFrame) -> str:
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Test-split retrieval results (mean; 95\% cluster-bootstrap CI over topic groups in parentheses). Relevance labels are synthetic/metadata-derived unless stated otherwise in the text.}",
             r"\label{tab:main}", r"\scriptsize", r"\begin{tabular}{lrcccc}", r"\toprule", r"Method & \#Q & P@5 & R@5 & MRR & NDCG@5 \\", r"\midrule"]
    for _, r in main.iterrows():
        cells = [f"{f3(r[m])} ({f3(r[m + '_lo'])}--{f3(r[m + '_hi'])})" for m in METRICS]
        lines.append(f"{tex(r['method'])} & {int(r['n_queries'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def table_ablation(abl: pd.DataFrame, selected: float) -> str:
    val = abl[abl.split == "validation"].set_index("config")
    test = abl[abl.split == "test"].set_index("config")
    order = ["semantic", "metadata"]
    for a in sorted(abl[abl.method == "hybrid"]["alpha"].dropna().unique()):
        order += [f"hybrid_a{a:g}_constrained", f"hybrid_a{a:g}_unconstrained"]
    def label(c: str) -> str:
        if c == "semantic":
            return "Semantic only"
        if c == "metadata":
            return "Metadata only"
        a = c.split("_a")[1].split("_")[0]
        suffix = "with constraint handling" if c.endswith("_constrained") else "no constraint handling"
        star = r"$^\dagger$" if (float(a) == selected and c.endswith("_constrained")) else ""
        return rf"Hybrid $\alpha$={a}, {suffix}{star}"
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Ablation. Validation NDCG@5 was used to select $\alpha$ ($^\dagger$); the test columns are reported for every configuration but were not used for selection.}",
             r"\label{tab:ablation}", r"\scriptsize", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Configuration & Val NDCG@5 & Test P@5 & Test R@5 & Test MRR & Test NDCG@5 \\", r"\midrule"]
    for c in order:
        if c not in test.index:
            continue
        lines.append(f"{label(c)} & {f3(val.loc[c, 'NDCG@5'])} & {f3(test.loc[c, 'P@5'])} & {f3(test.loc[c, 'R@5'])} & {f3(test.loc[c, 'MRR'])} & {f3(test.loc[c, 'NDCG@5'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def table_bytype(bq: pd.DataFrame) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Test NDCG@5 by query type (topic-only vs.\ explicit constraints).}", r"\label{tab:bytype}", r"\scriptsize",
             r"\begin{tabular}{lcc}", r"\toprule", r"Method & Topic-only & With constraints \\", r"\midrule"]
    for m, g in bq.groupby("method", sort=False):
        a = g[g.query_type == "topic-only queries"]
        b = g[g.query_type == "queries with explicit constraints"]
        lines.append(f"{tex(m)} & {f3(a['NDCG@5'].iloc[0]) if len(a) else '--'} ({int(a['n_queries'].iloc[0]) if len(a) else 0}) & "
                     f"{f3(b['NDCG@5'].iloc[0]) if len(b) else '--'} ({int(b['n_queries'].iloc[0]) if len(b) else 0}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_paired(pc: pd.DataFrame) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Paired differences on test queries (tuned hybrid minus comparator), NDCG@5, with 95\% cluster-bootstrap CI.}",
             r"\label{tab:paired}", r"\scriptsize", r"\begin{tabular}{lccc}", r"\toprule", r"Comparator & $\Delta$ & 95\% CI & CI excludes 0 \\", r"\midrule"]
    for _, r in pc[pc.metric == "NDCG@5"].iterrows():
        comp = r["comparison"].split(" - ")[1]
        lines.append(f"{tex(comp)} & {r['mean_difference']:+.3f} & [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {'yes' if r['ci_excludes_zero'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_missing(report: dict) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Missing values in important catalogue fields (after cleaning).}", r"\label{tab:missing}", r"\scriptsize",
             r"\begin{tabular}{lrr}", r"\toprule", r"Field & Missing & Proportion \\", r"\midrule"]
    for f, v in report["missingness"].items():
        lines.append(f"{tex(f)} & {v['missing']} & {100 * v['proportion']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def pending_table(label: str, caption: str, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\fbox{{\\parbox{{0.85\\linewidth}}{{\\centering {PENDING}\\\\"
            f"This table is generated from experimental output files and is intentionally empty until the experiments are run.}}}}\n\\end{{{env}}}\n")


# ----------------------------------------------------------------------------- findings sentence

def findings_sentence(main: pd.DataFrame, pc: pd.DataFrame, sel: float) -> str:
    """Purely factual, auto-generated summary (so prose cannot contradict the tables)."""
    best = {m: main.loc[main[m].idxmax(), "method"] for m in METRICS}
    row = main[main["method"].str.contains("tuned")].iloc[0]
    vs = pc[(pc.metric == "NDCG@5")]
    parts = []
    for _, r in vs.iterrows():
        comp = r["comparison"].split(" - ")[1]
        if comp.startswith("hybrid"):
            continue
        verdict = "above" if r["mean_difference"] > 0 else "below"
        sig = "interval excludes zero" if r["ci_excludes_zero"] else "interval includes zero"
        parts.append(f"{verdict} {tex(comp)} by {abs(r['mean_difference']):.3f} ({sig})")
    return (f"On the test split the tuned hybrid ($\\alpha={sel:g}$) reached NDCG@5 = {row['NDCG@5']:.3f}; its mean NDCG@5 was " + "; ".join(parts) + ". "
            f"Best NDCG@5: {tex(best['NDCG@5'])}; best MRR: {tex(best['MRR'])}.")


# ----------------------------------------------------------------------------- main

def valid_run(res: Path) -> tuple[bool, str, dict]:
    meta_path = res / "run_metadata.json"
    needed = ["main_results.csv", "ablation.csv", "paired_comparisons.csv", "by_query_type.csv", "alpha_selection.json"]
    if not meta_path.exists():
        return False, "no run_metadata.json", {}
    meta = json.loads(meta_path.read_text())
    if meta.get("is_fixture_run"):
        return False, "run used the synthetic fixture", meta
    if not meta.get("semantic_is_pretrained_minilm"):
        return False, "run used the LSA stand-in, not MiniLM", meta
    missing = [f for f in needed if not (res / f).exists()]
    if missing:
        return False, f"missing files: {missing}", meta
    return True, "ok", meta


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=cfg["paths"]["results_dir"])
    ap.add_argument("--catalog-report", default=f"{cfg['paths']['processed_dir']}/cleaning_report.json")
    ap.add_argument("--tables-dir", default=cfg["paths"]["paper_tables_dir"])
    ap.add_argument("--figures-dir", default=cfg["paths"]["paper_figures_dir"])
    ap.add_argument("--out-figures-dir", default=cfg["paths"]["figures_dir"])
    args = ap.parse_args()

    res = resolve_dir(args.results)
    tdir = resolve_dir(args.tables_dir)
    fdir = resolve_dir(args.figures_dir)
    odir = resolve_dir(args.out_figures_dir)
    architecture_figure(fdir / "architecture.pdf", odir / "architecture.png")
    (fdir / "architecture.png").write_bytes((odir / "architecture.png").read_bytes())

    ok, why, meta = valid_run(res)
    rep_path = resolve(args.catalog_report)
    report = json.loads(rep_path.read_text()) if rep_path.exists() else None
    real_report = report is not None and report.get("n_synthetic_records", 0) == 0

    m = "% AUTO-GENERATED by src/generate_paper_tables.py - do not edit by hand\n"
    m += macro("resultsstatus", "final" if ok else "pending")
    pend = r"\textbf{[PENDING]}"
    m += macro("catsize", str(report["n_final"]) if real_report else pend)
    m += macro("catraw", str(report["n_raw"]) if real_report else pend)
    if ok:
        qs = meta["queries_per_split"]
        m += macro("nqueries", str(meta["n_queries"]))
        m += macro("ntrain", str(qs.get("train", 0)))
        m += macro("nval", str(qs.get("validation", 0)))
        m += macro("ntest", str(qs.get("test", 0)))
        m += macro("njudg", str(meta["n_relevance_judgments_listed"]))
        m += macro("nhuman", str(meta["human_verified_judgments"]))
        m += macro("ntopics", str(meta["n_topic_groups"]))
        m += macro("selalpha", f"{meta['selected_alpha']:g}")
        m += macro("qrelskind", tex(meta["qrels_kind"]))
        t = meta["timing_seconds"]
        m += macro("tindex", f"{t['index_and_embedding_build']:.1f}")
        m += macro("tquery", f"{1000 * t['mean_end_to_end_explained_query']:.0f}")
        m += macro("hwdesc", tex(meta["hardware"]["platform"]) + ", CPU only")
        main_df = pd.read_csv(res / "main_results.csv")
        abl = pd.read_csv(res / "ablation.csv")
        pc = pd.read_csv(res / "paired_comparisons.csv")
        bq = pd.read_csv(res / "by_query_type.csv")
        sel = meta["selected_alpha"]
        m += macro("findings", findings_sentence(main_df, pc, sel))
        (tdir / "table_main.tex").write_text(table_main(main_df))
        (tdir / "table_ablation.tex").write_text(table_ablation(abl, sel))
        (tdir / "table_bytype.tex").write_text(table_bytype(bq))
        (tdir / "table_paired.tex").write_text(table_paired(pc))
        alpha_figure(abl, sel, fdir / "alpha_sweep.pdf", odir / "alpha_sweep.png")
        (fdir / "alpha_sweep.png").write_bytes((odir / "alpha_sweep.png").read_bytes())
        methods_figure(main_df, fdir / "methods_bar.pdf", odir / "methods_bar.png")
        (fdir / "methods_bar.png").write_bytes((odir / "methods_bar.png").read_bytes())
        print("[tables] final results tables and figures generated")
    else:
        for k in ("nqueries", "ntrain", "nval", "ntest", "njudg", "nhuman", "ntopics", "selalpha", "qrelskind", "tindex", "tquery", "hwdesc"):
            m += macro(k, pend)
        m += macro("findings", r"Quantitative findings are \textbf{pending}: the experiments have not been run on the real catalogue with the pretrained model.")
        (tdir / "table_main.tex").write_text(pending_table("tab:main", "Test-split retrieval results (pending)."))
        (tdir / "table_ablation.tex").write_text(pending_table("tab:ablation", "Ablation results (pending)."))
        (tdir / "table_bytype.tex").write_text(pending_table("tab:bytype", "NDCG@5 by query type (pending).", wide=False))
        (tdir / "table_paired.tex").write_text(pending_table("tab:paired", "Paired differences (pending).", wide=False))
        for name in ("alpha_sweep", "methods_bar"):  # placeholder figures so LaTeX compiles
            fig, ax = plt.subplots(figsize=(3.4, 1.6))
            ax.axis("off")
            ax.text(0.5, 0.5, "PENDING\n(run experiments, then\nsrc/generate_paper_tables.py)", ha="center", va="center", fontsize=8, color="#B71C1C")
            fig.savefig(fdir / f"{name}.pdf", bbox_inches="tight")
            fig.savefig(odir / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"[tables] results PENDING ({why}); placeholders written")

    if real_report:
        (tdir / "table_missing.tex").write_text(table_missing(report))
    else:
        (tdir / "table_missing.tex").write_text(pending_table("tab:missing", "Missing values in important catalogue fields (pending).", wide=False))
    (tdir / "macros.tex").write_text(m)


if __name__ == "__main__":
    main()
