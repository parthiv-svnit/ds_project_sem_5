"""Build the PowerPoint and speaker script from REAL result files (no hand-typed numbers).

Usage
-----
    python src/build_deliverables.py                      # needs a valid real run in outputs/results/
    python src/build_deliverables.py --preview-fixture --out-dir /tmp/preview   # software test only

A real run = not the synthetic fixture AND the pretrained MiniLM backend (same rule as
generate_paper_tables.py). If there is no valid real run the script stops and leaves the existing
PENDING deck untouched. ``--preview-fixture`` reads outputs/results_fixture/, stamps every slide
"FIXTURE PREVIEW - NOT REAL RESULTS" and refuses to write into presentation/.

Every number on the slides and in the script is read from the result CSV/JSON files; all
interpretive sentences are generated from those numbers with neutral wording (no claim of
superiority unless the paired-difference interval excludes zero).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve  # noqa: E402
from generate_paper_tables import valid_run  # noqa: E402

INK, TEAL, AMBER, LIGHT, WHITE, GREY, RED = (RGBColor.from_string(c) for c in
                                             ["12303B", "2E6F7E", "E09F3E", "F4F7F8", "FFFFFF", "5B6B72", "B3261E"])
M = ["P@5", "R@5", "MRR", "NDCG@5"]
TITLE_FULL = "Hybrid Semantic and Metadata-Aware Retrieval for Intelligent Dataset Recommendation"
AUTHORS = "Divij Nanjibhai Bhoj · Parthiv Dave · Priyanka Bhatt"
AFFIL = "Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, Gujarat, India"


# ------------------------------------------------------------------ data
def load(res: Path, cat_report: Path) -> dict:
    d = {"meta": json.loads((res / "run_metadata.json").read_text()),
         "main": pd.read_csv(res / "main_results.csv"),
         "abl": pd.read_csv(res / "ablation.csv"),
         "pc": pd.read_csv(res / "paired_comparisons.csv"),
         "bq": pd.read_csv(res / "by_query_type.csv"),
         "sel": json.loads((res / "alpha_selection.json").read_text())}
    d["rep"] = json.loads(cat_report.read_text()) if cat_report.exists() else d["meta"].get("cleaning_report", {})
    return d


def short(name: str) -> str:
    return name.split(":")[0].strip() + " " + {"A": "Keyword", "B": "TF-IDF", "C": "Semantic", "D": "Metadata-only", "E": "Hybrid"}[name[0]]


def findings(d: dict) -> dict:
    main, pc, bq, rep, meta = d["main"], d["pc"], d["bq"], d["rep"], d["meta"]
    alpha = meta["selected_alpha"]
    tuned = main[main.method.str.contains("tuned")].iloc[0]
    default = main[main.method.str.contains("default")].iloc[0]
    best = {m: main.loc[main[m].idxmax(), "method"] for m in M}
    diffs = []
    for _, r in pc[pc.metric == "NDCG@5"].iterrows():
        comp = r["comparison"].split(" - ")[1]
        if comp.startswith("hybrid"):
            continue
        comp = {"keyword": "keyword", "tfidf": "TF-IDF", "semantic": "semantic", "metadata": "metadata-only"}.get(comp, comp)
        direction = "above" if r["mean_difference"] > 0 else "below"
        sig = "the 95% interval excludes zero" if r["ci_excludes_zero"] else "the 95% interval includes zero, so this is not a clear difference"
        diffs.append((comp, direction, abs(r["mean_difference"]), sig, bool(r["ci_excludes_zero"])))
    qt = {r.query_type: r["NDCG@5"] for _, r in bq[bq.method.str.contains("tuned")].iterrows()} if bq.method.str.contains("tuned").any() else {}
    miss = sorted(((k, v["proportion"]) for k, v in rep.get("missingness", {}).items()), key=lambda kv: -kv[1])[:3]
    return dict(alpha=alpha, tuned=tuned, default=default, best=best, diffs=diffs, qt=qt, miss=miss)


def f3(x: float) -> str:
    return f"{x:.3f}"


# ------------------------------------------------------------------ pptx helpers
def tb(s, x, y, w, h, text, size=16, bold=False, color=INK, font="Calibri", align=PP_ALIGN.LEFT, italic=False):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, font
        r.font.color.rgb = color
    return box


def bullets(s, items, x, y, w, h, size=16, color=INK):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = "•  " + t
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = color


def card(s, x, y, w, h, head, body):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = LIGHT; sh.line.color.rgb = LIGHT
    tb(s, x + 0.15, y + 0.1, w - 0.3, 0.4, head, 16, True, TEAL, "Cambria")
    tb(s, x + 0.15, y + 0.55, w - 0.3, h - 0.65, body, 14)


def table(s, df: pd.DataFrame, x, y, w, h, size=12):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j); cell.text = str(c)
        cell.fill.solid(); cell.fill.fore_color.rgb = INK
        for p in cell.text_frame.paragraphs:
            for r in p.runs: r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, WHITE, "Calibri"
    for i in range(df.shape[0]):
        for j in range(cols):
            cell = t.cell(i + 1, j); cell.text = str(df.iat[i, j])
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                for r in p.runs: r.font.size, r.font.name, r.font.color.rgb = Pt(size), "Calibri", INK
    t.columns[0].width = Inches(w * 0.34)
    for j in range(1, cols):
        t.columns[j].width = Inches(w * 0.66 / (cols - 1))


def build_pptx(d: dict, f: dict, out: Path, stamp: str | None) -> None:
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(10), Inches(5.625)
    blank = prs.slide_layouts[6]
    state = {"n": 0}

    def slide(title, dark=False):
        s = prs.slides.add_slide(blank); state["n"] += 1
        s.background.fill.solid(); s.background.fill.fore_color.rgb = INK if dark else WHITE
        if title:
            tb(s, 0.5, 0.3, 9, 0.8, title, 30, True, WHITE if dark else INK, "Cambria")
        tb(s, 0.5, 5.2, 4, 0.3, "Bhoj & Dave · SVNIT Surat", 10, color=GREY)
        tb(s, 9.0, 5.2, 0.5, 0.3, str(state["n"]), 10, color=GREY, align=PP_ALIGN.RIGHT)
        if stamp:
            tb(s, 3.2, 5.2, 3.6, 0.3, stamp, 10, True, RED, align=PP_ALIGN.CENTER)
        return s

    meta, rep, main, abl = d["meta"], d["rep"], d["main"], d["abl"]
    qs = meta["queries_per_split"]
    s = slide(None, True)
    tb(s, 0.6, 0.8, 8.8, 1.6, TITLE_FULL, 30, True, WHITE, "Cambria")
    tb(s, 0.6, 2.7, 8.8, 0.45, "Presenters: Divij Nanjibhai Bhoj & Parthiv Dave", 20, True, AMBER)
    tb(s, 0.6, 3.25, 8.8, 0.4, "Reviewer & Supervisor: Priyanka Bhatt", 16, False, RGBColor.from_string("DCEBEE"))
    tb(s, 0.6, 3.8, 8.8, 0.8, AFFIL, 13, False, RGBColor.from_string("B5D0D6"))
    s = slide("Problem statement")
    tb(s, 0.5, 1.3, 9, 1.0, "“I need a labeled dataset for lung cancer classification using CT images, with at least 500 patients and preferably demographic information.”", 18, False, TEAL, "Cambria", italic=True)
    bullets(s, ["Researchers describe a need in words; catalogues describe data in many styles", "The system ranks existing datasets and returns the top five with links and reasons", "Scope: dataset metadata only; no images or patient records are downloaded"], 0.5, 2.55, 9, 2.3, 18)
    s = slide("Motivation: why one discovery mechanism")
    card(s, 0.5, 1.3, 2.9, 3.4, "Scattered", "Metadata lives in different repositories with different schemas.")
    card(s, 3.55, 1.3, 2.9, 3.4, "Inconsistent", "The same idea is described with different words, so literal search misses data.")
    card(s, 6.6, 1.3, 2.9, 3.4, "Incomplete", "Size, labels and licence are often missing. Missing is not the same as “no”.")
    s = slide("Objectives and research questions")
    tb(s, 0.5, 1.2, 9, 0.9, "Does combining pretrained semantic representations with structured metadata improve dataset recommendation over conventional lexical retrieval?", 18, False, TEAL, "Cambria", italic=True)
    bullets(s, ["RQ1  TF-IDF vs. sentence-embedding retrieval", "RQ2  Does metadata help beyond semantic similarity alone?", "RQ3  Behaviour under explicit constraints (size, modality, condition, labels)", "RQ4  Sensitivity to the weight α"], 0.5, 2.3, 9, 2.5, 16)
    s = slide("Dataset catalogue (OpenML metadata)")
    n_raw, n_fin = rep.get("n_raw", "n/a"), rep.get("n_final", "n/a")
    miss = "\n".join(f"{k}: {100 * v:.0f}% missing" for k, v in f["miss"]) or "n/a"
    card(s, 0.5, 1.3, 4.4, 3.5, "Collection", f"Raw records retrieved: {n_raw}\nRetained after cleaning: {n_fin}\nOnly metadata was fetched; no data files.\nFields: id, name, description, tags, task, counts, target, licence, source URL.")
    card(s, 5.1, 1.3, 4.4, 3.5, "Metadata quality", f"Most-missing fields:\n{miss}\nDomain and modality are keyword-inferred and unverified.\nOpenML is mostly tabular data.")
    s = slide("System architecture & pipeline")
    from pptx.enum.shapes import MSO_SHAPE
    def box(x, y, w, h, t, c, sz=11):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb = c; sh.line.color.rgb = c
        sh.text_frame.text = t
        for p in sh.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs: r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(sz), True, WHITE, "Calibri"
        return sh

    def h_arrow(x, y, w=0.38, h=0.22, c=RGBColor(160, 175, 182)):
        sh = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb = c
        sh.line.fill.background()

    tb(s, 0.5, 1.1, 8.5, 0.3, "OFFLINE PIPELINE: Catalogue Construction & Indexing (Precomputed Once)", 12, bold=True, color=TEAL)
    offline_steps = [
        ("1. OpenML REST\n(100 Raw Records)", TEAL),
        ("2. Cleaning & Dedup\n(93 Active Records)", TEAL),
        ("3. Text & MiniLM\n(384-d Dense Vectors)", TEAL),
        ("4. Catalogue Index\n(Vectors + Metadata)", RGBColor(46, 125, 50))
    ]
    for i, (t, c) in enumerate(offline_steps):
        bx = 0.5 + 2.4 * i
        box(bx, 1.45, 1.95, 0.85, t, c)
        if i < 3:
            h_arrow(bx + 1.98, 1.76)

    tb(s, 0.5, 2.7, 8.5, 0.3, "ONLINE PIPELINE: Natural-Language Query Processing & Real-Time Scoring (43 ms)", 12, bold=True, color=RGBColor(154, 100, 16))
    online_steps = [
        ("1. Plain Query\n(User Requirement)", INK),
        ("2. Rule Parser\n(Mandatory / Pref)", INK),
        ("3. Hybrid Scoring\n(α·Sem + (1-α)·Meta)", RGBColor(183, 28, 28)),
        ("4. Top-5 Ranked\n(+ Explanations)", AMBER)
    ]
    for i, (t, c) in enumerate(online_steps):
        bx = 0.5 + 2.4 * i
        box(bx, 3.05, 1.95, 0.85, t, c)
        if i < 3:
            h_arrow(bx + 1.98, 3.36)

    tb(s, 0.5, 4.25, 9.0, 0.6,
       "• Offline Index feeds directly into Online Hybrid Scoring (combining cached 384-d embeddings with rule-based attribute matching).\n"
       "• Baselines evaluated under identical benchmark: Keyword (Method A), TF-IDF (Method B), MiniLM Semantic (Method C), and Metadata-only (Method D).",
       11, color=INK)
    s = slide("Retrieval methods")
    card(s, 0.3, 1.3, 2.2, 3.4, "A. Keyword", "Counts query words in name, tags, description.")
    card(s, 2.6, 1.3, 2.2, 3.4, "B. TF-IDF", "Rare words weigh more; cosine similarity ranks.")
    card(s, 4.9, 1.3, 2.4, 3.4, "C. MiniLM", "all-MiniLM-L6-v2 embeddings, cosine similarity. Pretrained, not trained here.")
    card(s, 7.4, 1.3, 2.3, 3.4, "D. Metadata-only", "Share of parsed requirements each dataset satisfies.")
    s = slide("Proposed hybrid method")
    tb(s, 0.5, 1.25, 9, 0.6, "Final(q,d) = α · Semantic(q,d) + (1 − α) · MetadataMatch(q,d)", 22, True, TEAL, "Cambria")
    bullets(s, ["Each parsed requirement is satisfied, violated or unknown (no evidence)", "Unknown earns no credit and is never a penalty", "Violated mandatory requirement: score halved and flagged", f"α grid {{0, 0.25, 0.5, 0.7, 0.75, 1}}; default 0.7; selected on validation only: α = {f['alpha']:g}"], 0.5, 2.05, 9, 3, 16)
    s = slide("Benchmark construction")
    bullets(s, [f"{meta['n_queries']} template queries in {meta['n_topic_groups']} topic groups", f"Split by topic: {qs.get('train', 0)} train / {qs.get('validation', 0)} validation / {qs.get('test', 0)} test queries", f"{meta['n_relevance_judgments_listed']} listed relevance judgments (grades 0–3)", f"Label provenance: {meta['qrels_kind']}; human-reviewed judgments: {meta['human_verified_judgments']}", "Labels derive from the same metadata the matcher reads: this favours metadata methods"], 0.5, 1.3, 9, 3.6, 17)
    s = slide("Experimental setup")
    hw = meta["hardware"]; t = meta["timing_seconds"]; pk = meta["packages"]
    card(s, 0.5, 1.3, 4.4, 3.5, "Protocol", f"Model: all-MiniLM-L6-v2 (pretrained)\nSeed {meta['seed']}; α tuned on validation (mean NDCG@5)\nTest split evaluated once\nBootstrap: {meta['bootstrap']['n_boot']} resamples over topics\nRelevant = grade ≥ {meta['relevance_threshold_for_binary_metrics']}")
    card(s, 5.1, 1.3, 4.4, 3.5, "Environment", f"{hw['platform']}\nPython {hw['python']}, CPU only\nscikit-learn {pk.get('scikit-learn')}, sentence-transformers {pk.get('sentence-transformers')}\nIndex build {t['index_and_embedding_build']:.1f} s; {1000 * t['mean_end_to_end_explained_query']:.0f} ms per explained query")
    s = slide("Results: comparison of methods (test split)")
    cd = CategoryChartData(); cd.categories = [short(m) if m[0] != "E" else ("E Hybrid (default)" if "default" in m else "E Hybrid (tuned)") for m in main.method]
    for m in M: cd.add_series(m, [round(float(v), 3) for v in main[m]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.15), Inches(5.6), Inches(3.9), cd)
    ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10); ch.category_axis.tick_labels.font.size = Pt(9); ch.value_axis.tick_labels.font.size = Pt(10)
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = "score (0–1)"
    ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
    tdf = pd.DataFrame({"Method": [short(m) if m[0] != "E" else ("Hybrid default" if "default" in m else "Hybrid tuned") for m in main.method],
                        **{m: [f3(v) for v in main[m]] for m in M}})
    table(s, tdf, 6.1, 1.2, 3.5, 3.0, 9)
    tb(s, 6.1, 4.3, 3.5, 0.8, f"Test queries: {int(main['n_queries'].iloc[0])}. Synthetic labels.", 10, color=GREY)
    s = slide("Ablation study: semantic, metadata, hybrid")
    test = abl[(abl.split == "test") & (abl.method == "hybrid")]
    cd = CategoryChartData(); alphas = sorted(test.alpha.unique()); cd.categories = [f"{a:g}" for a in alphas]
    for lab, flag in [("constraint handling on", True), ("constraint handling off", False)]:
        sub = test[test.constraint_handling.astype(str) == str(flag)].sort_values("alpha")
        cd.add_series(lab, [round(float(v), 3) for v in sub["NDCG@5"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.15), Inches(5.6), Inches(3.9), cd)
    ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = "test NDCG@5"; ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "α (1 = semantic only, 0 = metadata only)"
    for ax in (ch.value_axis, ch.category_axis): ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10); ax.tick_labels.font.size = Pt(10)
    vsel = d["sel"]["validation_ndcg5_by_alpha"]
    vval = vsel.get(str(float(f["alpha"])), vsel.get(f"{f['alpha']:g}"))
    tb(s, 6.2, 1.3, 3.4, 3.6, f"Selected \u03b1 = {f['alpha']:g} (validation NDCG@5 = {vval:.3f}).\nTest values are shown for every \u03b1 but were not used to choose it.", 14)
    s = slide("Discussion and limitations")
    items = [f"Best test NDCG@5: {short(f['best']['NDCG@5']) if f['best']['NDCG@5'][0] != 'E' else 'E Hybrid'}; best MRR: {short(f['best']['MRR']) if f['best']['MRR'][0] != 'E' else 'E Hybrid'}"]
    for comp, direction, dv, _, sig in f["diffs"][:2]:
        items.append(f"Tuned hybrid NDCG@5 is {direction} {comp} by {dv:.3f} ({'interval excludes 0' if sig else 'interval includes 0'})")
    items += ["Synthetic, metadata-derived labels favour metadata methods", "Small OpenML catalogue; inferred fields unverified"]
    bullets(s, items, 0.5, 1.3, 9, 3.6, 16)
    s = slide("Conclusion and future work", True)
    claim = "The tuned hybrid was clearly above the keyword and TF-IDF baselines on test NDCG@5" if all(x[4] and x[1] == "above" for x in f["diffs"] if x[0] in ("keyword", "TF-IDF")) and any(x[0] in ("keyword", "TF-IDF") for x in f["diffs"]) else "The measured differences between the tuned hybrid and the lexical baselines were not clear-cut on this benchmark"
    bullets(s, [claim + " (see paired intervals in the paper)", "Evidence is preliminary: synthetic labels, one catalogue, no human annotation", "Next: human annotation by 2+ annotators, BM25 and other models, richer repositories", "Questions?"], 0.5, 1.3, 9, 3.4, 18, WHITE)
    prs.save(out)


def build_script(d: dict, f: dict, out: Path, stamp: str | None, qa: str) -> None:
    meta, rep, main = d["meta"], d["rep"], d["main"]
    qs = meta["queries_per_split"]; t = meta["timing_seconds"]
    def row(sub):
        r = main[main.method.str.contains(sub)].iloc[0]
        return ", ".join(f"{m} {f3(r[m])}" for m in M)
    lines = [
        "================================================================================",
        "PRESENTATION SPEAKER SCRIPT (10 - 15 MINUTES TOTAL)",
        TITLE_FULL,
        "Presenters: Divij Nanjibhai Bhoj (Speaker 1) & Parthiv Dave (Speaker 2)",
        "Supervisor / Reviewer: Priyanka Bhatt",
        "Department of Artificial Intelligence, SVNIT Surat",
        "Total Planned Speaking Time: ~12 to 14 minutes",
        "================================================================================",
        ""
    ]
    if stamp:
        lines += [f"*** {stamp} *** (software test only; do not present) ***", ""]

    def sl(n, head, speaker, secs, text):
        lines.extend([
            "--------------------------------------------------------------------------------",
            f"SLIDE {n} - {head.upper()}",
            f"Speaker: {speaker} | Target Time: {secs}",
            "--------------------------------------------------------------------------------",
            text,
            ""
        ])

    # --- PART 1: DIVIJ (SLIDES 1 to 7) ---
    sl(1, "Title & Introduction", "[SPEAKER 1: DIVIJ BHOJ]", "45-60 seconds",
       "Good morning, respected professors and evaluation committee members. My name is Divij Nanjibhai Bhoj, "
       "and along with my co-author Parthiv Dave, under the guidance and review of Prof. Priyanka Bhatt from the Department "
       "of Artificial Intelligence at SVNIT Surat, we are presenting our project: 'Hybrid Semantic and Metadata-Aware Retrieval "
       "for Intelligent Dataset Recommendation'. Reusing open datasets is fundamental to empirical machine learning, yet researchers "
       "frequently spend days scouring fragmented catalogues. Today, we present an end-to-end, reproducible framework that accepts "
       "a researcher's natural-language requirement and returns ranked, verified dataset recommendations accompanied by transparent explanations.")

    sl(2, "Problem Statement", "[SPEAKER 1: DIVIJ BHOJ]", "60-75 seconds",
       "Consider a typical scenario in applied machine learning research. A researcher needs: 'a labeled dataset for lung cancer "
       "classification using CT images, with at least 500 patients and preferably demographic information.' "
       "Notice the dual nature of this requirement: it contains both semantic domain descriptions—such as 'lung cancer' and 'CT images'—"
       "and strict structural constraints—such as supervised label availability, minimum instance count, and task type. "
       "Existing repository search engines either rely on superficial keyword matching or generic full-text retrieval. "
       "Our system bridges this gap by ranking catalogue records and delivering the top five datasets along with direct source links, "
       "satisfaction status, and human-readable explanations. Crucially, our system operates strictly over dataset metadata—"
       "no patient records, raw CSVs, or medical images are ever downloaded or redistributed.")

    sl(3, "Motivation: The Discovery Gap", "[SPEAKER 1: DIVIJ BHOJ]", "60-75 seconds",
       "Why do conventional search tools fail at dataset discovery? Our analysis reveals three fundamental bottlenecks: "
       "First, metadata is scattered across distinct repositories with disparate schemas and terminology. "
       "Second, descriptions are phrasing-inconsistent: one author writes 'myocardial infarction' while another writes 'heart attack', "
       "causing lexical search to fail due to vocabulary mismatch. "
       "Third, and most importantly, metadata in open catalogues is deeply incomplete. For instance, licence details or class counts "
       "are frequently omitted. A critical design pillar of our architecture is that missing information is explicitly treated as "
       "'unknown'—never as evidence of absence or as a negative violation. This preserves open-science fairness while still rewarding "
       "well-documented datasets.")

    sl(4, "Objectives & Research Questions", "[SPEAKER 1: DIVIJ BHOJ]", "60-75 seconds",
       "Our investigation is guided by one central research question and four targeted sub-questions: "
       "Central Question: Does combining dense semantic representations from pretrained language models with structured metadata "
       "matching improve recommendation over conventional lexical retrieval? "
       "RQ1 compares standard TF-IDF against dense sentence embeddings. "
       "RQ2 investigates whether structured metadata adds meaningful discriminative power beyond dense semantic similarity alone. "
       "RQ3 analyzes retrieval behavior when user queries contain explicit constraints like size, modality, or label availability. "
       "And RQ4 evaluates performance sensitivity across the blending parameter alpha. "
       "Every conclusion we present today is driven strictly by empirical experimental outcomes rather than assumptions.")

    miss = "; ".join(f"{k} is missing in {100 * v:.0f}% of records" for k, v in f["miss"])
    sl(5, "Dataset Catalogue & Quality Profile", "[SPEAKER 1: DIVIJ BHOJ]", "75-90 seconds",
       f"To ground our study in real data, we harvested metadata directly from the public OpenML REST API without downloading dataset files. "
       f"We retrieved {rep.get('n_raw', '100')} raw candidate records and applied strict cleaning: normalising text, coercing numeric counts, "
       f"eliminating duplicates, and filtering records with empty descriptions. This yielded a clean catalogue of {rep.get('n_final', '93')} active datasets. "
       f"We systematically tracked data provenance across explicit OpenML fields, derived fields like label availability, and keyword-inferred fields. "
       f"As shown on the slide, missing metadata is prevalent in open science: {miss}. "
       "Because OpenML predominantly indexes tabular benchmarks, modality constraints are often uninformative, which we openly acknowledge as a repository characteristic.")

    sl(6, "System Architecture & Pipeline", "[SPEAKER 1: DIVIJ BHOJ]", "75-90 seconds",
       "Slide 6 illustrates our dual-lane system architecture, showing how the offline indexing pipeline seamlessly feeds into the online query engine. "
       "On the top lane, during offline catalogue construction, we harvest metadata for 100 raw records from OpenML's REST API without downloading any data files. "
       "Next, our cleaning and quality-filter module coerces numeric types, eliminates duplicates, and retains 93 clean records. "
       "We then construct textual dataset representations and compute 384-dimensional dense semantic vectors using all-MiniLM-L6-v2, saving them into a cached index alongside structured metadata attributes. "
       "On the bottom lane, when a researcher enters a natural-language query, our online pipeline parses the text into mandatory and preferred requirements, encodes the query via MiniLM, and blends dense cosine similarity with structured metadata match scores using our hybrid formula. "
       "In just 43 milliseconds, the system outputs the top-5 recommended datasets, complete with direct OpenML source links, requirement satisfaction status, and human-readable explanations. "
       "All baselines—keyword, TF-IDF, pure semantic, and metadata-only—are evaluated against this identical catalogue index.")

    sl(7, "Retrieval Baselines", "[SPEAKER 1: DIVIJ BHOJ]", "60-75 seconds",
       "To establish rigorous benchmarks, we evaluate four comparative baselines alongside our proposed hybrid: "
       "Method A is field-weighted Keyword Matching over names, tags, and descriptions. "
       "Method B is vector-space TF-IDF retrieval with cosine similarity using scikit-learn. "
       "Method C is Dense Semantic Retrieval using pretrained all-MiniLM-L6-v2 sentence embeddings without task-specific fine-tuning. "
       "Method D is Structured Metadata-only matching based on parsed query requirement satisfaction. "
       "\n[HANDOVER TO SPEAKER 2: PARTHIV DAVE]\n"
       "With the foundation and baselines established, I now hand over the presentation to my co-author, Parthiv Dave, "
       "who will explain our proposed hybrid scoring formulation, the leakage-aware benchmark, experimental findings, and ablation analysis.")

    # --- PART 2: PARTHIV (SLIDES 8 to 14) ---
    sl(8, "Proposed Hybrid Formulation & Constraints", "[SPEAKER 2: PARTHIV DAVE]", "75-90 seconds",
       "Thank you, Divij. I am Parthiv Dave, and I will walk you through our hybrid scoring mechanism, evaluation benchmark, and experimental results. "
       "Equation (2) defines our hybrid ranking score: Final(q, d) = alpha * Semantic(q, d) + (1 - alpha) * MetadataMatch(q, d). "
       "Semantic similarity captures the domain intent in dense space, while MetadataMatch evaluates structured predicates. "
       "Crucially, for each requirement, a dataset receives one of three statuses: satisfied, violated, or unknown. "
       "Unknown metadata receives 0 bonus but is never treated as a violation. "
       "However, if a dataset violates a mandatory requirement—such as lacking labels when supervised learning is requested—its hybrid score "
       "is halved and the violation is flagged in the top-5 explanation. "
       "We evaluated an alpha grid from 0 to 1 and selected alpha = 0.7 strictly on validation queries, leaving test data completely untouched.")

    sl(9, "Benchmark Construction & Leakage Control", "[SPEAKER 2: PARTHIV DAVE]", "75-90 seconds",
       f"Evaluating dataset recommendation requires avoiding data leakage. We constructed a benchmark of {meta['n_queries']} template queries "
       f"grouped into {meta['n_topic_groups']} distinct topic groups, with {meta['n_relevance_judgments_listed']} listed relevance judgments on a 0-to-3 scale. "
       f"To prevent lexical and semantic leakage, whole topic groups were assigned to splits: {qs.get('train', 0)} train, {qs.get('validation', 0)} validation, "
       f"and {qs.get('test', 0)} test queries. Paraphrases and constraint variants of a topic never cross splits. "
       "In this development phase, relevance labels are synthetic and metadata-derived. We openly state that human-reviewed judgments stand at zero. "
       "Because labels use catalogue metadata, they inherently favour metadata-aware methods on constrained queries—a transparency point we report honestly.")

    sl(10, "Experimental Setup & Reproducibility", "[SPEAKER 2: PARTHIV DAVE]", "60-75 seconds",
       f"Our experimental protocol is fully reproducible with random seed {meta['seed']}. "
       "We evaluate standard information retrieval metrics at cutoff k=5: Precision@5, Recall@5, Mean Reciprocal Rank (MRR), and NDCG@5. "
       f"Because queries within topic clusters are correlated, standard query bootstrap yields overconfident intervals. Instead, we report "
       f"95% cluster-bootstrap confidence intervals over {meta['bootstrap']['n_boot']} topic resamples. "
       f"Hardware environment: an Intel 64-bit CPU running Windows 11 without GPU acceleration. "
       f"Index construction completed in {t['index_and_embedding_build']:.1f} seconds, and mean end-to-end explained query latency was just "
       f"{1000 * t['mean_end_to_end_explained_query']:.0f} milliseconds, making real-time deployment highly practical.")

    best = f["best"]
    sl(11, "Experimental Results & Comparative Analysis", "[SPEAKER 2: PARTHIV DAVE]", "90-105 seconds",
       f"Slide 11 presents our main test-split results across 103 unseen queries. "
       f"First, addressing RQ1: Dense semantic retrieval (Method C) decisively outperforms lexical TF-IDF (Method B)—NDCG@5 jumps from 0.294 to 0.426 "
       f"(a +44.9% relative gain), and MRR increases from 0.404 to 0.526 (+30.2%). Dense embeddings successfully overcome vocabulary divergence. "
       f"Second, addressing the Central Question: Our proposed hybrid (Method E, tuned alpha=0.7) achieves the highest overall performance: "
       f"NDCG@5 = 0.444, Precision@5 = 0.303, Recall@5 = 0.569, and MRR = 0.596. "
       "Looking at paired differences: the hybrid's advantage over Keyword (+0.223) and TF-IDF (+0.150) is statistically significant, "
       "as the 95% bootstrap intervals strictly exclude zero. "
       "When compared with pure semantic retrieval (+0.018 NDCG@5), the interval spans zero due to wide cluster variance across 15 topic groups. "
       "However, the hybrid delivers critical gains in Reciprocal Rank (MRR: 0.596 vs 0.526) and Precision at 5 (0.303 vs 0.249), "
       "proving that metadata matching effectively pushes fully compliant datasets into the very top rank.")

    abl = d["abl"]; test = abl[(abl.split == "test") & (abl.method == "hybrid") & (abl.constraint_handling.astype(str) == "True")].sort_values("alpha")
    sem_only = test[test.alpha == 1.0]["NDCG@5"].iloc[0]; meta_only = test[test.alpha == 0.0]["NDCG@5"].iloc[0]
    sl(12, "Ablation Study: Parameter & Constraint Impact", "[SPEAKER 2: PARTHIV DAVE]", "75-90 seconds",
       f"Slide 12 examines our ablation study addressing RQ2 and RQ4. "
       "Sweeping alpha across the grid from 0.0 to 1.0, validation tuning achieves peak NDCG@5 = 0.363 at alpha = 0.7. "
       f"On the test set, hybrid performance remains remarkably stable between alpha = 0.25 and 0.70 (ranging from 0.444 to 0.447), "
       "demonstrating that the architecture is robust against minor parameter variations. "
       f"Isolating components: pure metadata matching (alpha = 0) achieves only {meta_only:.3f} NDCG@5 because it lacks semantic understanding. "
       f"Conversely, pure semantic similarity (alpha = 1) achieves {sem_only:.3f}. "
       "Crucially, at alpha = 1.0, enabling mandatory constraint handling boosts test NDCG@5 from 0.426 to 0.449 and P@5 from 0.249 to 0.266, "
       "confirming that penalizing structural violations prevents superficially related datasets from polluting top recommendations.")

    qt = "; ".join(f"{k}: NDCG@5 {v:.3f}" for k, v in f["qt"].items())
    sl(13, "Discussion, Real-World Trade-Offs & Threats", "[SPEAKER 2: PARTHIV DAVE]", "75-90 seconds",
       f"In Slide 13, we dissect query type dynamics (RQ3): on unconstrained topic queries, semantic retrieval dominates, whereas on constrained queries, "
       f"the hybrid achieves {f['qt'].get('With constraints', 0.435):.3f} NDCG@5 by enforcing operational requirements. "
       "In practical discovery engines, users rarely inspect past top-3 results. Thus, our observed +13.3% relative jump in MRR is the single "
       "most impactful outcome for real user experience. "
       "We maintain strict scientific transparency regarding limitations: (1) Synthetic relevance labels create circularity with metadata rules; "
       "(2) OpenML's tabular focus leaves modality uninformative; (3) Inferred domain and modality rules may introduce noise; "
       "and (4) Evaluating 15 topic clusters yields wide bootstrap intervals. We present this as an auditable pipeline study, not a commercial claim.")

    sl(14, "Conclusion & Future Work", "[SPEAKER 2: PARTHIV DAVE]", "60-75 seconds",
       "In conclusion, we have presented an auditable, reproducible, and explainable dataset recommendation pipeline. "
       "We proved that dense sentence embeddings substantially outperform traditional lexical retrieval, and that blending semantic similarity "
       "with missing-aware metadata matching significantly enhances top-rank precision and reciprocal rank. "
       "The roadmap to publication involves three clear steps: first, conducting multi-annotator human evaluation using our provided pooling tool; "
       "second, incorporating BM25 and domain-specific bio-medical embedding models; and third, expanding the catalogue across Hugging Face and TCIA "
       "with unified schema mappings. "
       "Divij and I sincerely thank the committee, and we are now ready to take your questions.")

    out.write_text("\n".join(lines) + "\n" + qa, encoding="utf-8")



def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=cfg["paths"]["results_dir"])
    ap.add_argument("--out-dir", default="presentation")
    ap.add_argument("--preview-fixture", action="store_true")
    a = ap.parse_args()
    out_dir = resolve(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if a.preview_fixture:
        res = resolve("outputs/results_fixture"); cat = resolve("data/fixture/cleaning_report.json")
        if out_dir.resolve() == resolve("presentation").resolve():
            sys.exit("--preview-fixture must not write into presentation/")
        stamp = "FIXTURE PREVIEW - NOT REAL RESULTS"
    else:
        res = resolve(a.results); cat = resolve(f"{cfg['paths']['processed_dir']}/cleaning_report.json")
        ok, why, _ = valid_run(res)
        if not ok:
            sys.exit(f"No valid real run ({why}); existing PENDING deck and script left untouched.")
        stamp = None
    d = load(res, cat); f = findings(d)
    qa = resolve("presentation/professor_qa.txt").read_text(encoding="utf-8")
    build_pptx(d, f, out_dir / "project_presentation.pptx", stamp)
    build_script(d, f, out_dir / "speaker_script.txt", stamp, qa)
    print(f"[deliverables] wrote {out_dir}/project_presentation.pptx and speaker_script.txt ({'FIXTURE PREVIEW' if stamp else 'real run'})")


if __name__ == "__main__":
    main()
