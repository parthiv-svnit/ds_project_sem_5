"""STEP (benchmark) - Build a SYNTHETIC, METADATA-DERIVED query benchmark + annotation workflow.

IMPORTANT RESEARCH-INTEGRITY NOTE
---------------------------------
Everything this script generates automatically is *synthetic / metadata-derived*:
  * queries come from templates over a topic list (src/benchmark_topics below);
  * relevance grades come from rules over catalogue metadata (name, tags, description, explicit
    quality counts). They are NOT human judgments and are stored with
    ``annotation_source = synthetic_metadata_derived``.
Because the ranking methods also read the same metadata, scores against these labels are biased
(in particular for the metadata matcher on constrained queries). Use them for development and
smoke testing; publication-quality claims need human review (see the 'annotate' workflow).

Sub-commands
------------
  generate         build queries + sparse qrels + splits   -> data/benchmark/
  make-pool        pool top-k results of every method for a sample of queries -> annotation sheet
  merge            merge a human-filled annotation sheet into qrels_reviewed.csv

Usage
-----
    python src/build_benchmark.py generate
    python src/build_benchmark.py generate --catalog data/fixture/catalog_text.csv --out data/fixture/benchmark
    python src/build_benchmark.py make-pool --n-queries 40
    python src/build_benchmark.py merge --annotated data/benchmark/annotation_sheet_filled.csv

Leakage control: queries are split by TOPIC GROUP (all paraphrases, templates and constraint
variants of one topic stay in the same split), so no near-duplicate query crosses splits.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
import pandas as pd

from common import contains_term, load_config, normalize_text, resolve, write_json
from query_parser import parse_query

# topic -> (surface forms used in queries, strong evidence terms, weak evidence terms)
BENCHMARK_TOPICS: dict[str, dict[str, list[str]]] = {
    "breast_cancer": {"surface": ["breast cancer", "breast tumor", "mammography"], "strong": ["breast cancer", "breast", "mammograph"], "weak": ["cancer", "tumor", "malignant", "benign"]},
    "lung_cancer": {"surface": ["lung cancer", "pulmonary nodule", "lung tumor"], "strong": ["lung cancer", "lung", "pulmonary"], "weak": ["cancer", "tumor", "malignant"]},
    "cancer_general": {"surface": ["cancer", "tumor", "malignancy"], "strong": ["cancer", "tumor", "tumour", "carcinoma", "malignant"], "weak": ["oncology", "benign", "biopsy"]},
    "diabetes": {"surface": ["diabetes", "diabetic patients", "blood glucose"], "strong": ["diabetes", "diabetic", "glucose"], "weak": ["insulin", "patient", "clinical"]},
    "heart_disease": {"surface": ["heart disease", "cardiac patients", "cardiovascular risk"], "strong": ["heart", "cardiac", "cardiovascular"], "weak": ["cholesterol", "blood pressure", "patient"]},
    "ecg_arrhythmia": {"surface": ["ECG arrhythmia", "electrocardiogram signals", "heartbeat classification"], "strong": ["ecg", "electrocardiogram", "arrhythmia", "heartbeat"], "weak": ["heart", "cardiac", "signal"]},
    "eeg": {"surface": ["EEG recordings", "brain signals", "electroencephalogram"], "strong": ["eeg", "electroencephalogram"], "weak": ["brain", "epilepsy", "seizure"]},
    "liver_kidney": {"surface": ["liver disease", "kidney disease", "hepatitis"], "strong": ["liver", "kidney", "hepatitis", "renal"], "weak": ["patient", "clinical", "disease"]},
    "thyroid_neuro": {"surface": ["thyroid disease", "Parkinson's disease", "Alzheimer's"], "strong": ["thyroid", "parkinson", "alzheimer"], "weak": ["patient", "disease", "clinical"]},
    "gene_expression": {"surface": ["gene expression", "microarray data", "genomic profiles"], "strong": ["gene", "microarray", "genomic", "expression"], "weak": ["dna", "rna", "bioinformatics"]},
    "protein_molecule": {"surface": ["protein sequences", "molecular activity", "chemical compounds"], "strong": ["protein", "molecular", "molecule", "compound"], "weak": ["chemical", "sequence", "bioinformatics"]},
    "credit_finance": {"surface": ["credit scoring", "loan default", "bank customers"], "strong": ["credit", "loan", "bank"], "weak": ["financial", "customer", "default"]},
    "fraud_churn": {"surface": ["fraud detection", "customer churn", "insurance claims"], "strong": ["fraud", "churn", "insurance"], "weak": ["customer", "transaction", "financial"]},
    "spam_text": {"surface": ["spam email", "text documents", "news articles"], "strong": ["spam", "email", "news", "text"], "weak": ["document", "corpus", "word"]},
    "sentiment": {"surface": ["sentiment analysis", "product reviews", "tweets"], "strong": ["sentiment", "review", "tweet"], "weak": ["opinion", "text", "language"]},
    "image_digits": {"surface": ["handwritten digits", "image recognition", "pixel images"], "strong": ["digit", "mnist", "image", "pixel"], "weak": ["handwritten", "visual", "picture"]},
    "speech_audio": {"surface": ["speech recognition", "audio recordings", "voice signals"], "strong": ["speech", "audio", "voice", "acoustic"], "weak": ["sound", "phoneme", "signal"]},
    "sensor_iot": {"surface": ["sensor readings", "IoT devices", "machine vibration"], "strong": ["sensor", "iot", "vibration"], "weak": ["machine", "telemetry", "signal"]},
    "network_security": {"surface": ["network intrusion", "cyber attack traffic", "malware"], "strong": ["intrusion", "network", "malware", "attack"], "weak": ["traffic", "security", "packet"]},
    "weather_climate": {"surface": ["weather", "climate measurements", "air pollution"], "strong": ["weather", "climate", "pollution", "rainfall"], "weak": ["temperature", "environment", "air"]},
    "housing_price": {"surface": ["house prices", "real estate", "property valuation"], "strong": ["house", "housing", "real estate", "property"], "weak": ["price", "estate", "value"]},
    "food_wine": {"surface": ["wine quality", "food composition", "recipes"], "strong": ["wine", "food", "recipe"], "weak": ["quality", "nutrition", "chemical"]},
    "census_social": {"surface": ["census income", "student performance", "survey responses"], "strong": ["census", "income", "student", "survey"], "weak": ["education", "population", "demographic"]},
    "plants_animals": {"surface": ["iris flowers", "mushroom species", "animal species"], "strong": ["iris", "mushroom", "species", "plant", "animal"], "weak": ["flower", "biology", "botanical"]},
    "cars_transport": {"surface": ["car evaluation", "traffic", "vehicle data"], "strong": ["car", "vehicle", "traffic"], "weak": ["transport", "automobile", "engine"]},
    "sports_games": {"surface": ["sports results", "chess games", "player statistics"], "strong": ["sport", "chess", "player", "game"], "weak": ["match", "team", "score"]},
}

TASKS = [("classification", ["classification", "classification tasks", "classification experiments"]),
         ("regression", ["regression", "regression tasks", "regression experiments"]),
         (None, [""])]
TEMPLATES_TASK = [
    "I need a dataset for {topic} {task}",
    "Find data about {topic} suitable for {task}",
    "I want {topic} data for {task}",
]
TEMPLATES_NOTASK = [
    "Looking for {topic} data for research",
    "Recommend a dataset on {topic}",
    "Find data about {topic}",
]
CONSTRAINTS: list[tuple[str, dict[str, Any]]] = [
    ("", {}),
    ("with at least {n} instances", {"min_instances": True}),
    ("that is labeled", {"labels": True}),
    ("with at least {n} instances and labeled", {"min_instances": True, "labels": True}),
    ("for binary classification", {"n_classes": 2}),
    ("and preferably with demographic features", {"demographic": True}),
    ("and it should be a large dataset", {"large": True}),
]
MIN_N_CHOICES = [100, 500, 1000, 5000]


# --------------------------------------------------------------------------- relevance rules

def topic_grade(topic: dict[str, list[str]], name: str, tags: str, desc: str) -> int:
    """Base topical grade from catalogue text (0-3). Rules are documented in the paper.

    3: strong term in name or tags      2: strong term in description (and not in name/tags)
    1: only weak related term anywhere  0: nothing
    """
    nt = f"{name} {tags}"
    if any(contains_term(nt, t) for t in topic["strong"]):
        return 3
    if any(contains_term(desc, t) for t in topic["strong"]):
        return 2
    if any(contains_term(nt + " " + desc, t) for t in topic["weak"]):
        return 1
    return 0


def graded_relevance(spec: dict[str, Any], cat: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """Apply constraint adjustments to the base topical grades.

    * violated explicit numeric/label constraint  -> grade capped at 1 (right topic, unusable)
    * task mismatch (classification asked, dataset has no derived classes) -> grade - 1
    * unknown values never change the grade (missing is not negative evidence)
    """
    g = base.copy()
    n_inst = pd.to_numeric(cat["number_of_instances"], errors="coerce").to_numpy()
    n_cls = pd.to_numeric(cat["number_of_classes"], errors="coerce").to_numpy()
    hl = cat["has_labels"].map(lambda v: np.nan if pd.isna(v) else float(str(v).lower() in {"true", "1", "1.0"})).to_numpy()
    c = spec["constraints"]
    if "min_instances" in c:
        g = np.where(~np.isnan(n_inst) & (n_inst < c["min_instances"]), np.minimum(g, 1), g)
    if "large" in c:
        g = np.where(~np.isnan(n_inst) & (n_inst < 10_000), np.minimum(g, 1), g)
    if c.get("labels"):
        g = np.where(hl == 0.0, np.minimum(g, 1), g)
    if "n_classes" in c:
        g = np.where(~np.isnan(n_cls) & (n_cls >= 1) & (n_cls != c["n_classes"]), np.minimum(g, 1), g)
    if spec.get("task") == "classification":
        g = np.where(~np.isnan(n_cls) & (n_cls == 0), np.maximum(g - 1, 0), g)
    return g.astype(int)


# --------------------------------------------------------------------------- generation

def generate(cat: pd.DataFrame, cfg: dict[str, Any], seed: int, n_queries: int, min_relevant: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    cat = cat.reset_index(drop=True)
    names = cat["dataset_name"].fillna("").astype(str).str.lower()
    tags = cat["tags"].fillna("").astype(str).str.replace("|", " ", regex=False).str.lower()
    desc = cat["description"].fillna("").astype(str).str.lower()

    # topical grades per topic; keep only topics with enough relevant catalogue items
    topic_base: dict[str, np.ndarray] = {}
    for tname, t in BENCHMARK_TOPICS.items():
        base = np.array([topic_grade(t, a, b, c) for a, b, c in zip(names, tags, desc)])
        if int((base >= 2).sum()) >= min_relevant:
            topic_base[tname] = base
    usable = sorted(topic_base)
    if not usable:
        raise SystemExit("No benchmark topic has enough relevant datasets in this catalogue; enlarge the catalogue.")

    # candidate queries per topic
    pool: list[dict[str, Any]] = []
    for tname in usable:
        t = BENCHMARK_TOPICS[tname]
        for surface, (task, task_phr), (c_phr, c_spec) in itertools.product(t["surface"], TASKS, CONSTRAINTS):
            if c_spec.get("n_classes") and task == "regression":
                continue  # contradictory combination
            tmpl = str(rng.choice(TEMPLATES_TASK if task else TEMPLATES_NOTASK))
            task_text = str(rng.choice(task_phr)) if task_phr[0] else ""
            constraints = dict(c_spec)
            n_val = int(rng.choice(MIN_N_CHOICES))
            if constraints.get("min_instances"):
                constraints["min_instances"] = n_val
            text = tmpl.format(topic=surface, task=task_text)
            phrase = c_phr.format(n=n_val) if c_phr else ""
            if c_spec.get("n_classes") and task is None:
                task = "classification"  # 'binary classification' implies the task
            text = normalize_text((text + " " + phrase).strip())
            pool.append({"topic": tname, "text": text, "task": task, "constraints": constraints})
    # per-topic quota, sampled without replacement and de-duplicated on text
    per_topic = max(1, int(np.ceil(n_queries / len(usable))))
    chosen: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for tname in usable:
        cand = [p for p in pool if p["topic"] == tname]
        order = rng.permutation(len(cand))
        taken = 0
        for i in order:
            if taken >= per_topic:
                break
            if cand[i]["text"].lower() in seen_text:
                continue
            seen_text.add(cand[i]["text"].lower())
            chosen.append(cand[i])
            taken += 1
    chosen = chosen[:max(n_queries, len(usable))] if len(chosen) > n_queries else chosen

    # splits by topic group
    fr = cfg["benchmark"]["split_fractions"]
    perm = list(rng.permutation(usable))
    n_t = len(perm)
    n_val = max(1, int(round(fr["validation"] * n_t))) if n_t >= 3 else 0
    n_test = max(1, int(round(fr["test"] * n_t))) if n_t >= 2 else 0
    test_topics = set(perm[:n_test])
    val_topics = set(perm[n_test:n_test + n_val])
    split_of = {t: ("test" if t in test_topics else "validation" if t in val_topics else "train") for t in usable}

    q_rows, r_rows = [], []
    for qi, spec in enumerate(chosen, start=1):
        qid = f"Q{qi:04d}"
        parsed = parse_query(spec["text"])
        grades = graded_relevance(spec, cat, topic_base[spec["topic"]])
        q_rows.append({
            "query_id": qid, "query_text": spec["text"], "topic_group": spec["topic"], "split": split_of[spec["topic"]],
            "has_constraints": bool(spec["constraints"]), "constraint_types": "|".join(sorted(spec["constraints"])) or "none",
            "n_parsed_requirements": len(parsed.requirements),
            "annotation_source": cfg["benchmark"]["annotation_source"],
            "query_origin": "synthetic_template",
        })
        for pos in np.flatnonzero(grades >= 1):
            r_rows.append({
                "query_id": qid, "query_text": spec["text"], "candidate_dataset_id": int(cat.loc[pos, "dataset_id"]),
                "relevance": int(grades[pos]), "split": split_of[spec["topic"]],
                "annotation_source": cfg["benchmark"]["annotation_source"],
                "annotation_notes": f"rule-based: topic={spec['topic']}; base={int(topic_base[spec['topic']][pos])}; constraints={spec['constraints']}",
            })
    queries = pd.DataFrame(q_rows)
    qrels = pd.DataFrame(r_rows)
    n_rel2 = qrels[qrels.relevance >= 2].groupby("query_id").size()
    keep = set(n_rel2[n_rel2 >= 1].index)  # evaluation requires >= 1 relevant (grade >= 2)
    dropped = len(queries) - len(keep)
    queries = queries[queries.query_id.isin(keep)].reset_index(drop=True)
    qrels = qrels[qrels.query_id.isin(keep)].reset_index(drop=True)
    summary = {
        "n_catalogue": int(len(cat)), "n_topics_usable": len(usable), "topics_dropped": sorted(set(BENCHMARK_TOPICS) - set(usable)),
        "n_queries": int(len(queries)), "queries_dropped_no_relevant": int(dropped),
        "n_relevance_judgments_listed": int(len(qrels)),
        "judgment_note": "sparse qrels: only grades >= 1 are listed; all unlisted pairs have grade 0 (implicit)",
        "n_implicit_zero_pairs": int(len(queries) * len(cat) - len(qrels)),
        "queries_per_split": queries["split"].value_counts().to_dict(),
        "topics_per_split": {s: sorted(t for t, v in split_of.items() if v == s) for s in ("train", "validation", "test")},
        "grade_counts_listed": qrels["relevance"].value_counts().sort_index().to_dict(),
        "annotation_source": cfg["benchmark"]["annotation_source"],
        "human_verified_judgments": 0,
        "seed": seed,
    }
    return queries, qrels, summary


# --------------------------------------------------------------------------- annotation workflow

ANNOTATION_COLUMNS = ["query_id", "query_text", "candidate_dataset_id", "dataset_name", "source_url", "dataset_description_excerpt",
                      "synthetic_grade", "human_relevance", "annotator_id", "annotation_notes"]


def make_pool(rankings_csv: str, queries: pd.DataFrame, qrels: pd.DataFrame, cat: pd.DataFrame, n_queries: int, seed: int, depth: int,
              splits: tuple[str, ...]) -> pd.DataFrame:
    """Pool the top-`depth` items of every method for a random sample of queries (TREC-style pooling)."""
    rk = pd.read_csv(resolve(rankings_csv))
    q = queries[queries.split.isin(splits)]
    rng = np.random.default_rng(seed)
    sample = q.sample(n=min(n_queries, len(q)), random_state=int(rng.integers(0, 2**31 - 1)))
    rk = rk[(rk.query_id.isin(sample.query_id)) & (rk["rank"] <= depth)]
    pooled = rk[["query_id", "dataset_id"]].drop_duplicates().rename(columns={"dataset_id": "candidate_dataset_id"})
    syn = qrels[["query_id", "candidate_dataset_id", "relevance"]].rename(columns={"relevance": "synthetic_grade"})
    out = pooled.merge(syn, how="left", on=["query_id", "candidate_dataset_id"]).fillna({"synthetic_grade": 0})
    out = out.merge(queries[["query_id", "query_text"]], on="query_id")
    c = cat[["dataset_id", "dataset_name", "source_url", "description"]].rename(columns={"dataset_id": "candidate_dataset_id"})
    out = out.merge(c, on="candidate_dataset_id", how="left")
    out["dataset_description_excerpt"] = out["description"].fillna("").str.slice(0, 300)
    out["human_relevance"] = ""
    out["annotator_id"] = ""
    out["annotation_notes"] = ""
    return out[ANNOTATION_COLUMNS].sort_values(["query_id", "candidate_dataset_id"]).reset_index(drop=True)


def merge_annotations(annotated: pd.DataFrame, qrels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Human grades (0-3) override synthetic grades for the annotated pairs only."""
    a = annotated.copy()
    a["human_relevance"] = pd.to_numeric(a["human_relevance"], errors="coerce")
    a = a[a["human_relevance"].isin([0, 1, 2, 3])]
    base = qrels.copy()
    base["annotation_source"] = base["annotation_source"].astype(str)
    key = ["query_id", "candidate_dataset_id"]
    base = base.merge(a[key + ["human_relevance", "annotator_id", "annotation_notes"]], on=key, how="outer", suffixes=("", "_h"))
    human = base["human_relevance"].notna()
    base["relevance"] = np.where(human, base["human_relevance"], base["relevance"]).astype(float)
    base["annotation_source"] = np.where(human, "human_reviewed", base["annotation_source"].replace("nan", "synthetic_metadata_derived"))
    base["annotation_notes"] = np.where(human, base["annotation_notes_h"].fillna(""), base["annotation_notes"].fillna(""))
    base["query_text"] = base["query_text"].fillna("")
    base = base[base["relevance"] >= 1]  # sparse convention
    stats = {"n_human_judgments": int(human.sum()), "n_queries_with_human_judgments": int(a["query_id"].nunique()),
             "annotators": sorted(a["annotator_id"].dropna().astype(str).unique().tolist())}
    return base.drop(columns=[c for c in base.columns if c.endswith("_h") or c in ("human_relevance", "annotator_id")], errors="ignore"), stats


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--catalog", default=f"{cfg['paths']['processed_dir']}/catalog_text.csv")
    g.add_argument("--out", default=cfg["paths"]["benchmark_dir"])
    g.add_argument("--n-queries", type=int, default=cfg["benchmark"]["n_queries"])
    g.add_argument("--seed", type=int, default=cfg["seed"])
    p = sub.add_parser("make-pool")
    p.add_argument("--catalog", default=f"{cfg['paths']['processed_dir']}/catalog_text.csv")
    p.add_argument("--bench", default=cfg["paths"]["benchmark_dir"])
    p.add_argument("--rankings", default=f"{cfg['paths']['results_dir']}/rankings_top10.csv")
    p.add_argument("--n-queries", type=int, default=40)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--seed", type=int, default=cfg["seed"])
    m = sub.add_parser("merge")
    m.add_argument("--bench", default=cfg["paths"]["benchmark_dir"])
    m.add_argument("--annotated", required=True)
    args = ap.parse_args()

    if args.cmd == "generate":
        cat = pd.read_csv(resolve(args.catalog))
        queries, qrels, summary = generate(cat, cfg, args.seed, args.n_queries)
        out = args.out
        queries.to_csv(resolve(f"{out}/queries.csv"), index=False)
        qrels.to_csv(resolve(f"{out}/qrels_synthetic.csv"), index=False)
        write_json(summary, f"{out}/benchmark_summary.json")
        template = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        template.to_csv(resolve(f"{out}/annotation_template.csv"), index=False)
        print(f"[benchmark] {summary['n_queries']} queries, {summary['n_relevance_judgments_listed']} listed judgments "
              f"(all synthetic/metadata-derived, 0 human-verified) -> {out}")
        print(f"[benchmark] queries per split: {summary['queries_per_split']}")
    elif args.cmd == "make-pool":
        cat = pd.read_csv(resolve(args.catalog))
        queries = pd.read_csv(resolve(f"{args.bench}/queries.csv"))
        qrels = pd.read_csv(resolve(f"{args.bench}/qrels_synthetic.csv"))
        pool = make_pool(args.rankings, queries, qrels, cat, args.n_queries, args.seed, args.depth, tuple(args.splits))
        path = resolve(f"{args.bench}/annotation_sheet.csv")
        pool.to_csv(path, index=False)
        print(f"[benchmark] annotation sheet with {len(pool)} pairs for {pool.query_id.nunique()} queries -> {path}")
        print("[benchmark] fill the 'human_relevance' column (0-3), set 'annotator_id', save as annotation_sheet_filled.csv")
    elif args.cmd == "merge":
        qrels = pd.read_csv(resolve(f"{args.bench}/qrels_synthetic.csv"))
        annotated = pd.read_csv(resolve(args.annotated))
        merged, stats = merge_annotations(annotated, qrels)
        merged.to_csv(resolve(f"{args.bench}/qrels_reviewed.csv"), index=False)
        write_json(stats, f"{args.bench}/annotation_stats.json")
        print(f"[benchmark] merged {stats['n_human_judgments']} human judgments -> {args.bench}/qrels_reviewed.csv")


if __name__ == "__main__":
    main()
