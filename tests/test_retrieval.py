import numpy as np
import pandas as pd
import pytest

from build_benchmark import generate, merge_annotations, topic_grade, BENCHMARK_TOPICS
from build_dataset_text import add_text_column
from clean_catalog import clean_catalog
from common import load_config
from query_parser import parse_query
from retrieval import (CatalogIndex, LSABackend, Recommender, SAT, UNK, VIO, combine_hybrid, make_backend,
                       metadata_match, rank_scores, requirement_status)


def make_catalog():
    def rec(i, name, desc, n, cls=2, target="class", tags=None, lic=None):
        return {"dataset_id": i, "name": name, "description": desc, "tags": tags or [], "default_target_attribute": target,
                "licence": lic, "qualities": {"NumberOfInstances": n, "NumberOfFeatures": 10, "NumberOfClasses": cls},
                "source_repository": "OpenML", "retrieved_at_utc": "x"}
    recs = [
        rec(1, "lung-cancer-ct", "Labeled CT scan features of lung cancer patients with age and sex demographic columns.", 900, tags=["lung cancer", "ct"]),
        rec(2, "lung-cancer-small", "Small lung cancer study with CT image features and patient age.", 40, tags=["lung cancer"]),
        rec(3, "breast-w", "Breast cancer diagnosis from fine needle aspirate measurements for patients.", 699, tags=["breast cancer"]),
        rec(4, "credit-g", "Credit risk of bank customers described by financial attributes and loan information.", 1000),
        rec(5, "spam-base", "Email spam classification based on word frequencies in documents.", 4601),
        rec(6, "unlabeled-lung-text", "Lung cancer related notes with no declared target attribute in this record.", 3000, target=None),
        rec(7, "house-prices", "Regression of real estate house sale prices from property attributes.", 1500, cls=0),
    ]
    cat, _ = clean_catalog(recs)
    return add_text_column(cat)


@pytest.fixture(scope="module")
def rec():
    cat = make_catalog()
    return Recommender(cat, LSABackend(n_components=5), load_config())


def test_parser_extracts_documented_requirements():
    p = parse_query("I need a labeled dataset for lung cancer classification using CT images, with at least 500 patients and preferably demographic information.")
    kinds = {r.kind: r for r in p.requirements}
    assert kinds["min_instances"].value == 500 and kinds["min_instances"].mandatory
    assert kinds["labels"].mandatory
    assert kinds["condition"].value == "lung cancer"
    assert kinds["modality"].value == "ct"
    assert kinds["task"].value == "classification"
    assert kinds["demographic"].mandatory is False
    assert kinds["domain"].value == "medical"


def test_parser_softens_preferably_and_handles_no_requirements():
    p = parse_query("preferably at least 200 instances")
    assert p.by_kind("min_instances")[0].mandatory is False
    assert parse_query("something interesting").requirements == []


def test_parser_k_suffix_and_binary():
    p = parse_query("a binary classification dataset with more than 10k rows")
    assert p.by_kind("min_instances")[0].value == 10_000
    assert p.by_kind("n_classes")[0].value == 2


def test_missing_values_are_unknown_not_violations(rec):
    idx = rec.idx
    st = requirement_status(parse_query("a labeled dataset").by_kind("labels")[0], idx)
    # record 6 has no target -> has_labels is unknown -> status UNK (0), never VIO
    pos6 = int(np.where(idx.ids == 6)[0][0])
    assert st[pos6] == UNK
    assert st[int(np.where(idx.ids == 1)[0][0])] == SAT


def test_min_instances_constraint_marks_violation(rec):
    p = parse_query("lung cancer dataset with at least 500 instances")
    meta = metadata_match(p, rec.idx)
    pos2 = int(np.where(rec.idx.ids == 2)[0][0])
    pos1 = int(np.where(rec.idx.ids == 1)[0][0])
    assert meta.hard_violation[pos2] and not meta.hard_violation[pos1]


def test_hard_constraint_cannot_be_hidden_by_high_text_similarity(rec):
    q = "lung cancer CT scan dataset with at least 500 instances"
    top = rec.recommend(q, k=5, alpha=0.9, apply_constraints=True)
    names = [r["dataset_name"] for r in top]
    assert names.index("lung-cancer-ct") < names.index("lung-cancer-small")
    small = next(r for r in top if r["dataset_name"] == "lung-cancer-small")
    assert small["hard_violation"] and any("min_instances" in u for u in small["unsatisfied"])


def test_exclude_policy_ranks_violators_last():
    cat = make_catalog()
    cfg = load_config()
    cfg["retrieval"]["hard_constraint_policy"] = "exclude"
    r = Recommender(cat, LSABackend(n_components=5), cfg)
    res = r.recommend("lung cancer dataset with at least 500 instances", k=7)
    assert res[-1]["hard_violation"] or all(not x["hard_violation"] for x in res)
    first_violation = next(i for i, x in enumerate(res) if x["hard_violation"])
    assert all(x["hard_violation"] for x in res[first_violation:])


def test_hybrid_formula_and_endpoints():
    sem = np.array([0.8, 0.2])
    class M:  # minimal stand-in with required attributes
        score = np.array([0.1, 0.9]); has_requirements = True; hard_violation = np.array([False, False])
    assert np.allclose(combine_hybrid(sem, M, 0.7, False), 0.7 * sem + 0.3 * M.score)
    assert np.allclose(combine_hybrid(sem, M, 1.0, False), sem)       # alpha=1 -> semantic only
    assert np.allclose(combine_hybrid(sem, M, 0.0, False), M.score)   # alpha=0 -> metadata only


def test_no_requirements_means_final_equals_semantic(rec):
    q = "something about measurements"
    comp = rec.components([q])
    s = rec.scores_from_components(comp, "hybrid", 0.3)[0]
    assert np.allclose(s, comp["semantic"][0])


def test_all_methods_return_top5_with_required_fields(rec):
    for method in ("keyword", "tfidf", "semantic", "metadata", "hybrid"):
        res = rec.recommend("lung cancer CT dataset with demographic information", k=5, method=method)
        assert len(res) == 5 and [r["rank"] for r in res] == [1, 2, 3, 4, 5]
        for r in res:
            for key in ("dataset_name", "source_url", "score", "metadata", "matched", "unsatisfied", "unknown", "explanation"):
                assert key in r
            assert str(r["source_url"]).startswith("http")


def test_metadata_only_ties_are_deterministic(rec):
    a = rec.recommend("an unrelated query xyz", k=5, method="metadata")
    b = rec.recommend("an unrelated query xyz", k=5, method="metadata")
    assert [r["dataset_id"] for r in a] == [r["dataset_id"] for r in b]


def test_rank_scores_tie_break_by_id():
    order = rank_scores(np.array([0.5, 0.5, 0.9]), np.array([30, 10, 20]))
    assert list(order) == [2, 1, 0]


def test_robust_to_missing_optional_columns():
    cat = make_catalog().drop(columns=["license", "modality", "feature_names", "subdomain"])
    r = Recommender(cat, LSABackend(n_components=5), load_config())
    assert len(r.recommend("lung cancer", k=3)) == 3


def test_missing_text_column_raises_clear_error():
    with pytest.raises(ValueError):
        Recommender(make_catalog().drop(columns=["text"]), None, load_config())


def test_unknown_backend_name():
    with pytest.raises(ValueError):
        make_backend("nope", load_config())


# ---------------------------------------------------------------- benchmark integrity

def test_benchmark_splits_have_no_topic_leakage_and_labels_are_flagged_synthetic():
    from make_fixture import make_raw_records
    cfg = load_config()
    cat, _ = clean_catalog(make_raw_records(1))
    cat = add_text_column(cat)
    q, qrels, summary = generate(cat, cfg, seed=7, n_queries=120)
    topic_splits = q.groupby("topic_group")["split"].nunique()
    assert (topic_splits == 1).all(), "a topic appears in more than one split"
    assert set(q["split"]) <= {"train", "validation", "test"}
    assert (qrels["annotation_source"] == "synthetic_metadata_derived").all()
    assert summary["human_verified_judgments"] == 0
    assert q["query_text"].str.lower().is_unique


def test_benchmark_is_reproducible():
    from make_fixture import make_raw_records
    cfg = load_config()
    cat = add_text_column(clean_catalog(make_raw_records(1))[0])
    q1, r1, _ = generate(cat, cfg, seed=3, n_queries=60)
    q2, r2, _ = generate(cat, cfg, seed=3, n_queries=60)
    pd.testing.assert_frame_equal(q1, q2)
    pd.testing.assert_frame_equal(r1, r2)


def test_topic_grade_rules():
    t = BENCHMARK_TOPICS["breast_cancer"]
    assert topic_grade(t, "breast-w", "", "") == 3
    assert topic_grade(t, "x", "", "study of breast cancer cases") == 2
    assert topic_grade(t, "x", "", "a malignant tumor sample") == 1
    assert topic_grade(t, "x", "", "credit card fraud") == 0


def test_human_annotation_overrides_only_annotated_pairs_and_is_marked():
    qrels = pd.DataFrame({"query_id": ["Q1", "Q1"], "query_text": ["q", "q"], "candidate_dataset_id": [1, 2], "relevance": [3, 2],
                          "split": ["test", "test"], "annotation_source": ["synthetic_metadata_derived"] * 2, "annotation_notes": ["a", "b"]})
    ann = pd.DataFrame({"query_id": ["Q1"], "candidate_dataset_id": [2], "human_relevance": [0], "annotator_id": ["A1"], "annotation_notes": ["not suitable"]})
    merged, stats = merge_annotations(ann, qrels)
    assert stats["n_human_judgments"] == 1
    assert set(merged["candidate_dataset_id"]) == {1}      # pair 2 now grade 0 -> dropped (sparse)
    r1 = merged[merged.candidate_dataset_id == 1].iloc[0]
    assert r1["annotation_source"] == "synthetic_metadata_derived" and r1["relevance"] == 3
