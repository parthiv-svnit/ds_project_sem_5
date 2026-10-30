import numpy as np
import pandas as pd

from build_dataset_text import build_text
from clean_catalog import clean_catalog, normalize_record
from common import normalize_text


def raw(i, name="Dataset", desc="A reasonably long description about breast cancer patients and tumours.", **kw):
    base = {"dataset_id": i, "name": name, "description": desc, "tags": ["cancer"], "default_target_attribute": "class",
            "licence": "Public", "qualities": {"NumberOfInstances": 569, "NumberOfFeatures": 31, "NumberOfClasses": 2},
            "source_repository": "OpenML", "retrieved_at_utc": "2026-01-01T00:00:00+00:00"}
    base.update(kw)
    return base


def test_normalize_text_strips_html_urls_whitespace():
    assert normalize_text("<p>Hello&nbsp;  <b>world</b> see https://x.org/a </p>") == "Hello world see"
    assert normalize_text(None) == "" and normalize_text(float("nan")) == ""


def test_missing_fields_do_not_crash_and_stay_missing():
    rec = {"dataset_id": 5, "name": "Bare", "description": None, "tags": None, "qualities": {}}
    row = normalize_record(rec)
    assert row["dataset_id"] == 5
    assert pd.isna(row["description"]) and pd.isna(row["tags"]) and pd.isna(row["license"])
    assert pd.isna(row["number_of_instances"])
    # missing target must give UNKNOWN labels, never False
    assert pd.isna(row["has_labels"])
    # text builder tolerates the missing fields
    assert "Dataset name: Bare." in build_text(pd.Series(row))


def test_impossible_numbers_become_missing():
    row = normalize_record(raw(1, qualities={"NumberOfInstances": -3, "NumberOfFeatures": 0, "NumberOfClasses": -1}))
    assert pd.isna(row["number_of_instances"]) and pd.isna(row["number_of_features"]) and pd.isna(row["number_of_classes"])


def test_provenance_flags():
    row = normalize_record(raw(1))
    assert row["task_type"] == "classification" and row["task_type_source"] == "derived"
    assert row["has_labels"] is True and row["has_labels_source"] == "derived"
    assert row["domain"] == "medical" and row["domain_source"] == "inferred_keyword"
    assert row["data_type_source"] == "repository_default"


def test_dedup_by_id_and_by_content_keeps_lowest_id():
    recs = [raw(10), raw(10), raw(11), raw(12, name="Other", desc="Totally different text about credit card fraud and banks, long enough.")]
    cat, rep = clean_catalog(recs)
    assert rep["dropped_duplicate_id"] == 1
    assert rep["dropped_content_duplicate"] == 1  # id 11 duplicates id 10's name + description
    assert sorted(cat["dataset_id"]) == [10, 12]


def test_quality_filter_documented_and_counted():
    recs = [raw(1), raw(2, desc="x", tags=[]), raw(3, name=""), raw(4, desc="x", tags=["t"], name="Tagged")]
    cat, rep = clean_catalog(recs, min_description_chars=40)
    assert set(cat["dataset_id"]) == {1, 4}            # thin-without-tags and nameless removed
    assert rep["dropped_quality_total"] == 2
    assert rep["n_final"] == 2 and rep["n_raw"] == 4


def test_report_contains_missingness_for_important_fields():
    _, rep = clean_catalog([raw(1), raw(2, name="B", desc="Another sufficiently long description of sensor measurements.")])
    assert "license" in rep["missingness"] and "number_of_instances" in rep["missingness"]
    assert rep["missingness"]["number_of_instances"]["missing"] == 0


def test_synthetic_flag_is_set_for_fixture_records():
    cat, rep = clean_catalog([raw(1, source_repository="SYNTHETIC_FIXTURE")])
    assert bool(cat["is_synthetic"].iloc[0]) and rep["n_synthetic_records"] == 1


def test_text_excludes_inferred_fields():
    row = pd.Series(normalize_record(raw(1)))
    text = build_text(row)
    assert "medical" not in text.lower().split("description:")[0]  # inferred domain is not injected
    assert "Task: classification with 2 classes." in text
