"""STEP 2 - Clean raw OpenML metadata into the dataset catalogue.

Usage
-----
    python src/clean_catalog.py                       # real run (data/raw/openml_raw_latest.jsonl)
    python src/clean_catalog.py --input data/fixture/fixture_raw.jsonl \
        --output data/fixture/catalog.csv --report data/fixture/cleaning_report.json   # fixture

What the cleaning does (all steps are counted in the cleaning report):
  1. Parse raw JSONL; keep the original identifiers and source URLs (provenance).
  2. Normalise text (HTML/URL stripping, whitespace); never fabricate text.
  3. Coerce numeric quality fields; impossible values (negative, zero instances) -> missing.
  4. Remove exact duplicate dataset ids, then content duplicates (same normalised name and
     description prefix; the lowest dataset id is kept - these are typically versions).
  5. Apply documented quality criteria: a record needs a name, and either a description of at
     least ``min_description_chars`` characters or at least one tag.
  6. Derive / infer optional fields and record HOW each was obtained in ``*_source`` columns:
        explicit            - taken verbatim from OpenML
        derived             - computed from an explicit field (e.g. has_labels from target)
        inferred_keyword    - keyword-lexicon inference from name/tags/description (NOT verified)
        repository_default  - property of the repository (OpenML stores tabular files)
     Missing information stays missing (NaN); it is never converted to a negative fact.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from common import (CONDITION_TERMS, DOMAIN_LEXICON, MODALITY_LEXICON, TASK_LEXICON, contains_term,
                    load_config, normalize_text, resolve, write_json)

CATALOG_COLUMNS = [
    "dataset_id", "dataset_name", "description", "domain", "subdomain", "tags", "task_type",
    "modality", "data_type", "number_of_instances", "number_of_features", "number_of_classes",
    "target_attribute", "has_labels", "license", "source_repository", "source_url",
    "original_data_url", "feature_names", "retrieved_at_utc", "is_synthetic",
    "domain_source", "subdomain_source", "task_type_source", "modality_source",
    "data_type_source", "has_labels_source",
]
IMPORTANT_FIELDS = [
    "description", "tags", "domain", "task_type", "modality", "number_of_instances",
    "number_of_features", "number_of_classes", "target_attribute", "has_labels", "license",
]
MAX_DESC_CHARS = 4000


# --------------------------------------------------------------------------- helpers

def _num(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _clean_missing_string(value: Any) -> str | float:
    s = normalize_text(value)
    return s if s and s.lower() not in {"none", "nan", "null", "unknown", "n/a", "na"} else np.nan


def _join(items: Any) -> str | float:
    if items is None or (isinstance(items, float) and np.isnan(items)):
        return np.nan
    if isinstance(items, str):
        items = [t for t in items.split("|") if t]
    items = [normalize_text(i) for i in items if normalize_text(i)]
    return "|".join(items) if items else np.nan


def infer_domain(text_lower: str) -> tuple[str | float, str | float]:
    """Keyword-lexicon domain inference. Returns (domain, subdomain); either may be NaN."""
    scores = {dom: sum(contains_term(text_lower, t) for t in terms) for dom, terms in DOMAIN_LEXICON.items()}
    best = max(scores.values()) if scores else 0
    domain = next((d for d, s in scores.items() if s == best), np.nan) if best > 0 else np.nan
    sub = next((c for c in sorted(CONDITION_TERMS, key=len, reverse=True) if contains_term(text_lower, c)), np.nan)
    if isinstance(sub, str) and domain != "medical":
        domain = "medical"  # a named medical condition implies the medical domain
    return domain, sub


def infer_modality(text_lower: str) -> str | float:
    hits = [m for m, terms in MODALITY_LEXICON.items() if m != "tabular" and any(contains_term(text_lower, t) for t in terms)]
    return "|".join(hits) if hits else np.nan


def infer_task(text_lower: str) -> str | float:
    for task, terms in TASK_LEXICON.items():
        if any(contains_term(text_lower, t) for t in terms):
            return task
    return np.nan


# --------------------------------------------------------------------------- core

def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn one raw metadata record into one catalogue row (before dedup / filtering)."""
    quals = raw.get("qualities") or {}
    name = normalize_text(raw.get("name"))
    desc = normalize_text(raw.get("description"), max_chars=MAX_DESC_CHARS)
    tags_joined = _join(raw.get("tags"))
    target = _clean_missing_string(raw.get("default_target_attribute"))
    feats = raw.get("feature_names")

    n_inst = _num(quals.get("NumberOfInstances"))
    n_feat = _num(quals.get("NumberOfFeatures"))
    n_cls = _num(quals.get("NumberOfClasses"))
    if n_inst <= 0:
        n_inst = np.nan
    if n_feat <= 0:
        n_feat = np.nan
    if n_cls < 0:
        n_cls = np.nan

    blob = " ".join(x for x in [name, str(tags_joined).replace("|", " ") if isinstance(tags_joined, str) else "", desc] if x).lower()

    domain, subdomain = infer_domain(blob)
    modality = infer_modality(blob)

    if isinstance(n_cls, float) and not np.isnan(n_cls) and n_cls >= 2:
        task, task_src = "classification", "derived"
    else:
        task = infer_task(blob)
        task_src = "inferred_keyword" if isinstance(task, str) else np.nan

    if isinstance(target, str):
        has_labels, has_labels_src = True, "derived"
    else:
        has_labels, has_labels_src = np.nan, np.nan  # unknown, NOT False

    source_repo = raw.get("source_repository") or "OpenML"
    return {
        "dataset_id": raw.get("dataset_id"),
        "dataset_name": name or np.nan,
        "description": desc or np.nan,
        "domain": domain,
        "subdomain": subdomain,
        "tags": tags_joined,
        "task_type": task,
        "modality": modality,
        "data_type": "tabular",
        "number_of_instances": n_inst,
        "number_of_features": n_feat,
        "number_of_classes": n_cls,
        "target_attribute": target,
        "has_labels": has_labels,
        "license": _clean_missing_string(raw.get("licence") or raw.get("license")),
        "source_repository": source_repo,
        "source_url": raw.get("source_url") or (f"https://www.openml.org/search?type=data&id={raw.get('dataset_id')}"),
        "original_data_url": _clean_missing_string(raw.get("original_data_url")),
        "feature_names": _join(feats),
        "retrieved_at_utc": raw.get("retrieved_at_utc"),
        "is_synthetic": str(source_repo).upper().startswith("SYNTHETIC"),
        "domain_source": "inferred_keyword" if isinstance(domain, str) else np.nan,
        "subdomain_source": "inferred_keyword" if isinstance(subdomain, str) else np.nan,
        "task_type_source": task_src,
        "modality_source": "inferred_keyword" if isinstance(modality, str) else np.nan,
        "data_type_source": "repository_default",
        "has_labels_source": has_labels_src,
    }


def clean_catalog(raw_records: list[dict[str, Any]], min_description_chars: int = 40) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw records into a catalogue DataFrame and return (catalog, report)."""
    report: dict[str, Any] = {"n_raw": len(raw_records)}
    rows = [normalize_record(r) for r in raw_records]
    df = pd.DataFrame(rows, columns=CATALOG_COLUMNS)

    # 1. drop rows without a usable id
    before = len(df)
    df = df[pd.to_numeric(df["dataset_id"], errors="coerce").notna()].copy()
    df["dataset_id"] = df["dataset_id"].astype(int)
    report["dropped_missing_id"] = before - len(df)

    # 2. exact duplicate ids
    before = len(df)
    df = df.drop_duplicates("dataset_id", keep="first")
    report["dropped_duplicate_id"] = before - len(df)

    # 3. quality criteria
    before = len(df)
    has_name = df["dataset_name"].notna()
    desc_len = df["description"].fillna("").str.len()
    ok = has_name & ((desc_len >= min_description_chars) | df["tags"].notna())
    report["dropped_quality_no_name"] = int((~has_name).sum())
    report["dropped_quality_thin_metadata"] = int((has_name & ~ok).sum())
    df = df[ok].copy()
    report["dropped_quality_total"] = before - len(df)

    # 4. content duplicates (versions of the same dataset); keep lowest id
    before = len(df)
    key = df["dataset_name"].fillna("").str.lower().str.strip() + "||" + df["description"].fillna("").str.lower().str[:500]
    df = df.assign(_k=key).sort_values("dataset_id").drop_duplicates("_k", keep="first").drop(columns="_k")
    report["dropped_content_duplicate"] = before - len(df)

    df = df.sort_values("dataset_id").reset_index(drop=True)
    report["n_final"] = int(len(df))
    report["min_description_chars"] = min_description_chars
    report["n_synthetic_records"] = int(df["is_synthetic"].sum())

    n = max(len(df), 1)
    report["missingness"] = {
        f: {"missing": int(df[f].isna().sum()), "proportion": round(float(df[f].isna().sum()) / n, 4)}
        for f in IMPORTANT_FIELDS
    }
    report["provenance_counts"] = {
        col: {str(k): int(v) for k, v in df[col].value_counts(dropna=True).items()}
        for col in ["domain_source", "task_type_source", "modality_source", "has_labels_source"]
    }
    return df, report


def load_raw(path: str) -> list[dict[str, Any]]:
    records = []
    with open(resolve(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=f"{cfg['paths']['raw_dir']}/openml_raw_latest.jsonl")
    ap.add_argument("--output", default=f"{cfg['paths']['processed_dir']}/catalog.csv")
    ap.add_argument("--report", default=f"{cfg['paths']['processed_dir']}/cleaning_report.json")
    args = ap.parse_args()

    raw = load_raw(args.input)
    catalog, report = clean_catalog(raw, cfg["collection"]["min_description_chars"])
    catalog.to_csv(resolve(args.output), index=False)
    write_json(report, args.report)
    print(f"[clean] {report['n_raw']} raw -> {report['n_final']} catalogue records -> {args.output}")
    print("[clean] missingness (important fields):")
    for f, v in report["missingness"].items():
        print(f"    {f:22s} {v['missing']:6d}  ({100 * v['proportion']:.1f}%)")


if __name__ == "__main__":
    main()
