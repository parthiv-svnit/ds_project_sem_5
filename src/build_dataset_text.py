"""STEP 3 - Build one text representation per dataset.

Design decision: the text uses only *explicitly supplied or derived* metadata (name, tags,
description, target attribute, class/size facts, feature names if collected). It deliberately
EXCLUDES keyword-inferred fields (domain, modality, subdomain) so that the text-based methods
and the structured metadata matcher are not fed the same inferred signal twice.

Usage
-----
    python src/build_dataset_text.py
    python src/build_dataset_text.py --input data/fixture/catalog.csv --output data/fixture/catalog_text.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import load_config, resolve


def _fmt_int(x) -> str | None:
    return None if pd.isna(x) else f"{int(x)}"


def build_text(row: pd.Series) -> str:
    """Concatenate the useful explicit fields of one catalogue row into a single string."""
    parts: list[str] = []
    if pd.notna(row.get("dataset_name")):
        parts.append(f"Dataset name: {row['dataset_name']}.")
    if pd.notna(row.get("tags")):
        parts.append("Tags: " + str(row["tags"]).replace("|", ", ") + ".")
    if pd.notna(row.get("description")):
        parts.append(f"Description: {row['description']}")
    if pd.notna(row.get("target_attribute")):
        parts.append(f"Target attribute: {row['target_attribute']}.")
    if pd.notna(row.get("task_type")) and row.get("task_type_source") == "derived":
        parts.append(f"Task: {row['task_type']} with {_fmt_int(row.get('number_of_classes'))} classes.")
    facts = []
    if _fmt_int(row.get("number_of_instances")):
        facts.append(f"{_fmt_int(row['number_of_instances'])} instances")
    if _fmt_int(row.get("number_of_features")):
        facts.append(f"{_fmt_int(row['number_of_features'])} features")
    if facts:
        parts.append("Size: " + ", ".join(facts) + ".")
    if pd.notna(row.get("feature_names")):
        parts.append("Features: " + str(row["feature_names"]).replace("|", ", ")[:500] + ".")
    return " ".join(parts)


def add_text_column(catalog: pd.DataFrame) -> pd.DataFrame:
    out = catalog.copy()
    out["text"] = out.apply(build_text, axis=1)
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=f"{cfg['paths']['processed_dir']}/catalog.csv")
    ap.add_argument("--output", default=f"{cfg['paths']['processed_dir']}/catalog_text.csv")
    args = ap.parse_args()
    df = pd.read_csv(resolve(args.input))
    df = add_text_column(df)
    df.to_csv(resolve(args.output), index=False)
    print(f"[text] wrote {len(df)} text representations -> {args.output}")
    print(f"[text] median text length: {int(df['text'].str.len().median())} characters")


if __name__ == "__main__":
    main()
