"""STEP 1 - Collect dataset METADATA from OpenML (no dataset files are downloaded).

Usage
-----
    python src/collect_openml_metadata.py --limit 100          # smoke test
    python src/collect_openml_metadata.py --limit 1500         # full catalogue

Outputs (in data/raw/):
    openml_listing_<UTC timestamp>.csv   the dataset listing returned by OpenML (qualities)
    openml_raw_<UTC timestamp>.jsonl     one JSON object per dataset (raw metadata)
    openml_raw_latest.jsonl              copy of the most recent raw file (read by clean_catalog.py)
    collection_log.json                  provenance: timestamps, package versions, counts, failures

API notes (checked against the openml-python 'main' reference pages while writing this
script; verify against your installed version with ``pip show openml``):
  * ``openml.datasets.list_datasets(data_id=None, offset=None, size=None, status=None,
    tag=None, output_format='dict', **kwargs)`` -> dict or DataFrame; quality columns such as
    NumberOfInstances / NumberOfFeatures / NumberOfClasses appear when qualities were computed.
  * ``openml.datasets.get_dataset(dataset_id, download_data=..., download_qualities=...,
    download_features_meta_data=...)`` -> OpenMLDataset. Passing ``download_data=False``
    fetches only the dataset description XML, NOT the data file.
Because attribute names can differ between versions, every attribute is read with getattr
and missing values are stored as null rather than raising.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from common import load_config, resolve, resolve_dir, write_json

# Quality columns we try to keep from the listing (names as used by OpenML qualities).
LISTING_QUALITIES = [
    "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures", "NumberOfMissingValues", "NumberOfInstancesWithMissingValues",
    "MajorityClassSize", "MinorityClassSize",
]


def _import_openml():
    try:
        import openml  # noqa: WPS433
        return openml
    except ImportError:
        sys.exit("The 'openml' package is not installed. Run: pip install -r requirements.txt")


def list_candidates(openml, status: str, max_candidates: int) -> pd.DataFrame:
    """Return the OpenML dataset listing as a DataFrame (one row per dataset id)."""
    listing = None
    for pool in [max_candidates, 2000, 1000, 500, 200]:
        if pool > max_candidates:
            continue
        try:
            try:
                listing = openml.datasets.list_datasets(status=status, size=pool, output_format="dataframe")
            except TypeError:
                # Newer openml-python versions may drop/rename ``output_format``.
                listing = openml.datasets.list_datasets(status=status, size=pool)
            break
        except Exception as exc:
            if pool <= 200:
                raise
            print(f"[collect] candidate pool {pool} returned {exc}; retrying with smaller pool...", flush=True)
            time.sleep(2)
    if isinstance(listing, dict):
        listing = pd.DataFrame.from_dict(listing, orient="index")
    listing = pd.DataFrame(listing).reset_index(drop=True)
    if "did" not in listing.columns:
        for alt in ("dataset_id", "id"):
            if alt in listing.columns:
                listing = listing.rename(columns={alt: "did"})
                break
    if "did" not in listing.columns:
        raise RuntimeError(f"Could not find dataset id column in listing; columns = {list(listing.columns)}")
    return listing.drop_duplicates("did").sort_values("did").reset_index(drop=True)


def fetch_one(openml, did: int, with_features: bool, max_retries: int, pause: float) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch metadata for one dataset. Returns (record, error_message)."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            ds = openml.datasets.get_dataset(
                int(did),
                download_data=False,
                download_qualities=False,
                download_features_meta_data=with_features,
            )
            tags = getattr(ds, "tag", None)
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            feature_names = None
            if with_features:
                try:
                    feats = getattr(ds, "features", None) or {}
                    feature_names = [getattr(f, "name", None) for f in feats.values()]
                except Exception:  # features are optional
                    feature_names = None
            rec = {
                "dataset_id": int(did),
                "name": getattr(ds, "name", None),
                "version": getattr(ds, "version", None),
                "description": getattr(ds, "description", None),
                "tags": list(tags) if tags else [],
                "default_target_attribute": getattr(ds, "default_target_attribute", None),
                "row_id_attribute": getattr(ds, "row_id_attribute", None),
                "licence": getattr(ds, "licence", None),
                "creator": getattr(ds, "creator", None),
                "collection_date": getattr(ds, "collection_date", None),
                "language": getattr(ds, "language", None),
                "format": getattr(ds, "format", None),
                "original_data_url": getattr(ds, "original_data_url", None),
                "paper_url": getattr(ds, "paper_url", None),
                "openml_url": getattr(ds, "url", None),
                "feature_names": feature_names,
            }
            time.sleep(pause)
            return rec, None
        except Exception as exc:  # network error, deactivated dataset, API change, ...
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 10))
    return None, last_err


def main() -> None:
    cfg = load_config()
    ccfg = cfg["collection"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=ccfg["limit"], help="number of datasets to keep")
    ap.add_argument("--status", default=ccfg["status"])
    ap.add_argument("--seed", type=int, default=cfg["seed"], help="seed for the reproducible sample of ids")
    ap.add_argument("--with-features", action="store_true",
                    help="also request feature-name metadata (extra API call per dataset; still no data download)")
    ap.add_argument("--candidate-pool", type=int, default=10_000, help="size of the listing to sample from")
    args = ap.parse_args()

    openml = _import_openml()
    started = dt.datetime.now(dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    raw_dir = resolve_dir(cfg["paths"]["raw_dir"])

    print(f"[collect] listing OpenML datasets (status={args.status}) ...", flush=True)
    try:
        listing = list_candidates(openml, args.status, args.candidate_pool)
    except Exception as exc:
        sys.exit(f"[collect] FAILED to list datasets from OpenML: {type(exc).__name__}: {exc}\n"
                 "Check your internet connection / firewall (https://www.openml.org must be reachable).")
    print(f"[collect] listing returned {len(listing)} datasets", flush=True)
    listing_path = raw_dir / f"openml_listing_{stamp}.csv"
    listing.to_csv(listing_path, index=False)

    # Reproducible sample: shuffle ids with a fixed seed, then walk until `limit` records succeed.
    rng = np.random.default_rng(args.seed)
    ids = listing["did"].to_numpy()
    order = rng.permutation(len(ids))
    qual_cols = [c for c in LISTING_QUALITIES if c in listing.columns]
    qual_lookup = listing.set_index("did")[qual_cols].to_dict(orient="index") if qual_cols else {}
    other_cols = [c for c in ("name", "status", "format", "version") if c in listing.columns]
    other_lookup = listing.set_index("did")[other_cols].to_dict(orient="index") if other_cols else {}

    raw_path = raw_dir / f"openml_raw_{stamp}.jsonl"
    kept, failures = 0, []
    with open(raw_path, "w", encoding="utf-8") as out:
        for pos in order:
            if kept >= args.limit:
                break
            did = int(ids[pos])
            rec, err = fetch_one(openml, did, args.with_features, ccfg["max_retries"], ccfg["request_pause_s"])
            if rec is None:
                failures.append({"dataset_id": did, "error": err})
                continue
            rec["qualities"] = {k: (None if pd.isna(v) else float(v)) for k, v in qual_lookup.get(did, {}).items()}
            rec["listing"] = {k: (None if pd.isna(v) else v) for k, v in other_lookup.get(did, {}).items()}
            rec["source_repository"] = "OpenML"
            rec["source_url"] = f"https://www.openml.org/search?type=data&id={did}"
            rec["retrieved_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            kept += 1
            if kept % 10 == 0 or kept == args.limit:
                print(f"[collect] {kept}/{args.limit} records retrieved ({len(failures)} failures)", flush=True)

    shutil.copyfile(raw_path, raw_dir / "openml_raw_latest.jsonl")
    log = {
        "started_utc": started.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limit_requested": args.limit,
        "records_retrieved": kept,
        "failures": len(failures),
        "failure_examples": failures[:20],
        "listing_size": int(len(listing)),
        "seed": args.seed,
        "with_features": args.with_features,
        "openml_version": getattr(openml, "__version__", "unknown"),
        "raw_file": raw_path.name,
        "listing_file": listing_path.name,
    }
    write_json(log, raw_dir / "collection_log.json")
    print(f"[collect] done: {kept} records -> {raw_path}")
    if kept < args.limit:
        print(f"[collect] WARNING: requested {args.limit} but only {kept} retrieved; report the actual number.")


if __name__ == "__main__":
    main()
