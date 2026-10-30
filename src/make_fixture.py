"""Generate a small SYNTHETIC FIXTURE catalogue for offline smoke tests and unit tests.

*** These records are fabricated by this script. They are NOT OpenML data. ***
They exist only so the pipeline and tests can run without internet access. Every record carries
source_repository = 'SYNTHETIC_FIXTURE', a 'synthetic://' URL, '[SYNTHETIC FIXTURE]' in its name
and is_synthetic = True. They are written to data/fixture/ and never to data/raw or
data/processed. Results computed on the fixture are pipeline checks, not scientific evidence.

Usage
-----
    python src/make_fixture.py
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from build_benchmark import BENCHMARK_TOPICS
from build_dataset_text import add_text_column
from clean_catalog import clean_catalog
from common import load_config, resolve, write_json

FIXTURE_ID_START = 900_001


def make_raw_records(seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    recs: list[dict] = []
    next_id = FIXTURE_ID_START

    def add(name, desc, tags, n_inst, n_feat, n_cls, target, licence):
        nonlocal next_id
        recs.append({
            "dataset_id": next_id, "name": f"[SYNTHETIC FIXTURE] {name}", "description": desc, "tags": tags,
            "default_target_attribute": target, "licence": licence, "format": "ARFF",
            "qualities": {"NumberOfInstances": n_inst, "NumberOfFeatures": n_feat, "NumberOfClasses": n_cls},
            "source_repository": "SYNTHETIC_FIXTURE", "source_url": f"synthetic://fixture/{next_id}",
            "retrieved_at_utc": now,
        })
        next_id += 1

    def size():
        n = int(np.exp(rng.uniform(np.log(60), np.log(60000))))
        return n, int(rng.integers(4, 80)), int(rng.choice([0, 2, 2, 3, 5]))

    for tname, t in BENCHMARK_TOPICS.items():
        strong, weak = t["strong"], t["weak"]
        for j in range(4):  # strong in name + tags
            n, f, c = size()
            word = strong[j % len(strong)]
            add(f"{word} study {j + 1}", f"Synthetic record about {word} measurements collected for methodological testing, entry {j + 1}.",
                [word, "synthetic"], n, f, c, "target" if rng.random() > 0.1 else None, None)
        for j in range(3):  # strong only in description
            n, f, c = size()
            word = strong[(j + 1) % len(strong)]
            add(f"collection {tname[:4]}{j + 1}", f"A generic table whose records relate to {word} in the description only; entry {j + 1} of the fixture.",
                [], n, f, c, "label", "CC-BY-4.0")
        for j in range(2):  # weak only
            n, f, c = size()
            word = weak[j % len(weak)]
            add(f"misc {tname[:4]}{j + 1}", f"Loosely related synthetic table mentioning {word} once; used as a partially relevant record {j + 1}.",
                ["misc"], n, f, c, None, None)
    for j in range(25):  # distractors / thin metadata
        n, f, c = size()
        add(f"distractor {j}", "Synthetic distractor record with unrelated content about arbitrary quantities and abstract measurements." if j % 5 else "x",
            [] if j % 5 == 0 else ["noise"], n, f, c, "y", None)
    # duplicates to exercise deduplication
    recs.append(dict(recs[0]))
    dup = dict(recs[1]); dup["dataset_id"] = next_id; recs.append(dup)
    return recs


def main() -> None:
    cfg = load_config()
    fixture_dir = cfg["paths"]["fixture_dir"]
    raw = make_raw_records(cfg["seed"])
    with open(resolve(f"{fixture_dir}/fixture_raw.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")
    cat, report = clean_catalog(raw, cfg["collection"]["min_description_chars"])
    cat.to_csv(resolve(f"{fixture_dir}/catalog.csv"), index=False)
    add_text_column(cat).to_csv(resolve(f"{fixture_dir}/catalog_text.csv"), index=False)
    write_json(report, f"{fixture_dir}/cleaning_report.json")
    (resolve(f"{fixture_dir}/README.txt")).write_text(
        "SYNTHETIC FIXTURE - fabricated records for smoke tests only. Not OpenML data. Never cite results computed on it.\n")
    print(f"[fixture] {len(raw)} raw synthetic records -> {report['n_final']} after cleaning -> {fixture_dir}/catalog_text.csv")


if __name__ == "__main__":
    main()
