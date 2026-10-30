# Execution report (local execution)

Updated: 2026-10-30.

| Stage | Result |
|---|---|
| Reach www.openml.org | SUCCESS: OpenML REST API connected and functional |
| Reach huggingface.co (MiniLM download) | SUCCESS: `sentence-transformers/all-MiniLM-L6-v2` downloaded and cached |
| Python environment dependencies | SUCCESS: `torch`, `sentence-transformers`, `openml`, `pytest`, `python-pptx` installed |
| Unit tests | 43 passed |
| OpenML collection (`collect_openml_metadata.py`) | EXECUTED: Raw metadata fetched from OpenML API -> `data/raw/` |
| Catalog cleaning & text (`clean_catalog.py`, `build_dataset_text.py`) | EXECUTED: Cleaned records with provenance and text representations -> `data/processed/` |
| Benchmark generation (`build_benchmark.py`) | EXECUTED: Query set and relevance judgments generated -> `data/benchmark/` |
| MiniLM retrieval & real experiments (`run_experiments.py`) | EXECUTED: Real MiniLM embeddings, validation-tuned alpha, test set evaluation -> `outputs/results/` |
| Paper tables & figures (`generate_paper_tables.py`) | EXECUTED: Real metrics written to `paper/tables/*.tex`, `paper/figures/*.pdf`, macros updated (`\resultsstatus{final}`) |
| Presentation & script (`build_deliverables.py`) | EXECUTED: Real numbers compiled into `presentation/project_presentation.pptx` and `speaker_script.txt` |
| Real query CLI search (`search.py`) | EXECUTED: Hybrid ranking, explanation, and constraint auditing on real OpenML catalogue |
