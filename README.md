# Hybrid Semantic and Metadata-Aware Retrieval for Intelligent Dataset Recommendation

**Authors:** Divij Nanjibhai Bhoj, Parthiv Dave, Priyanka Bhatt
**Affiliation:** Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, Gujarat, India.

A researcher describes a machine-learning problem in plain language. The system searches a **catalogue of dataset metadata** (OpenML) and returns the top five datasets, each with its source link, metadata, the requirements it satisfies, the ones it fails, the ones it cannot verify, and a short explanation. Only metadata is collected; **no dataset files, images or patient records are downloaded.**

---------------------------------------------------------------------------------------------------

## 0. Current status

| Item | Status |
|---|---|
| Unit tests (cleaning, parser, retrieval, constraints, metrics, collector) | **Completed and executed**: 43 passed |
| OpenML metadata collection (`collect_openml_metadata.py`) | **Executed**: Real metadata harvested from OpenML API (100 raw -> 93 catalogue records) |
| MiniLM (`all-MiniLM-L6-v2`) semantic retrieval | **Executed**: Pretrained MiniLM weights downloaded, cached, and evaluated |
| Real-catalogue experiments and paper/slide result numbers | **Completed and executed**: Real metrics generated into `outputs/results/`, `paper/tables/`, and `paper/main.tex` |
| `build_deliverables.py` (slides + script from results) | **Executed on real results**: Updated `presentation/project_presentation.pptx` and `speaker_script.txt` |
| Human relevance annotation | **Future work**: Benchmarking uses synthetic/metadata-derived judgments (`human_verified_judgments = 0`) |
| Learned ranker, multi-repository catalogue, LLM query parser | Suggested future work |

---------------------------------------------------------------------------------------------------

## 1. Research questions

**Central:** Does combining semantic representations from pretrained language models with structured dataset metadata improve dataset recommendation compared with conventional lexical retrieval?

* **RQ1** TF-IDF vs. pretrained sentence-embedding retrieval.
* **RQ2** Does structured metadata help beyond semantic similarity alone?
* **RQ3** How does the hybrid behave when queries carry explicit constraints (size, modality, condition, labels)?
* **RQ4** How sensitive is performance to the weight alpha?

These are questions, not conclusions; the experiments decide.

## 2. Architecture

```
OpenML REST/openml-python (metadata only)
  -> collect_openml_metadata.py   -> data/raw/ (raw JSONL + listing + collection_log.json)
  -> clean_catalog.py             -> data/processed/catalog.csv (+ cleaning_report.json, *_source provenance columns)
  -> build_dataset_text.py        -> data/processed/catalog_text.csv (one text per dataset)
  -> build_benchmark.py generate  -> data/benchmark/ (queries, synthetic qrels, splits, annotation template)
  -> run_experiments.py           -> outputs/results/ (CSV + JSON)
  -> generate_paper_tables.py     -> paper/tables/*.tex, paper/figures/*.pdf
  -> paper/main.tex (IEEEtran)    -> paper/main.pdf
Query path: query -> rule-based parser (query_parser.py) -> {keyword, tfidf, semantic, metadata, hybrid} (retrieval.py) -> top-5 + explanation
```

Methods: **A** keyword, **B** TF-IDF + cosine, **C** MiniLM + cosine, **D** metadata-only, **E (proposed)** hybrid
`Final = alpha * Semantic + (1 - alpha) * MetadataMatch`.

Score definitions (none is a calibrated probability):
* `Semantic = max(0, cosine(q, d))` in [0,1].
* `MetadataMatch = sum_r w_r * 1[r satisfied] / sum_r w_r` over requirements parsed from the query. Each requirement is *satisfied*, *violated* or *unknown*. **Unknown (missing metadata) earns 0 credit and is never treated as a violation.**
* If the query has no parseable requirement, `Final = Semantic`.
* Mandatory requirements (minimum size, labels, binary): a *violated* one multiplies the score by 0.5 (`penalize`, default) or ranks the item last (`exclude`), and is flagged in the explanation. Config: `config/config.yaml`.
* At alpha = 1 the formula reduces to semantic similarity (plus constraint handling); at alpha = 0 it is metadata match only.

## 3. Repository structure

```
README.md  requirements.txt  environment.txt  .gitignore
config/config.yaml            all paths, seed, model, alpha grid, constraint policy
data/README.md                data provenance and licence notes
data/raw|processed|benchmark  produced by the pipeline (git-ignored)
data/fixture/                 SYNTHETIC fixture (fabricated, labelled, never mixed with real data)
src/   common.py collect_openml_metadata.py clean_catalog.py build_dataset_text.py
       query_parser.py retrieval.py build_benchmark.py evaluate.py run_experiments.py
       generate_paper_tables.py build_deliverables.py make_fixture.py search.py app.py
tests/ test_clean_catalog.py test_collect.py test_retrieval.py test_metrics.py
outputs/results|tables|figures        real-run outputs;  outputs/results_fixture/ = fixture check only
paper/ main.tex references.bib tables/ figures/
presentation/ project_presentation.pptx speaker_script.txt
```

## 4. Installation

Requires Python 3.10+ (developed on 3.13), CPU only, no paid API, no patient data.

**Local**
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # expect: all tests pass
```

**Google Colab** (upload the project ZIP to the Colab session first)
```python
!unzip -q dataset-recommender.zip && cd dataset-recommender && pip install -r requirements.txt
%cd /content/dataset-recommender
!python -m pytest -q
```
Colab already ships numpy/pandas/scikit-learn/torch; `pip install -r requirements.txt` adds `openml`, `sentence-transformers`, `streamlit`.

## 5. Run order (exact commands)

```bash
# 5.1  Smoke test: small catalogue (minutes)
python src/collect_openml_metadata.py --limit 100
python src/clean_catalog.py
python src/build_dataset_text.py
python src/build_benchmark.py generate
python src/run_experiments.py --semantic-backend minilm --n-boot 200

# 5.2  Full catalogue (target 500-2000 records; use the number actually retained)
python src/collect_openml_metadata.py --limit 1500          # add --with-features for feature names
python src/clean_catalog.py
python src/build_dataset_text.py
python src/build_benchmark.py generate
python src/run_experiments.py --semantic-backend minilm
```
Expected files: `data/raw/openml_raw_latest.jsonl`, `data/raw/collection_log.json` (timestamps, package versions, failures), `data/processed/catalog.csv`, `cleaning_report.json`, `catalog_text.csv`, `embeddings_minilm.npy` (+ `.json` fingerprint), `data/benchmark/queries.csv`, `qrels_synthetic.csv`, `annotation_template.csv`, `benchmark_summary.json`, and in `outputs/results/`: `run_metadata.json`, `main_results.csv`, `ablation.csv`, `alpha_selection.json`, `paired_comparisons.csv`, `by_query_type.csv`, `per_query_metrics.csv`, `rankings_top10.csv`.
The first MiniLM run downloads the model once (~90 MB) and caches embeddings.

**Run each method on one query**
```bash
python src/search.py --method keyword  --query "I need a dataset for breast cancer classification"
python src/search.py --method tfidf    --query "I need a dataset for breast cancer classification"
python src/search.py --method semantic --query "I need a dataset for breast cancer classification"
python src/search.py --method metadata --query "labeled binary dataset with at least 500 instances"
python src/search.py --method hybrid --alpha 0.7 --query "I need a labeled dataset for lung cancer classification using CT images, with at least 500 patients and preferably demographic information"
```
**Offline check with no internet** (synthetic fixture, LSA stand-in; *never* report these as results):
```bash
python src/make_fixture.py
python src/search.py --fixture --backend lsa --method hybrid --query "ECG arrhythmia detection"
python src/run_experiments.py --fixture --semantic-backend lsa
```

## 6. Benchmark and relevance annotation

`build_benchmark.py generate` creates ~400 template queries over a topic list, graded 0-3 by **rules over catalogue metadata**. These labels are **synthetic / metadata-derived** (`annotation_source = synthetic_metadata_derived`), not human judgments. Because the metadata matcher reads the same metadata, scores against them are **biased in favour of the metadata methods**, especially on constrained queries. Use them for development; report them as preliminary.

**Leakage control:** queries are split by *topic group* (train/validation/test = 50/20/30% of topics); all paraphrases, templates and constraint variants of a topic stay in one split. Alpha is tuned on validation only; the test split is evaluated once. The same topic list is used for query generation and label rules, which is an additional circularity to disclose.

**Manual annotation workflow (required before any publication claim)**
```bash
python src/run_experiments.py --semantic-backend minilm            # produces rankings_top10.csv
python src/build_benchmark.py make-pool --n-queries 40 --depth 5   # pools top-5 of every method
```
1. Open `data/benchmark/annotation_sheet.csv` (written by `make-pool`) in a spreadsheet.
2. For each row read the dataset name, source link and description excerpt, then fill `human_relevance` with 0, 1, 2 or 3 (0 irrelevant, 1 partial, 2 relevant, 3 highly relevant), your `annotator_id`, and optional `annotation_notes`. Use at least two annotators on a shared subset and report agreement (e.g. Cohen's kappa).
3. Save as `data/benchmark/annotation_sheet_filled.csv`, then:
```bash
python src/build_benchmark.py merge --annotated data/benchmark/annotation_sheet_filled.csv
python src/run_experiments.py --semantic-backend minilm --qrels reviewed
```
Disclose how many queries and judgments were reviewed and how they were selected. Never describe labels as human-verified unless they were.

## 7. Streamlit demo
```bash
streamlit run src/app.py
```
Enter a requirement; see the extracted requirements, top-5 matches, source links, scores, metadata, matched / unsatisfied / unknown requirements and explanations. In Colab, run it with a tunnel (e.g. `npx localtunnel --port 8501`) or demo with `search.py`.

## 8. Reproducing tables and figures, and compiling the paper
```bash
python src/generate_paper_tables.py          # reads outputs/results/, writes paper/tables and paper/figures
cd paper && latexmk -pdf main.tex            # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```
**Slides and speaking script from real results** (after a valid real run only; it refuses fixture or LSA runs):
```bash
python src/build_deliverables.py     # rewrites presentation/project_presentation.pptx and speaker_script.txt with the measured numbers
```
It reads `outputs/results/` and `presentation/professor_qa.txt`; a software-only preview is `python src/build_deliverables.py --preview-fixture --out-dir /tmp/preview`.

Needs a TeX distribution with `IEEEtran` (TeX Live full, MiKTeX, or Overleaf: upload `paper/`). Until a valid real run exists, the PDF shows PENDING placeholders. Everything quoted in the prose comes from `paper/tables/macros.tex`.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `openml` import error / `output_format` TypeError | `pip install -r requirements.txt`; the collector already falls back if `output_format` is gone. Check `pip show openml` |
| Collection is slow or some IDs fail | Normal; failures are logged in `collection_log.json`. Lower `--limit` first. Rerun is safe |
| `Catalogue not found` | Run the pipeline in Section 5 in order, or use `--fixture` |
| Model download fails | Needs internet once for huggingface.co. Offline: use `--backend lsa` only for pipeline checks |
| `generate_paper_tables.py` says PENDING | Run `run_experiments.py` with `--semantic-backend minilm` and no `--fixture` |
| `IEEEtran.cls not found` | Install `texlive-publishers` (Linux) or use Overleaf |
| Streamlit shows no catalogue | Run the pipeline, or select "Synthetic fixture" after `python src/make_fixture.py` |

## 10. Sources
* OpenML: https://www.openml.org/ , docs https://docs.openml.org/ . Vanschoren et al., "OpenML: networked science in machine learning," *SIGKDD Explorations* 15(2), 2013 (see `paper/references.bib` for verification notes).
* Reimers & Gurevych, "Sentence-BERT," EMNLP-IJCNLP 2019. Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
* scikit-learn (TfidfVectorizer, cosine similarity): Pedregosa et al., JMLR 2011.
All bibliography entries carry a verification status at the top of `references.bib`; **re-check the entries marked PARTIALLY VERIFIED before submission.**

## 11. Known limitations
* OpenML is mostly tabular benchmark data; clinical, imaging and genomic datasets (TCIA, GDC, PhysioNet) are **not** in this catalogue, so queries such as "CT images" can only be matched against whatever OpenML records mention them. The catalogue is neither exclusively medical nor complete.
* Domain, modality and subdomain are **keyword-inferred** (see `*_source` columns) and unverified; licence is often missing.
* Synthetic queries and metadata-derived labels are circular and biased (Section 6).
* The rule-based parser handles a limited vocabulary and phrasing.
* Finding a dataset is not permission to use or redistribute it: check each source's licence.
* A catalogue of ~1,000 records and ~400 template queries cannot support strong statistical claims.

## 12. Before this can be submitted as a research paper
1. Run the real collection and experiments; record the real counts.
2. Human-annotate pooled results (two or more annotators, report agreement) and re-run with `--qrels reviewed`.
3. Re-verify every entry marked PARTIALLY VERIFIED in `references.bib`; fill the PENDING items.
4. Add a supervisor/affiliation check, ethics statement review and your institution's AI-use disclosure; confirm the venue's template.
5. Consider broader catalogues (Hugging Face, TCIA, GDC, PhysioNet) with a documented schema mapping.
