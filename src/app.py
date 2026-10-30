"""Streamlit demo: natural-language dataset requirement -> top-5 explained recommendations.

Launch:
    streamlit run src/app.py

The app only reads catalogue METADATA; it never downloads the underlying datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, resolve  # noqa: E402
from retrieval import Recommender, make_backend  # noqa: E402

cfg = load_config()
st.set_page_config(page_title="Dataset Recommender", layout="wide")
st.title("Hybrid Semantic and Metadata-Aware Dataset Recommender")
st.caption("Divij Nanjibhai Bhoj and Parthiv Dave - Department of Artificial Intelligence, SVNIT Surat. "
           "Research prototype: rankings are scores, not probabilities, and relevance is not verified.")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Catalogue", ["Real OpenML catalogue", "Synthetic fixture (demo only)"], index=0)
    backend_kind = st.radio("Semantic model", ["MiniLM (all-MiniLM-L6-v2)", "LSA offline stand-in"], index=0)
    alpha = st.slider("alpha (weight on semantic similarity)", 0.0, 1.0, float(cfg["retrieval"]["alpha_default"]), 0.05)
    constraints_on = st.checkbox("Apply hard-constraint handling", value=True)
    policy = st.selectbox("Hard-constraint policy", ["penalize", "exclude"], index=0)

catalog_path = (cfg["paths"]["processed_dir"] + "/catalog_text.csv") if source.startswith("Real") else (cfg["paths"]["fixture_dir"] + "/catalog_text.csv")
if not resolve(catalog_path).exists():
    st.error(f"Catalogue file not found: {catalog_path}. Run the pipeline first "
             "(see README) or choose the synthetic fixture after running `python src/make_fixture.py`.")
    st.stop()
if not source.startswith("Real"):
    st.warning("SYNTHETIC FIXTURE: fabricated records for demonstration. Not real datasets.")
if backend_kind.startswith("LSA"):
    st.warning("LSA stand-in is NOT the pretrained MiniLM model.")


@st.cache_resource(show_spinner="Building indices (first run downloads/encodes with MiniLM)...")
def load_recommender(path: str, kind: str, pol: str) -> Recommender:
    df = pd.read_csv(resolve(path))
    c = dict(cfg)
    c["retrieval"] = {**cfg["retrieval"], "hard_constraint_policy": pol}
    backend = make_backend("minilm" if kind.startswith("MiniLM") else "lsa", c, cache_path=None if not source.startswith("Real") else None)
    return Recommender(df, backend, c)


try:
    rec = load_recommender(catalog_path, backend_kind, policy)
except Exception as exc:  # e.g. sentence-transformers missing or model download blocked
    st.error(f"Could not initialise the retriever: {type(exc).__name__}: {exc}")
    st.stop()

st.write(f"Catalogue size: **{len(rec.cat)}** dataset records.")
example = "I need a labeled dataset for lung cancer classification using CT images, with at least 500 patients and preferably demographic information."
query = st.text_area("Describe the dataset you need", value=example, height=100)

if st.button("Recommend", type="primary") and query.strip():
    parsed = rec.parse(query)
    with st.expander("Requirements extracted from your query (rule-based)", expanded=True):
        if parsed.requirements:
            st.table(pd.DataFrame([{"requirement": r.kind, "value": r.value, "mandatory": r.mandatory, "triggered by": r.evidence} for r in parsed.requirements]))
        else:
            st.info("No structured requirements found; ranking uses semantic similarity only.")
    results = rec.recommend(query, k=5, method="hybrid", alpha=alpha, apply_constraints=constraints_on)
    for r in results:
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                title = f"#{r['rank']}  {r['dataset_name']}"
                url = r["source_url"]
                st.subheader(title)
                if str(url).startswith("http"):
                    st.markdown(f"Source: [{url}]({url})")
                else:
                    st.markdown(f"Source: `{url}`")
                st.write(r["explanation"])
            with c2:
                st.metric("Final score (ranking score)", f"{r['score']:.3f}")
                st.caption(f"semantic {r['semantic_score']:.3f} | metadata {('%.3f' % r['metadata_score']) if r['metadata_score'] is not None else 'n/a'} | "
                           f"TF-IDF {r['tfidf_score']:.3f} | keyword {r['keyword_score']:.3f}")
            m1, m2, m3 = st.columns(3)
            m1.markdown("**Matched**\n\n" + ("\n".join(f"- {x}" for x in r["matched"]) or "-"))
            m2.markdown("**Not satisfied**\n\n" + ("\n".join(f"- {x}" for x in r["unsatisfied"]) or "-"))
            m3.markdown("**Unknown (no metadata evidence)**\n\n" + ("\n".join(f"- {x}" for x in r["unknown"]) or "-"))
            if r["hard_violation"]:
                st.error("Violates a mandatory requirement (score penalised / excluded according to the selected policy).")
            st.json({k: v for k, v in r["metadata"].items() if v is not None}, expanded=False)
    st.caption("Metadata marked 'inferred' in the catalogue (domain, modality, subdomain) comes from keyword rules and is not verified. "
               "Discovering a dataset here does not grant access to, or redistribution rights for, the underlying data - check the source license.")
