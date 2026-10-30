"""STEPS 4-8 - Retrieval methods, structured metadata matching, hybrid ranking, explanations.

Methods (names used throughout the project)
-------------------------------------------
  keyword   (A) weighted keyword matching on name / tags / description
  tfidf     (B) TF-IDF + cosine similarity (scikit-learn)
  semantic  (C) pretrained sentence embeddings + cosine similarity
  metadata  (D) structured-requirement matching only (no text similarity)
  hybrid    (E) FinalScore = alpha * Semantic + (1 - alpha) * MetadataMatch  (+ constraint handling)

Score definitions (none of these is a calibrated probability)
-------------------------------------------------------------
  Semantic(q,d)   = max(0, cosine(embedding(q), embedding(d)))  in [0, 1]
  MetadataMatch   = sum_r w_r * 1[status_r = satisfied] / sum_r w_r  in [0, 1]
                    over the requirements r parsed from the query. A requirement's status is
                    'satisfied', 'violated' or 'unknown' (catalogue has no evidence). Unknown
                    earns ``unknown_credit`` (default 0), i.e. missing metadata is neither
                    rewarded nor treated as a negative fact.
  If the query yields NO requirements the metadata component is undefined and
                    Final = Semantic (alpha is effectively 1 for that query).
  Hard constraints: requirements flagged mandatory by the parser. If a dataset is *violated*
                    on a mandatory requirement, the policy 'penalize' multiplies Final by
                    ``hard_constraint_penalty`` and flags the violation; policy 'exclude'
                    ranks it last. 'unknown' never triggers the policy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from common import QUERY_FILLER, contains_term, normalize_text, resolve, tokenize
from query_parser import LARGE_THRESHOLD, ParsedQuery, Requirement, parse_query

SAT, UNK, VIO = 1, 0, -1
REQUIREMENT_WEIGHTS = {
    "min_instances": 2.0, "labels": 1.5, "n_classes": 1.5, "min_classes": 1.0,
    "condition": 2.0, "modality": 1.5, "task": 1.0, "domain": 1.0,
    "demographic": 0.5, "large": 0.5,
}
DEMOGRAPHIC_TERMS = ["age", "sex", "gender", "race", "ethnicity", "demographic", "demographics"]
METHODS = ("keyword", "tfidf", "semantic", "metadata", "hybrid")


def _stem(tok: str) -> str:
    """Very light plural stripping so 'images' matches 'image'."""
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def query_terms(query: str) -> list[str]:
    toks = [_stem(t) for t in tokenize(query) if t not in QUERY_FILLER and t not in ENGLISH_STOP_WORDS and not t.isdigit()]
    seen: set[str] = set()
    return [t for t in toks if not (t in seen or seen.add(t))]


# =========================================================================== catalogue index

class CatalogIndex:
    """Pre-computed numpy views of the catalogue used for vectorised requirement checks."""

    def __init__(self, cat: pd.DataFrame):
        self.cat = cat.reset_index(drop=True)
        n = len(self.cat)
        self.n = n
        self.ids = self.cat["dataset_id"].to_numpy()
        g = lambda c: self.cat[c] if c in self.cat.columns else pd.Series([np.nan] * n)
        self.n_inst = pd.to_numeric(g("number_of_instances"), errors="coerce").to_numpy(dtype=float)
        self.n_cls = pd.to_numeric(g("number_of_classes"), errors="coerce").to_numpy(dtype=float)
        hl = g("has_labels")
        self.has_labels = np.array([np.nan if pd.isna(v) else (1.0 if str(v).lower() in {"true", "1", "1.0"} else 0.0) for v in hl])
        self.task = g("task_type").fillna("").astype(str).str.lower().to_numpy()
        self.domain = g("domain").fillna("").astype(str).str.lower().to_numpy()
        self.modality = [set(m.split("|")) if isinstance(m, str) and m else set() for m in g("modality")]
        self.name = g("dataset_name").fillna("").astype(str).str.lower().to_numpy()
        self.tags = g("tags").fillna("").astype(str).str.replace("|", " ", regex=False).str.lower().to_numpy()
        self.desc = g("description").fillna("").astype(str).str.lower().to_numpy()
        self.feat = g("feature_names").fillna("").astype(str).str.replace("|", " ", regex=False).str.lower().to_numpy()
        self.desc_len = np.array([len(d) for d in self.desc])
        self.full = np.array([f"{a} {b} {c}" for a, b, c in zip(self.name, self.tags, self.desc)])
        self._contains_cache: dict[str, np.ndarray] = {}

    def contains(self, term: str, include_features: bool = False) -> np.ndarray:
        key = f"{term}|{include_features}"
        if key not in self._contains_cache:
            hay = (self.full + " " + self.feat) if include_features else self.full
            self._contains_cache[key] = np.array([contains_term(h, term) for h in hay])
        return self._contains_cache[key]


# =========================================================================== requirement matching

def requirement_status(req: Requirement, idx: CatalogIndex) -> np.ndarray:
    """Return an int8 array (+1 satisfied / 0 unknown / -1 violated) for one requirement."""
    n = idx.n
    out = np.zeros(n, dtype=np.int8)
    k, v = req.kind, req.value
    if k in ("min_instances", "large"):
        known = ~np.isnan(idx.n_inst)
        thr = float(v)
        out[known & (idx.n_inst >= thr)] = SAT
        out[known & (idx.n_inst < thr)] = VIO
    elif k == "labels":
        known = ~np.isnan(idx.has_labels)
        out[known & (idx.has_labels == 1.0)] = SAT
        out[known & (idx.has_labels == 0.0)] = VIO
    elif k == "n_classes":
        known = ~np.isnan(idx.n_cls) & (idx.n_cls >= 1)
        out[known & (idx.n_cls == float(v))] = SAT
        out[known & (idx.n_cls != float(v))] = VIO
    elif k == "min_classes":
        known = ~np.isnan(idx.n_cls) & (idx.n_cls >= 1)
        out[known & (idx.n_cls >= float(v))] = SAT
        out[known & (idx.n_cls < float(v))] = VIO
    elif k == "task":
        known = idx.task != ""
        out[known & (idx.task == str(v))] = SAT
        out[known & (idx.task != str(v))] = VIO
    elif k == "domain":
        known = idx.domain != ""
        out[known & (idx.domain == str(v))] = SAT
        out[known & (idx.domain != str(v))] = VIO
    elif k == "condition":
        found = idx.contains(str(v))
        out[found] = SAT
        out[~found & (idx.desc_len >= 40)] = VIO  # absent from a reasonably long description
    elif k == "modality":
        if str(v) == "tabular":
            out[:] = SAT  # repository default (OpenML stores tabular files)
        else:
            known = np.array([len(m) > 0 for m in idx.modality])
            has = np.array([str(v) in m for m in idx.modality])
            out[has] = SAT
            out[known & ~has] = VIO
    elif k == "demographic":
        hit = np.zeros(n, dtype=bool)
        for t in DEMOGRAPHIC_TERMS:
            hit |= idx.contains(t, include_features=True)
        out[hit] = SAT  # positive evidence only; absence stays 'unknown'
    return out


@dataclass
class MetadataResult:
    score: np.ndarray                 # (n,) in [0,1]; zeros if no requirements
    has_requirements: bool
    hard_violation: np.ndarray        # (n,) bool
    statuses: list[tuple[Requirement, np.ndarray]]


def metadata_match(parsed: ParsedQuery, idx: CatalogIndex, unknown_credit: float = 0.0) -> MetadataResult:
    reqs = parsed.requirements
    n = idx.n
    if not reqs:
        return MetadataResult(np.zeros(n), False, np.zeros(n, dtype=bool), [])
    statuses, num, den = [], np.zeros(n), 0.0
    hard = np.zeros(n, dtype=bool)
    for r in reqs:
        st = requirement_status(r, idx)
        w = REQUIREMENT_WEIGHTS.get(r.kind, 1.0)
        num += w * (st == SAT) + w * unknown_credit * (st == UNK)
        den += w
        if r.mandatory:
            hard |= st == VIO
        statuses.append((r, st))
    return MetadataResult(num / den, True, hard, statuses)


def combine_hybrid(sem: np.ndarray, meta: MetadataResult, alpha: float, apply_constraints: bool = True,
                   policy: str = "penalize", penalty: float = 0.5) -> np.ndarray:
    """FinalScore = alpha*sem + (1-alpha)*meta, then optional hard-constraint handling."""
    if not meta.has_requirements:
        return sem.copy()
    final = alpha * sem + (1.0 - alpha) * meta.score
    if apply_constraints:
        if policy == "exclude":
            final = np.where(meta.hard_violation, -1.0, final)
        else:
            final = np.where(meta.hard_violation, final * penalty, final)
    return final


def rank_scores(scores: np.ndarray, ids: np.ndarray, k: int | None = None) -> np.ndarray:
    """Indices sorted by descending score; ties broken by ascending dataset_id (deterministic)."""
    order = np.lexsort((ids, -scores))
    return order if k is None else order[:k]


# =========================================================================== semantic backends

class SemanticBackend:
    name = "abstract"

    def fit(self, texts: Sequence[str]) -> None: ...

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray: ...

    def doc_matrix(self) -> np.ndarray: ...

    def similarity(self, queries: Sequence[str]) -> np.ndarray:
        """Cosine similarity matrix (n_queries, n_docs); embeddings are L2-normalised."""
        return self.encode_queries(queries) @ self.doc_matrix().T


class MiniLMBackend(SemanticBackend):
    """sentence-transformers/all-MiniLM-L6-v2 (pretrained; never trained here)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_path: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.name = "all-MiniLM-L6-v2"
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._model = None
        self._docs: np.ndarray | None = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("sentence-transformers is not installed (pip install -r requirements.txt)") from exc
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @staticmethod
    def _fingerprint(texts: Sequence[str], model_name: str) -> str:
        h = hashlib.sha256(model_name.encode())
        for t in texts:
            h.update(t.encode("utf-8", "ignore"))
        return h.hexdigest()

    def fit(self, texts: Sequence[str]) -> None:
        fp = self._fingerprint(texts, self.model_name)
        if self.cache_path:
            npy, meta = Path(resolve(self.cache_path)), Path(resolve(self.cache_path)).with_suffix(".json")
            if npy.exists() and meta.exists():
                try:
                    if json.loads(meta.read_text())["fingerprint"] == fp:
                        self._docs = np.load(npy)
                        return
                except Exception:
                    pass
        model = self._get_model()
        self._docs = model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        if self.cache_path:
            np.save(npy, self._docs)
            meta.write_text(json.dumps({"fingerprint": fp, "model": self.model_name, "n": len(texts)}))

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._get_model().encode(list(queries), batch_size=self.batch_size, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    def doc_matrix(self) -> np.ndarray:
        assert self._docs is not None, "call fit() first"
        return self._docs


class LSABackend(SemanticBackend):
    """OFFLINE STAND-IN ONLY: TF-IDF + truncated SVD (latent semantic analysis).

    This is NOT the pretrained MiniLM model. It exists so that the full hybrid code path can be
    executed and tested without network access. Results produced with it must be labelled
    'LSA (offline stand-in)' and must never be reported as MiniLM results.
    """
    name = "lsa_offline_standin"

    def __init__(self, n_components: int = 64, seed: int = 42):
        self.n_components, self.seed = n_components, seed
        self._vec: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._docs: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> None:
        self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        X = self._vec.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self._docs = normalize(self._svd.fit_transform(X))

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return normalize(self._svd.transform(self._vec.transform(list(queries))))

    def doc_matrix(self) -> np.ndarray:
        assert self._docs is not None, "call fit() first"
        return self._docs


def make_backend(kind: str, cfg: dict[str, Any], cache_path: str | None = None) -> SemanticBackend:
    if kind == "minilm":
        return MiniLMBackend(cfg["model"]["name"], cache_path or cfg["model"]["embedding_cache"], cfg["model"]["batch_size"])
    if kind == "lsa":
        return LSABackend(seed=cfg["seed"])
    raise ValueError(f"unknown semantic backend '{kind}' (use 'minilm' or 'lsa')")


# =========================================================================== lexical retrievers

class KeywordRetriever:
    """Method A: weighted presence of query terms in name (3), tags (2), description (1)."""
    FIELD_WEIGHTS = (3.0, 2.0, 1.0)

    def __init__(self, idx: CatalogIndex):
        self.idx = idx
        f = lambda arr: [set(_stem(t) for t in tokenize(s)) for s in arr]
        self.name_t, self.tag_t, self.desc_t = f(idx.name), f(idx.tags), f(idx.desc)

    def score(self, query: str) -> np.ndarray:
        terms = query_terms(query)
        out = np.zeros(self.idx.n)
        if not terms:
            return out
        wmax = sum(self.FIELD_WEIGHTS)
        for i in range(self.idx.n):
            s = 0.0
            for t in terms:
                s += (3.0 * (t in self.name_t[i]) + 2.0 * (t in self.tag_t[i]) + 1.0 * (t in self.desc_t[i])) / wmax
            out[i] = s / len(terms)
        return out


class TfidfRetriever:
    """Method B: TF-IDF (scikit-learn) + cosine similarity."""

    def __init__(self, texts: Sequence[str]):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.docs = self.vec.fit_transform(texts)

    def similarity(self, queries: Sequence[str]) -> np.ndarray:
        q = self.vec.transform(list(queries))
        return (q @ self.docs.T).toarray()  # both L2-normalised -> cosine


# =========================================================================== recommender facade

class Recommender:
    """Holds the catalogue, all retrievers and caches; produces score matrices and explained top-k."""

    def __init__(self, catalog: pd.DataFrame, backend: SemanticBackend | None = None, cfg: dict[str, Any] | None = None,
                 text_col: str = "text"):
        if text_col not in catalog.columns:
            raise ValueError(f"catalogue needs a '{text_col}' column (run build_dataset_text.py)")
        self.cat = catalog.reset_index(drop=True)
        self.cfg = cfg or {}
        rcfg = self.cfg.get("retrieval", {})
        self.alpha_default = rcfg.get("alpha_default", 0.7)
        self.policy = rcfg.get("hard_constraint_policy", "penalize")
        self.penalty = rcfg.get("hard_constraint_penalty", 0.5)
        self.idx = CatalogIndex(self.cat)
        self.texts = self.cat[text_col].fillna("").astype(str).tolist()
        self.keyword = KeywordRetriever(self.idx)
        self.tfidf = TfidfRetriever(self.texts)
        self.backend = backend
        if backend is not None:
            backend.fit(self.texts)
        self._parse_cache: dict[str, ParsedQuery] = {}

    # -- parsing / components -------------------------------------------------------------
    def parse(self, q: str) -> ParsedQuery:
        if q not in self._parse_cache:
            self._parse_cache[q] = parse_query(q)
        return self._parse_cache[q]

    def semantic_scores(self, queries: Sequence[str]) -> np.ndarray:
        if self.backend is None:
            raise RuntimeError("no semantic backend configured")
        return np.clip(self.backend.similarity(queries), 0.0, 1.0)

    def components(self, queries: Sequence[str]) -> dict[str, Any]:
        """All per-query component scores, computed once and reused across methods/alphas."""
        comp: dict[str, Any] = {
            "keyword": np.vstack([self.keyword.score(q) for q in queries]),
            "tfidf": self.tfidf.similarity(queries),
            "meta": [metadata_match(self.parse(q), self.idx) for q in queries],
        }
        if self.backend is not None:
            comp["semantic"] = self.semantic_scores(queries)
        return comp

    def scores_from_components(self, comp: dict[str, Any], method: str, alpha: float | None = None,
                               apply_constraints: bool = True) -> np.ndarray:
        alpha = self.alpha_default if alpha is None else alpha
        n_q = comp["keyword"].shape[0]
        if method in ("keyword", "tfidf", "semantic"):
            return comp[method]
        if method == "metadata":
            return np.vstack([m.score for m in comp["meta"]])
        if method == "hybrid":
            return np.vstack([
                combine_hybrid(comp["semantic"][i], comp["meta"][i], alpha, apply_constraints, self.policy, self.penalty)
                for i in range(n_q)
            ])
        raise ValueError(f"unknown method {method}")

    # -- single-query explained recommendation --------------------------------------------
    def recommend(self, query: str, k: int = 5, method: str = "hybrid", alpha: float | None = None,
                  apply_constraints: bool = True) -> list[dict[str, Any]]:
        alpha = self.alpha_default if alpha is None else alpha
        comp = self.components([query])
        scores = self.scores_from_components(comp, method, alpha, apply_constraints)[0]
        order = rank_scores(scores, self.idx.ids, k)
        meta: MetadataResult = comp["meta"][0]
        results = []
        for rank, i in enumerate(order, start=1):
            row = self.cat.iloc[i]
            matched, unsatisfied, unknown = [], [], []
            for req, st in meta.statuses:
                (matched if st[i] == SAT else unsatisfied if st[i] == VIO else unknown).append(req)
            sem_val = float(comp["semantic"][0][i]) if "semantic" in comp else None
            results.append({
                "rank": rank,
                "dataset_id": int(row["dataset_id"]),
                "dataset_name": row["dataset_name"],
                "source_url": row["source_url"],
                "score": float(scores[i]),
                "semantic_score": sem_val,
                "tfidf_score": float(comp["tfidf"][0][i]),
                "keyword_score": float(comp["keyword"][0][i]),
                "metadata_score": float(meta.score[i]) if meta.has_requirements else None,
                "hard_violation": bool(meta.hard_violation[i]),
                "matched": [r.label() for r in matched],
                "unsatisfied": [r.label() for r in unsatisfied],
                "unknown": [r.label() for r in unknown],
                "metadata": {c: (None if pd.isna(row.get(c)) else row.get(c)) for c in
                             ["domain", "subdomain", "task_type", "modality", "number_of_instances", "number_of_features",
                              "number_of_classes", "target_attribute", "has_labels", "license", "source_repository"] if c in row.index},
                "explanation": explain(rank, method, alpha, sem_val, meta, i, scores[i], matched, unsatisfied, unknown),
            })
        return results


def explain(rank: int, method: str, alpha: float, sem: float | None, meta: MetadataResult, i: int, score: float,
            matched: list[Requirement], unsatisfied: list[Requirement], unknown: list[Requirement]) -> str:
    bits = [f"Rank {rank} by '{method}' (score {score:.3f}; a ranking score, not a probability)."]
    if method == "hybrid" and sem is not None:
        if meta.has_requirements:
            bits.append(f"= {alpha:.2f} x semantic {sem:.3f} + {1 - alpha:.2f} x metadata {meta.score[i]:.3f}.")
        else:
            bits.append(f"No structured requirements were found in the query, so score = semantic similarity {sem:.3f}.")
    if matched:
        bits.append("Matched: " + "; ".join(r.label() for r in matched) + ".")
    if unsatisfied:
        bits.append("NOT satisfied: " + "; ".join(r.label() for r in unsatisfied) + ".")
    if unknown:
        bits.append("Unknown (no metadata evidence): " + "; ".join(r.label() for r in unknown) + ".")
    if meta.hard_violation[i]:
        bits.append("WARNING: violates a mandatory requirement.")
    return " ".join(bits)
