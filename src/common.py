"""Shared helpers: configuration loading, path handling, text normalisation, lexicons.

Every other script imports from here so that paths, seeds and lexicons are defined once.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config/config.yaml (or a custom path)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | Path) -> Path:
    """Resolve a project-relative path and create parent directories."""
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_dir(rel: str | Path) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    with open(resolve(path), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- text

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def normalize_text(value: Any, max_chars: int | None = None) -> str:
    """Lower-case-preserving cleanup: unescape HTML, strip tags/URLs, collapse whitespace.

    Returns '' for None/NaN so callers never have to special-case missing text.
    Case is preserved here (embedding models are case-aware); use ``tokenize`` for
    lexical methods.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- lexicons
# Transparent keyword lexicons used for (a) inferring catalogue fields and
# (b) parsing queries. Every inferred field is flagged as 'inferred_keyword'
# in the catalogue provenance columns. These are deliberately small and auditable.

DOMAIN_LEXICON: dict[str, list[str]] = {
    "medical": [
        "cancer", "tumor", "tumour", "carcinoma", "malignant", "benign", "patient",
        "patients", "clinical", "medical", "disease", "diagnosis", "diabetes", "diabetic",
        "heart", "cardiac", "ecg", "eeg", "hepatitis", "liver", "kidney", "thyroid",
        "parkinson", "alzheimer", "stroke", "oncology", "biopsy", "mammography",
        "arrhythmia", "blood pressure", "cholesterol", "hospital", "symptom", "survival",
    ],
    "biology": [
        "gene", "genes", "genomic", "genome", "protein", "dna", "rna", "microarray",
        "expression", "species", "bioinformatics", "molecular", "sequence",
    ],
    "finance": [
        "credit", "loan", "bank", "banking", "fraud", "stock", "financial", "insurance",
        "churn", "customer", "marketing", "price",
    ],
    "text_nlp": [
        "text", "document", "spam", "sentiment", "review", "news", "email", "language",
        "word", "corpus", "tweet",
    ],
    "vision": ["image", "images", "pixel", "pixels", "digit", "digits", "photo", "visual"],
    "environment": ["weather", "climate", "air quality", "pollution", "temperature", "forest"],
    "engineering": ["sensor", "machine", "vibration", "network", "intrusion", "iot", "telemetry"],
    "social": ["census", "income", "education", "student", "survey", "population"],
}

MODALITY_LEXICON: dict[str, list[str]] = {
    "ct": ["ct", "ct scan", "ct scans", "computed tomography"],
    "mri": ["mri", "magnetic resonance"],
    "xray": ["x-ray", "xray", "radiograph", "radiography"],
    "ecg": ["ecg", "ekg", "electrocardiogram"],
    "eeg": ["eeg", "electroencephalogram"],
    "image": ["image", "images", "pixel", "pixels", "photo", "photograph"],
    "text": ["text", "document", "documents", "corpus", "sentence", "tweet", "email"],
    "audio": ["audio", "speech", "sound", "voice", "acoustic"],
    "time_series": ["time series", "time-series", "temporal", "sequence of measurements"],
    "genomic": ["gene expression", "genomic", "genome", "microarray", "dna", "rna"],
    "tabular": ["tabular", "table", "spreadsheet", "attributes", "features"],
}

TASK_LEXICON: dict[str, list[str]] = {
    "classification": ["classification", "classify", "classifier", "classes", "class label",
                       "diagnosis", "detection", "detect", "categorize", "categorise"],
    "regression": ["regression", "predict price", "continuous target", "forecast value"],
    "clustering": ["clustering", "cluster", "unsupervised"],
    "survival": ["survival", "time-to-event", "time to event", "prognosis"],
    "time_series": ["time series", "time-series", "forecasting"],
}

# conditions / topics shared between query parsing and benchmark generation
CONDITION_TERMS: list[str] = [
    "lung cancer", "breast cancer", "prostate cancer", "skin cancer", "colon cancer",
    "liver cancer", "cervical cancer", "leukemia", "melanoma", "cancer",
    "diabetes", "heart disease", "arrhythmia", "hepatitis", "thyroid", "parkinson",
    "alzheimer", "kidney disease", "stroke", "epilepsy",
]

QUERY_FILLER = {
    "i", "need", "needs", "find", "want", "looking", "look", "for", "a", "an", "the", "of",
    "dataset", "datasets", "data", "suitable", "research", "project", "that", "with", "and",
    "or", "to", "in", "on", "at", "least", "least,", "preferably", "please", "using", "use",
    "me", "my", "some", "any", "have", "has", "should", "must", "be", "is", "are", "which",
    "containing", "contains", "include", "including", "information", "about", "from", "can",
    "available", "good", "recommend", "show", "get", "also", "more", "than", "over", "least",
}


def contains_term(text_lower: str, term: str) -> bool:
    """Whole-word / whole-phrase containment test (avoids 'ct' matching 'object')."""
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def lexicon_hits(text_lower: str, lexicon: dict[str, list[str]]) -> list[str]:
    return [label for label, terms in lexicon.items() if any(contains_term(text_lower, t) for t in terms)]
