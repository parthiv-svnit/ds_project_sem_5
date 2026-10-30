"""Rule-based extraction of structured requirements from a natural-language query.

The parser is intentionally transparent: every extracted requirement records the
phrase that triggered it, and requirements are split into *mandatory* (hard) and
*preferred* (soft) according to a fixed, documented rule:

  mandatory : numeric minimum size ("at least 500 patients"), label availability
              ("labeled"), class-count constraints ("binary").
  preferred : domain, condition, modality, task, data type, demographic features,
              "large dataset".

Phrases such as "preferably" downgrade a requirement to preferred.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from common import (CONDITION_TERMS, DOMAIN_LEXICON, MODALITY_LEXICON, TASK_LEXICON,
                    contains_term, lexicon_hits)

LARGE_THRESHOLD = 10_000  # instances; used for the soft 'large dataset' requirement

_NUM_WORDS = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}
_MIN_PATTERNS = [
    r"(?:at least|minimum of|minimum|no fewer than|not fewer than|more than|over|greater than|>=?)\s*([\d][\d,\.]*)\s*(k|thousand|m|million)?",
    r"([\d][\d,\.]*)\s*(k|thousand|m|million)?\s*\+\s*(?:samples|instances|patients|records|rows|examples|subjects|images)",
]
_SOFT_MARKERS = ("preferably", "ideally", "if possible", "optionally", "would be nice")


@dataclass
class Requirement:
    kind: str                 # e.g. 'min_instances', 'labels', 'modality', ...
    value: Any
    mandatory: bool
    evidence: str             # phrase in the query that triggered the rule

    def label(self) -> str:
        return f"{self.kind}={self.value}" + (" [mandatory]" if self.mandatory else "")


@dataclass
class ParsedQuery:
    text: str
    requirements: list[Requirement] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Requirement]:
        return [r for r in self.requirements if r.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "requirements": [asdict(r) for r in self.requirements]}


def _to_number(raw: str, suffix: str | None) -> int | None:
    raw = raw.replace(",", "")
    raw = raw.rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    if suffix:
        val *= _NUM_WORDS.get(suffix.lower(), 1)
    return int(val)


def _is_soft(text_lower: str, span_start: int) -> bool:
    window = text_lower[max(0, span_start - 40): span_start + 5]
    return any(m in window for m in _SOFT_MARKERS)


def parse_query(text: str) -> ParsedQuery:
    """Extract structured requirements from ``text``."""
    low = text.lower()
    reqs: list[Requirement] = []

    # --- minimum number of instances (mandatory unless softened)
    for pat in _MIN_PATTERNS:
        m = re.search(pat, low)
        if m:
            n = _to_number(m.group(1), m.group(2))
            if n is not None and n > 0:
                reqs.append(Requirement("min_instances", n, not _is_soft(low, m.start()), m.group(0).strip()))
                break

    # --- large dataset (soft)
    if re.search(r"\b(large|big|massive)\b", low) and not any(r.kind == "min_instances" for r in reqs):
        reqs.append(Requirement("large", LARGE_THRESHOLD, False, "large"))

    # --- labels required (mandatory unless softened)
    m = re.search(r"\b(labeled|labelled|annotated|supervised|with labels|ground truth)\b", low)
    if m:
        reqs.append(Requirement("labels", True, not _is_soft(low, m.start()), m.group(0)))

    # --- class count
    if re.search(r"\bbinary\b", low):
        reqs.append(Requirement("n_classes", 2, True, "binary"))
    elif re.search(r"\b(multi-?class|multi-?label)\b", low):
        reqs.append(Requirement("min_classes", 3, False, "multiclass"))

    # --- task
    for task, terms in TASK_LEXICON.items():
        hit = next((t for t in terms if contains_term(low, t)), None)
        if hit:
            reqs.append(Requirement("task", task, False, hit))
            break

    # --- condition (longest match first so 'lung cancer' beats 'cancer')
    for cond in sorted(CONDITION_TERMS, key=len, reverse=True):
        if contains_term(low, cond):
            reqs.append(Requirement("condition", cond, False, cond))
            break

    # --- modality
    for mod, terms in MODALITY_LEXICON.items():
        if mod == "tabular":
            continue
        hit = next((t for t in terms if contains_term(low, t)), None)
        if hit:
            reqs.append(Requirement("modality", mod, False, hit))
            break
    if contains_term(low, "tabular") and not any(r.kind == "modality" for r in reqs):
        reqs.append(Requirement("modality", "tabular", False, "tabular"))

    # --- domain (only medical/biology are used as explicit requirement; others too if stated)
    domain_hits = lexicon_hits(low, DOMAIN_LEXICON)
    explicit_medical = re.search(r"\b(medical|clinical|biomedical|healthcare|patients?)\b", low)
    if explicit_medical:
        reqs.append(Requirement("domain", "medical", False, explicit_medical.group(0)))
    elif any(r.kind == "condition" for r in reqs):
        reqs.append(Requirement("domain", "medical", False, "implied by condition"))
    elif domain_hits and len(domain_hits) == 1 and domain_hits[0] in {"finance", "text_nlp", "vision", "environment", "biology"}:
        reqs.append(Requirement("domain", domain_hits[0], False, "domain keyword"))

    # --- demographic features (soft; evidence can only be positive)
    m = re.search(r"\b(demographic|demographics|age and (?:sex|gender)|age,? (?:sex|gender))\b", low)
    if m:
        reqs.append(Requirement("demographic", True, False, m.group(0)))

    return ParsedQuery(text=text, requirements=reqs)
