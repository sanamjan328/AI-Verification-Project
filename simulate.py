"""
AI Verification pipeline — Stages 1 & 2: source clinical data + AI-generated/processed record.

We simulate a clinical verification workload. Each unit of work is a "case":
a source clinical record (structured facts) is passed to a (simulated) AI engine
that produces a processed record (e.g. a discharge summary / extracted problem
list). The AI is imperfect: it sometimes fabricates facts (hallucination),
drops facts (omission), or corrupts values (error). Ground-truth labels are
recorded so downstream verification / detection / scoring stages can be
evaluated against known truth.

Design choices grounded in current practice:
- Fabricated ("unsupported") vs corrupted vs omitted mirror the error taxonomy
  used in clinical-summary hallucination benchmarks (FaithBench-style).
- Each atomic fact is verifiable independently — the "factual splitting" idea
  from RAG hallucination pipelines (HalluSearch, SemEval-2025 Task 3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Controlled clinical vocabulary (small, synthetic — no real PHI)
# ---------------------------------------------------------------------------

CONDITIONS = [
    "type 2 diabetes", "hypertension", "atrial fibrillation", "COPD",
    "chronic kidney disease", "hypothyroidism", "asthma", "osteoarthritis",
    "GERD", "depression",
]
MEDICATIONS = [
    "metformin", "lisinopril", "apixaban", "salbutamol", "levothyroxine",
    "atorvastatin", "amlodipine", "omeprazole", "sertraline", "furosemide",
]
ALLERGIES = ["penicillin", "sulfa", "aspirin", "latex", "none recorded"]
LABEL = Literal["supported", "fabricated", "corrupted"]

FactValue = str


@dataclass
class Fact:
    """An atomic, independently verifiable clinical claim."""
    field: str          # e.g. "condition", "medication", "lab:HbA1c"
    value: FactValue
    label: LABEL = "supported"   # ground truth vs the source record


@dataclass
class Case:
    case_id: str
    source_facts: list[Fact]        # ground-truth source clinical data (Stage 1)
    ai_facts: list[Fact]            # AI-processed record (Stage 2)
    # Metadata used later for drift / cost / latency monitoring:
    site: str
    model_version: str
    n_source_facts: int = 0
    processing_ms: float = 0.0
    cost_usd: float = 0.0

    def __post_init__(self):
        self.n_source_facts = len(self.source_facts)

    def to_row(self) -> dict:
        return {
            "case_id": self.case_id,
            "site": self.site,
            "model_version": self.model_version,
            "n_source_facts": self.n_source_facts,
            "n_ai_facts": len(self.ai_facts),
            "n_fabricated": sum(f.label == "fabricated" for f in self.ai_facts),
            "n_corrupted": sum(f.label == "corrupted" for f in self.ai_facts),
            "processing_ms": round(self.processing_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


def _make_source_record(rng: random.Random) -> list[Fact]:
    facts: list[Fact] = []
    for c in rng.sample(CONDITIONS, k=rng.randint(2, 5)):
        facts.append(Fact("condition", c))
    for m in rng.sample(MEDICATIONS, k=rng.randint(2, 5)):
        facts.append(Fact("medication", m))
    facts.append(Fact("allergy", rng.choice(ALLERGIES)))
    facts.append(Fact("lab:HbA1c", f"{rng.uniform(5.0, 12.0):.1f}%"))
    facts.append(Fact("lab:eGFR", f"{rng.randint(15, 120)}"))
    return facts


@dataclass
class EngineConfig:
    """Behaviour of the simulated AI engine. Drift is injected by shifting these."""
    fabricate_rate: float = 0.06     # P(add an unsupported fact per source fact)
    corrupt_rate: float = 0.05       # P(corrupt a value)
    omit_rate: float = 0.08          # P(drop a source fact)
    base_latency_ms: float = 220.0
    latency_jitter_ms: float = 60.0
    cost_per_fact_usd: float = 8e-5


def _corrupt_value(fact: Fact, rng: random.Random) -> FactValue:
    if fact.field.startswith("lab:HbA1c"):
        return f"{rng.uniform(5.0, 12.0):.1f}%"
    if fact.field.startswith("lab:eGFR"):
        return f"{rng.randint(15, 120)}"
    if fact.field == "condition":
        return rng.choice(CONDITIONS)
    if fact.field == "medication":
        return rng.choice(MEDICATIONS)
    if fact.field == "allergy":
        return rng.choice(ALLERGIES)
    return fact.value + "?"


def run_engine(source: list[Fact], cfg: EngineConfig, rng: random.Random) -> tuple[list[Fact], float, float]:
    """Simulate the AI engine transforming a source record into a processed record."""
    out: list[Fact] = []
    for f in source:
        if rng.random() < cfg.omit_rate:
            continue  # omission
        if rng.random() < cfg.corrupt_rate:
            out.append(Fact(f.field, _corrupt_value(f, rng), label="corrupted"))
        else:
            out.append(Fact(f.field, f.value, label="supported"))
    # fabrications (unsupported additions)
    for _ in source:
        if rng.random() < cfg.fabricate_rate:
            field = rng.choice(["condition", "medication", "allergy"])
            value = {"condition": CONDITIONS, "medication": MEDICATIONS,
                     "allergy": ALLERGIES}[field]
            out.append(Fact(field, rng.choice(value), label="fabricated"))
    latency = max(20.0, rng.gauss(cfg.base_latency_ms, cfg.latency_jitter_ms))
    cost = len(source) * cfg.cost_per_fact_usd
    return out, latency, cost


def generate_dataset(
    n_cases: int = 4000,
    seed: int = 7,
    drift_start: int | None = 3000,
    drift_cfg: EngineConfig | None = None,
) -> list[Case]:
    """
    Generate a stream of cases. Optionally inject model drift after `drift_start`
    by switching the engine config (higher fabrication/latency) and bumping the
    model_version tag — this gives the drift-detection stage something to catch.
    """
    rng = random.Random(seed)
    base = EngineConfig()
    drift = drift_cfg or EngineConfig(
        fabricate_rate=0.16, corrupt_rate=0.11, omit_rate=0.10,
        base_latency_ms=300.0, latency_jitter_ms=90.0,
    )
    sites = ["site_A_toronto", "site_B_montreal", "site_C_vancouver"]
    cases: list[Case] = []
    for i in range(n_cases):
        drifted = drift_start is not None and i >= drift_start
        cfg = drift if drifted else base
        version = "engine-v1.3" if not drifted else "engine-v1.4"
        source = _make_source_record(rng)
        ai_facts, latency, cost = run_engine(source, cfg, rng)
        cases.append(Case(
            case_id=f"C{i:05d}",
            source_facts=source,
            ai_facts=ai_facts,
            site=rng.choice(sites),
            model_version=version,
            processing_ms=latency,
            cost_usd=cost,
        ))
    return cases


if __name__ == "__main__":
    ds = generate_dataset(20, drift_start=None)
    for c in ds[:3]:
        print(c.to_row())
