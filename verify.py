"""
AI Verification pipeline — Stages 3 & 4: verification + hallucination/error detection.

Verification: each atomic fact in the AI record is checked against the source
record (the "reference"). This is the reference-based / claim-verification
paradigm (RefChecker, HalluSearch, textual-entailment checkers) reduced to its
core: split the output into atomic claims, then adjudicate each claim against a
grounded reference as SUPPORTED / UNSUPPORTED / CONTRADICTED.

Because a real engine gives us only free text (not gold labels), we ALSO
implement a second, self-contained detector that needs no reference:
self-consistency via multiple samples (SelfCheckGPT-style). We simulate resampling
and measure claim stability; unstable claims are flagged. In production you would
run both and combine (ensemble voting), which is what `detect()` does.

References for the approach:
- RefChecker / claim-triplet checking (SemEval-2025 Task 3).
- SelfCheckGPT self-consistency & entropy (keepitsimple, SemEval-2025).
- Ensemble adjudication / probability voting (MSA, SemEval-2025).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from simulate import Case, Fact, EngineConfig, run_engine

Verdict = Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


@dataclass
class ClaimJudgement:
    fact: Fact
    verdict: Verdict
    stability: float          # self-consistency score in [0,1]; low = suspicious
    is_flagged: bool          # detector's final call (predicted hallucination/error)
    true_is_error: bool       # ground truth: was this actually fabricated/corrupted


# ---------------------------------------------------------------------------
# Reference-based verification (Stage 3)
# ---------------------------------------------------------------------------

def _index_source(source: list[Fact]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for f in source:
        idx.setdefault(f.field, set()).add(f.value)
    return idx


def verify_claim(fact: Fact, source_idx: dict[str, set[str]],
                 ref_noise: float = 0.0, rng: random.Random | None = None) -> Verdict:
    """
    Adjudicate one atomic claim against the grounded source reference.

    `ref_noise` models an imperfect reference (retrieval miss, OCR error,
    incomplete source extraction): with probability ref_noise the verifier
    returns the wrong verdict, which is what creates false positives/negatives
    in a realistic system. Set to 0 for an oracle reference.
    """
    vals = source_idx.get(fact.field)
    if not vals:
        verdict: Verdict = "UNSUPPORTED"
    elif fact.value in vals:
        verdict = "SUPPORTED"
    else:
        verdict = "CONTRADICTED"

    if ref_noise and rng is not None and rng.random() < ref_noise:
        # flip supported<->unsupported to simulate reference errors
        verdict = "UNSUPPORTED" if verdict == "SUPPORTED" else "SUPPORTED"
    return verdict


# ---------------------------------------------------------------------------
# Reference-free self-consistency detection (Stage 4)
# ---------------------------------------------------------------------------

def self_consistency_stability(
    case: Case, fact: Fact, n_samples: int, rng: random.Random,
    cfg: EngineConfig,
) -> float:
    """
    SelfCheckGPT-style: re-run the engine on the same source several times and
    measure how often the same (field, value) claim reappears. Genuine
    (supported) facts are stable across resamples; fabricated/corrupted ones are
    stochastic and diverge, yielding low stability.
    """
    hits = 0
    for _ in range(n_samples):
        resampled, _, _ = run_engine(case.source_facts, cfg, rng)
        if any(rf.field == fact.field and rf.value == fact.value for rf in resampled):
            hits += 1
    return hits / n_samples


# ---------------------------------------------------------------------------
# Ensemble detector: combine reference verdict + self-consistency
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    n_samples: int = 5
    stability_threshold: float = 0.5   # below this, flag as unstable
    ref_noise: float = 0.03            # imperfect reference rate
    # Weighting of the two evidence sources when they disagree:
    trust_reference: bool = True


def detect(
    case: Case,
    detector_cfg: DetectorConfig,
    engine_cfg: EngineConfig,
    rng: random.Random,
) -> list[ClaimJudgement]:
    source_idx = _index_source(case.source_facts)
    out: list[ClaimJudgement] = []
    for fact in case.ai_facts:
        verdict = verify_claim(fact, source_idx, detector_cfg.ref_noise, rng)
        stability = self_consistency_stability(
            case, fact, detector_cfg.n_samples, rng, engine_cfg
        )
        ref_flag = verdict in ("UNSUPPORTED", "CONTRADICTED")
        cons_flag = stability < detector_cfg.stability_threshold
        # Ensemble vote: reference evidence is authoritative when available;
        # self-consistency is a fallback that also catches subtle drift.
        if detector_cfg.trust_reference:
            flagged = ref_flag or (cons_flag and verdict != "SUPPORTED")
        else:
            flagged = ref_flag or cons_flag
        out.append(ClaimJudgement(
            fact=fact,
            verdict=verdict,
            stability=round(stability, 3),
            is_flagged=flagged,
            true_is_error=(fact.label in ("fabricated", "corrupted")),
        ))
    return out
