"""
AI Verification pipeline — Stage 5: quality score + evaluation + anomaly/outlier detection.

Three things live here:

1. QUALITY SCORE per case. A calibrated scalar in [0,1] combining the fraction
   of AI claims that are grounded/supported, penalised by flagged errors and by
   omissions relative to the source. This is the number surfaced to clinicians.

2. DETECTOR EVALUATION. Precision, recall, F1 of the hallucination/error
   detector against ground-truth labels (per-claim), plus quality-score
   calibration (does a low score actually mean a bad record?).

3. ANOMALY / OUTLIER DETection at the case level over operational + quality
   features (latency, cost, fabrication count, quality score) using robust
   z-scores and IsolationForest — the "flag unusual cases for human review" step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from simulate import Case
from verify import ClaimJudgement


# ---------------------------------------------------------------------------
# 1. Quality score
# ---------------------------------------------------------------------------

@dataclass
class QualityWeights:
    w_grounded: float = 1.0
    w_flag_penalty: float = 0.6
    w_omission_penalty: float = 0.4


def quality_score(case: Case, judgements: list[ClaimJudgement],
                  w: QualityWeights = QualityWeights()) -> float:
    n_ai = max(1, len(judgements))
    grounded = sum(j.verdict == "SUPPORTED" for j in judgements) / n_ai
    flagged = sum(j.is_flagged for j in judgements) / n_ai
    # omission: source facts that never made it into the AI record
    src_pairs = {(f.field, f.value) for f in case.source_facts}
    ai_pairs = {(j.fact.field, j.fact.value) for j in judgements}
    omitted = len(src_pairs - ai_pairs) / max(1, len(src_pairs))
    raw = (w.w_grounded * grounded
           - w.w_flag_penalty * flagged
           - w.w_omission_penalty * omitted)
    return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Detector evaluation (precision / recall / F1 + calibration)
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def evaluate_detector(all_judgements: list[list[ClaimJudgement]]) -> DetectionMetrics:
    tp = fp = fn = tn = 0
    for js in all_judgements:
        for j in js:
            if j.is_flagged and j.true_is_error:
                tp += 1
            elif j.is_flagged and not j.true_is_error:
                fp += 1
            elif not j.is_flagged and j.true_is_error:
                fn += 1
            else:
                tn += 1
    return DetectionMetrics(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# 3. Anomaly / outlier detection at the case level
# ---------------------------------------------------------------------------

def robust_z(series: pd.Series) -> pd.Series:
    """Median/MAD-based z-score; robust to the heavy tails common in latency/cost."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad


def flag_operational_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    df = df.copy()
    for col in ["processing_ms", "cost_usd", "n_fabricated"]:
        df[f"rz_{col}"] = robust_z(df[col])
    df["robust_outlier"] = (
        df[[c for c in df.columns if c.startswith("rz_")]].abs() > z_thresh
    ).any(axis=1)
    return df


def isolation_forest_outliers(df: pd.DataFrame, features: list[str],
                              contamination: float = 0.03,
                              seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    X = df[features].to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=seed)
    df["iso_pred"] = iso.fit_predict(X)          # -1 = outlier
    df["iso_outlier"] = df["iso_pred"] == -1
    df["iso_score"] = iso.score_samples(X)       # lower = more anomalous
    return df
