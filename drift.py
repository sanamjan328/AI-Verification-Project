"""
AI Verification pipeline — model/data drift detection.

Compares a reference window (early, trusted production data) against a current
window using two standard distribution-shift tests:

- Population Stability Index (PSI): binned distribution shift. Rules of thumb:
  PSI < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant.
- Kolmogorov-Smirnov (KS) test: nonparametric two-sample test on continuous
  features; small p-value => distributions differ.

These are the workhorse tests used by Evidently and most production monitoring
stacks. We run them on quality score, latency, cost and fabrication count so that
a real degradation in the engine surfaces as drift on the quality signal — the
"alert on input drift AND an eval drop" pattern recommended for LLM systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    # Quantile bins from the reference so each ref bin has ~equal mass.
    quantiles = np.quantile(ref, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    ref_pct = np.clip(ref_counts / max(1, ref_counts.sum()), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(1, cur_counts.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_label(psi: float) -> str:
    if psi < 0.1:
        return "no_shift"
    if psi < 0.2:
        return "moderate_shift"
    return "significant_shift"


@dataclass
class DriftResult:
    feature: str
    psi: float
    psi_label: str
    ks_stat: float
    ks_pvalue: float
    drifted: bool


def detect_drift(ref_df, cur_df, features: list[str],
                 psi_thresh: float = 0.2, ks_alpha: float = 0.05) -> list[DriftResult]:
    results: list[DriftResult] = []
    for feat in features:
        r = ref_df[feat].to_numpy()
        c = cur_df[feat].to_numpy()
        psi = population_stability_index(r, c)
        ks_stat, ks_p = stats.ks_2samp(r, c)
        drifted = (psi >= psi_thresh) or (ks_p < ks_alpha)
        results.append(DriftResult(
            feature=feat, psi=round(psi, 4), psi_label=psi_label(psi),
            ks_stat=round(float(ks_stat), 4), ks_pvalue=round(float(ks_p), 6),
            drifted=drifted,
        ))
    return results
