"""
AI Verification pipeline — Stage 6 (human/clinical decision) + full orchestration.

Runs the end-to-end experiment on the simulated dataset and writes results/plots
that the research paper reports on:
  1. generate cases (with drift injected in the tail)
  2. verify + detect hallucinations/errors (ensemble)
  3. quality score each case
  4. evaluate detector precision/recall/F1
  5. anomaly / outlier detection on operational features
  6. drift detection: reference window vs current window
  7. clinical decision policy + sealed ledger, then integrity check (incl. a
     tamper demo)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import generate_dataset, EngineConfig
from verify import detect, DetectorConfig
from quality import (
    quality_score, evaluate_detector, flag_operational_outliers,
    isolation_forest_outliers,
)
from drift import detect_drift
from ledger import SealedLedger, content_hash

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

N_CASES = 4000
DRIFT_START = 3000


# ---------------------------------------------------------------------------
# Stage 6: clinical decision policy
# ---------------------------------------------------------------------------

def clinical_decision(qscore: float, n_flagged: int) -> str:
    """
    Map a quality score + flag count onto an action for the clinician.
    Tunable thresholds; conservative because this is clinical data.
    """
    if n_flagged == 0 and qscore >= 0.85:
        return "auto_accept"           # high confidence, no flags
    if qscore >= 0.6:
        return "clinician_review"      # usable, needs a human glance
    return "escalate_reject"           # low quality, block + senior review


def main():
    rng = random.Random(20260811)
    base_cfg = EngineConfig()
    # detector must resample under the *base* engine behaviour it expects
    det_cfg = DetectorConfig(n_samples=5, stability_threshold=0.5)

    cases = generate_dataset(N_CASES, seed=7, drift_start=DRIFT_START)

    rows = []
    all_judgements = []
    ledger = SealedLedger()

    for case in cases:
        judgements = detect(case, det_cfg, base_cfg, rng)
        all_judgements.append(judgements)
        q = quality_score(case, judgements)
        n_flagged = sum(j.is_flagged for j in judgements)
        decision = clinical_decision(q, n_flagged)

        ledger.append(
            case_id=case.case_id,
            source_hash=content_hash([(f.field, f.value) for f in case.source_facts]),
            ai_record_hash=content_hash([(f.field, f.value) for f in case.ai_facts]),
            quality_score=q,
            n_flagged=n_flagged,
            decision=decision,
        )

        r = case.to_row()
        r["quality_score"] = round(q, 4)
        r["n_flagged"] = n_flagged
        r["decision"] = decision
        r["drifted_regime"] = case_index_drifted(case.case_id)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cases.csv", index=False)

    # ---- 4. detector evaluation (overall + per regime) ----
    overall = evaluate_detector(all_judgements).as_dict()
    pre = evaluate_detector(all_judgements[:DRIFT_START]).as_dict()
    post = evaluate_detector(all_judgements[DRIFT_START:]).as_dict()

    # ---- 5. anomaly / outlier detection ----
    df = flag_operational_outliers(df, z_thresh=3.5)
    df = isolation_forest_outliers(
        df, features=["processing_ms", "cost_usd", "n_fabricated", "quality_score"],
        contamination=0.03,
    )
    df.to_csv(OUT / "cases_scored.csv", index=False)

    # ---- 6. drift detection: reference (first 1500) vs current (last 1000) ----
    ref_df = df.iloc[:1500]
    cur_df = df.iloc[-1000:]
    drift_feats = ["quality_score", "processing_ms", "cost_usd", "n_fabricated"]
    drift = detect_drift(ref_df, cur_df, drift_feats)
    drift_records = [d.__dict__ for d in drift]

    # ---- 7. ledger integrity + tamper demo ----
    ok, bad, msg = ledger.verify_integrity()
    # tamper: mutate one historical event's quality score, re-verify
    tampered_records = ledger.to_records()
    ledger._events[1000].quality_score = 0.99  # silent edit
    ok_after, bad_after, msg_after = ledger.verify_integrity()

    summary = {
        "n_cases": N_CASES,
        "drift_start_index": DRIFT_START,
        "detector_overall": overall,
        "detector_pre_drift": pre,
        "detector_post_drift": post,
        "decision_mix": df["decision"].value_counts().to_dict(),
        "decision_mix_pre": df.iloc[:DRIFT_START]["decision"].value_counts().to_dict(),
        "decision_mix_post": df.iloc[DRIFT_START:]["decision"].value_counts().to_dict(),
        "n_robust_outliers": int(df["robust_outlier"].sum()),
        "n_iso_outliers": int(df["iso_outlier"].sum()),
        "mean_quality_pre": round(float(df.iloc[:DRIFT_START]["quality_score"].mean()), 4),
        "mean_quality_post": round(float(df.iloc[DRIFT_START:]["quality_score"].mean()), 4),
        "mean_latency_pre": round(float(df.iloc[:DRIFT_START]["processing_ms"].mean()), 2),
        "mean_latency_post": round(float(df.iloc[DRIFT_START:]["processing_ms"].mean()), 2),
        "drift": drift_records,
        "ledger_len": len(ledger),
        "ledger_integrity_before_tamper": {"ok": ok, "msg": msg},
        "ledger_integrity_after_tamper": {"ok": ok_after, "first_bad_seq": bad_after, "msg": msg_after},
    }
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    (OUT / "summary.json").write_text(json.dumps(_clean(summary), indent=2))

    make_plots(df)
    print(json.dumps(_clean(summary), indent=2))
    return summary


def case_index_drifted(case_id: str) -> bool:
    return int(case_id[1:]) >= DRIFT_START


def make_plots(df: pd.DataFrame):
    # Quality score over time
    fig, ax = plt.subplots(figsize=(8, 3.2))
    roll = df["quality_score"].rolling(100, min_periods=20).mean()
    ax.plot(df.index, roll, color="#1f6f6f", lw=1.6)
    ax.axvline(DRIFT_START, color="#c0392b", ls="--", lw=1.2, label="drift injected")
    ax.set_title("Rolling mean quality score (window=100)")
    ax.set_xlabel("case index"); ax.set_ylabel("quality score")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "quality_over_time.png", dpi=130); plt.close(fig)

    # Latency distribution pre/post
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(df.iloc[:DRIFT_START]["processing_ms"], bins=40, alpha=0.6,
            label="pre-drift", color="#1f6f6f")
    ax.hist(df.iloc[DRIFT_START:]["processing_ms"], bins=40, alpha=0.6,
            label="post-drift", color="#c0392b")
    ax.set_title("Processing latency distribution")
    ax.set_xlabel("ms"); ax.set_ylabel("count"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "latency_hist.png", dpi=130); plt.close(fig)

    # Decision mix pre/post
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    mix = pd.DataFrame({
        "pre": df.iloc[:DRIFT_START]["decision"].value_counts(normalize=True),
        "post": df.iloc[DRIFT_START:]["decision"].value_counts(normalize=True),
    }).fillna(0)
    mix.plot(kind="bar", ax=ax, color=["#1f6f6f", "#c0392b"])
    ax.set_title("Clinical decision mix (proportion)")
    ax.set_ylabel("proportion"); ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(OUT / "decision_mix.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
