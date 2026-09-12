# Verification Pipeline (simulation + reference implementation)

End-to-end simulation of a clinical AI verification service, matching the Engineer scope:

    source clinical data -> AI-generated/processed record -> verification
      -> hallucination/error detection -> quality score -> human/clinical decision

## Stages / modules
| Stage | Module | What it does |
|---|---|---|
| 1-2 Source + AI record | `simulate.py` | Synthetic clinical records; imperfect AI engine (fabricate/corrupt/omit); drift injection |
| 3 Verification | `verify.py` | Atomic-claim checking vs grounded source reference (SUPPORTED/UNSUPPORTED/CONTRADICTED) |
| 4 Hallucination/error detection | `verify.py` | Self-consistency (SelfCheckGPT-style) + ensemble with reference |
| 5 Quality score + eval + anomaly | `quality.py` | Calibrated quality score, precision/recall/F1, robust-z + IsolationForest outliers |
| — Drift | `drift.py` | PSI + KS two-sample tests on quality/latency/cost/fabrication |
| — Sealed ledger | `ledger.py` | SHA-256 hash-chained append-only audit log + integrity verify |
| 6 Clinical decision + orchestration | `run_experiment.py` | Decision policy, runs the full experiment, writes outputs/ |

## Run
```
pip install -r requirements.txt
python -m AI_pipeline.run_experiment
```
Outputs (CSV, plots, summary.json) are written to `outputs/`.
