"""
Load the MedHallu benchmark (UTAustin-AIHealth/MedHallu).

MedHallu is 10,000 PubMedQA-derived medical QA pairs. Each row has a ground-truth
answer and a paired hallucinated answer, plus difficulty (easy/medium/hard) and a
hallucination category. Two Hub configs:

- pqa_labeled     — 1,000 human-annotated samples
- pqa_artificial  — 9,000 pipeline-generated samples

Loading prefers a local parquet cache under data/medhallu/. If the files are
missing they are downloaded from Hugging Face. The Hugging Face `datasets`
library is optional; pandas + parquet is the default path so this works even
when `datasets` is not installed or is incompatible with the local pyarrow.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from simulate import Case, Fact

__all__ = [
    "load_medhallu",
    "to_binary_pairs",
    "to_cases",
    "summary_tables",
    "token_jaccard",
]

HF_DATASET = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_labeled", "pqa_artificial")
SPLIT = "train"

PARQUET_URL = (
    "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu/"
    "resolve/refs%2Fconvert%2Fparquet/{config}/train/0000.parquet"
)

COLUMN_MAP = {
    "Question": "question",
    "Knowledge": "knowledge",
    "Ground Truth": "ground_truth",
    "Difficulty Level": "difficulty",
    "Hallucinated Answer": "hallucinated_answer",
    "Category of Hallucination": "hallucination_category",
}

DIFFICULTY_ORDER = ("easy", "medium", "hard")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = REPO_ROOT / "data" / "medhallu"

_WORD = re.compile(r"[a-z0-9]+")
# Require uppercase after punctuation to avoid splitting on medical abbreviations
# (e.g. "Dr.", "approx.", "mg."). Newlines always split regardless.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")
CLAIM_FIELD = "claim"
ALIGN_THRESHOLD = 0.5


def _as_str_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(x) for x in list(value) if x is not None and str(x).strip()]
    except TypeError:
        return [str(value)]


def _download_parquet(config: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = PARQUET_URL.format(config=config)
    tmp = dest.with_suffix(".parquet.part")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            last_exc = exc
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]


def _load_config_parquet(config: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{config}.parquet"
    _download_parquet(config, path)
    return pd.read_parquet(path)


def _load_config_datasets(config: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for source='datasets'. "
            "Install it with: pip install datasets  — or use source='parquet' instead."
        ) from exc
    ds = load_dataset(HF_DATASET, config, split=SPLIT)
    return ds.to_pandas()


def _normalize(df: pd.DataFrame, config: str) -> pd.DataFrame:
    rename = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    out = df.rename(columns=rename)
    missing = [c for c in COLUMN_MAP.values() if c not in out.columns]
    if missing:
        raise ValueError(f"{config}: missing columns {missing}; got {list(out.columns)}")

    out["config"] = config
    out["question"] = out["question"].astype(str)
    out["ground_truth"] = out["ground_truth"].astype(str)
    out["hallucinated_answer"] = out["hallucinated_answer"].astype(str)
    out["difficulty"] = out["difficulty"].astype(str).str.lower().str.strip()
    out["hallucination_category"] = (
        out["hallucination_category"].astype(str).str.replace("#Question#", "Question", regex=False)
    )
    out["knowledge"] = out["knowledge"].map(_as_str_list)
    out["knowledge_text"] = out["knowledge"].map(lambda xs: "\n".join(xs))
    out["n_knowledge"] = out["knowledge"].map(len)
    out["question_chars"] = out["question"].str.len()
    out["ground_truth_chars"] = out["ground_truth"].str.len()
    out["hallucinated_chars"] = out["hallucinated_answer"].str.len()
    out["knowledge_chars"] = out["knowledge_text"].str.len()
    out["gt_hallu_jaccard"] = [
        token_jaccard(gt, ha)
        for gt, ha in zip(out["ground_truth"], out["hallucinated_answer"])
    ]
    return out.reset_index(drop=True)


def token_jaccard(a: str, b: str) -> float:
    """Word-set Jaccard between two strings (lowercased alphanumeric tokens)."""
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_medhallu(
    configs: Iterable[str] = CONFIGS,
    cache_dir: str | Path | None = None,
    source: str = "parquet",
) -> pd.DataFrame:
    """
    Load one or both MedHallu configs into a single DataFrame.

    Parameters
    ----------
    configs : iterable
        Hub configs to load. Default both `pqa_labeled` and `pqa_artificial`.
    cache_dir : path
        Local parquet cache. Default `data/medhallu/` next to this file.
    source : {'parquet', 'datasets'}
        `parquet` downloads Hub parquet shards (no `datasets` package needed).
        `datasets` uses `datasets.load_dataset` if that stack works locally.
    """
    if source not in ("parquet", "datasets"):
        raise ValueError(f"source must be 'parquet' or 'datasets', got {source!r}")
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    frames = []
    for config in configs:
        if config not in CONFIGS:
            raise ValueError(f"Unknown config {config!r}; expected one of {CONFIGS}")
        raw = _load_config_datasets(config) if source == "datasets" else _load_config_parquet(config, cache)
        frames.append(_normalize(raw, config))
    return pd.concat(frames, ignore_index=True)


def to_binary_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each MedHallu row into two labelled answers (accurate vs hallucinated).

    The Hub dump stores both answers on one row. Detection eval
    wants one answer per row with a binary gold label.
    """
    df = df.reset_index(drop=True)
    base = df[[
        "config", "question", "knowledge_text", "ground_truth",
        "difficulty", "hallucination_category",
    ]].copy()
    accurate = base.assign(
        answer=df["ground_truth"],
        gold_label="accurate",
        is_hallucinated=False,
    )
    hallucinated = base.assign(
        answer=df["hallucinated_answer"],
        gold_label="hallucinated",
        is_hallucinated=True,
    )
    return pd.concat([accurate, hallucinated], ignore_index=True)


def split_sentences(text: str) -> list[str]:
    """Split a passage into atomic claim-sized sentences. Empty input → []."""
    raw = str(text).strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(raw) if p.strip()]
    return parts or [raw]


def lexical_stability(case: Case, fact: Fact) -> float:
    """
    Max token-Jaccard of this claim against source facts.

    Used as the SelfCheckGPT stand-in on MedHallu: there is no generative
    engine to resample, so overlap with the gold-answer reference is the
    available stability signal.
    """
    if not case.source_facts:
        return 0.0
    return max(token_jaccard(fact.value, src.value) for src in case.source_facts)


def _align_to_source(text: str, source_values: list[str], threshold: float) -> str:
    """Rewrite a claim to the matched source sentence when Jaccard is high enough
    so `verify_claim` exact-match can return SUPPORTED for paraphrases."""
    if not source_values:
        return text
    best_v, best_j = text, -1.0
    for value in source_values:
        score = token_jaccard(text, value)
        if score > best_j:
            best_v, best_j = value, score
    return best_v if best_j >= threshold else text


def to_cases(
    df: pd.DataFrame,
    align_threshold: float = ALIGN_THRESHOLD,
    seed: int = 7,
) -> list[Case]:
    """
    Map MedHallu binary pairs onto `Case` objects.

    Source facts = sentences of the gold answer (the analogue of a source
    clinical record). AI facts = sentences of the candidate answer (gold copy
    or hallucinated). Hallucinated claims are labelled `fabricated`.

    Knowledge passages stay off this sealed path: they are context, not the
    gold reference, and lexical overlap with knowledge does not separate
    MedHallu's paired answers.
    """
    pairs = to_binary_pairs(df)
    rng = random.Random(seed)
    cases: list[Case] = []
    for i, row in enumerate(pairs.itertuples(index=False)):
        source_sents = split_sentences(row.ground_truth)
        if not source_sents:
            source_sents = [str(row.ground_truth).strip() or "empty"]
        source_facts = [Fact(CLAIM_FIELD, sent, label="supported") for sent in source_sents]
        source_values = [f.value for f in source_facts]

        answer_sents = split_sentences(row.answer)
        if not answer_sents:
            answer_sents = [str(row.answer).strip() or "empty"]
        label = "fabricated" if row.is_hallucinated else "supported"
        ai_facts = [
            Fact(CLAIM_FIELD, _align_to_source(sent, source_values, align_threshold), label=label)
            for sent in answer_sents
        ]

        latency = max(20.0, rng.gauss(80.0 + 0.15 * len(str(row.answer)), 25.0))
        cost = len(source_facts) * 8e-5
        case = Case(
            case_id=f"M{i:05d}",
            source_facts=source_facts,
            ai_facts=ai_facts,
            site=str(row.config),
            model_version=f"medhallu-{row.gold_label}",
            processing_ms=latency,
            cost_usd=cost,
            metadata={
                "difficulty": str(row.difficulty),
                "gold_label": str(row.gold_label),
                "is_hallucinated": bool(row.is_hallucinated),
                "hallucination_category": str(row.hallucination_category),
            },
        )
        cases.append(case)
    return cases


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Counts used by the EDA notebook and CLI."""
    difficulty = (
        df.groupby(["config", "difficulty"]).size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c in DIFFICULTY_ORDER if c in df["difficulty"].unique()])
    )
    category = df.groupby(["config", "hallucination_category"]).size().unstack(fill_value=0)
    cross = pd.crosstab(df["difficulty"], df["hallucination_category"])
    lengths = df[
        ["question_chars", "ground_truth_chars", "hallucinated_chars", "knowledge_chars", "n_knowledge", "gt_hallu_jaccard"]
    ].describe().T
    jaccard_by_diff = (
        df.groupby("difficulty")["gt_hallu_jaccard"]
        .agg(["count", "mean", "median", "std"])
        .reindex([d for d in DIFFICULTY_ORDER if d in set(df["difficulty"])])
    )
    return {
        "difficulty": difficulty,
        "category": category,
        "difficulty_x_category": cross,
        "lengths": lengths,
        "jaccard_by_difficulty": jaccard_by_diff,
    }


def main() -> None:
    df = load_medhallu()
    print(f"loaded {len(df):,} rows from {HF_DATASET}")
    print(df.groupby("config").size().to_string())
    print()
    tables = summary_tables(df)
    print("difficulty")
    print(tables["difficulty"].to_string())
    print()
    print("hallucination category")
    print(tables["category"].to_string())
    print()
    print("GT vs hallucinated token Jaccard by difficulty")
    print(tables["jaccard_by_difficulty"].round(3).to_string())
    cases = to_cases(df.head(20))
    print(f"to_cases smoke test: {len(cases)} cases, first id={cases[0].case_id}")
    print()
    out = REPO_ROOT / "outputs" / "medhallu_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_rows": int(len(df)),
        "by_config": df.groupby("config").size().to_dict(),
        "difficulty": df["difficulty"].value_counts().to_dict(),
        "hallucination_category": df["hallucination_category"].value_counts().to_dict(),
        "mean_gt_hallu_jaccard_by_difficulty": {
            k: round(float(v), 4)
            for k, v in df.groupby("difficulty")["gt_hallu_jaccard"].mean().items()
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
