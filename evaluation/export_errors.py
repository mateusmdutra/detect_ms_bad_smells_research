"""
Export LLM errors (FP and FN) to CSV for qualitative analysis.

Matches exactly the methodology of data-analysis.ipynb:
  - Results from: ../results/**/<system>/<smell>/run_*.json
  - Ground truth:  ../ground-truth/final-ground-truth.csv
  - Binary label:  mean confidence across all runs >= 3 → smell present
  - Explanation:   taken from the latest run for each model/system/smell

Output: evaluation/llm_errors.csv
Columns: model, diagram, sys_id, smell, error_type, confidence, gt_score, explanation
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"
GT_CSV      = Path(__file__).parent.parent / "ground-truth" / "final-ground-truth.csv"
OUTPUT_CSV  = Path(__file__).parent / "llm_errors.csv"
THRESHOLD   = 3

SMELLS = {
    "cyclic_dependency",
    "microservice_greedy",
    "nano_service",
    "no_api_gateway",
    "shared_database",
}

GT_COL_MAP = {
    "CD":  "cyclic_dependency",
    "MG":  "microservice_greedy",
    "NS":  "nano_service",
    "NAG": "no_api_gateway",
    "SD":  "shared_database",
}

MODEL_DISPLAY_NAMES = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8-e81efef6": "llama-4-maverick",
}

# ── Ground truth ──────────────────────────────────────────────────────────────

def load_ground_truth() -> pd.DataFrame:
    gt_raw = pd.read_csv(GT_CSV)
    gt_raw = gt_raw[pd.to_numeric(gt_raw["DIAGRAMS"], errors="coerce").notna()].copy()
    gt_raw["sys_id"] = gt_raw["DIAGRAMS"].astype(int).astype(str)

    rows = []
    for col, smell in GT_COL_MAP.items():
        sub = gt_raw[["sys_id", col]].copy()
        sub.columns = ["sys_id", "score"]
        sub["smell"] = smell
        rows.append(sub)

    gt_long = pd.concat(rows, ignore_index=True)
    gt_long["score"] = pd.to_numeric(gt_long["score"], errors="coerce")

    gt_agg = (
        gt_long.groupby(["sys_id", "smell"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "gt_score"})
    )
    gt_agg["gt_label"] = (gt_agg["gt_score"] >= THRESHOLD).astype(int)
    return gt_agg


# ── Predictions ───────────────────────────────────────────────────────────────

def _resolve_model_name(raw: str) -> str:
    for fragment, display in MODEL_DISPLAY_NAMES.items():
        if fragment in raw:
            return display
    return raw


def load_predictions() -> pd.DataFrame:
    rows = []

    for run_file in sorted(RESULTS_DIR.rglob("run_*.json")):
        smell_dir = run_file.parent
        if smell_dir.name not in SMELLS:
            continue
        sys_dir = smell_dir.parent
        m = re.match(r"^(\d+)", sys_dir.name)
        if not m:
            continue

        try:
            with open(run_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        confidence = data.get("confidence")
        if confidence is None:
            continue

        raw_model = sys_dir.parent.name
        model = _resolve_model_name(raw_model)

        rows.append(
            {
                "model":       model,
                "sys_id":      m.group(1),
                "diagram":     sys_dir.name,
                "smell":       smell_dir.name,
                "confidence":  float(confidence),
                "explanation": data.get("explanation", ""),
                "run_file":    str(run_file),
            }
        )

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gt = load_ground_truth()
    pred_df = load_predictions()

    if pred_df.empty:
        print("No prediction files found.", file=sys.stderr)
        sys.exit(1)

    # Average confidence across runs → binary label (same as notebook)
    pred_agg = (
        pred_df.groupby(["model", "sys_id", "smell", "diagram"])["confidence"]
        .mean()
        .reset_index()
        .rename(columns={"confidence": "confidence_mean"})
    )
    pred_agg["pred_label"] = (pred_agg["confidence_mean"] >= THRESHOLD).astype(int)

    # Latest run explanation for qualitative reading
    latest_explanation = (
        pred_df.sort_values("run_file")
        .drop_duplicates(["model", "sys_id", "smell"], keep="last")
        [["model", "sys_id", "smell", "explanation"]]
    )
    pred_agg = pred_agg.merge(latest_explanation, on=["model", "sys_id", "smell"], how="left")

    merged = pred_agg.merge(gt, on=["sys_id", "smell"], how="inner")

    fp = merged[(merged["pred_label"] == 1) & (merged["gt_label"] == 0)].copy()
    fp["error_type"] = "FP"

    fn = merged[(merged["pred_label"] == 0) & (merged["gt_label"] == 1)].copy()
    fn["error_type"] = "FN"

    errors = (
        pd.concat([fp, fn], ignore_index=True)
        [["model", "diagram", "sys_id", "smell", "error_type", "confidence_mean", "gt_score", "explanation"]]
        .rename(columns={"confidence_mean": "confidence"})
        .sort_values(["model", "error_type", "smell", "sys_id"])
        .reset_index(drop=True)
    )

    errors.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(errors)} errors → {OUTPUT_CSV}")
    print(f"\nBy error type:\n{errors['error_type'].value_counts().to_string()}")
    print(f"\nBy model:\n{errors.groupby('model')['error_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
