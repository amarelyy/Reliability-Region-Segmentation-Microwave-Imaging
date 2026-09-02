"""
metrics_result/check_low_confidence_dominance.py

Computes ALL narrative placeholder values (X, Y, Z, N) from metrics_raw.csv
and prints them ready to copy into the paper.

Input:  rrs_mbi/results/metrics_raw.csv
Output: Terminal only
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"

THRESH_LOW = 0.30
THRESH_HIGH = 0.70


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run 'python -m rrs_mbi.main' first to generate metrics_raw.csv."
        )

    df = pd.read_csv(CSV_PATH)

    if "reliability_score" not in df.columns:
        raise ValueError(
            f"Column 'reliability_score' not found in {CSV_PATH}. "
            f"Available columns: {df.columns.tolist()}"
        )

    scores: np.ndarray = np.asarray(
        df["reliability_score"].dropna().to_numpy(dtype=np.float64),
        dtype=np.float64,
    )

    # ================================================================
    # CORE COUNTS
    # ================================================================
    n_total = len(scores)
    n_high = int(np.sum(scores >= THRESH_HIGH))
    n_med = int(np.sum((scores >= THRESH_LOW) & (scores < THRESH_HIGH)))
    n_low = int(np.sum(scores < THRESH_LOW))

    pct_high = n_high / n_total * 100
    pct_med = n_med / n_total * 100
    pct_low = n_low / n_total * 100

    # ================================================================
    # STATISTICS
    # ================================================================
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))

    # ================================================================
    # LOCALIZATION METRICS
    # ================================================================
    loc_error = df["localization_error_mm"].dropna().to_numpy(dtype=np.float64) if "localization_error_mm" in df.columns else np.array([])
    n_loc_valid = len(loc_error)
    mean_loc_error = float(np.mean(loc_error)) if n_loc_valid > 0 else float("nan")
    std_loc_error = float(np.std(loc_error)) if n_loc_valid > 0 else float("nan")

    det_thresh = 15.0
    n_detected = int(np.sum(loc_error <= det_thresh)) if n_loc_valid > 0 else 0
    detection_rate = n_detected / n_loc_valid * 100 if n_loc_valid > 0 else float("nan")

    # ================================================================
    # SIZE METRICS
    # ================================================================
    size_error = df["size_error_mm"].dropna().to_numpy(dtype=np.float64) if "size_error_mm" in df.columns else np.array([])
    n_size_valid = len(size_error)
    mean_size_error = float(np.mean(size_error)) if n_size_valid > 0 else float("nan")

    size_det_thresh = 5.0
    n_size_detected = int(np.sum(size_error <= size_det_thresh)) if n_size_valid > 0 else 0
    size_detection_rate = n_size_detected / n_size_valid * 100 if n_size_valid > 0 else float("nan")

    # ================================================================
    # SIGNAL QUALITY
    # ================================================================
    def safe_mean(col_name: str) -> float:
        if col_name in df.columns:
            vals = df[col_name].dropna().to_numpy(dtype=np.float64)
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")
        return float("nan")

    mean_scr = safe_mean("scr_db")
    mean_cnr = safe_mean("cnr")
    mean_cf = safe_mean("cf_at_peak")

    # ================================================================
    # PRINT EVERYTHING
    # ================================================================
    print("=" * 70)
    print("ALL NARRATIVE PLACEHOLDER VALUES")
    print("=" * 70)

    print(f"\n--- SCAN COUNTS ---")
    print(f"  [N]  Total evaluated scans:       {n_total}")
    print(f"  [X]  High confidence (>={THRESH_HIGH}):   {n_high} ({pct_high:.1f}%)")
    print(f"  [Y]  Medium confidence ({THRESH_LOW}-{THRESH_HIGH}): {n_med} ({pct_med:.1f}%)")
    print(f"  [Z]  Low confidence (<{THRESH_LOW}):    {n_low} ({pct_low:.1f}%)")

    print(f"\n--- RELIABILITY SCORE STATISTICS ---")
    print(f"  Mean:   {mean_score:.4f}")
    print(f"  Median: {median_score:.4f}")
    print(f"  Std:    {std_score:.4f}")

    print(f"\n--- LOCALIZATION PERFORMANCE ---")
    print(f"  Valid localization errors:  {n_loc_valid}/{n_total}")
    print(f"  Mean localization error:    {mean_loc_error:.2f} mm")
    print(f"  Std localization error:     {std_loc_error:.2f} mm")
    print(f"  Detection rate (<={det_thresh}mm):   {detection_rate:.1f}% ({n_detected}/{n_loc_valid})")

    print(f"\n--- SIZE ESTIMATION ---")
    print(f"  Valid size estimates:       {n_size_valid}/{n_total}")
    print(f"  Mean size error:            {mean_size_error:.2f} mm")
    print(f"  Size detection rate (<={size_det_thresh}mm): {size_detection_rate:.1f}% ({n_size_detected}/{n_size_valid})")

    print(f"\n--- SIGNAL QUALITY ---")
    print(f"  Mean SCR (dB):              {mean_scr:.2f}")
    print(f"  Mean CNR:                   {mean_cnr:.2f}")
    print(f"  Mean CF at Peak:            {mean_cf:.4f}")

    print(f"\n{'=' * 70}")
    print("READY-TO-COPY NARRATIVE VALUES")
    print(f"{'=' * 70}")
    print(f"  [N] = {n_total}")
    print(f"  [X] = {pct_high:.1f}")
    print(f"  [Y] = {pct_med:.1f}")
    print(f"  [Z] = {pct_low:.1f}")
    print(f"  Mean Loc Error = {mean_loc_error:.2f} mm")
    print(f"  Std Loc Error  = {std_loc_error:.2f} mm")
    print(f"  Detection Rate = {detection_rate:.1f}%")
    print(f"  Mean Size Error = {mean_size_error:.2f} mm")
    print(f"  Size Det Rate  = {size_detection_rate:.1f}%")
    print(f"  Mean SCR       = {mean_scr:.2f} dB")
    print(f"  Mean CNR       = {mean_cnr:.2f}")
    print(f"  Mean CF        = {mean_cf:.4f}")
    print(f"{'=' * 70}")

    print(f"\nNARRATIVE INSERTS:")
    print(f"  Section 3.1:")
    print(f"    \"Approximately {pct_high:.1f}% of scans achieved high confidence\"")
    print(f"    \"(>= 0.70), {pct_med:.1f}% fell within medium confidence\"")
    print(f"    \"(0.30-0.70), and {pct_low:.1f}% were classified as low\"")
    print(f"    \"confidence (< 0.30).\"")
    print(f"")
    print(f"  Section 3.1 (clinical justification):")
    print(f"    \"...without it, clinicians would receive uniformly presented\"")
    print(f"    \"images with no indication that over {pct_low:.1f}% are unreliable\"")
    print(f"    \"for localization decisions.\"")
    print(f"")
    print(f"  Figure 1 caption:")
    print(f"    \"Distribution of RRS-MBI reliability scores across {n_total}\"")
    print(f"    \"evaluated tumor-bearing scans.\"")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()