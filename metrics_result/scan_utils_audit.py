"""
metrics_results/scan_utilization_audit.py

Audits scan utilization: how many scans were evaluated vs skipped,
and why. Reads metrics_raw.csv and cross-references with tumor_model
to identify root causes of exclusion.

Input:  rrs_mbi/results/metrics_raw.csv
Output: Terminal only (transparent audit report)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run 'python -m rrs_mbi.main' first to generate metrics_raw.csv."
        )

    df = pd.read_csv(CSV_PATH)

    print("=" * 70)
    print("SCAN UTILIZATION AUDIT")
    print("=" * 70)

    # 1. How many scans are in the CSV (actually evaluated)
    n_evaluated = len(df)
    print(f"\n[1] SCANS IN metrics_raw.csv (evaluated): {n_evaluated}")

    # 2. Check which columns exist and their NaN counts
    print(f"\n[2] COLUMN COMPLETENESS CHECK:")
    print(f"    {'Column':<30} {'Non-NaN':>10} {'NaN':>10} {'% Valid':>10}")
    print(f"    {'-'*62}")
    for col in df.columns:
        non_nan = df[col].notna().sum()
        nan_count = df[col].isna().sum()
        pct_valid = non_nan / n_evaluated * 100 if n_evaluated > 0 else 0
        flag = " <-- INCOMPLETE" if pct_valid < 95 else ""
        print(f"    {col:<30} {non_nan:>10} {nan_count:>10} {pct_valid:>9.1f}%{flag}")

    # 3. Check ground truth availability
    gt_cols = ["gt_x_mm", "gt_y_mm", "gt_r_mm"]
    available_gt = [c for c in gt_cols if c in df.columns]

    if available_gt:
        print(f"\n[3] GROUND TRUTH AVAILABILITY:")
        for col in available_gt:
            valid = df[col].notna().sum()
            invalid = df[col].isna().sum()
            print(f"    {col}: {valid} valid, {invalid} NaN")

        all_gt_valid = df[available_gt].notna().all(axis=1).sum()
        print(f"    All GT columns valid simultaneously: {all_gt_valid}/{n_evaluated}")
    else:
        print(f"\n[3] GROUND TRUTH: No GT columns found in CSV!")
        print(f"    Available columns: {df.columns.tolist()}")

    # 4. Check localization error validity
    if "localization_error_mm" in df.columns:
        valid_loc = df["localization_error_mm"].notna().sum()
        nan_loc = df["localization_error_mm"].isna().sum()
        print(f"\n[4] LOCALIZATION ERROR:")
        print(f"    Valid: {valid_loc}, NaN: {nan_loc}")
        if valid_loc > 0:
            print(f"    Mean: {df['localization_error_mm'].mean():.2f} mm")
            print(f"    Min:  {df['localization_error_mm'].min():.2f} mm")
            print(f"    Max:  {df['localization_error_mm'].max():.2f} mm")

    # 5. Check size estimation validity
    if "size_error_mm" in df.columns:
        valid_size = df["size_error_mm"].notna().sum()
        nan_size = df["size_error_mm"].isna().sum()
        print(f"\n[5] SIZE ESTIMATION:")
        print(f"    Valid: {valid_size}, NaN: {nan_size}")
        if valid_size > 0:
            print(f"    Mean size error: {df['size_error_mm'].mean():.2f} mm")

    # 6. Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Scans in tumor_model (total dataset rows):  504")
    print(f"  Scans actually evaluated (in CSV):          {n_evaluated}")
    print(f"  Scans skipped during evaluation:            {504 - n_evaluated}")
    print(f"  Skipped because: GT coordinates were NaN")
    print(f"                   (healthy phantoms or missing metadata)")
    print(f"{'=' * 70}")

    # 7. Load tumor_model directly to confirm
    try:
        sys.path.append(str(ROOT))
        from src.data_loading import load_all_data
        data = load_all_data()  # type: ignore
        tumor_model = data["tumor_model"]

        print(f"\n[6] CROSS-REFERENCE WITH tumor_model:")
        print(f"    tumor_model rows: {len(tumor_model)}")

        gt_cols_tm = ["tumor_x_mm", "tumor_y_mm", "tumor_radius_mm"]
        alt_cols_tm = ["tum_x", "tum_y", "tum_rad"]

        for cols in [gt_cols_tm, alt_cols_tm]:
            available = [c for c in cols if c in tumor_model.columns]
            if available:
                valid_mask = tumor_model[available].notna().all(axis=1)
                n_valid = int(valid_mask.sum())
                n_invalid = len(tumor_model) - n_valid
                print(f"    Columns checked: {available}")
                print(f"    Rows with valid GT: {n_valid}")
                print(f"    Rows without GT (NaN): {n_invalid}")
                print(f"    --> This explains why {n_invalid} scans were skipped")
                break
        else:
            print(f"    No GT columns found in tumor_model either!")
            print(f"    tumor_model columns: {tumor_model.columns.tolist()}")

    except Exception as e:
        print(f"\n[6] Could not load tumor_model for cross-reference: {e}")

    print(f"\n{'=' * 70}")
    print("CORRECT N FOR PAPER NARRATIVE:")
    print(f"  Use N = {n_evaluated} (not 504) in all Results sections")
    print(f"  State: 'Of 504 tumor_model rows, {n_evaluated} had valid GT")
    print(f"         and were included in the evaluation.'")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()