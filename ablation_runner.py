"""
ablation_runner.py (ALL-IN CORRECT PHYSICS STACK)

Runs comparison matrix using the most geometrically precise model (Snellius + Effective Velocity)
while varying preprocessing components to isolate their impact.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Define frequency axis for bandpass filtering
FREQ_START_HZ = 1e9
FREQ_STOP_HZ = 8e9
N_FREQ_PTS = 1001
freq_axis = np.linspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ_PTS)

VARIANTS = [
    # --- BASELINE: Correct Physics but Minimal Preprocessing ---
    dict(name="Snellius_Raw",             beamformer="das", use_snellius=True, use_cf=False, use_bandpass=False, use_depth_gain=False),
    dict(name="Snellius_CF",              beamformer="das", use_snellius=True, use_cf=True,  use_bandpass=False, use_depth_gain=False),
    
    # --- COMPONENT TESTING: Adding Bandpass (4-6 GHz) ---
    dict(name="Snellius_CF_Bandpass",     beamformer="das", use_snellius=True, use_cf=True,  use_bandpass=True,  use_depth_gain=False),
    
    # --- FULL STACK: Adding Depth Gain Compensation ---
    dict(name="Full_Stack_DAS",           beamformer="das", use_snellius=True, use_cf=True,  use_bandpass=True,  use_depth_gain=True),
    dict(name="Full_Stack_DMAS",          beamformer="dmas",use_snellius=True, use_cf=True,  use_bandpass=True,  use_depth_gain=True),
]

def _run_one(idx, variant, s21, tumor_model):
    try:
        r = reconstruct_scan(
            idx, s21, tumor_model, freq_axis=freq_axis,
            beamformer=variant["beamformer"],
            use_snellius=variant["use_snellius"],
            use_cf=variant["use_cf"],
            use_bandpass=variant.get("use_bandpass", False),
            use_depth_gain=variant.get("use_depth_gain", False),
        )
        r.pop("diagnostics", None)
        return ("ok", idx, r)
    except Exception as e:
        return ("fail", idx, str(e))

def run_variant(variant, s21, tumor_model, n_scans, n_jobs=1):
    outcomes = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(idx, variant, s21, tumor_model) for idx in range(n_scans)
    )
    rows = [r for status, idx, r in outcomes if status == "ok"]
    failed = [(idx, r) for status, idx, r in outcomes if status == "fail"]
    df = pd.DataFrame(rows)
    return df, failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scans", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    n_scans = args.n_scans or d["n_valid_scans"]
    print(f"Running improved physics ablation on {n_scans} scans.\n")

    summary_rows = []
    for variant in VARIANTS:
        print(f"Running variant: {variant['name']}")
        df, failed = run_variant(variant, s21, tumor_model, n_scans, n_jobs=args.n_jobs)

        if failed:
            print(f"  {len(failed)} scans failed (e.g. {failed[0]})")

        out_path = RESULTS_DIR / f"ablation_{variant['name']}.csv"
        df.to_csv(out_path, index=False)

        if len(df) > 0:
            summary_rows.append(dict(
                variant=variant["name"],
                n_scans=len(df),
                mean_le_mm=df["localization_error_mm"].mean(),
                median_le_mm=df["localization_error_mm"].median(),
                detection_rate_20mm=(df["localization_error_mm"] <= 20).mean(),
                mean_scr_db=df["scr_db"].mean(),
            ))
        print(f"  -> mean LE: {df['localization_error_mm'].mean():.2f}mm "
              f"| detection@20mm: {(df['localization_error_mm'] <= 20).mean():.1%}\n"
              if len(df) > 0 else "  -> no successful scans\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "ablation_summary_correct_physics.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 70)
    print("ABLATION SUMMARY (CORRECT PHYSICS STACK)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()