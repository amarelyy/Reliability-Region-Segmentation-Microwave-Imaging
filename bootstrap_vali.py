"""
bootstrap_gap_ci.py

Standalone bootstrap CI on the localization-error gap between the full
dataset and the top-N most-reliable scans. Reads the already-generated
rrs_mbi/results/metrics_raw.csv directly -- does NOT re-run the
reconstruction/evaluation pipeline, so this runs in seconds.

Usage:
    python bootstrap_gap_ci.py
    python bootstrap_gap_ci.py --n-top 50 --n-boot 10000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"


def bootstrap_gap_ci(df: pd.DataFrame, n_top: int = 50, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)

    errors_all = df["localization_error"].dropna().to_numpy()
    top_n = (
        df.sort_values("reliability_score", ascending=False)
        .head(n_top)["localization_error"]
        .dropna()
        .to_numpy()
    )

    gaps = np.empty(n_boot)
    for b in range(n_boot):
        boot_top = rng.choice(top_n, size=len(top_n), replace=True)
        boot_all = rng.choice(errors_all, size=len(errors_all), replace=True)
        gaps[b] = boot_all.mean() - boot_top.mean()

    point = errors_all.mean() - top_n.mean()
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    excludes_zero = ci_lo > 0

    print("=" * 60)
    print(f"BOOTSTRAP CI: baseline mean LE - top-{n_top} mean LE")
    print("=" * 60)
    print(f"  n_all={len(errors_all)}  n_top={len(top_n)}  n_boot={n_boot}")
    print(f"  Baseline mean LE:   {errors_all.mean():.2f} mm")
    print(f"  Top-{n_top} mean LE:     {top_n.mean():.2f} mm")
    print(f"  Point gap:          {point:+.2f} mm")
    print(f"  95% CI:             [{ci_lo:+.2f}, {ci_hi:+.2f}] mm")
    print(f"  CI excludes zero:   {excludes_zero}")
    print("=" * 60)

    return {
        "n_top": n_top,
        "n_boot": n_boot,
        "point_gap_mm": float(point),
        "ci_lower_mm": float(ci_lo),
        "ci_upper_mm": float(ci_hi),
        "excludes_zero": bool(excludes_zero),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-top", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "This needs rrs_mbi/results/metrics_raw.csv -- run "
            "'python -m rrs_mbi.main' once beforehand if it's missing."
        )

    df = pd.read_csv(CSV_PATH)
    result = bootstrap_gap_ci(df, n_top=args.n_top, n_boot=args.n_boot, seed=args.seed)

    out_path = ROOT / "metrics_result" / "bootstrap_gap_ci.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()