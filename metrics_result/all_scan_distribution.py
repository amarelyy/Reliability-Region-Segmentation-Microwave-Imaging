"""
metrics_results/all_scan_distribution.py (IMPROVED VERSION)

Generates Figure 1 with two panels:
- Left: Linear scale showing dominance of low-confidence scans
- Right: Log scale revealing structure in the tail distribution

Input:  rrs_mbi/results/metrics_raw.csv
Output: metrics_image/fig1_reliability_distribution.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"
OUTPUT_PATH = ROOT / "metrics_image" / "fig1_reliability_distribution.png"

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
    n_total = len(scores)

    if n_total == 0:
        raise ValueError("No valid reliability scores found in CSV.")

    n_high = int(np.sum(scores >= THRESH_HIGH))
    n_med = int(np.sum((scores >= THRESH_LOW) & (scores < THRESH_HIGH)))
    n_low = int(np.sum(scores < THRESH_LOW))

    pct_high = n_high / n_total * 100
    pct_med = n_med / n_total * 100
    pct_low = n_low / n_total * 100

    print(f"Total scans: {n_total}")
    print(f"High   (>={THRESH_HIGH}): {n_high:>4} ({pct_high:.1f}%)")
    print(f"Medium ({THRESH_LOW}-<{THRESH_HIGH}): {n_med:>4} ({pct_med:.1f}%)")
    print(f"Low    (<{THRESH_LOW}): {n_low:>4} ({pct_low:.1f}%)")
    print(
        f"Mean: {float(np.mean(scores)):.4f} | "
        f"Median: {float(np.median(scores)):.4f} | "
        f"Std: {float(np.std(scores)):.4f}"
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Common histogram parameters
    bins = 50
    hist_kwargs = dict(bins=bins, edgecolor="white", alpha=0.85)

    # LEFT PANEL: Linear scale
    ax1.hist(scores, color="#4C72B0", **hist_kwargs)
    for patch in ax1.patches:
        left_edge = patch.get_x()
        if left_edge >= THRESH_HIGH:
            patch.set_facecolor("#2ca02c")
        elif left_edge >= THRESH_LOW:
            patch.set_facecolor("#ff7f0e")
        else:
            patch.set_facecolor("#d62728")

    ax1.axvline(THRESH_LOW, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Medium/Low ({THRESH_LOW})")
    ax1.axvline(THRESH_HIGH, color="#2ca02c", linestyle="--", linewidth=2, label=f"High/Medium ({THRESH_HIGH})")

    y_max_1 = ax1.get_ylim()[1]
    ax1.annotate(f"Low\n{pct_low:.1f}%", xy=(THRESH_LOW / 2, y_max_1 * 0.85), ha="center", fontsize=11, fontweight="bold", color="#d62728")
    ax1.annotate(f"Med\n{pct_med:.1f}%", xy=((THRESH_LOW + THRESH_HIGH) / 2, y_max_1 * 0.85), ha="center", fontsize=11, fontweight="bold", color="#ff7f0e")
    ax1.annotate(f"High\n{pct_high:.1f}%", xy=(min(THRESH_HIGH + 0.15, 1.0), y_max_1 * 0.85), ha="center", fontsize=11, fontweight="bold", color="#2ca02c")

    ax1.set_xlabel("Reliability Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Scans", fontsize=12, fontweight="bold")
    ax1.set_title("Linear Scale: Dominance of Low-Confidence Scans", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # RIGHT PANEL: Log scale to reveal tail structure
    ax2.hist(scores, color="#4C72B0", **hist_kwargs)
    for patch in ax2.patches:
        left_edge = patch.get_x()
        if left_edge >= THRESH_HIGH:
            patch.set_facecolor("#2ca02c")
        elif left_edge >= THRESH_LOW:
            patch.set_facecolor("#ff7f0e")
        else:
            patch.set_facecolor("#d62728")

    ax2.axvline(THRESH_LOW, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Medium/Low ({THRESH_LOW})")
    ax2.axvline(THRESH_HIGH, color="#2ca02c", linestyle="--", linewidth=2, label=f"High/Medium ({THRESH_HIGH})")

    ax2.set_yscale("log")
    ax2.set_xlabel("Reliability Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Scans (log scale)", fontsize=12, fontweight="bold")
    ax2.set_title("Log Scale: Exponential Decay Structure", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, which="both")

    plt.suptitle(
        "Distribution of RRS-MBI Reliability Scores (N = {})".format(n_total),
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()