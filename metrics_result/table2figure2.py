"""
metrics_result/table2figure2.py

Generates Table 2 (selective classification performance CSV) and
Figure 2 (selective classification curve with CI and threshold).

Input:  rrs_mbi/results/metrics_raw.csv
Output: metrics_result/table2_selective_classification.csv
        metrics_result/fig2_selective_classification.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"
OUT_DIR = ROOT / "metrics_result"

LOC_DET_THRESH_MM = 15.0
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
KEEP_RATIOS = np.linspace(1.0, 0.1, 10)


def bootstrap_ci(errors: np.ndarray) -> tuple[float, float]:
    if len(errors) == 0:
        return 0.0, 0.0
    e = np.asarray(errors, dtype=np.float64).ravel()
    means = [
        float(np.mean(np.random.choice(e, size=len(e), replace=True)))
        for _ in range(N_BOOTSTRAP)
    ]
    alpha = 1.0 - CONFIDENCE
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_table2(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by="reliability_score", ascending=False
    ).reset_index(drop=True)
    total = len(df_sorted)
    rows = []

    for ratio in KEEP_RATIOS:
        n_keep = max(1, int(total * ratio))
        subset = df_sorted.iloc[:n_keep]

        loc_errors = subset["localization_error"].dropna().to_numpy(dtype=np.float64)

        if len(loc_errors) == 0:
            continue

        ci_lo, ci_hi = bootstrap_ci(loc_errors)

        loc_detected = int(np.sum(loc_errors <= LOC_DET_THRESH_MM))
        loc_det_rate = loc_detected / len(loc_errors) * 100

        rows.append({
            "keep_ratio": round(float(ratio), 2),
            "scans_kept": n_keep,
            "actual_threshold": float(subset["reliability_score"].iloc[-1]),
            "mean_loc_error_mm": float(np.mean(loc_errors)),
            "std_loc_error_mm": float(np.std(loc_errors)),
            "ci_lower_mm": ci_lo,
            "ci_upper_mm": ci_hi,
            "loc_detection_rate_pct": round(loc_det_rate, 1),
            "loc_detected_count": loc_detected,
        })

    return pd.DataFrame(rows)


def plot_figure2(t2: pd.DataFrame, save_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.fill_between(
        t2["scans_kept"],
        t2["ci_lower_mm"],
        t2["ci_upper_mm"],
        color="#1f77b4",
        alpha=0.2,
        label="95% CI",
    )
    ax1.plot(
        t2["scans_kept"],
        t2["mean_loc_error_mm"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Mean Localization Error",
    )
    ax1.set_xlabel("Scans Kept (Sorted by Reliability)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Localization Error (mm)", color="#1f77b4", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        t2["scans_kept"],
        t2["actual_threshold"],
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Min Reliability Threshold",
    )
    ax2.set_ylabel(
        "Min Reliability Threshold",
        color="#d62728",
        fontsize=12,
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor="#d62728")

    baseline_det = t2[t2["keep_ratio"] == 1.0]["loc_detection_rate_pct"].values[0]
    filtered_det = t2[t2["keep_ratio"] == 0.1]["loc_detection_rate_pct"].values[0]
    ax1.annotate(
        f"Detection Rate:\n{baseline_det:.1f}% -> {filtered_det:.1f}%\n(+{filtered_det - baseline_det:.1f} pp)",
        xy=(t2["scans_kept"].iloc[-1], t2["mean_loc_error_mm"].iloc[-1]),
        xytext=(t2["scans_kept"].iloc[-1] + 40, t2["mean_loc_error_mm"].iloc[-1] - 5),
        fontsize=10,
        fontweight="bold",
        color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c"),
    )

    baseline_std = t2[t2["keep_ratio"] == 1.0]["std_loc_error_mm"].values[0]
    filtered_std = t2[t2["keep_ratio"] == 0.1]["std_loc_error_mm"].values[0]
    var_reduction = (1 - filtered_std / baseline_std) * 100
    ax1.annotate(
        f"Std Reduction:\n{baseline_std:.2f} -> {filtered_std:.2f} mm\n(-{var_reduction:.1f}%)",
        xy=(t2["scans_kept"].iloc[0], t2["mean_loc_error_mm"].iloc[0]),
        xytext=(t2["scans_kept"].iloc[0] - 120, t2["mean_loc_error_mm"].iloc[0] + 4),
        fontsize=10,
        fontweight="bold",
        color="#ff7f0e",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="#ff7f0e"),
    )

    plt.title(
        "Selective Classification Performance",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        framealpha=0.9,
        fontsize=10,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run 'python -m rrs_mbi.main' first to generate metrics_raw.csv."
        )

    df = pd.read_csv(CSV_PATH)

    required_cols = ["reliability_score", "localization_error"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    print(f"Loaded {len(df)} scans from {CSV_PATH}")

    t2 = compute_table2(df)

    csv_out = OUT_DIR / "table2_selective_classification.csv"
    t2.to_csv(csv_out, index=False)

    print("\n" + "=" * 80)
    print("TABLE 2: SELECTIVE CLASSIFICATION PERFORMANCE")
    print("=" * 80)
    print(
        f"{'Keep%':<7} | {'N':<5} | {'Thresh':<8} | {'MeanErr':<9} | "
        f"{'Std':<7} | {'95% CI':<16} | {'DetRate'}"
    )
    print("-" * 80)
    for _, r in t2.iterrows():
        ci_str = f"[{r['ci_lower_mm']:.1f}-{r['ci_upper_mm']:.1f}]"
        print(
            f"{r['keep_ratio']*100:<6.0f}% | {r['scans_kept']:<5} | "
            f"{r['actual_threshold']:<8.4f} | {r['mean_loc_error_mm']:<9.2f} | "
            f"{r['std_loc_error_mm']:<7.2f} | {ci_str:<16} | "
            f"{r['loc_detection_rate_pct']:.1f}%"
        )
    print("=" * 80)

    baseline = t2[t2["keep_ratio"] == 1.0].iloc[0]
    filtered = t2[t2["keep_ratio"] == 0.1].iloc[0]
    det_abs = filtered["loc_detection_rate_pct"] - baseline["loc_detection_rate_pct"]
    det_rel = det_abs / baseline["loc_detection_rate_pct"] * 100
    var_red = (1 - filtered["std_loc_error_mm"] / baseline["std_loc_error_mm"]) * 100

    print(f"\nKEY FINDINGS:")
    print(f"  Detection rate:  {baseline['loc_detection_rate_pct']:.1f}% -> {filtered['loc_detection_rate_pct']:.1f}% (+{det_abs:.1f} pp abs, +{det_rel:.1f}% rel)")
    print(f"  Std reduction:   {baseline['std_loc_error_mm']:.2f} -> {filtered['std_loc_error_mm']:.2f} mm (-{var_red:.1f}%)")
    print(f"  Mean error:      {baseline['mean_loc_error_mm']:.2f} -> {filtered['mean_loc_error_mm']:.2f} mm (stable)")

    fig_out = OUT_DIR / "fig2_selective_classification.png"
    plot_figure2(t2, fig_out)

    print(f"\nTable saved to: {csv_out}")
    print(f"Figure saved to: {fig_out}")


if __name__ == "__main__":
    main()