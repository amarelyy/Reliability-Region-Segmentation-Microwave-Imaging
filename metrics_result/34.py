"""
metrics_result/table3_and_fig4.py

Generates Table Z (signal quality by confidence tier) and
Figure 4 (scatter plots: reliability vs all metrics).

Input:  rrs_mbi/results/metrics_raw.csv
Output: metrics_result/table3_signal_quality.csv
        metrics_result/fig4_correlation_scatter.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"
OUT_DIR = ROOT / "metrics_result"

THRESH_LOW = 0.30
THRESH_HIGH = 0.70


def assign_tier(score: float) -> str:
    if score >= THRESH_HIGH:
        return "High"
    elif score >= THRESH_LOW:
        return "Medium"
    return "Low"


def compute_table_z(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["reliability_score"].apply(assign_tier)

    metric_cols = ["scr_db", "cnr", "cf_at_peak", "localization_error"]
    available = [c for c in metric_cols if c in df.columns]

    rows = []
    for tier in ["High", "Medium", "Low", "All"]:
        subset = df if tier == "All" else df[df["tier"] == tier]
        row = {"tier": tier, "n_scans": len(subset)}
        for col in available:
            vals = subset[col].dropna().to_numpy(dtype=np.float64)
            row[f"mean_{col}"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            row[f"std_{col}"] = float(np.std(vals)) if len(vals) > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    targets = ["localization_error", "scr_db", "cnr", "cf_at_peak"]
    available = [t for t in targets if t in df.columns]

    rows = []
    for metric in available:
        valid = df[["reliability_score", metric]].dropna()
        if len(valid) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(valid["reliability_score"], valid[metric])
        spearman_rho, spearman_p = stats.spearmanr(valid["reliability_score"], valid[metric])
        rows.append({
            "metric": metric,
            "n": len(valid),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": pearson_p,
            "spearman_rho": round(spearman_rho, 4),
            "spearman_p": spearman_p,
        })

    return pd.DataFrame(rows)


def plot_figure4(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path) -> None:
    targets = ["localization_error", "scr_db", "cnr", "cf_at_peak"]
    available = [t for t in targets if t in df.columns]

    labels = {
        "localization_error": "Localization Error (mm)",
        "scr_db": "SCR (dB)",
        "cnr": "CNR",
        "cf_at_peak": "CF at Peak",
    }

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    colors = {"High": "#2ca02c", "Medium": "#ff7f0e", "Low": "#d62728"}
    df_plot = df.copy()
    df_plot["tier"] = df_plot["reliability_score"].apply(assign_tier)

    for idx, metric in enumerate(available):
        ax = axes[idx]
        valid = df_plot[["reliability_score", metric, "tier"]].dropna()

        for tier_name, color in colors.items():
            tier_data = valid[valid["tier"] == tier_name]
            ax.scatter(
                tier_data["reliability_score"],
                tier_data[metric],
                alpha=0.4, s=12, edgecolors="none",
                color=color, label=tier_name if idx == 0 else None,
            )

        corr_row = corr_df[corr_df["metric"] == metric]
        if len(corr_row) > 0:
            r_val = corr_row.iloc[0]["pearson_r"]
            rho_val = corr_row.iloc[0]["spearman_rho"]
            ax.set_title(
                f"{labels.get(metric, metric)}\nr={r_val:.3f}, ρ={rho_val:.3f}",
                fontsize=11, fontweight="bold",
            )
        else:
            ax.set_title(labels.get(metric, metric), fontsize=11, fontweight="bold")

        ax.set_xlabel("Reliability Score", fontsize=10)
        if idx == 0:
            ax.set_ylabel(labels.get(metric, metric), fontsize=10)
        ax.grid(True, alpha=0.3)

    if n > 0:
        axes[0].legend(loc="upper left", fontsize=9, framealpha=0.85)

    plt.suptitle(
        "Reliability Score vs Performance and Signal Quality Metrics",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run 'python -m rrs_mbi.main' first."
        )

    df = pd.read_csv(CSV_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table Z
    tz = compute_table_z(df)
    tz_path = OUT_DIR / "table3_signal_quality.csv"
    tz.to_csv(tz_path, index=False)

    print("=" * 75)
    print("TABLE Z: SIGNAL QUALITY BY CONFIDENCE TIER")
    print("=" * 75)

    display_cols = ["tier", "n_scans"]
    for col in tz.columns:
        if col.startswith("mean_"):
            display_cols.append(col)
    available_display = [c for c in display_cols if c in tz.columns]

    print(tz[available_display].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved to: {tz_path}")

    # Correlations
    corr = compute_correlations(df)
    corr_path = OUT_DIR / "correlation_coefficients.csv"
    corr.to_csv(corr_path, index=False)

    print("\n" + "=" * 75)
    print("CORRELATIONS: Reliability Score vs Metrics")
    print("=" * 75)
    print(f"{'Metric':<25} {'Pearson r':>10} {'p-value':>12} {'Spearman rho':>13} {'p-value':>12}")
    print("-" * 75)
    for _, row in corr.iterrows():
        sig = "***" if row["pearson_p"] < 0.001 else ("**" if row["pearson_p"] < 0.01 else ("*" if row["pearson_p"] < 0.05 else "ns"))
        print(
            f"{row['metric']:<25} {row['pearson_r']:>10.4f} {row['pearson_p']:>12.2e} "
            f"{row['spearman_rho']:>13.4f} {row['spearman_p']:>12.2e} {sig}"
        )
    print(f"\nSaved to: {corr_path}")

    # Figure 4
    fig_path = OUT_DIR / "fig4_correlation_scatter.png"
    plot_figure4(df, corr, fig_path)
    print(f"Figure saved to: {fig_path}")

    # Summary for narrative
    print("\n" + "=" * 75)
    print("NARRATIVE SUMMARY")
    print("=" * 75)

    high_row = tz[tz["tier"] == "High"].iloc[0] if len(tz[tz["tier"] == "High"]) > 0 else None
    low_row = tz[tz["tier"] == "Low"].iloc[0] if len(tz[tz["tier"] == "Low"]) > 0 else None

    if high_row is not None and low_row is not None:
        for col in tz.columns:
            if col.startswith("mean_") and "localization_error" not in col:
                h_val = high_row[col]
                l_val = low_row[col]
                if not np.isnan(h_val) and not np.isnan(l_val) and l_val != 0:
                    ratio = h_val / l_val
                    print(f"  {col}: High={h_val:.2f}, Low={l_val:.2f}, Ratio={ratio:.1f}x")

    loc_corr = corr[corr["metric"] == "localization_error"]
    scr_corr = corr[corr["metric"] == "scr_db"]
    if len(loc_corr) > 0 and len(scr_corr) > 0:
        print(f"\n  Correlation with loc error:  r = {loc_corr.iloc[0]['pearson_r']:.4f}")
        print(f"  Correlation with SCR:        r = {scr_corr.iloc[0]['pearson_r']:.4f}")
        print(f"  --> Signal quality correlates {abs(scr_corr.iloc[0]['pearson_r']) / max(abs(loc_corr.iloc[0]['pearson_r']), 0.001):.1f}x stronger with reliability than localization error does")

    print("=" * 75)


if __name__ == "__main__":
    main()