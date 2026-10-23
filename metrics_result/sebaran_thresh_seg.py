"""Figure: Distribution of RRS-MBI reliability scores with confidence level thresholds."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})


def plot_reliability_distribution_thresholds(metrics_path="rrs_mbi/results/metrics_raw.csv",
                                             save_path="reliability_score_distribution.png"):
    df = pd.read_csv(metrics_path)
    rel = df["reliability_score"].to_numpy(dtype=float)
    valid = ~np.isnan(rel)
    rel = rel[valid]
    n = len(rel)

    low_thresh = 0.30
    high_thresh = 0.70

    n_low = (rel < low_thresh).sum()
    n_med = ((rel >= low_thresh) & (rel < high_thresh)).sum()
    n_high = (rel >= high_thresh).sum()

    pct_low = n_low / n * 100
    pct_med = n_med / n * 100
    pct_high = n_high / n * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 1, 41)
    counts, edges = np.histogram(rel, bins=bins)

    colors = []
    for left_edge in edges[:-1]:
        mid = left_edge + (edges[1] - edges[0]) / 2
        if mid < low_thresh:
            colors.append('#d62728')
        elif mid < high_thresh:
            colors.append('#ff7f0e')
        else:
            colors.append('#2ca02c')

    # --- Left: Linear scale ---
    ax = axes[0]
    ax.bar(edges[:-1], counts, width=np.diff(edges), color=colors,
           edgecolor='white', linewidth=0.5, align='edge')
    ax.axvline(x=low_thresh, color='#ff7f0e', linestyle='--', linewidth=2,
               label=f'Medium/Low ({low_thresh})')
    ax.axvline(x=high_thresh, color='#2ca02c', linestyle='--', linewidth=2,
               label=f'High/Medium ({high_thresh})')

    ax.text(0.12, 0.85, f'Low\n{pct_low:.1f}%', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='#d62728', ha='center')
    ax.text(0.50, 0.85, f'Med\n{pct_med:.1f}%', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='#ff7f0e', ha='center')
    ax.text(0.85, 0.85, f'High\n{pct_high:.1f}%', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='#2ca02c', ha='center')

    ax.set_xlabel("Reliability Score")
    ax.set_ylabel("Number of Scans")
    ax.set_title("Linear Scale: Dominance of Low-Confidence Scans")
    ax.legend(loc="upper right")

    # --- Right: Log scale ---
    ax = axes[1]
    ax.bar(edges[:-1], counts, width=np.diff(edges), color=colors,
           edgecolor='white', linewidth=0.5, align='edge')
    ax.set_yscale('log')
    ax.axvline(x=low_thresh, color='#ff7f0e', linestyle='--', linewidth=2,
               label=f'Medium/Low ({low_thresh})')
    ax.axvline(x=high_thresh, color='#2ca02c', linestyle='--', linewidth=2,
               label=f'High/Medium ({high_thresh})')

    ax.set_xlabel("Reliability Score")
    ax.set_ylabel("Number of Scans (log scale)")
    ax.set_title("Log Scale: Exponential Decay Structure")
    ax.legend(loc="upper right")

    plt.suptitle(f"Distribution of RRS-MBI Reliability Scores (N = {n})",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    print(f"\nLow  (< {low_thresh}):  {n_low} scans ({pct_low:.1f}%)")
    print(f"Med  ({low_thresh}-{high_thresh}): {n_med} scans ({pct_med:.1f}%)")
    print(f"High (>= {high_thresh}): {n_high} scans ({pct_high:.1f}%)")
    plt.close(fig)


if __name__ == "__main__":
    plot_reliability_distribution_thresholds()