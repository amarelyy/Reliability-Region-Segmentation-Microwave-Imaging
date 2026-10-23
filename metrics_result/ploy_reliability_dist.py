"""Figure 3: Reliability score distribution and correlation with localization error."""
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


def plot_reliability_distribution(metrics_path="rrs_mbi/results/metrics_raw.csv",
                                  save_path="reliability_distribution.png"):
    df = pd.read_csv(metrics_path)
    rel = df["reliability_score"].to_numpy(dtype=float)
    err = df["localization_error"].to_numpy(dtype=float)
    valid = ~(np.isnan(rel) | np.isnan(err))
    rel_v = rel[valid]
    err_v = err[valid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Histogram
    axes[0].hist(rel_v, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(x=np.mean(rel_v), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean = {np.mean(rel_v):.3f}')
    axes[0].set_xlabel("Reliability Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) Distribution of Reliability Scores")
    axes[0].legend(loc="upper right")

    # (b) Scatter
    axes[1].scatter(rel_v, err_v, s=10, alpha=0.4, edgecolors='none', c='steelblue')
    z = np.polyfit(rel_v, err_v, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rel_v.min(), rel_v.max(), 100)
    axes[1].plot(x_line, p(x_line), 'r--', linewidth=1.5,
                 label=f'Linear fit (slope={z[0]:.1f})')
    axes[1].set_xlabel("Reliability Score")
    axes[1].set_ylabel("Localization Error (mm)")
    axes[1].set_title("(b) Reliability vs Localization Error")
    axes[1].legend(loc="upper right")

    from scipy.stats import spearmanr
    rho, p_val = spearmanr(rel_v, err_v)
    axes[1].text(0.05, 0.05, f'ρ = {rho:.3f}, p = {p_val:.3f}',
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_reliability_distribution()