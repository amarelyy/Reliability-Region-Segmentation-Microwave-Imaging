"""Figure 2: SVM-ranked vs argmax localization error comparison."""
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


def plot_svm_vs_argmax(candidates_path="svm_candidates_scored.csv", save_path="metrics_image/svm_vs_argmax.png"):
    df = pd.read_csv(candidates_path)

    # Argmax = candidate #0 per scan (brightest-first order)
    argmax = df.groupby("scan_idx").first().reset_index()
    argmax_le = argmax["le_to_gt_mm"].to_numpy()

    # SVM-ranked = highest oof_score per scan
    svm_ranked = df.loc[df.groupby("scan_idx")["oof_score"].idxmax()].reset_index()
    svm_le = svm_ranked["le_to_gt_mm"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CDF comparison
    axes[0].ecdf(argmax_le, label=f"Argmax (mean={argmax_le.mean():.1f} mm)", linewidth=2)
    axes[0].ecdf(svm_le, label=f"SVM-ranked (mean={svm_le.mean():.1f} mm)", linewidth=2)
    axes[0].axvline(x=20, color='gray', linestyle='--', alpha=0.7, label="Detection threshold (20 mm)")
    axes[0].set_xlabel("Localization Error (mm)")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].set_title("(a) Cumulative Distribution of Localization Error")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(0, 80)

    # Right: Paired scatter
    min_len = min(len(argmax_le), len(svm_le))
    axes[1].scatter(argmax_le[:min_len], svm_le[:min_len],
                    s=8, alpha=0.4, edgecolors='none', c='steelblue')
    lim_max = max(argmax_le.max(), svm_le.max()) * 1.05
    axes[1].plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, label="y = x (equal performance)")
    axes[1].set_xlabel("Argmax LE (mm)")
    axes[1].set_ylabel("SVM-Ranked LE (mm)")
    axes[1].set_title("(b) Per-Scan Comparison")
    axes[1].legend(loc="upper left")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_svm_vs_argmax()