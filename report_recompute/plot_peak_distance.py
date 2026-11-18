"""
report_recompute/plot_peak_distance.py

Box plot of the selected-blob peak distance from the shell centre, per band of the ground-truth
distance from the shell centre (0-10, 10-20, 20-30, 30-40, >40 mm), all 504 scans.
Each box sits at the band's median GT distance (numeric x axis) so the y = x reference line is meaningful;
the black diamond is the median GT distance of the band (it lies on y = x by construction).

Input : reconstruction/results/per_scan_metrics.csv (written by reconstruction/reconstruct_images.py --all)
Output: metrics_image/peak_distance_by_band.png (300 dpi), report_recompute/peak_distance_by_band.csv

Run from the repo root:  python report_recompute/plot_peak_distance.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

d = pd.read_csv(REPO / "reconstruction" / "results" / "per_scan_metrics.csv")
assert len(d) == 504 and d["phant_id"].nunique() == 66
gt, pk = d["gt_dist_from_centre"].to_numpy(float), d["peak_dist_from_centre"].to_numpy(float)

edges, names = [0, 10, 20, 30, 40, np.inf], ["0-10", "10-20", "20-30", "30-40", ">40"]
rows, data, pos = [], [], []
for lo, hi, nm in zip(edges[:-1], edges[1:], names):
    m = (gt >= lo) & (gt < hi)
    q1, med, q3 = np.percentile(pk[m], [25, 50, 75])
    rows.append(dict(gt_distance_band_mm=nm, n=int(m.sum()), phantoms=int(d.loc[m, "phant_id"].nunique()),
                     peak_dist_median_mm=med, peak_dist_q1_mm=q1, peak_dist_q3_mm=q3, peak_dist_iqr_mm=q3 - q1,
                     peak_dist_mean_mm=pk[m].mean(), median_gt_dist_mm=float(np.median(gt[m]))))
    data.append(pk[m])
    pos.append(float(np.median(gt[m])))
tab = pd.DataFrame(rows)
tab.to_csv(HERE / "peak_distance_by_band.csv", index=False)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 1.0})
fig, ax = plt.subplots(figsize=(7, 5))
lim = float(np.ceil(max(max(pos), pk.max()) / 10) * 10) + 5
ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.5, label="y = x (peak at the true distance)")
bp = ax.boxplot(data, positions=pos, widths=3.6, patch_artist=True, showfliers=True, manage_ticks=False,
                medianprops=dict(color="red", linewidth=2.0), whiskerprops=dict(color="#1f77b4"),
                capprops=dict(color="#1f77b4"), boxprops=dict(edgecolor="#1f77b4", facecolor="#1f77b4", alpha=0.35),
                flierprops=dict(marker="o", markersize=3, markerfacecolor="#1f77b4", markeredgecolor="white",
                                markeredgewidth=0.4, alpha=0.6))
ax.plot(pos, pos, linestyle="none", marker="D", markersize=8, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.4, zorder=5, label="Median GT distance of the band")
ax.plot([], [], color="red", linewidth=2.0, label="Median peak distance")
ax.set_xticks(pos)
ax.set_xticklabels([f"{n} mm\n(n={k})" for n, k in zip(names, tab["n"])])
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_title("Selected-Blob Peak Distance vs. True Tumor Distance")
ax.set_xlabel("Tumor distance from shell centre, GT band (positioned at band median GT distance)")
ax.set_ylabel("Peak distance from shell centre (mm)")
ax.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
(REPO / "metrics_image").mkdir(exist_ok=True)
plt.savefig(REPO / "metrics_image" / "peak_distance_by_band.png", dpi=300, bbox_inches="tight")
print(tab.round(2).to_string(index=False))
print("scans:", len(d), "all-scan median peak dist", round(float(np.median(pk)), 2), "IQR",
      np.round(np.percentile(pk, [25, 75]), 2))
