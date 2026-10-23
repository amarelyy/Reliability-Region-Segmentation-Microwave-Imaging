"""Figure 1: Distribution of tumor locations relative to array center."""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
})

with open(Path("data/metadata_gen_two.pickle"), "rb") as f:
    meta = pickle.load(f)

tum_x = np.array([e.get("tum_x", np.nan) for e in meta], dtype=float)
tum_y = np.array([e.get("tum_y", np.nan) for e in meta], dtype=float)
valid = ~(np.isnan(tum_x) | np.isnan(tum_y))
tum_x, tum_y = tum_x[valid], tum_y[valid]

if np.median(np.sqrt(tum_x**2 + tum_y**2)) < 5:
    tum_x *= 10; tum_y *= 10

dist = np.sqrt(tum_x**2 + tum_y**2)

zones = ["< 5 mm", "5–10 mm", "10–20 mm", "20–30 mm", "> 30 mm"]
masks = [dist < 5, (dist >= 5) & (dist < 10), (dist >= 10) & (dist < 20),
         (dist >= 20) & (dist < 30), dist >= 30]
counts = [m.sum() for m in masks]
pcts = [c / len(dist) * 100 for c in counts]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(zones, pcts, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'],
              edgecolor='white', width=0.6)
for bar, pct, n in zip(bars, pcts, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=11)
ax.set_ylabel("Percentage of Tumors (%)")
ax.set_xlabel("Distance from Array Center")
ax.set_title("Tumor Location Distribution (N = 504)")
ax.set_ylim(0, max(pcts) * 1.2)
plt.tight_layout()
fig.savefig("tumor_location_distribution.png", dpi=300, bbox_inches='tight')
print("Saved: tumor_location_distribution.png")
plt.close(fig)