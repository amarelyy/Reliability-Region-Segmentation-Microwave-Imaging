"""SENSITIVITY ANALYSIS (not part of fixes.patch): how much of the fixed-pipeline error is the
shell/skin reflection?  For each cached reconstruction set, compare
  - pipeline blob centroid (as evaluated),
  - global argmax of the image,
  - argmax restricted to the breast disc (r <= R_shell) and to r <= 0.9 R_shell.
Cache sets: baseline (empty-chamber subtraction), fixed (empty-chamber), fixed_adisub
(adipose-reference subtraction).
"""
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
SETS = {"baseline": A / "baseline/cache/recon_tumour",
        "fixed": A / "fixed/cache/recon_tumour",
        "fixed_adisub": A / "fixed/cache/recon_tumour_adisub"}

rows, summ = [], []
for name, base in SETS.items():
    if not base.with_suffix(".npz").exists():
        continue
    meta = pd.read_csv(base.with_suffix(".csv"))
    z = np.load(base.with_suffix(".npz"))
    for i, r in meta.iterrows():
        img, ax = z[f"img_{i}"], z[f"axis_{i}"]
        gx, gy = np.meshgrid(ax, ax)
        rr = np.hypot(gx, gy)
        out = dict(set=name, i=i, phant_id=r.phant_id, R=r.breast_radius_mm,
                   gt_r=np.hypot(r.tumor_x_mm, r.tumor_y_mm), LE_blob=r.localization_error_mm,
                   pred_r_blob=np.hypot(r.peak_x_mm, r.peak_y_mm))
        for lab, lim in (("argmax", np.inf), ("argmax_inR", 1.0), ("argmax_in0.9R", 0.9)):
            iy, ix = np.unravel_index(np.argmax(np.where(rr <= lim * r.breast_radius_mm, img, -np.inf)), img.shape)
            out[f"LE_{lab}"] = np.hypot(ax[ix] - r.tumor_x_mm, ax[iy] - r.tumor_y_mm)
            out[f"pred_r_{lab}"] = np.hypot(ax[ix], ax[iy])
        rows.append(out)
df = pd.DataFrame(rows)
df.to_csv(A / "sensitivity_per_scan.csv", index=False)
for name, g in df.groupby("set", sort=False):
    for m in ("blob", "argmax", "argmax_inR", "argmax_in0.9R"):
        le = g[f"LE_{m}"]
        summ.append(dict(set=name, locator=m, n=len(g), mean_LE=le.mean(), median_LE=le.median(),
                         det15_pct=(le <= 15).mean() * 100, det15_n=int((le <= 15).sum()),
                         mean_pred_r=g[f"pred_r_{m}"].mean(),
                         frac_pred_outside_R=(g[f"pred_r_{m}"] > g.R).mean()))
s = pd.DataFrame(summ)
s.to_csv(A / "sensitivity_summary.csv", index=False)
print(s.round(3).to_string(index=False))
