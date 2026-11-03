"""Final comparison: report | baseline (report code) | fixes.patch only | final (all fixes),
plus a diagnostic of what the predicted point sits on (fibroglandular insert vs tumour)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
REPO = A.parent
sys.path.insert(0, str(A)); sys.path.insert(0, str(REPO))
from summarize import summarize
from src.data_loading import load_all_data

RUNS = {"baseline_report_code": A / "baseline/sat5/rrs_mbi/results/metrics_raw.csv",
        "fixes_patch_only": A / "fixed/sat20/rrs_mbi/results/metrics_raw.csv",
        "final_all_fixes": A / "final/sat20/rrs_mbi/results/metrics_raw.csv"}
report = pd.read_csv(A / "baseline/comparison_report_vs_baseline.csv", index_col=0)["report"]
tm = load_all_data()["tumor_model"]

cols = {"report": report}
for k, p in RUNS.items():
    d = pd.read_csv(p)
    s = summarize(p)
    if "trivial_center_error" in d:
        triv = d.trivial_center_error
    else:  # older runs: trivial = always predict the chamber origin
        triv = np.hypot(d.true_x_mm, d.true_y_mm)
    s["trivial_mean_LE"] = triv.mean()
    s["trivial_det15_pct"] = (triv <= 15).mean() * 100
    s["pct_scans_beating_trivial"] = (d.localization_error < triv).mean() * 100
    top = d.nlargest(int(len(d) * 0.1), "reliability_score")
    s["top10_trivial_LE"] = (top.trivial_center_error if "trivial_center_error" in top
                             else np.hypot(top.true_x_mm, top.true_y_mm)).mean()
    cx = d.shell_cx_mm if "shell_cx_mm" in d else 0.0
    cy = d.shell_cy_mm if "shell_cy_mm" in d else 0.0
    s["mean_pred_dist_from_shell_centre"] = np.hypot(d.pred_x_mm - cx, d.pred_y_mm - cy).mean()
    s["mean_gt_dist_from_shell_centre"] = np.hypot(d.true_x_mm - cx, d.true_y_mm - cy).mean()
    cols[k] = pd.Series(s)
t = pd.DataFrame(cols)
t.index.name = "metric"
(A / "final").mkdir(exist_ok=True)
t.to_csv(A / "final/comparison_report_baseline_patch_final.csv")
pd.set_option("display.width", 220)
print(t.round(4).to_string())

# --- What does the predicted point sit on? (final pipeline) ---
d = pd.read_csv(RUNS["final_all_fixes"])
m = tm.iloc[d.scan_idx.to_numpy()].reset_index(drop=True)
fx, fy = m.fib_x.to_numpy() * 10, m.fib_y.to_numpy() * 10   # fib insert offset, cm -> mm
px, py = d.pred_x_mm.to_numpy(), d.pred_y_mm.to_numpy()
d_fib = np.hypot(px - fx, py - fy)
rng = np.random.default_rng(0)
ctrl = np.mean([np.hypot(px - fx[p], py - fy[p]).mean() for p in (rng.permutation(len(px)) for _ in range(2000))])
diag = pd.DataFrame([dict(
    mean_dist_pred_to_fib_centre=d_fib.mean(), shuffled_control=ctrl,
    mean_dist_pred_to_shell_centre=np.hypot(px - d.shell_cx_mm, py - d.shell_cy_mm).mean(),
    mean_dist_pred_to_tumour=d.localization_error.mean(),
    mean_dist_tumour_to_fib_centre=np.hypot(d.true_x_mm - fx, d.true_y_mm - fy).mean(),
    LE_tumour_in_fib=d.localization_error[m.tum_in_fib == 1].mean(),
    LE_tumour_not_in_fib=d.localization_error[m.tum_in_fib == 0].mean(),
    n_in_fib=int((m.tum_in_fib == 1).sum()))])
diag.to_csv(A / "final/diag_prediction_vs_fib.csv", index=False)
print("\n", diag.round(2).T.to_string())

# LE by tumour size and by distance band (final)
d["tum_rad"] = m.tumor_radius_mm.values
d["gt_r"] = np.hypot(d.true_x_mm - d.shell_cx_mm, d.true_y_mm - d.shell_cy_mm)
by = pd.concat({
    "by_tumour_radius": d.groupby("tum_rad").agg(n=("localization_error", "size"), LE=("localization_error", "mean"),
                                                 det15=("localization_error", lambda v: (v <= 15).mean() * 100),
                                                 trivial=("trivial_center_error", "mean")),
    "by_gt_distance": d.groupby(pd.cut(d.gt_r, [0, 10, 20, 30, 40, 80])).agg(
        n=("localization_error", "size"), LE=("localization_error", "mean"),
        det15=("localization_error", lambda v: (v <= 15).mean() * 100), trivial=("trivial_center_error", "mean"))})
by.to_csv(A / "final/LE_by_size_and_distance.csv")
print("\n", by.round(1).to_string())
