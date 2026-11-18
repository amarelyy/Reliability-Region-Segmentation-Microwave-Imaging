"""Three fixed D-normalisation settings on the saved report-time run (REAL data, R_sat = 5 mm run).
Reads results_audit/baseline/report_original/metrics_raw.csv (read only). Writes only to results_dnorm/.
Step 1 uses D ONLY to fix the settings; LE / R enter only in step 3.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rrs_mbi.metrics import normalise_dominance
OUT = REPO / "results_dnorm"
SRC = REPO / "results_audit/baseline/report_original/metrics_raw.csv"
df = pd.read_csv(SRC); N = len(df); assert N == 504

# ---- step 1: settings from the distribution of D only (fixed before any result is looked at)
D = df.peak_dominance.to_numpy(float)
p95 = float(np.percentile(D, 95))
SETTINGS = [("a_linear_sat5_old", "linear", 5.0), ("b_linear_p95", "linear", p95), ("c_log_p95", "log", p95)]
git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
(OUT / "settings_fixed_from_D_only.json").write_text(json.dumps(dict(
    D_p95_of_504=p95, percentile_method="numpy default (linear interpolation)", settings=SETTINGS,
    source=str(SRC.relative_to(REPO)), git_commit=git("rev-parse", "HEAD"), git_dirty=git("status", "--short")), indent=2))
print(f"D p95 over {N} scans = {p95:.4f}", flush=True)

# ---- step 2: rebuild the other three factors exactly (report-time geometry kept)
cc = df.central_concentration.to_numpy(float)
cf = df.cf_at_peak.to_numpy(float)
cff = np.where(np.isnan(cf), 1.0, np.clip((cf - 0.1) / 0.7, 0.1, 1.0))
assert (D >= 5).all()  # D_norm_old == 1 for every scan -> G recoverable exactly from the saved R
geo = df.reliability_score.to_numpy(float) / ((1 - cc) * cff)
assert np.isfinite(geo).all() and (geo <= 1 + 1e-9).all()

# ---- step 3: metrics per setting
le = df.localization_error.to_numpy(float)
ratios = np.linspace(1.0, 0.1, 10)
rows, scans = [], df[["scan_idx", "scan_id", "localization_error"]].copy()
for name, mode, sat in SETTINGS:
    dn = np.array([normalise_dominance(x, sat, mode) for x in D])
    R = dn * (1 - cc) * geo * cff
    if name.startswith("a_"):
        assert np.abs(R - df.reliability_score.to_numpy()).max() < 1e-12, "setting (a) must reproduce the saved R"
    scans[f"Dnorm_{name}"] = dn; scans[f"R_{name}"] = R
    rho, p = spearmanr(R, le)
    r = dict(setting=name, d_mode=mode, d_saturation=sat,
             frac_Dnorm_eq_1_pct=100 * (dn >= 1.0).mean(), Dnorm_min=dn.min(), Dnorm_median=np.median(dn), Dnorm_max=dn.max(),
             R_mean=R.mean(), R_min=R.min(), R_max=R.max(),
             n_lt_030=int((R < .3).sum()), n_030_070=int(((R >= .3) & (R < .7)).sum()), n_ge_070=int((R >= .7).sum()),
             spearman_R_vs_LE=rho, spearman_p=p)
    t = pd.DataFrame(dict(R=R, LE=le)).sort_values("R", ascending=False).reset_index(drop=True)  # same call as evaluator.analyze_thresholds
    for ratio, tag in [(ratios[-2], "top20"), (ratios[-1], "top10")]:
        k = max(1, int(N * ratio)); sub = t.iloc[:k]
        r[f"{tag}_n"] = k; r[f"{tag}_meanLE"] = sub.LE.mean(); r[f"{tag}_det15_pct"] = 100 * (sub.LE <= 15).mean()
        r[f"{tag}_threshold"] = sub.R.iloc[-1]; r[f"{tag}_ties_at_cut"] = int((t.R == sub.R.iloc[-1]).sum())
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv(OUT / "dnorm_settings_summary.csv", index=False)
scans.to_csv(OUT / "dnorm_per_scan.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(res.T.to_string(float_format=lambda x: f"{x:.4f}"))
print("all-scan mean LE %.2f, det15 %.1f%%" % (le.mean(), 100 * (le <= 15).mean()))
