"""Step 3: report | baseline | fixed table, confusion matrices, chi-square on triage tiers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from summarize import summarize

RUNS = {
    "baseline_sat5": A / "baseline/sat5/rrs_mbi/results/metrics_raw.csv",
    "baseline_sat20": A / "baseline/sat20/rrs_mbi/results/metrics_raw.csv",
    "fixed_sat5": A / "fixed/sat5/rrs_mbi/results/metrics_raw.csv",
    "fixed_sat20": A / "fixed/sat20/rrs_mbi/results/metrics_raw.csv",
}
report = pd.read_csv(A / "baseline/comparison_report_vs_baseline.csv", index_col=0)["report"]

table = pd.DataFrame({"report": report, **{k: pd.Series(summarize(p)) for k, p in RUNS.items()}})
table.index.name = "metric"
table.to_csv(A / "fixed/comparison_report_baseline_fixed.csv")


def tiers(R, scheme):
    if scheme == "percentile_33_67":
        lo, hi = R.quantile(0.33), R.quantile(0.67)
    else:
        lo, hi = 0.30, 0.70
    return np.where(R >= hi, "high", np.where(R < lo, "low", "medium")), lo, hi


rows, cm_rows = [], []
for run, p in RUNS.items():
    d = pd.read_csv(p)
    acc = np.where(d.localization_error <= 15.0, "accurate", "inaccurate")
    for scheme in ("percentile_33_67", "fixed_0.30_0.70"):
        t, lo, hi = tiers(d.reliability_score, scheme)
        ct = pd.crosstab(pd.Categorical(t, ["low", "medium", "high"]),
                         pd.Categorical(acc, ["accurate", "inaccurate"]), dropna=False)
        for tier in ct.index:
            n = int(ct.loc[tier].sum())
            cm_rows.append(dict(run=run, scheme=scheme, low_thr=lo, high_thr=hi, tier=tier, n=n,
                                accurate=int(ct.loc[tier, "accurate"]),
                                inaccurate=int(ct.loc[tier, "inaccurate"]),
                                det_pct=ct.loc[tier, "accurate"] / n * 100 if n else np.nan,
                                mean_LE=d.localization_error[t == tier].mean()))
        ct_nz = ct[(ct.sum(axis=1) > 0)]
        ct_nz = ct_nz.loc[:, ct_nz.sum(axis=0) > 0]
        if ct_nz.shape[0] >= 2 and ct_nz.shape[1] >= 2:
            chi2, p_chi, dof, exp = chi2_contingency(ct_nz.values)
            min_exp = exp.min()
        else:
            chi2 = p_chi = dof = min_exp = np.nan
        rows.append(dict(run=run, scheme=scheme, low_thr=lo, high_thr=hi, tiers_nonempty=ct_nz.shape[0],
                         chi2=chi2, dof=dof, p=p_chi, min_expected=min_exp))

cm = pd.DataFrame(cm_rows)
chi = pd.DataFrame(rows)
cm.to_csv(A / "fixed/confusion_matrices.csv", index=False)
chi.to_csv(A / "fixed/chi_square_tiers.csv", index=False)

pd.set_option("display.width", 220)
print(table.round(4).to_string())
print()
print(cm.round(3).to_string(index=False))
print()
print(chi.round(4).to_string(index=False))
