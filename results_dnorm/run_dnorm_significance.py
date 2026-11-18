"""Permutation p + bootstrap gap CI for the three D settings, using the report's own procedure.
Primary procedure = results_audit/summarize.py lines 11-15, 33 (reproduces 0.0983 and [-1.4233, +6.0937]).
Cross-checks = rrs_mbi/evaluator.run_permutation_test and bootstrap_vali.bootstrap_gap_ci (different random-stream order).
Sources are imported read-only (bytecode writing off). New files only in results_dnorm/.
"""
import io, contextlib, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "results_audit"))
from summarize import summarize                 # the function that produced the report-table p and CI
from rrs_mbi.evaluator import run_permutation_test
import bootstrap_vali

OUT = REPO / "results_dnorm"
base = pd.read_csv(REPO / "results_audit/baseline/report_original/metrics_raw.csv")
per = pd.read_csv(OUT / "dnorm_per_scan.csv")
assert (base.scan_idx.values == per.scan_idx.values).all() and np.allclose(base.localization_error, per.localization_error)
names = ["a_linear_sat5_old", "b_linear_p95", "c_log_p95"]
quiet = lambda f, *a, **k: (lambda b: (contextlib.redirect_stdout(b).__enter__(), f(*a, **k))[1])(io.StringIO())

def one(name):
    d = base.copy(); d["reliability_score"] = per[f"R_{name}"].values
    p = OUT / f"metrics_raw_{name}.csv"; d.to_csv(p, index=False)
    s = summarize(p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pe = run_permutation_test(d, n_top=50, n_perm=10000)          # evaluator version (seed 42)
        bv = bootstrap_vali.bootstrap_gap_ci(d, n_top=50, n_boot=10000, seed=42)  # standalone version
    top = d.nlargest(50, "reliability_score").localization_error
    return dict(setting=name, n_top=50, top50_meanLE=top.mean(), all_meanLE=d.localization_error.mean(),
                point_gap_mm=d.localization_error.mean() - top.mean(),
                perm_p=s["perm_p_top50"], gap_CI_lo=s["gap_CI_lo"], gap_CI_hi=s["gap_CI_hi"],
                xcheck_evaluator_perm_p=pe, xcheck_bootstrap_vali_lo=bv["ci_lower_mm"], xcheck_bootstrap_vali_hi=bv["ci_upper_mm"])

# ---- step 2: setting (a) must reproduce the report-run values before anything else is computed
ra = one(names[0])
ref = pd.read_csv(REPO / "results_audit/baseline/comparison_report_vs_baseline.csv").set_index("metric").baseline_sat5
print("setting (a):", {k: round(v, 4) for k, v in ra.items() if isinstance(v, float)})
ok = (abs(ra["perm_p"] - ref["perm_p_top50"]) < 1e-9 and abs(ra["gap_CI_lo"] - ref["gap_CI_lo"]) < 1e-9
      and abs(ra["gap_CI_hi"] - ref["gap_CI_hi"]) < 1e-9)
print("reference (baseline_sat5 in comparison_report_vs_baseline.csv): p=%.4f CI=[%.4f, %.4f]  -> reproduces: %s"
      % (ref["perm_p_top50"], ref["gap_CI_lo"], ref["gap_CI_hi"], ok), flush=True)
if not ok:
    sys.exit("STOP: setting (a) does not reproduce the report-run p / CI")
rows = [ra] + [one(n) for n in names[1:]]
res = pd.DataFrame(rows); res.to_csv(OUT / "dnorm_significance.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.T.to_string(float_format=lambda x: f"{x:.4f}"))
