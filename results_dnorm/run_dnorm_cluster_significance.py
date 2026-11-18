"""Phantom-cluster bootstrap (the final pipeline's evaluator method) for the three D settings.
Uses rrs_mbi.evaluator.cluster_bootstrap + top_fraction_gap UNCHANGED (imported read-only).
Gap = mean LE of all scans - mean LE of the top-50 scans by R (mm). p = fraction of bootstrap gaps <= 0 (evaluator's definition).
"""
import sys
sys.dont_write_bytecode = True
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rrs_mbi.evaluator import cluster_bootstrap, top_fraction_gap
OUT = REPO / "results_dnorm"
names = ["a_linear_sat5_old", "b_linear_p95", "c_log_p95"]
dfs = {n: pd.read_csv(OUT / f"metrics_raw_{n}.csv") for n in names}
assert all(len(d) == 504 and d.scan_id.nunique() == 66 for d in dfs.values())

rows, tops = [], {}
for n in names:
    d = dfs[n]
    tops[n] = set(d.nlargest(50, "reliability_score").scan_idx)
    ks = []
    def stat(x):
        ks.append(max(1, int(len(x) * (50 / len(x)))))
        return top_fraction_gap(x, 50 / len(x))
    est = stat(d); ks.clear()
    r = dict(setting=n, point_gap_mm=est, n_phantoms_in_top50=d.nlargest(50, "reliability_score").scan_id.nunique())
    for nb in (10000, 2000):
        ks.clear()
        b = cluster_bootstrap(d, stat, n_boot=nb, seed=42)
        lo, hi = np.percentile(b, [2.5, 97.5])
        r[f"n{nb}_CI_lo"], r[f"n{nb}_CI_hi"], r[f"n{nb}_p_one_sided"] = lo, hi, float(np.mean(b <= 0))
        if nb == 10000:
            r["resamples_with_k_ne_50_pct"] = 100 * float(np.mean(np.array(ks) != 50))
    rows.append(r)
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
res = pd.DataFrame(rows)
res["p_raw"] = res.n10000_p_one_sided; res["p_bonferroni_x3"] = np.minimum(1.0, 3 * res.p_raw)
res.to_csv(OUT / "dnorm_cluster_significance.csv", index=False)
ov = pd.DataFrame([dict(pair=f"{a} & {b}", shared_of_50=len(tops[a] & tops[b])) for a, b in combinations(names, 2)]
                  + [dict(pair="all three", shared_of_50=len(tops[names[0]] & tops[names[1]] & tops[names[2]]))])
ov.to_csv(OUT / "dnorm_top50_overlap.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.T.to_string(float_format=lambda x: f"{x:.4f}")); print(ov.to_string(index=False))
