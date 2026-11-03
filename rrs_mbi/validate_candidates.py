"""
rrs_mbi/validate_candidates.py

Test candidate RRS-MBI v2 metrics as a FAIL-SAFE: can a ground-truth-free score separate
successful localizations (LE <= 15 mm) from failures, on held-out phantoms?

  python -m rrs_mbi.validate_candidates compute [--n-jobs 4]   # per-scan metrics -> CSV
  python -m rrs_mbi.validate_candidates analyze                # AUROC, composites, risk-coverage

Pre-specified (fixed before evaluation):
  - success = LE <= 15 mm; metric directions in candidate_metrics.DIRECTION
  - composite A = mean of direction-adjusted percentile ranks of the candidate metrics (no fitting)
  - composite B = logistic regression, phantom-grouped 5-fold CV, out-of-fold scores only
  - all CIs: phantom-cluster bootstrap (scans of one phantom are not independent)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rrs_mbi import candidate_metrics as cm
from rrs_mbi.evaluator import cluster_bootstrap, compute_localization_error
from rrs_mbi.metrics import evaluate_scan_reliability

OUT = Path(__file__).resolve().parent.parent / "results_audit" / "logic_refactor"
SUCCESS_MM = 15.0
GATE_NS = 0.70
SUBBANDS = ((1e9, 4.5e9), (4.5e9, 8e9))
CANDIDATES = ["sa_half_dist", "sa_interleave_dist", "subband_dist", "artifact_prior", "peak_margin", "cf_at_peak"]


# ----------------------------------------------------------------------------------------------
# compute
# ----------------------------------------------------------------------------------------------
def _scan_row(i, s21, tm, idm, freq_axis, ref_cache):
    from src.pipeline import reconstruct_scan
    r = reconstruct_scan(i, s21, tm, idm, freq_axis=freq_axis, gate_ns=GATE_NS, return_diagnostics=True)
    g = r["diagnostics"]
    img, axis, ts, dg = g["image"], g["axis_mm"], g["time_signal"], g["delay_grid"]
    c = (r["shell_cx_mm"], r["shell_cy_mm"])
    region = cm.breast_disc(axis, c, r["breast_radius_mm"])
    p = (r["peak_x_mm"], r["peak_y_mm"])
    rel = evaluate_scan_reliability(img, axis_mm=axis, cf_at_peak=r["cf_at_peak"],
                                    saturation_radius_mm=20.0, peak_xy=p, center_xy=c)

    from src.signal_processing import get_time_axis
    sa_half, sa_inter = cm.subaperture_distances(ts, get_time_axis(ts.shape[0]), dg, axis, region, GATE_NS)

    sub = [reconstruct_scan(i, s21, tm, idm, freq_axis=freq_axis, gate_ns=GATE_NS, bandpass_mode=b)
           for b in SUBBANDS]
    subband_dist = compute_localization_error((sub[0]["peak_x_mm"], sub[0]["peak_y_mm"]),
                                              (sub[1]["peak_x_mm"], sub[1]["peak_y_mm"]))

    row = tm.iloc[i]
    key = (int(row["adi_ref_id"]), row["phant_id"])
    if key not in ref_cache:   # tumour-free image of the same phantom on the same grid
        ref_row = tm.iloc[[i]].copy()
        ref_row["original_s21_idx"] = idm[int(row["adi_ref_id"])]
        rr = reconstruct_scan(0, s21, ref_row, idm, freq_axis=freq_axis, gate_ns=GATE_NS,
                              ref_col="emp_ref_id", return_diagnostics=True)
        ref_cache[key] = rr["diagnostics"]["image"]
    art = cm.artifact_prior(ref_cache[key], axis, region, p)

    return dict(
        scan_idx=i, phant_id=row["phant_id"],
        localization_error=r["localization_error_mm"],
        trivial_center_error=compute_localization_error(c, (r["gt_x_mm"], r["gt_y_mm"])),
        pred_dist=float(np.hypot(p[0] - c[0], p[1] - c[1])),
        R_old=rel["reliability_score"], D_norm=min(1.0, rel["peak_dominance"] / 5.0),
        one_minus_CC=1 - rel["central_concentration"], G=rel["geo_plausibility"],
        cf_at_peak=r["cf_at_peak"], sa_half_dist=sa_half, sa_interleave_dist=sa_inter,
        subband_dist=subband_dist, artifact_prior=art, peak_margin=cm.peak_margin(img, axis, region, p),
    )


def _run_chunk(indices, s21, tm, idm, freq_axis):
    ref_cache, rows = {}, []
    for i in indices:
        try:
            rows.append(_scan_row(i, s21, tm, idm, freq_axis, ref_cache))
        except Exception as e:  # keep going; report failures explicitly
            rows.append(dict(scan_idx=i, error=str(e)))
    return rows


def compute(n_jobs=4):
    from joblib import Parallel, delayed
    from src.data_loading import load_all_data
    d = load_all_data()
    tm = d["tumor_model"]
    order = tm.sort_values("phant_id").index.to_numpy()        # group by phantom -> delay-cache hits
    chunks = [c for c in np.array_split(order, n_jobs * 4) if len(c)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_chunk)(c, d["s21"], tm, d["id_to_original_idx"], d["freq_axis"]) for c in chunks)
    df = pd.DataFrame([r for rows in res for r in rows]).sort_values("scan_idx").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "candidate_metrics_per_scan.csv", index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"saved {len(df)} rows ({n_err} errors) -> {OUT / 'candidate_metrics_per_scan.csv'}")


# ----------------------------------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------------------------------
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if len(np.unique(y[m])) == 2 else float("nan")


def _adjusted(df, cols):
    return pd.DataFrame({c: cm.DIRECTION[c] * df[c] for c in cols})


def composite_rank(df, cols=CANDIDATES):
    """Composite A: mean percentile rank of direction-adjusted metrics. No parameters fitted."""
    return _adjusted(df, cols).rank(pct=True).mean(axis=1, skipna=True)


def composite_logistic_oof(df, cols=CANDIDATES, n_splits=5, seed=42):
    """Composite B: logistic regression, phantom-grouped CV, out-of-fold probabilities only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    X = _adjusted(df, cols).to_numpy()
    y = (df["localization_error"] <= SUCCESS_MM).astype(int).to_numpy()
    oof = np.full(len(df), np.nan)
    coefs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=df["phant_id"]):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        coefs.append(model[-1].coef_[0])
    return oof, pd.DataFrame(coefs, columns=cols)


def risk_coverage(df, score, coverages=np.linspace(1.0, 0.1, 10)):
    s = df.assign(_s=score).sort_values("_s", ascending=False)
    rows = []
    for cov in coverages:
        k = max(1, int(round(len(s) * cov)))
        top = s.iloc[:k]
        succ = top["localization_error"] <= SUCCESS_MM
        rows.append(dict(coverage=round(cov, 2), n=k, mean_LE=top["localization_error"].mean(),
                         trivial_LE=top["trivial_center_error"].mean(), det_pct=succ.mean() * 100,
                         successes_retained_pct=succ.sum() / max(1, (s["localization_error"] <= SUCCESS_MM).sum()) * 100))
    return pd.DataFrame(rows)


def analyze(n_boot=1000):
    df = pd.read_csv(OUT / "candidate_metrics_per_scan.csv")
    if "error" in df:
        bad = df["error"].notna()
        print(f"dropping {int(bad.sum())} scans with errors")
        df = df[~bad].reset_index(drop=True)
    df["success"] = (df["localization_error"] <= SUCCESS_MM).astype(int)
    df["comp_A_rank"] = composite_rank(df)
    df["comp_B_logit_oof"], coefs = composite_logistic_oof(df)
    print(f"n={len(df)}  successes={df.success.sum()} ({df.success.mean() * 100:.1f}%)  phantoms={df.phant_id.nunique()}")

    # AUROC per score (direction-adjusted for candidates; baselines as-is, +1 = higher is "reliable")
    scores = {c: cm.DIRECTION[c] * df[c] for c in CANDIDATES}
    scores.update({"R_old": df["R_old"], "D_norm": df["D_norm"], "one_minus_CC": df["one_minus_CC"],
                   "G": df["G"], "pred_dist (far=reliable)": df["pred_dist"],
                   "comp_A_rank": df["comp_A_rank"], "comp_B_logit_oof": df["comp_B_logit_oof"]})
    rows = []
    for name, s in scores.items():
        fn = lambda d, name=name: _auc(d["success"].to_numpy(), scores_of(d)[name].to_numpy())
        b = cluster_bootstrap(df, fn, n_boot=n_boot, group_col="phant_id")
        rows.append(dict(score=name, AUROC=_auc(df["success"].to_numpy(), s.to_numpy()),
                         ci_lo=np.nanpercentile(b, 2.5), ci_hi=np.nanpercentile(b, 97.5)))
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / "auroc.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAUROC for success (LE<=15 mm); 0.5 = no information; phantom-cluster 95% CI")
    print(auc.round(3).to_string(index=False))

    print("\nComposite B logistic coefficients per fold (direction-adjusted, standardised):")
    print(coefs.round(2).to_string(index=False))
    coefs.to_csv(OUT / "comp_B_fold_coefficients.csv", index=False)

    rc = pd.concat({k: risk_coverage(df, df[k]) for k in ("R_old", "comp_A_rank", "comp_B_logit_oof")})
    rc.to_csv(OUT / "risk_coverage.csv")
    print("\nRisk-coverage (accepted = top-coverage by score):")
    print(rc.round(2).to_string())

    gap_rows = []
    for k in ("R_old", "comp_A_rank", "comp_B_logit_oof"):
        for cov in (0.2, 0.1):
            fn = lambda d, k=k, cov=cov: float(
                d["localization_error"].mean() - d.nlargest(max(1, int(len(d) * cov)), k)["localization_error"].mean())
            b = cluster_bootstrap(df, fn, n_boot=n_boot, group_col="phant_id")
            gap_rows.append(dict(score=k, coverage=cov, LE_gap_all_minus_accepted=fn(df),
                                 ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5)))
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT / "coverage_gaps.csv", index=False)
    print("\nLE gap (all - accepted; >0 = score helps), phantom-cluster 95% CI:")
    print(gaps.round(2).to_string(index=False))
    df.to_csv(OUT / "candidate_metrics_scored.csv", index=False)


def scores_of(d):
    """Scores recomputed on a (bootstrap) frame. Composite B keeps its out-of-fold values."""
    s = {c: cm.DIRECTION[c] * d[c] for c in CANDIDATES}
    s.update({"R_old": d["R_old"], "D_norm": d["D_norm"], "one_minus_CC": d["one_minus_CC"], "G": d["G"],
              "pred_dist (far=reliable)": d["pred_dist"], "comp_A_rank": d["comp_A_rank"],
              "comp_B_logit_oof": d["comp_B_logit_oof"]})
    return s


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "compute":
        n_jobs = int(sys.argv[sys.argv.index("--n-jobs") + 1]) if "--n-jobs" in sys.argv else 4
        compute(n_jobs)
    else:
        analyze()
