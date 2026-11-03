"""Summarize an evaluator metrics_raw.csv into the CLAUDE.md report-table metrics."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

def summarize(path):
    d = pd.read_csv(path)
    le, R = d.localization_error, d.reliability_score
    ph = d.groupby("scan_id").agg(e=("localization_error", "mean"), r=("reliability_score", "mean"),
                                  det=("localization_error", lambda x: (x <= 15).any()))
    top = lambda f: d.nlargest(int(len(d) * f), "reliability_score")
    rng = np.random.default_rng(42)
    t50 = d.nlargest(50, "reliability_score").localization_error.to_numpy()
    gaps = [rng.choice(le, len(le)).mean() - rng.choice(t50, 50).mean() for _ in range(10000)]
    rng2 = np.random.default_rng(42)
    p_perm = np.mean([rng2.choice(le.to_numpy(), 50, replace=False).mean() <= t50.mean() for _ in range(10000)])
    return {
        "n_scans": len(d),
        "mean_LE": le.mean(), "median_LE": le.median(), "sd_LE": le.std(),
        "det15_pct": (le <= 15).mean() * 100, "det15_n": int((le <= 15).sum()),
        "mean_pred_dist": np.hypot(d.pred_x_mm, d.pred_y_mm).mean(),
        "mean_gt_dist": np.hypot(d.true_x_mm, d.true_y_mm).mean(),
        "phantom_mean_err": ph.e.mean(), "phantom_sd_err": ph.e.std(),
        "phantoms_with_det": int(ph.det.sum()), "n_phantoms": len(ph),
        "mean_CC": d.central_concentration.mean(), "median_CC": d.central_concentration.median(),
        "sd_CC": d.central_concentration.std(), "mean_CF_peak": d.cf_at_peak.mean(),
        "R_mean": R.mean(), "R_min": R.min(), "R_max": R.max(),
        "R_ge_0.70": int((R >= 0.70).sum()), "R_0.30_0.70": int(((R >= 0.30) & (R < 0.70)).sum()),
        "R_lt_0.30": int((R < 0.30).sum()),
        "top10_LE": top(0.1).localization_error.mean(), "top10_thr": top(0.1).reliability_score.min(),
        "top10_det_pct": (top(0.1).localization_error <= 15).mean() * 100,
        "top20_LE": top(0.2).localization_error.mean(),
        "perm_p_top50": p_perm,
        "gap_CI_lo": np.percentile(gaps, 2.5), "gap_CI_hi": np.percentile(gaps, 97.5),
        "spearman_scan": spearmanr(R, le)[0], "spearman_scan_p": spearmanr(R, le)[1],
        "pearson_scan": pearsonr(R, le)[0],
        "spearman_phantom": spearmanr(ph.r, ph.e)[0], "pearson_phantom": pearsonr(ph.r, ph.e)[0],
        "mean_SCR_dB": d.scr_db.mean(),
    }
