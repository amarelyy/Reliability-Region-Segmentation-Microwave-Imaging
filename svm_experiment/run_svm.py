"""
svm_experiment/run_svm.py

Candidate verifier on the current pipeline: a linear SVM ranks the blob candidates of each scan
(svm_candidates_v2.csv from build_candidates.py); its pick is compared with the pipeline peak,
the image argmax and two centre baselines. Phantom-grouped 5-fold CV (GroupKFold by phant_id),
phantom-cluster bootstrap (resample phantoms, recompute every statistic), 4 feature combinations.

Run from the repo root:  python svm_experiment/run_svm.py
Outputs (metrics_result/): svm_current_le.csv, svm_current_paired.csv, svm_current_weights.csv,
                           svm_current_auc.csv, svm_current_results.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "metrics_result"
N_BOOT, SEED, N_SPLITS = 2000, 42, 5
DETS = (10.0, 15.0, 20.0)

FEATS = ["f_cf_mean", "f_peak_rel", "f_dist_ratio", "f_area_frac"]
COMBOS = {
    "C1 all 4": FEATS,
    "C2 no position (-dist_ratio)": ["f_cf_mean", "f_peak_rel", "f_area_frac"],
    "C3 position only (dist_ratio)": ["f_dist_ratio"],
    "C4 no area (-area_frac)": ["f_cf_mean", "f_peak_rel", "f_dist_ratio"],
}


def stop(msg):
    print("STOP:", msg)
    sys.exit(1)


# --------------------------------------------------------------------------------------- load + checks
cand = pd.read_csv(HERE / "svm_candidates_v2.csv")
LABEL_MM = 20.0                       # label = centroid within 20 mm of the tumour centre (same as build_candidates.py)
cand["label"] = (cand["le_mm"] <= LABEL_MM).astype(int)
scans = pd.read_csv(HERE / "svm_scans_v2.csv").sort_values("scan_i").reset_index(drop=True)
if len(scans) != 504 or scans["phant_id"].nunique() != 66:
    stop(f"expected 504 scans / 66 phantoms, got {len(scans)} / {scans['phant_id'].nunique()}")
if cand["scan_i"].nunique() + int((scans["n_cand"] == 0).sum()) != 504:
    stop("candidate table does not cover all scans")

pipe_rows = cand[cand["is_pipeline"] == 1].set_index("scan_i")["le_mm"]
chk = scans.set_index("scan_i")["pipe_le"].reindex(pipe_rows.index)
if float((pipe_rows - chk).abs().max()) > 1e-6:
    stop("largest-blob candidate LE differs from pipeline LE")
ref = pd.read_csv(REPO / "reconstruction" / "results" / "per_scan_metrics.csv").set_index("i")
ref_diff = float((ref["LE"].reindex(scans["scan_i"]).to_numpy() - scans["pipe_le"].to_numpy()).__abs__().max())
if ref_diff > 1e-6:
    stop(f"pipeline LE differs from reconstruction/results/per_scan_metrics.csv (max abs diff {ref_diff:.3g})")

# --------------------------------------------------------------------------------------- CV
groups = cand["phant_id"].to_numpy()
y = cand["label"].to_numpy()
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(cand, y, groups))
for k, (tr, te) in enumerate(folds):
    if set(groups[tr]) & set(groups[te]):
        stop(f"fold {k}: phantom in both train and test")
    if len(te) == 0 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        stop(f"fold {k}: empty or single-class fold")
if sorted(np.concatenate([te for _, te in folds]).tolist()) != list(range(len(cand))):
    stop("folds do not cover every candidate exactly once")


def fit_oof(cols):
    X = cand[cols].to_numpy(float)
    oof = np.zeros(len(cand))
    aucs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = SVC(kernel="linear", class_weight="balanced", random_state=SEED).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
        aucs.append(roc_auc_score(y[te], oof[te]))
    sc = StandardScaler().fit(X)
    full = SVC(kernel="linear", class_weight="balanced", random_state=SEED).fit(sc.transform(X), y)
    return oof, aucs, dict(zip(cols, full.coef_[0]))


# --------------------------------------------------------------------------------------- per-scan LE per method
scan_ids = scans["scan_i"].to_numpy()
le = {"Pipeline peak (largest blob)": scans["pipe_le"].to_numpy(),
      "Plain argmax (image max pixel)": scans["argmax_le"].to_numpy(),
      "Centre guess (shell centre point)": scans["centre_guess_le"].to_numpy()}


def pick(score_col):
    """LE of the candidate with the largest value of score_col per scan; scans without any blob fall back to the pipeline peak."""
    d = cand.assign(_s=score_col)
    best = d.loc[d.groupby("scan_i")["_s"].idxmax(), ["scan_i", "le_mm"]].set_index("scan_i")["le_mm"]
    return best.reindex(scan_ids).fillna(pd.Series(scans["pipe_le"].to_numpy(), index=scan_ids)).to_numpy()


le["Candidate nearest shell centre"] = pick(-cand["f_dist_ratio"].to_numpy())
oof_all, aucs_tab, weights_tab = {}, [], []
for name, cols in COMBOS.items():
    oof, aucs, w = fit_oof(cols)
    oof_all[name] = oof
    le["SVM-ranked OOF: " + name] = pick(oof)
    aucs_tab.append(dict(combo=name, features=", ".join(cols), auc_mean_fold=np.mean(aucs),
                         auc_fold_sd=np.std(aucs, ddof=1), auc_overall_oof=roc_auc_score(y, oof),
                         **{f"auc_fold{k+1}": a for k, a in enumerate(aucs)}))
    for f, v in w.items():
        weights_tab.append(dict(combo=name, feature=f, std_weight=v, favours="real tumour" if v > 0 else "false candidate"))
oracle = cand.groupby("scan_i")["le_mm"].min().reindex(scan_ids)
le["Oracle (best candidate)"] = oracle.fillna(pd.Series(scans["pipe_le"].to_numpy(), index=scan_ids)).to_numpy()

# --------------------------------------------------------------------------------------- bootstrap (phantom clusters)
ph = scans["phant_id"].to_numpy()
uniq = np.unique(ph)
cl = [np.where(ph == p)[0] for p in uniq]
rng = np.random.default_rng(SEED)
boots = [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(N_BOOT)]


def stats(v):
    s = dict(mean=float(np.mean(v)), median=float(np.median(v)))
    for t in DETS:
        s[f"det{int(t)}"] = float((v <= t).mean() * 100)
    return s


def ci(fn):
    pt = fn(np.arange(len(scan_ids)))
    bs = np.array([fn(ix) for ix in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi


KEYS = ["mean", "median", "det10", "det15", "det20"]
rows = []
for m, v in le.items():
    pt, lo, hi = ci(lambda ix, v=v: np.array([stats(v[ix])[k] for k in KEYS]))
    r = {"method": m, "n_scans": len(v)}
    for k, a, b, c in zip(KEYS, pt, lo, hi):
        r[k] = a
        r[k + "_lo"] = b
        r[k + "_hi"] = c
    rows.append(r)
tab_le = pd.DataFrame(rows)

svm_names = [m for m in le if m.startswith("SVM")]
base_names = ["Pipeline peak (largest blob)", "Plain argmax (image max pixel)",
              "Centre guess (shell centre point)", "Candidate nearest shell centre"]
prow = []
for s in svm_names:
    for b in base_names:
        d = lambda ix, s=s, b=b: np.array([np.mean(le[s][ix] - le[b][ix]),
                                          np.median(le[s][ix]) - np.median(le[b][ix]),
                                          ((le[s][ix] <= 20).mean() - (le[b][ix] <= 20).mean()) * 100])
        pt, lo, hi = ci(d)
        prow.append(dict(svm=s.replace("SVM-ranked OOF: ", ""), vs=b,
                         mean_diff_mm=pt[0], mean_lo=lo[0], mean_hi=hi[0],
                         median_diff_mm=pt[1], median_lo=lo[1], median_hi=hi[1],
                         det20_diff_pp=pt[2], det20_lo=lo[2], det20_hi=hi[2]))
tab_paired = pd.DataFrame(prow)
tab_auc = pd.DataFrame(aucs_tab)
tab_w = pd.DataFrame(weights_tab)

tab_le.to_csv(OUT / "svm_current_le.csv", index=False)
tab_paired.to_csv(OUT / "svm_current_paired.csv", index=False)
tab_w.to_csv(OUT / "svm_current_weights.csv", index=False)
tab_auc.to_csv(OUT / "svm_current_auc.csv", index=False)


# --------------------------------------------------------------------------------------- markdown (compact)
def f(x, d=2):
    return f"{x:.{d}f}"


def cell(r, k, d=2):
    return f"{f(r[k], d)} [{f(r[k + '_lo'], d)}, {f(r[k + '_hi'], d)}]"


SHORT = {"Pipeline peak (largest blob)": "Pipeline peak", "Plain argmax (image max pixel)": "Plain argmax",
         "Centre guess (shell centre point)": "Centre guess", "Candidate nearest shell centre": "Nearest-centre cand.",
         "Oracle (best candidate)": "Oracle"}
md = ["# SVM candidate verifier on the current pipeline", "",
      f"{len(scans)} scans, {scans['phant_id'].nunique()} phantoms, {len(cand)} candidates "
      f"({int(cand['label'].sum())} positive = centroid within {LABEL_MM:.0f} mm of the tumour). "
      f"Estimate [95% CI], phantom-cluster bootstrap ({N_BOOT}, seed {SEED}); SVM = OOF, {N_SPLITS}-fold GroupKFold by phantom.", "",
      "## Table 1 - LE (mm) and detection (%)", "",
      "| Method | Mean LE | Median LE | det@15 | det@20 |", "|---|---|---|---|---|"]
for _, r in tab_le.iterrows():
    name = SHORT.get(r["method"], r["method"].replace("SVM-ranked OOF: ", "SVM "))
    md.append(f"| {name} | {cell(r,'mean')} | {f(r['median'])} | {cell(r,'det15',1)} | {cell(r,'det20',1)} |")
md += ["", "## Table 2 - SVM minus baseline (mean LE in mm, det@20 in pp)", "",
       "| SVM | vs | Mean LE diff | det@20 diff |", "|---|---|---|---|"]
for _, r in tab_paired.iterrows():
    if r["vs"] in ("Pipeline peak (largest blob)", "Centre guess (shell centre point)"):
        md.append(f"| {r['svm']} | {SHORT[r['vs']]} | {f(r['mean_diff_mm'])} [{f(r['mean_lo'])}, {f(r['mean_hi'])}] | "
                  f"{f(r['det20_diff_pp'],1)} [{f(r['det20_lo'],1)}, {f(r['det20_hi'],1)}] |")
md += ["", "## Table 3 - AUC (OOF) and standardised SVM weights", "",
       "| Combination | AUC fold-mean | AUC overall | " + " | ".join(FEATS) + " |", "|---|---|---|" + "---|" * len(FEATS)]
for _, r in tab_auc.iterrows():
    w = tab_w[tab_w["combo"] == r["combo"]].set_index("feature")["std_weight"]
    md.append(f"| {r['combo']} | {f(r['auc_mean_fold'],3)} | {f(r['auc_overall_oof'],3)} | "
              + " | ".join(f(w[x], 2) if x in w.index else "-" for x in FEATS) + " |")
(OUT / "svm_current_results.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")

print("pipeline LE reproduced vs reconstruction/results/per_scan_metrics.csv: max abs diff", ref_diff)
print("\n".join(md))
