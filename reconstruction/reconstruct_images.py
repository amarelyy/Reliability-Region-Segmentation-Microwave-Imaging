"""
reconstruction/reconstruct_images.py

Reconstruction images + evaluation ON TOP of the ORIGINAL repo pipeline. Nothing in src/, rrs_mbi/ or
unet/ is changed:

  base image = reconstruct_scan(...)["diagnostics"]["image"]   (exactly the call of
               rrs_mbi/evaluator.py: TVSVD, RMS norm, full band, gate 0.70 ns, DAS x CF^1,
               adipose-reference subtraction, shell-centred delay model, breast disc)
  peak       = the pipeline's own peak, the ONLY detection (src.blob_detection: LoG sigma 2-5, Otsu,
               largest blob, weighted centroid on the raw image) -> its LE is the evaluator's LE
  display    = visual enhancement only (MER p, gamma from config.json), never used for the peak

  python reconstruction/reconstruct_images.py --all [--n-jobs 4] [--reuse]   # evaluate every scan ->
                                                       # results/ tables, render the top 10 -> images/
  python reconstruction/reconstruct_images.py --paper  # best / worst / median -> images/paper/
  python reconstruction/reconstruct_images.py --ci     # full bootstrap CIs (per-scan CSV from --all) -> results/evaluation_ci.md
  python reconstruction/reconstruct_images.py          # 5 random scans (seed 42) -> images/
  python reconstruction/reconstruct_images.py --scans 44 220 --gamma-sweep

"Best" (top-k) = mean percentile rank of (a) low LE and (b) a clean display image (peak margin:
1 - strongest competing intensity farther than 10 mm / intensity at the peak). It is chosen with the
ground truth, so these are best-case examples, not representative ones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan
from src import beamforming as bf, signal_processing as sp
import enhancement as en

CFG = json.loads((HERE / "config.json").read_text())
OUT = HERE / "images"
RESULTS = HERE / "results"
DISPLAY_INTERP = "bicubic"   # display only; data and LE are unchanged


# ----------------------------------------------------------------------------------------------
# reconstruction
# ----------------------------------------------------------------------------------------------
def peak_margin(img, axis, region, xy, excl=10.0, near=3.0):
    """1 - (strongest in-breast intensity farther than `excl` mm from xy) / (intensity at xy)."""
    gx, gy = np.meshgrid(axis, axis)
    d = np.hypot(gx - xy[0], gy - xy[1])
    at = (d <= near) & region
    far = region & (d > excl)
    if not at.any() or not far.any() or not img[at].max() > 0:
        return float("nan")
    return float(np.clip(1.0 - img[far].max() / img[at].max(), -1.0, 1.0))


def enhance(res, power, gamma, cfg=CFG):
    """Visual enhancement stages (display layer only)."""
    img, region = res["img"], res["region"]
    mer_img = en.mer(img, power, region)
    final = en.gamma_correct(mer_img, gamma)
    return dict(power=power, gamma=gamma, final=final, stages={
        "Raw (linear norm)": en.raw_normalised(img, region),
        "Percentile clip %d-%d" % tuple(cfg["percentile_clip"]):
            en.percentile_clip(img, *cfg["percentile_clip"], region=region),
        "MER (power=%.1f)" % power: mer_img,
        "MER + gamma(%.2g)" % gamma: final,
    })


def reconstruct(i, data, cfg=CFG, cf_power=None):
    tm = data["tumor_model"]
    # identical call to rrs_mbi/evaluator.run_evaluation()
    r = reconstruct_scan(i, data["s21"], tm, data["id_to_original_idx"], freq_axis=data["freq_axis"],
                         use_tvsvd=True, use_rms_norm=True, bandpass_mode="full", gate_ns=0.70,
                         return_diagnostics=True)
    g = r["diagnostics"]
    axis, img = g["axis_mm"], g["image"]
    centre = (r["shell_cx_mm"], r["shell_cy_mm"])
    R = r["breast_radius_mm"]
    gx, gy = np.meshgrid(axis, axis)
    region = np.hypot(gx - centre[0], gy - centre[1]) <= R

    if cf_power is not None:   # opt-in: changes the displayed image only, the peak stays the pipeline's
        _, _, img = bf.das_coherent_cf(g["time_signal"], sp.get_time_axis(g["time_signal"].shape[0]),
                                       g["delay_grid"], cf_power=cf_power, gate_ns=0.70)
        img = np.where(region, img, 0.0)

    row = tm.iloc[i]
    gt = (float(row["tumor_x_mm"]), float(row["tumor_y_mm"]))
    pipe_xy = (float(r["peak_x_mm"]), float(r["peak_y_mm"]))
    dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    metrics = {k: float(r[k]) for k in ("iou", "dice", "scr_db", "smr_db", "cnr", "cf_at_peak",
                                        "blob_area_px", "blob_compactness")}
    res = dict(i=i, phant_id=row["phant_id"], birads=row.get("birads", np.nan), axis=axis, img=img,
               region=region, centre=centre, R=R, gt=gt, gt_r=float(row["tumor_radius_mm"]),
               pipe_xy=pipe_xy, mask=g["tumor_mask"], m=metrics, cf_power=cf_power,
               LE=float(r["localization_error_mm"]), LE_trivial=dist(centre, gt))
    res.update(enhance(res, cfg["mer_power"], cfg["gamma"], cfg))
    res["margin"] = peak_margin(res["final"], axis, region, pipe_xy)
    return res


def evaluate_row(res):
    axis, gt, c, m = res["axis"], res["gt"], res["centre"], res["m"]
    px = abs(float(axis[1] - axis[0]))
    ix, iy = int(np.argmin(np.abs(axis - gt[0]))), int(np.argmin(np.abs(axis - gt[1])))
    d = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    area = m["blob_area_px"]
    pred_r = float(np.sqrt(area / np.pi) * px) if area > 0 else float("nan")
    return dict(i=res["i"], phant_id=res["phant_id"], birads=res["birads"], tumour_r_mm=res["gt_r"],
                gt_x=gt[0], gt_y=gt[1], shell_cx=c[0], shell_cy=c[1], breast_R=res["R"],
                gt_dist_from_centre=d(gt, c), peak_margin=res["margin"],
                LE=res["LE"], LE_trivial=res["LE_trivial"], peak_x=res["pipe_xy"][0], peak_y=res["pipe_xy"][1],
                peak_dist_from_centre=d(res["pipe_xy"], c), dice=m["dice"], iou=m["iou"], scr_db=m["scr_db"],
                smr_db=m["smr_db"], cnr=m["cnr"], cf_at_peak=m["cf_at_peak"], blob_area_px=area,
                compactness=m["blob_compactness"], gt_in_blob=float(bool(res["mask"][iy, ix])),
                size_err_mm=abs(pred_r - res["gt_r"]))


def _chunk(indices, s21, tm, idm, freq_axis):
    data = dict(s21=s21, tumor_model=tm, id_to_original_idx=idm, freq_axis=freq_axis)
    rows = []
    for i in indices:
        try:
            rows.append(evaluate_row(reconstruct(int(i), data)))
        except Exception as e:   # keep going, report explicitly
            rows.append(dict(i=int(i), error=str(e)))
    return rows


def evaluate_all(data, n_jobs=4):
    from joblib import Parallel, delayed
    tm = data["tumor_model"]
    order = tm.sort_values("phant_id").index.to_numpy()        # grouped by phantom -> delay-cache hits
    chunks = [c for c in np.array_split(order, n_jobs * 4) if len(c)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_chunk)(c, data["s21"], tm, data["id_to_original_idx"], data["freq_axis"]) for c in chunks)
    df = pd.DataFrame([r for rows in res for r in rows]).sort_values("i").reset_index(drop=True)
    if "error" in df and df["error"].notna().any():
        print("ERRORS:", df.loc[df["error"].notna(), ["i", "error"]].to_string())
        df = df[df["error"].isna()].drop(columns="error").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
def md(df, floatfmt="%.2f"):
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx in df.index:
        cells = [str(idx)]
        for c in df.columns:
            v = df.at[idx, c]
            if isinstance(v, (float, np.floating)):
                cells.append((floatfmt % v) if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def mode_bin(x, width=5.0):
    x = np.asarray(x, float)
    bins = np.arange(0, np.ceil(x.max() / width) * width + width, width)
    h, _ = np.histogram(x, bins)
    k = int(h.argmax())
    return "%.0f-%.0f (n=%d)" % (bins[k], bins[k + 1], h[k])


def describe(x):
    x = pd.Series(x).dropna()
    return dict(min=x.min(), p25=x.quantile(.25), median=x.median(), mean=x.mean(), p75=x.quantile(.75),
                max=x.max(), sd=x.std())


def build_tables(df):
    from rrs_mbi.evaluator import cluster_bootstrap
    d = df.rename(columns={"phant_id": "scan_id"})
    parts = ["Scans evaluated: %d, phantoms: %d. Original pipeline image and peak (enhancement is display only)." %
             (len(df), d["scan_id"].nunique()), ""]

    rows = {}
    for name, col in (("Pipeline peak", "LE"), ("Centre guess (trivial baseline)", "LE_trivial")):
        x = df[col]
        bs = cluster_bootstrap(d, lambda t, c=col: float(t[c].mean()), n_boot=2000)
        rows[name] = dict(describe(x), mode_5mm=mode_bin(x),
                          mean_CI95="[%.1f, %.1f]" % tuple(np.percentile(bs, [2.5, 97.5])),
                          det10=(x <= 10).mean() * 100, det15=(x <= 15).mean() * 100, det20=(x <= 20).mean() * 100)
    A = pd.DataFrame(rows).T[["min", "p25", "median", "mean", "mean_CI95", "p75", "max", "sd", "mode_5mm",
                              "det10", "det15", "det20"]]
    A.index.name = "Localisation error (mm)"
    parts += [md(A.rename(columns={"det10": "det@10 (%)", "det15": "det@15 (%)", "det20": "det@20 (%)"})), ""]

    spec = [("SCR (dB)", "scr_db"), ("SMR (dB)", "smr_db"), ("CNR", "cnr"), ("Dice", "dice"), ("IoU", "iou"),
            ("Blob area (px, 1 mm/px)", "blob_area_px"), ("Size error (mm)", "size_err_mm"),
            ("CF at peak", "cf_at_peak"), ("Peak distance from centre (mm)", "peak_dist_from_centre"),
            ("GT distance from centre (mm)", "gt_dist_from_centre"),
            ("Peak margin (cleanliness, 1 = clean)", "peak_margin")]
    B = {nm: {k: v for k, v in describe(df[key]).items() if k in ("mean", "median", "min", "max", "sd")}
         for nm, key in spec}
    Bdf = pd.DataFrame(B).T[["mean", "median", "min", "max", "sd"]]
    Bdf.index.name = "Image quality and segmentation (pipeline blob)"
    parts += [md(Bdf), ""]
    rates = pd.DataFrame({"GT inside blob (% scans)": [df["gt_in_blob"].mean() * 100],
                          "Size error <= 5 mm (% scans)": [(df["size_err_mm"] <= 5).mean() * 100],
                          "Dice = 0 (% scans)": [(df["dice"] == 0).mean() * 100]}, index=["pipeline blob"])
    rates.index.name = "Blob"
    parts += [md(rates), ""]

    fn = lambda t: float((t["LE"] - t["LE_trivial"]).mean())
    bs = cluster_bootstrap(d, fn, n_boot=2000)
    P = pd.DataFrame({"Difference (mm)": ["%+.2f" % fn(d)],
                      "95% CI (phantom-cluster bootstrap)": ["[%+.2f, %+.2f]" % tuple(np.percentile(bs, [2.5, 97.5]))],
                      "Pipeline better (% scans)": [(df["LE"] < df["LE_trivial"]).mean() * 100]},
                     index=["pipeline - centre guess"])
    P.index.name = "Paired mean LE"
    parts += [md(P), "Positive means the pipeline is worse than the centre guess on average.", ""]

    def grp(g):
        return pd.DataFrame({"n": g.size(), "mean LE (mm)": g["LE"].mean(),
                             "mean centre-guess LE (mm)": g["LE_trivial"].mean(),
                             "det@15 (%)": g["LE"].apply(lambda v: (v <= 15).mean() * 100),
                             "mean Dice": g["dice"].mean(), "mean IoU": g["iou"].mean(),
                             "mean SCR (dB)": g["scr_db"].mean(), "mean SMR (dB)": g["smr_db"].mean()})
    C1 = grp(df.groupby("tumour_r_mm"))
    C1.index = C1.index.astype(int)
    C2 = grp(df.groupby(pd.cut(df["gt_dist_from_centre"], [-0.1, 10, 20, 30, 40, 200],
                               labels=["0-10", "10-20", "20-30", "30-40", ">40"]), observed=True))
    C1.index.name, C2.index.name = "Tumour radius (mm)", "True distance from centre (mm)"
    parts += [md(C2), "", md(C1), ""]
    tabs = dict(A=A, B=Bdf, R=rates, P=P, C2=C2, C1=C1)
    has_birads = df["birads"].notna()
    if has_birads.any():
        C3 = grp(df[has_birads].groupby("birads"))
        C3.index = C3.index.astype(int)
        C3.index.name = "BI-RADS"
        parts += [md(C3), "%d scans have no BI-RADS and are excluded (n = %d)." %
                  ((~has_birads).sum(), has_birads.sum()), ""]
        tabs["C3"] = C3
    return "\n".join(parts), tabs


# ----------------------------------------------------------------------------------------------
# full bootstrap confidence intervals (phantom-cluster percentile bootstrap)
# ----------------------------------------------------------------------------------------------
BOOT_SEED = 42
QUALITY = [("SCR (dB)", "scr_db"), ("SMR (dB)", "smr_db"), ("CNR", "cnr"), ("Dice", "dice"), ("IoU", "iou"),
           ("Blob area (px)", "blob_area_px"), ("Size error (mm)", "size_err_mm"), ("CF at peak", "cf_at_peak"),
           ("Peak distance from centre (mm)", "peak_dist_from_centre")]


def _clusters(phant):
    """Row-index arrays, one per phantom (scans of one phantom are not independent)."""
    codes, uniq = pd.factorize(pd.Series(phant).reset_index(drop=True))
    return [np.flatnonzero(codes == k) for k in range(len(uniq))]


def bootstrap_ci(stats, clusters, n_boot=2000, seed=BOOT_SEED):
    """Phantom-cluster percentile bootstrap. `stats` maps name -> f(ix) over row indices. Each resample
    draws phantoms with replacement and recomputes EVERY statistic on the pooled rows.
    Returns name -> (estimate, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    n_cl = len(clusters)
    est = {k: float(f(np.concatenate(clusters))) for k, f in stats.items()}
    draws = {k: np.empty(n_boot) for k in stats}
    for b in range(n_boot):
        ix = np.concatenate([clusters[j] for j in rng.integers(0, n_cl, n_cl)])
        for k, f in stats.items():
            draws[k][b] = f(ix)
    return {k: (est[k], *np.nanpercentile(draws[k], [2.5, 97.5])) for k in stats}


def _fmt(t):
    est, lo, hi = t
    big = max(abs(est), abs(lo), abs(hi))
    dec = 3 if big < 1 else (1 if big >= 100 else 2)
    return "%.*f [%.*f, %.*f]" % (dec, est, dec, lo, dec, hi)


def _arrays(d):
    return {c: d[c].to_numpy(float) for c in d.columns if d[c].dtype.kind in "fiub"}


def _overall_stats(A):
    S = {}
    for tag, col in (("pipe", "LE"), ("centre", "LE_trivial")):
        x = A[col]
        S[tag + "|mean"] = lambda ix, x=x: x[ix].mean()
        S[tag + "|median"] = lambda ix, x=x: np.median(x[ix])
        S[tag + "|p25"] = lambda ix, x=x: np.percentile(x[ix], 25)
        S[tag + "|p75"] = lambda ix, x=x: np.percentile(x[ix], 75)
        S[tag + "|sd"] = lambda ix, x=x: x[ix].std(ddof=1)
        for t in (10, 15, 20):
            S[tag + "|det%d" % t] = lambda ix, x=x, t=t: (x[ix] <= t).mean() * 100
    a, b = A["LE"], A["LE_trivial"]
    S["diff|mean"] = lambda ix: (a[ix] - b[ix]).mean()
    S["diff|median"] = lambda ix: np.median(a[ix] - b[ix])
    S["diff|better"] = lambda ix: (a[ix] < b[ix]).mean() * 100
    S["diff|det15"] = lambda ix: ((a[ix] <= 15).mean() - (b[ix] <= 15).mean()) * 100
    for _, col in QUALITY:
        x = A[col]
        S[col + "|mean"] = lambda ix, x=x: x[ix].mean()
        S[col + "|median"] = lambda ix, x=x: np.median(x[ix])
    S["rate|gt_in_blob"] = lambda ix: A["gt_in_blob"][ix].mean() * 100
    S["rate|size5"] = lambda ix: (A["size_err_mm"][ix] <= 5).mean() * 100
    S["rate|dice0"] = lambda ix: (A["dice"][ix] == 0).mean() * 100
    return S


def _group_stats(A):
    a, b = A["LE"], A["LE_trivial"]
    return {"LE": lambda ix: a[ix].mean(), "centre": lambda ix: b[ix].mean(),
            "diff": lambda ix: (a[ix] - b[ix]).mean(), "det15": lambda ix: (a[ix] <= 15).mean() * 100,
            "dice": lambda ix: A["dice"][ix].mean(), "scr": lambda ix: A["scr_db"][ix].mean()}


def build_ci_tables(df, n_boot=2000):
    """All evaluation numbers with phantom-cluster bootstrap 95% CIs. Returns (markdown, long DataFrame)."""
    parts = ["Phantom-cluster percentile bootstrap: %d resamples of %d phantoms drawn with replacement (seed %d); "
             "every statistic is recomputed in each resample. Format: estimate [95%% CI]. Group tables resample "
             "the phantoms that contribute to that group." % (n_boot, df["phant_id"].nunique(), BOOT_SEED), ""]
    long = []

    R = bootstrap_ci(_overall_stats(_arrays(df)), _clusters(df["phant_id"]), n_boot)
    for k, t in R.items():
        long.append(dict(table="overall", row=k.split("|")[0], stat=k.split("|")[-1], estimate=t[0], ci_lo=t[1], ci_hi=t[2]))

    labs = [("mean", "mean"), ("median", "median"), ("p25", "p25"), ("p75", "p75"), ("SD", "sd"),
            ("det@10 (%)", "det10"), ("det@15 (%)", "det15"), ("det@20 (%)", "det20")]
    T1 = pd.DataFrame({name: {lab: _fmt(R[tag + "|" + k]) for lab, k in labs}
                       for name, tag in (("Pipeline peak", "pipe"), ("Centre guess", "centre"))}).T
    T1.index.name = "Localisation error (mm)"
    T2 = pd.DataFrame({"Mean diff (mm)": [_fmt(R["diff|mean"])], "Median diff (mm)": [_fmt(R["diff|median"])],
                       "Pipeline better (% scans)": [_fmt(R["diff|better"])],
                       "det@15 diff (pct. points)": [_fmt(R["diff|det15"])]}, index=["pipeline - centre guess"])
    T2.index.name = "Paired difference"
    T3 = pd.DataFrame({"mean [95% CI]": [_fmt(R[c + "|mean"]) for _, c in QUALITY],
                       "median [95% CI]": [_fmt(R[c + "|median"]) for _, c in QUALITY]},
                      index=[lab for lab, _ in QUALITY])
    T3.index.name = "Image quality and segmentation (pipeline blob)"
    T4 = pd.DataFrame({"GT inside blob (% scans)": [_fmt(R["rate|gt_in_blob"])],
                       "Size error <= 5 mm (% scans)": [_fmt(R["rate|size5"])],
                       "Dice = 0 (% scans)": [_fmt(R["rate|dice0"])]}, index=["pipeline blob"])
    T4.index.name = "Blob"
    parts += [md(T1), "", md(T2), "Positive means the pipeline is worse than the centre guess.", "", md(T3), "", md(T4), ""]

    def group_table(title, groups):
        rows = {}
        for label, sub in groups:
            r = bootstrap_ci(_group_stats(_arrays(sub)), _clusters(sub["phant_id"]), n_boot)
            rows[label] = {"n": len(sub), "phantoms": sub["phant_id"].nunique(),
                           "mean LE (mm)": _fmt(r["LE"]), "centre-guess LE (mm)": _fmt(r["centre"]),
                           "diff (mm)": _fmt(r["diff"]), "det@15 (%)": _fmt(r["det15"]),
                           "mean Dice": _fmt(r["dice"]), "mean SCR (dB)": _fmt(r["scr"])}
            for k, t in r.items():
                long.append(dict(table=title, row=label, stat=k, estimate=t[0], ci_lo=t[1], ci_hi=t[2]))
        T = pd.DataFrame(rows).T
        T.index.name = title
        return T

    edges, names = [-0.1, 10, 20, 30, 40, 200], ["0-10", "10-20", "20-30", "30-40", ">40"]
    band = pd.cut(df["gt_dist_from_centre"], edges, labels=names)
    parts += [md(group_table("True distance from centre (mm)", [(nm, df[band == nm]) for nm in names])), ""]
    parts += [md(group_table("Tumour radius (mm)",
                             [(str(int(r)), df[df["tumour_r_mm"] == r]) for r in sorted(df["tumour_r_mm"].unique())])), ""]
    has = df["birads"].notna()
    parts += [md(group_table("BI-RADS", [(str(int(b)), df[df["birads"] == b]) for b in sorted(df.loc[has, "birads"].unique())])),
              "%d scans have no BI-RADS and are excluded (n = %d)." % ((~has).sum(), has.sum()), ""]
    return "\n".join(parts), pd.DataFrame(long)


def select_top(df, k=10):
    lr = df["LE"].rank(pct=True)                                    # low LE = good
    cr = (-df["peak_margin"]).rank(pct=True, na_option="bottom")    # high margin = clean = good
    return df.assign(rank_score=(lr + cr) / 2).nsmallest(k, "rank_score").reset_index(drop=True)


def paper_scans(df):
    """Best / worst / median case by LE (median = closest to the median LE)."""
    med = df["LE"].median()
    return {"best": int(df.loc[df["LE"].idxmin(), "i"]), "worst": int(df.loc[df["LE"].idxmax(), "i"]),
            "median": int(df.loc[(df["LE"] - med).abs().idxmin(), "i"])}


# ----------------------------------------------------------------------------------------------
# plotting (turbo, black background, lime dashed GT circle, white + = pipeline peak)
# ----------------------------------------------------------------------------------------------
def _style_axes(ax):
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("white")


def _crop(ax, res, margin=1.15):
    lim = margin * res["R"]
    ax.set_xlim(res["centre"][0] - lim, res["centre"][0] + lim)
    ax.set_ylim(res["centre"][1] - lim, res["centre"][1] + lim)


def _gt_circle(ax, res, label=True):
    ax.add_patch(plt.Circle(res["gt"], res["gt_r"], fill=False, edgecolor="lime", linewidth=2, linestyle="--",
                            label=("GT (%.0f,%.0f) r=%.0fmm" % (*res["gt"], res["gt_r"])) if label else None))


def _show(ax, res, image):
    a = res["axis"]
    return ax.imshow(image, extent=[a[0], a[-1], a[0], a[-1]], origin="lower", cmap="turbo", vmin=0, vmax=1,
                     aspect="equal", interpolation=DISPLAY_INTERP)


def plot_main(res, path, rank=None, dpi=200, label=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = _show(ax, res, res["final"])
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Enhanced Intensity", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    _gt_circle(ax, res)
    ax.plot(*res["pipe_xy"], "w+", markersize=14, markeredgewidth=2,
            label="Det, pipeline peak (%.0f,%.0f)" % res["pipe_xy"])
    _crop(ax, res)
    b = "" if np.isnan(res["birads"]) else " | BI-RADS %.0f" % res["birads"]
    cf = "" if res["cf_power"] is None else " [cf_power=%.1f]" % res["cf_power"]
    rk = "" if rank is None else "#%d  " % rank
    if label:
        rk = label + " | "
    ax.set_title("%sDAS+CF (MER p=%.1f, gamma=%.2g)%s | %s%s\nLE=%.1fmm | centre guess=%.1fmm" %
                 (rk, res["power"], res["gamma"], cf, res["phant_id"], b, res["LE"], res["LE_trivial"]),
                 fontsize=10.5, fontweight="bold", color="white")
    ax.set_xlabel("X (mm)", color="white")
    ax.set_ylabel("Y (mm)", color="white")
    _style_axes(ax)
    ax.legend(loc="upper right", fontsize=8, facecolor="black", edgecolor="white", labelcolor="white")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def plot_steps(res, path):
    names = list(res["stages"])
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, name in zip(axes[:4], names):
        _show(ax, res, res["stages"][name])
        _gt_circle(ax, res, label=False)
        ax.set_title(name, color="white", fontsize=10)
    ax = axes[4]
    _show(ax, res, res["final"])
    a = res["axis"]
    ax.contour(a, a, res["mask"].astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    _gt_circle(ax, res, label=False)
    ax.plot(*res["pipe_xy"], "w+", markersize=15, markeredgewidth=2.5)
    ax.set_title("pipeline blob (LoG 2-5 > Otsu > largest) and peak (+)", color="white", fontsize=9)
    for ax in axes:
        _crop(ax, res)
        ax.set_xticks([]); ax.set_yticks([])
        _style_axes(ax)
    fig.suptitle("scan %d | %s | tumour r=%.0f mm (green dashed) | LE %.1f mm" %
                 (res["i"], res["phant_id"], res["gt_r"], res["LE"]),
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)


def plot_paper(results, labels, path, dpi=300):
    """One row, one panel per case, shared colorbar, same style as plot_main."""
    fig, axes = plt.subplots(1, len(results), figsize=(6.6 * len(results) + 1.2, 6.4), squeeze=False)
    axes = axes[0]
    for k, (ax, res, lab) in enumerate(zip(axes, results, labels)):
        im = _show(ax, res, res["final"])
        ax.add_patch(plt.Circle(res["gt"], res["gt_r"], fill=False, edgecolor="lime", linewidth=2, linestyle="--",
                                label="True tumour" if k == 0 else None))
        ax.plot(*res["pipe_xy"], "w+", markersize=14, markeredgewidth=2, label="Pipeline peak" if k == 0 else None)
        _crop(ax, res)
        ax.set_title("(%s) %s | %s, scan %d\nLE %.1f mm (centre guess %.1f mm)" %
                     (chr(97 + k), lab, res["phant_id"], res["i"], res["LE"], res["LE_trivial"]),
                     fontsize=11, fontweight="bold", color="white")
        ax.set_xlabel("X (mm)", color="white")
        if k == 0:
            ax.set_ylabel("Y (mm)", color="white")
        _style_axes(ax)
    axes[0].legend(loc="upper right", fontsize=9, facecolor="black", edgecolor="white", labelcolor="white")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.012)
    cb.set_label("Enhanced Intensity", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


SWEEP = [(2.0, 1.0), (2.0, 0.7), (2.0, 0.5), (2.0, 0.35), (2.0, 0.25)]   # net exponent = p * gamma


def gamma_sweep(results, path_png, path_csv):
    fig, axes = plt.subplots(len(results), len(SWEEP), figsize=(4.3 * len(SWEEP), 4.3 * len(results)),
                             squeeze=False)
    rows = []
    for r_i, res in enumerate(results):
        for c_i, (p, g) in enumerate(SWEEP):
            v = enhance(res, p, g)
            ax = axes[r_i][c_i]
            _show(ax, res, v["final"])
            _gt_circle(ax, res, label=False)
            ax.plot(*res["pipe_xy"], "w+", markersize=12, markeredgewidth=2)
            _crop(ax, res, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
            _style_axes(ax)
            bg = float(np.median(v["final"][res["region"]]))
            ax.set_title("p=%.0f, gamma=%.2g  (net exponent %.2g)\nbackground median %.3f" % (p, g, p * g, bg),
                         color="white", fontsize=8.5)
            if c_i == 0:
                ax.set_ylabel("scan %d  %s" % (res["i"], res["phant_id"]), color="white", fontsize=10)
            rows.append(dict(scan=res["i"], phant_id=res["phant_id"], mer_power=p, gamma=g, net_exponent=p * g,
                             background_median=bg))
    fig.suptitle("Effect of gamma after MER p=2 (display layer only; the peak does not change)", color="white",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path_png, dpi=130, facecolor="black")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(path_csv, index=False)


# ----------------------------------------------------------------------------------------------
def render(res, rank=None):
    prefix = "" if rank is None else "top%02d_" % rank
    tag = "" if res["cf_power"] is None else "_cf%.1f" % res["cf_power"]
    plot_main(res, OUT / ("%sscan_%03d_%s%s_LE%.0fmm.png" % (prefix, res["i"], res["phant_id"], tag, res["LE"])), rank)
    (OUT / "steps").mkdir(parents=True, exist_ok=True)
    plot_steps(res, OUT / "steps" / ("%sscan_%03d_%s%s_steps.png" % (prefix, res["i"], res["phant_id"], tag)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scans", type=int, nargs="*")
    ap.add_argument("--all", action="store_true", help="evaluate every scan, render the top-k")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse results/per_scan_metrics.csv")
    ap.add_argument("--paper", action="store_true", help="best / worst / median -> images/paper/")
    ap.add_argument("--ci", action="store_true",
                    help="full phantom-cluster bootstrap CIs from results/per_scan_metrics.csv -> results/evaluation_ci.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--gamma-sweep", action="store_true")
    ap.add_argument("--cf-power", type=float, nargs="?", const=CFG["cf_power"], default=None,
                    help="opt-in: display image re-beamformed with this CF exponent (peak unchanged)")
    args = ap.parse_args()

    if args.ci:   # needs only the per-scan table, no data loading
        RESULTS.mkdir(parents=True, exist_ok=True)
        text, long = build_ci_tables(pd.read_csv(RESULTS / "per_scan_metrics.csv"), args.n_boot)
        (RESULTS / "evaluation_ci.md").write_text(text + "\n", encoding="utf-8")
        long.to_csv(RESULTS / "ci_long.csv", index=False)
        print(text)
        return

    data = load_all_data()
    tm = data["tumor_model"]
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / "per_scan_metrics.csv"

    if args.paper:
        pdir = OUT / "paper"
        pdir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(cache)
        pick = paper_scans(df)
        res = {k: reconstruct(i, data) for k, i in pick.items()}
        for n, k in enumerate(("best", "worst", "median"), 1):
            r = res[k]
            plot_main(r, pdir / ("%d_%s_scan_%03d_%s_LE%.0fmm.png" % (n, k, r["i"], r["phant_id"], r["LE"])),
                      dpi=300, label=k.capitalize() + " case")
        order = ("best", "median", "worst")
        plot_paper([res[k] for k in order], [k.capitalize() for k in order], pdir / "figure_best_median_worst.png")
        rows = df[df["i"].isin(pick.values())].copy()
        rows.insert(0, "case", rows["i"].map({v: k for k, v in pick.items()}))
        rows.to_csv(pdir / "paper_scans.csv", index=False)
        print(rows.set_index("case").T.to_string())
        print("saved ->", pdir)
        return

    if args.all:
        if args.reuse and cache.exists():
            df = pd.read_csv(cache)
        else:
            df = evaluate_all(data, args.n_jobs)
            df.to_csv(cache, index=False)
        text, tabs = build_tables(df)
        (RESULTS / "evaluation_tables.md").write_text(text + "\n", encoding="utf-8")
        for k, t in tabs.items():
            t.to_csv(RESULTS / ("table_%s.csv" % k))
        top = select_top(df, args.top)
        top.to_csv(RESULTS / ("top%d.csv" % args.top), index=False)
        print(text)
        print("\nTOP %d:" % args.top)
        print(top[["i", "phant_id", "tumour_r_mm", "LE", "LE_trivial", "peak_margin", "gt_dist_from_centre",
                   "rank_score"]].round(2).to_string(index=False))
        if not args.metrics_only:
            for rank, i in enumerate(top["i"], 1):
                render(reconstruct(int(i), data), rank)
            print("rendered %d images -> %s" % (len(top), OUT))
        return

    scans = args.scans or sorted(int(k) for k in np.random.default_rng(args.seed).choice(len(tm), args.n, replace=False))
    print("scans:", scans)
    results = []
    for i in scans:
        res = reconstruct(i, data, cf_power=args.cf_power)
        results.append(res)
        if not args.gamma_sweep:
            render(res)
            print("  scan %d (%s): LE %.1f mm" % (i, res["phant_id"], res["LE"]), flush=True)
    if args.gamma_sweep:
        gamma_sweep(results, OUT / "gamma_sweep.png", OUT / "gamma_sweep.csv")
        print("saved", OUT / "gamma_sweep.png")


if __name__ == "__main__":
    main()
