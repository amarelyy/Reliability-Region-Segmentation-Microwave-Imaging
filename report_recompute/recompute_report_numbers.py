"""
report_recompute/recompute_report_numbers.py

Recomputes three stale report items from the CURRENT pipeline over all 504 scans (nothing retuned,
pipeline untouched):
  1. SCR evaluated AT THE GROUND-TRUTH tumour location vs true tumour distance (+ figure)
  3. central concentration of the DAS+CF image

Every variant uses the same call as the main results:
  reconstruct_scan(..., use_tvsvd=True, use_rms_norm=True, bandpass_mode="full", gate_ns=0.70)
Bootstrap = reconstruction.reconstruct_images.bootstrap_ci (phantom clusters, 2000, seed 42).

Run from the repo root:  python report_recompute/recompute_report_numbers.py --n-jobs 4 [--reuse]
Outputs: report_recompute/per_scan.csv, report_numbers.csv, report_numbers.md, metrics_image/signal_death_curve.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reconstruction"))

from src.data_loading import load_all_data                       # noqa: E402
from src.pipeline import reconstruct_scan                        # noqa: E402
from rrs_mbi.metrics import extract_breast_mask, calculate_central_concentration   # noqa: E402
import reconstruct_images as ri                                  # noqa: E402  (bootstrap helpers)

VARIANTS = {"DAS+CF": ("das", True)}   # main pipeline only (no beamformer ablation)
N_BOOT = 2000


def gt_disc_scr(img, axis, x, y, r, region):
    """10*log10( mean(img in disc of the TRUE tumour radius around the GT) / mean(img in rest of breast) )."""
    gx, gy = np.meshgrid(axis, axis)
    disc = ((gx - x) ** 2 + (gy - y) ** 2 <= max(r, 1.0) ** 2) & region
    if disc.sum() == 0:
        return np.nan
    return float(10 * np.log10(img[disc].mean() / (img[region & ~disc].mean() + 1e-12)))


def gt_roi_scr_legacy(img, axis, x, y, region, roi=2.0):
    """Legacy baseline_argmax_test definition: 20*log10( max(img within 2 mm of GT) / mean(img elsewhere in breast) )."""
    gx, gy = np.meshgrid(axis, axis)
    d = ((gx - x) ** 2 + (gy - y) ** 2 <= roi ** 2) & region
    if d.sum() == 0:
        return np.nan
    return float(20 * np.log10(img[d].max() / (img[region & ~d].mean() + 1e-12)))


def work(i, s21, tm, idm, freq_axis):
    out = dict(i=i)
    for name, (bfm, cf) in VARIANTS.items():
        r = reconstruct_scan(i, s21, tm, idm, freq_axis=freq_axis, beamformer=bfm, use_cf=cf, use_tvsvd=True,
                             use_rms_norm=True, bandpass_mode="full", gate_ns=0.70,
                             return_diagnostics=(name == "DAS+CF"))
        out[name + "|le"] = float(r["localization_error_mm"])
        out[name + "|scr"] = float(r["scr_db"])
        out[name + "|runtime"] = float(r["runtime_sec"])
        if name == "DAS+CF":
            g = r["diagnostics"]
            img, axis = g["image"], g["axis_mm"]
            sx, sy, R = float(r["shell_cx_mm"]), float(r["shell_cy_mm"]), float(r["breast_radius_mm"])
            gx, gy, gr = float(r["gt_x_mm"]), float(r["gt_y_mm"]), float(r["gt_r_mm"])
            xx, yy = np.meshgrid(axis, axis)
            region = np.hypot(xx - sx, yy - sy) <= R
            m_img = extract_breast_mask(img)
            out.update(
                phant_id=r["phant_id"], gt_x=gx, gt_y=gy, gt_r=gr, shell_cx=sx, shell_cy=sy, breast_R=R,
                gt_dist_shell=float(np.hypot(gx - sx, gy - sy)), gt_dist_array=float(np.hypot(gx, gy)),
                scr_gt_disc_db=gt_disc_scr(img, axis, gx, gy, gr, region),
                scr_gt_roi2mm_legacy_db=gt_roi_scr_legacy(img, axis, gx, gy, region),
                cc_main=calculate_central_concentration(img, m_img, axis, (sx, sy)),          # evaluator definition
                cc_array_centre=calculate_central_concentration(img, m_img, axis, (0.0, 0.0)),
                cc_disc_mask=calculate_central_concentration(img, region, axis, (sx, sy)))
    return out


def _chunk(idx, s21, tm, idm, freq_axis):
    rows, errs = [], []
    for i in idx:
        try:
            rows.append(work(int(i), s21, tm, idm, freq_axis))
        except Exception as e:
            errs.append((int(i), repr(e)))
    return rows, errs


def fmt(t, d=2):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse per_scan.csv")
    args = ap.parse_args()
    csv = HERE / "per_scan.csv"

    if args.reuse and csv.exists():
        df = pd.read_csv(csv)
    else:
        from joblib import Parallel, delayed
        data = load_all_data()
        tm = data["tumor_model"]
        order = tm.sort_values("phant_id").index.to_numpy()
        chunks = [c for c in np.array_split(order, args.n_jobs * 4) if len(c)]
        res = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_chunk)(c, data["s21"], tm, data["id_to_original_idx"], data["freq_axis"]) for c in chunks)
        errs = [e for _, es in res for e in es]
        if errs:
            print("ERRORS:", errs)
            sys.exit("STOP: some scans failed")
        df = pd.DataFrame([r for rows, _ in res for r in rows]).sort_values("i").reset_index(drop=True)
        df.to_csv(csv, index=False)
    assert len(df) == 504 and df["phant_id"].nunique() == 66, (len(df), df["phant_id"].nunique())

    cl = ri._clusters(df["phant_id"])
    A = {c: df[c].to_numpy(float) for c in df.columns if df[c].dtype.kind in "fiub"}
    long, md = [], ["# Recomputed report numbers (current pipeline, 504 scans, 66 phantoms)", "",
                    f"Estimate [95% CI]: phantom-cluster percentile bootstrap ({N_BOOT} resamples, seed {ri.BOOT_SEED}), "
                    "same routine as reconstruction/results/evaluation_ci.md.", ""]

    # ---------------------------------------------------------------- item 1: SCR at GT vs distance
    d, s, s_leg = A["gt_dist_shell"], A["scr_gt_disc_db"], A["scr_gt_roi2mm_legacy_db"]
    ok = np.isfinite(s) & np.isfinite(d)
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 1e9)]
    names = ["0-10", "10-20", "20-30", "30-40", ">40"]
    rows = []
    for (lo, hi), nm in zip(bins, names):
        m = ok & (d >= lo) & (d < hi)
        ix = np.flatnonzero(m)
        sub_cl = ri._clusters(df.loc[m, "phant_id"])
        t = ri.bootstrap_ci({"m": lambda j, v=s[m]: v[j].mean()}, sub_cl, N_BOOT)["m"] if m.sum() > 1 else (np.nan,) * 3
        rows.append(dict(distance_bin_mm=nm, n=int(m.sum()), phantoms=int(df.loc[m, "phant_id"].nunique()),
                         scr_gt_disc_mean=s[m].mean(), scr_ci=fmt(t), scr_legacy_roi2mm_mean=np.nanmean(s_leg[m]), n_legacy=int(np.isfinite(s_leg[m]).sum()),
                         blob_scr_mean=A["DAS+CF|scr"][m].mean()))
    t1 = pd.DataFrame(rows)
    # OLS zero-crossing + binned crossing
    slope, icpt = np.polyfit(d[ok], s[ok], 1)
    zero_ols = -icpt / slope
    mids5 = np.arange(2.5, d.max() + 5, 5.0)
    b5 = pd.DataFrame([dict(mid=mm, n=int(((d >= mm - 2.5) & (d < mm + 2.5) & ok).sum()),
                            mean=s[(d >= mm - 2.5) & (d < mm + 2.5) & ok].mean()) for mm in mids5]).query("n >= 3")
    cross_bin = np.nan
    for a, b in zip(b5.itertuples(), list(b5.itertuples())[1:]):
        if a.mean > 0 >= b.mean:
            cross_bin = a.mid + (a.mean - 0) / (a.mean - b.mean) * (b.mid - a.mid)
            break
    d_mean = float(d.mean())          # mean true tumour distance from the shell centre (= centre-guess LE mean)
    near = ok & (np.abs(d - d_mean) <= 2.5)
    r_p = float(np.corrcoef(d[ok], s[ok])[0, 1])
    md += ["## 1. SCR evaluated at the ground-truth location vs true tumour distance", "",
           "Definition: image = DAS x CF (breast-masked, same as the main results); signal = disc of the TRUE tumour radius "
           "(min 1 mm) centred on the ground truth; clutter = all other pixels of the breast disc (radius R, shell centre); "
           "SCR = 10 log10(mean signal / mean clutter). Distance = true tumour distance from the estimated shell centre.", "",
           "| True distance (mm) | n | phantoms | mean SCR at GT (dB) [95% CI] | mean legacy SCR (dB, 20log10 peak-in-2mm / mean; valid n) | mean blob SCR (dB) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['distance_bin_mm']} | {r['n']} | {r['phantoms']} | {r['scr_ci']} | {r['scr_legacy_roi2mm_mean']:.2f} (n={r['n_legacy']}) | {r['blob_scr_mean']:.2f} |")
    pred = icpt + slope * d_mean
    md += ["", f"- OLS fit: SCR = {icpt:.2f} + ({slope:.3f}) x distance, Pearson r = {r_p:.2f}; zero crossing (OLS) = {zero_ols:.1f} mm.",
           f"- Zero crossing from 5-mm binned means (linear interpolation) = {cross_bin:.1f} mm.",
           f"- SCR at the mean tumour distance ({d_mean:.2f} mm): OLS {pred:.2f} dB; mean of scans within +-2.5 mm = {s[near].mean():.2f} dB (n={int(near.sum())}).",
           f"- Excluded from item 1: {int((~ok).sum())} scans whose ground-truth disc lies entirely outside the breast disc (true distance > R); the legacy 2-mm-ROI column additionally drops scans whose ROI has no breast pixel.",
           f"- Scans with SCR at GT < 0 dB: {np.mean(s[ok] < 0) * 100:.1f}%; min {s[ok].min():.1f} dB, max {s[ok].max():.1f} dB.", ""]
    for r in rows:
        long.append(dict(item=1, row=f"SCR_at_GT bin {r['distance_bin_mm']} mm", n=r["n"], value=r["scr_ci"]))
    long += [dict(item=1, row="OLS slope dB/mm", n=int(ok.sum()), value=f"{slope:.4f}"),
             dict(item=1, row="OLS zero crossing mm", n=int(ok.sum()), value=f"{zero_ols:.2f}"),
             dict(item=1, row="binned zero crossing mm", n=int(ok.sum()), value=f"{cross_bin:.2f}"),
             dict(item=1, row=f"SCR at {d_mean:.2f} mm (OLS)", n=int(ok.sum()), value=f"{pred:.2f}")]

    # figure (same filename and style as the old metrics_image/signal_death_curve.png, dpi 300)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
                         "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d[ok], s[ok], color="#1f77b4", alpha=0.6, s=25, edgecolor="white", linewidth=0.5, label="Scan Measurements")
    xt = np.linspace(d[ok].min(), d[ok].max(), 100)
    ax.plot(xt, icpt + slope * xt, color="red", linewidth=2.0, label=f"Degradation Trendline (Slope: {slope:.2f} dB/mm)")
    ax.plot(b5["mid"], b5["mean"], "ks-", ms=5, lw=1.2, label="Mean per 5 mm bin")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Detectability Threshold (0 dB)")
    ax.set_title("Signal Death Curve: SCR vs. Tumor Radial Distance")
    ax.set_xlabel("Distance of Tumor Center from Shell Centre (mm)")
    ax.set_ylabel("Signal-to-Clutter Ratio (SCR) [dB]")
    ax.set_ylim(min(-30, np.floor(s[ok].min() - 1)), max(15, np.ceil(s[ok].max() + 1)))
    ax.legend(loc="upper right", frameon=True, edgecolor="black", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    (REPO / "metrics_image").mkdir(exist_ok=True)
    plt.savefig(REPO / "metrics_image" / "signal_death_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------- item 3: central concentration
    md += ["## 3. Central concentration of the DAS+CF image", "",
           "Main definition (rrs_mbi/metrics.py `calculate_central_concentration`, as called by rrs_mbi/evaluator.py): "
           "CC = sum(I^2 over mask pixels within 10 mm of the centre) / sum(I^2 over the mask); energy = intensity squared of "
           "the final DAS x CF image; mask = `extract_breast_mask` (pixels above the 15th percentile of positive pixels, holes "
           "filled, largest connected component); centre = estimated shell centre. Variants: centre = array centre (0, 0); "
           "mask = geometric breast disc (radius R around the shell centre).", "",
           "| Definition | Mean | Median | SD |", "|---|---|---|---|"]
    for lab, col in (("Main (image mask, shell centre)", "cc_main"), ("Array centre (0,0), image mask", "cc_array_centre"),
                     ("Breast-disc mask, shell centre", "cc_disc_mask")):
        x = A[col]
        md.append(f"| {lab} | {x.mean():.4f} | {np.median(x):.4f} | {x.std(ddof=1):.4f} |")
        long.append(dict(item=3, row=lab, n=504, value=f"mean {x.mean():.4f}; median {np.median(x):.4f}; SD {x.std(ddof=1):.4f}"))

    (HERE / "report_numbers.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    pd.DataFrame(long).to_csv(HERE / "report_numbers.csv", index=False)
    print(chr(10).join(md))


if __name__ == "__main__":
    main()
