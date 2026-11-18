"""
reconstruction/reconstruct_images.py

Reconstruction images with the tuned method from rr's slides (config.json):
corrected src pipeline (geometry, delay model, TVSVD) -> DAS x CF^1.5 -> MER (p=2) -> gamma 0.5
-> LoG sigma 3-7 -> Otsu -> largest blob -> weighted centroid with bias_alpha 0.3.
Image style follows rr's visualize_final.py (turbo, black background, green dashed GT circle,
white + for the detected point).

  python reconstruction/reconstruct_images.py                    # 5 random scans (seed 42)
  python reconstruction/reconstruct_images.py --n 10 --seed 7    # other random draw
  python reconstruction/reconstruct_images.py --scans 0 17 42    # specific rows of tumor_model
  python reconstruction/reconstruct_images.py --all --metrics-only   # LE table for all scans, no figures

Outputs: reconstruction/images/scan_XXX_<phantom>_LE<mm>mm.png (main image),
reconstruction/images/steps/scan_XXX_<phantom>_steps.png (pipeline step by step),
summary_metrics.csv, summary_aggregate.txt. The ground truth is drawn for evaluation only.
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
from src import beamforming as bf, signal_processing as sp, blob_detection as bd
import enhancement as en

CFG = json.loads((HERE / "config.json").read_text())
OUT = HERE / "images"
DISPLAY_INTERP = "bicubic"   # display only (smooth look of the reference images); data/LE unchanged


# ----------------------------------------------------------------------------------------------
# reconstruction
# ----------------------------------------------------------------------------------------------
def reconstruct(i, data, cfg=CFG):
    tm = data["tumor_model"]
    r = reconstruct_scan(i, data["s21"], tm, data["id_to_original_idx"], freq_axis=data["freq_axis"],
                         use_tvsvd=True, use_rms_norm=True, bandpass_mode="full",
                         gate_ns=cfg["gate_ns"], grid_step_mm=cfg["grid_step_mm"], return_diagnostics=True)
    g = r["diagnostics"]
    axis, ts, dg = g["axis_mm"], g["time_signal"], g["delay_grid"]
    centre = (r["shell_cx_mm"], r["shell_cy_mm"])
    R = r["breast_radius_mm"]
    gx, gy = np.meshgrid(axis, axis)
    region = np.hypot(gx - centre[0], gy - centre[1]) <= R

    # DAS x CF^cf_power, restricted to the breast disc exactly like the main pipeline
    _, _, img = bf.das_coherent_cf(ts, sp.get_time_axis(ts.shape[0]), dg,
                                   cf_power=cfg["cf_power"], gate_ns=cfg["gate_ns"])
    img = np.where(region, img, 0.0)

    mer_img = en.mer(img, cfg["mer_power"], region)
    final = en.gamma_correct(mer_img, cfg["gamma"])
    stages = {
        "Raw (linear norm)": en.raw_normalised(img, region),
        "Percentile clip %d-%d" % tuple(cfg["percentile_clip"]):
            en.percentile_clip(img, *cfg["percentile_clip"], region=region),
        "MER (power=%.1f)" % cfg["mer_power"]: mer_img,
        "MER + gamma(%.1f)" % cfg["gamma"]: final,
    }

    mask, _, _ = bd.select_blob_mask_log(final, sigma_list=tuple(cfg["log_sigma"]))
    iw_xy, biased_xy = en.weighted_centroid_biased(mask, final, axis, axis, cfg["bias_alpha"])

    row = tm.iloc[i]
    gt = (float(row["tumor_x_mm"]), float(row["tumor_y_mm"]))
    dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    birads = row.get("birads", np.nan)
    return dict(
        i=i, phant_id=row["phant_id"], birads=birads, axis=axis, stages=stages, final=final, mask=mask,
        centre=centre, R=R, gt=gt, gt_r=float(row["tumor_radius_mm"]), iw_xy=iw_xy, det_xy=biased_xy,
        LE_pipeline_old=float(r["localization_error_mm"]),   # src blob extraction on CF^1 image, no enhancement
        LE_alpha0=dist(iw_xy, gt), LE=dist(biased_xy, gt), LE_trivial=dist(centre, gt),
        blob_area_px=int(mask.sum()),
    )


# ----------------------------------------------------------------------------------------------
# plotting (style of rr's visualize_final.py)
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


def _title(res, name):
    b = "" if np.isnan(res["birads"]) else " | BI-RADS %.0f" % res["birads"]
    return "%s | %s%s\nLE=%.1fmm" % (name, res["phant_id"], b, res["LE"])


def plot_main(res, path):
    a = res["axis"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(res["final"], extent=[a[0], a[-1], a[0], a[-1]], origin="lower", cmap="turbo",
                   vmin=0, vmax=1, aspect="equal", interpolation=DISPLAY_INTERP)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Enhanced Intensity", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    _gt_circle(ax, res)
    ax.plot(*res["det_xy"], "w+", markersize=14, markeredgewidth=2,
            label="Det (%.0f,%.0f)" % res["det_xy"])
    _crop(ax, res)
    ax.set_title(_title(res, "DAS+CF (MER + gamma%.1f)" % CFG["gamma"]), fontsize=12, fontweight="bold",
                 color="white")
    ax.set_xlabel("X (mm)", color="white")
    ax.set_ylabel("Y (mm)", color="white")
    _style_axes(ax)
    ax.legend(loc="upper right", fontsize=9, facecolor="black", edgecolor="white", labelcolor="white")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def plot_steps(res, path):
    a = res["axis"]
    names = list(res["stages"])
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, name in zip(axes[:4], names):
        ax.imshow(res["stages"][name], extent=[a[0], a[-1], a[0], a[-1]], origin="lower", cmap="turbo",
                  vmin=0, vmax=1, aspect="equal", interpolation=DISPLAY_INTERP)
        _gt_circle(ax, res, label=False)
        ax.set_title(name, color="white", fontsize=10)
    ax = axes[4]
    ax.imshow(res["final"], extent=[a[0], a[-1], a[0], a[-1]], origin="lower", cmap="turbo", vmin=0, vmax=1,
              aspect="equal", interpolation=DISPLAY_INTERP)
    ax.contour(a, a, res["mask"].astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    _gt_circle(ax, res, label=False)
    ax.plot(*res["iw_xy"], "x", color="white", markersize=9, markeredgewidth=2)
    ax.plot(*res["det_xy"], "+", color="white", markersize=15, markeredgewidth=2.5)
    ax.set_title("LoG %s > Otsu > blob > centroid (x) > bias-corrected (+)" % "-".join(
        map(str, (CFG["log_sigma"][0], CFG["log_sigma"][-1]))), color="white", fontsize=9)
    for ax in axes:
        _crop(ax, res)
        ax.set_xticks([]); ax.set_yticks([])
        _style_axes(ax)
    fig.suptitle("scan %d | %s | tumour r=%.0f mm (green dashed) | LE: centroid %.1f mm, bias-corrected %.1f mm" %
                 (res["i"], res["phant_id"], res["gt_r"], res["LE_alpha0"], res["LE"]), color="white",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------------------------
def aggregate(df):
    from rrs_mbi.evaluator import cluster_bootstrap
    lines = ["n scans = %d, phantoms = %d" % (len(df), df.phant_id.nunique()), ""]
    lines.append("%-40s %8s %8s %8s" % ("localiser", "meanLE", "medLE", "det@15"))
    for name, col in [("src pipeline (CF^1, no enhancement)", "LE_pipeline_old"),
                      ("tuned centroid (bias_alpha=0)", "LE_alpha0"),
                      ("tuned + bias_alpha=%.1f (final)" % CFG["bias_alpha"], "LE"),
                      ("trivial: always predict shell centre", "LE_trivial")]:
        lines.append("%-40s %8.2f %8.2f %7.1f%%" % (name, df[col].mean(), df[col].median(),
                                                    (df[col] <= 15).mean() * 100))
    if df.phant_id.nunique() >= 5:
        lines += ["", "paired difference in mean LE (mm), phantom-cluster bootstrap 95% CI (negative = first is better):"]
        d = df.rename(columns={"phant_id": "scan_id"})
        for a, b in [("LE", "LE_trivial"), ("LE_alpha0", "LE_trivial"), ("LE", "LE_alpha0"), ("LE", "LE_pipeline_old")]:
            fn = lambda x, a=a, b=b: float((x[a] - x[b]).mean())
            bs = cluster_bootstrap(d, fn, n_boot=2000)
            lines.append("  %-10s - %-16s %+6.2f  [%+.2f, %+.2f]" % (a, b, fn(d), *np.percentile(bs, [2.5, 97.5])))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scans", type=int, nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    data = load_all_data()
    tm = data["tumor_model"]
    if args.all:
        scans = list(range(len(tm)))
    elif args.scans:
        scans = args.scans
    else:
        scans = sorted(int(k) for k in np.random.default_rng(args.seed).choice(len(tm), args.n, replace=False))
    print("scans:", scans if len(scans) <= 30 else "%d scans" % len(scans))
    (OUT / "steps").mkdir(parents=True, exist_ok=True)

    rows = []
    for k, i in enumerate(scans):
        res = reconstruct(i, data)
        rows.append({c: res[c] for c in ("i", "phant_id", "gt_r", "LE_pipeline_old", "LE_alpha0", "LE",
                                         "LE_trivial", "blob_area_px")} |
                    dict(gt_x=res["gt"][0], gt_y=res["gt"][1], det_x=res["det_xy"][0], det_y=res["det_xy"][1],
                         shell_cx=res["centre"][0], shell_cy=res["centre"][1]))
        if not args.metrics_only:
            plot_main(res, OUT / ("scan_%03d_%s_LE%.0fmm.png" % (i, res["phant_id"], res["LE"])))
            plot_steps(res, OUT / "steps" / ("scan_%03d_%s_steps.png" % (i, res["phant_id"])))
            print("  scan %d (%s): LE %.1f mm" % (i, res["phant_id"], res["LE"]), flush=True)
        elif k % 25 == 0:
            print("  %d/%d" % (k + 1, len(scans)), flush=True)

    df = pd.DataFrame(rows)
    suffix = "_all" if args.all else ""
    df.to_csv(OUT / ("summary_metrics%s.csv" % suffix), index=False)
    text = aggregate(df)
    (OUT / ("summary_aggregate%s.txt" % suffix)).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
