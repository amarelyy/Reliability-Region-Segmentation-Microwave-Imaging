"""Step 4: rebuild report Figures 5 and 6 from real reconstructions.

Figure 6: SCR vs tumour distance from centre, measured
  (a) at the predicted blob (pipeline's scr_db: blob mask vs rest), and
  (b) in a disc of the true tumour radius around the ground truth.
Figure 5: mean CF map of the adipose-only reference scans (adi_ref_id), on a
  grid normalised by each phantom's own shell radius.
Reads the caches written by recon_cache.py for <tag> in {baseline, fixed}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress, pearsonr

A = Path(__file__).resolve().parent
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6})


def gt_disc_scr(img, axis, x, y, r, region=None):
    """SCR of a disc of the true tumour radius vs the rest of `region` (whole image if None)."""
    gx, gy = np.meshgrid(axis, axis)
    disc = (gx - x) ** 2 + (gy - y) ** 2 <= max(r, 1.0) ** 2
    region = np.ones_like(disc) if region is None else region
    disc &= region
    if disc.sum() == 0:
        return np.nan
    return 10 * np.log10(img[disc].mean() / (img[region & ~disc].mean() + 1e-12))


def breast_region(axis, cx, cy, R, restricted):
    if not restricted:
        return None
    gx, gy = np.meshgrid(axis, axis)
    return np.hypot(gx - cx, gy - cy) <= R


def fit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    lr = linregress(x[m], y[m])
    return dict(slope_db_per_mm=lr.slope, intercept_db=lr.intercept, pearson_r=lr.rvalue,
                p=lr.pvalue, zero_cross_mm=-lr.intercept / lr.slope if lr.slope else np.nan,
                n=int(m.sum()), min_db=float(np.min(y[m])), frac_below_0db=float(np.mean(y[m] < 0)))


def figure6(tag):
    meta = pd.read_csv(A / tag / "cache/recon_tumour.csv")
    z = np.load(A / tag / "cache/recon_tumour.npz")
    if "shell_cx_mm" not in meta:
        meta["shell_cx_mm"] = meta["shell_cy_mm"] = 0.0
    restricted = tag == "final"
    meta["scr_gt_disc_db"] = [gt_disc_scr(z[f"img_{i}"], z[f"axis_{i}"], r.tumor_x_mm, r.tumor_y_mm,
                                          r.tumor_radius_mm,
                                          breast_region(z[f"axis_{i}"], r.shell_cx_mm, r.shell_cy_mm,
                                                        r.breast_radius_mm, restricted))
                              for i, r in meta.iterrows()]
    meta["gt_dist_mm"] = np.hypot(meta.tumor_x_mm - meta.shell_cx_mm, meta.tumor_y_mm - meta.shell_cy_mm)
    meta["gt_dist_norm"] = meta.gt_dist_mm / meta.breast_radius_mm
    meta.to_csv(A / tag / "fig6_scr_vs_distance.csv", index=False)
    stats = []
    for col in ("scr_db", "scr_gt_disc_db"):
        for xcol in ("gt_dist_mm", "gt_dist_norm"):
            stats.append(dict(tag=tag, scr=col, x=xcol, **fit(meta[xcol].to_numpy(), meta[col].to_numpy())))
    return meta, pd.DataFrame(stats)


def figure5(tag, n=121):
    meta = pd.read_csv(A / tag / "cache/recon_adipose.csv")
    z = np.load(A / tag / "cache/recon_adipose.npz")
    u = np.linspace(-1.5, 1.5, n)
    U, V = np.meshgrid(u, u)
    maps, imgs, rows = [], [], []
    if "shell_cx_mm" not in meta:
        meta["shell_cx_mm"] = meta["shell_cy_mm"] = 0.0
    for i, r in meta.iterrows():
        axis, cf, img = z[f"axis_{i}"], z[f"cf_{i}"], z[f"img_{i}"]
        R, cx, cy = r.breast_radius_mm, r.shell_cx_mm, r.shell_cy_mm
        # (y, x) for interpolator on (axis_y, axis_x); normalised grid centred on the shell centre
        pts = np.c_[cy + V.ravel() * R, cx + U.ravel() * R]
        kw = dict(bounds_error=False, fill_value=np.nan)
        maps.append(RegularGridInterpolator((axis, axis), cf, **kw)(pts).reshape(n, n))
        imgs.append(RegularGridInterpolator((axis, axis), img / img.max(), **kw)(pts).reshape(n, n))
        gx, gy = np.meshgrid(axis, axis)
        rr = np.hypot(gx - cx, gy - cy)
        inside = rr <= R
        iy, ix = np.unravel_index(np.argmax(np.where(inside, img, -np.inf)), img.shape)
        ky, kx = np.unravel_index(np.argmax(np.where(inside, cf, -np.inf)), cf.shape)
        rows.append(dict(tag=tag, adi_ref_id=r.adi_ref_id, phant_id=r.phant_id, R_mm=R,
                         img_peak_dist_mm=np.hypot(axis[ix] - cx, axis[iy] - cy),
                         img_peak_dist_norm=np.hypot(axis[ix] - cx, axis[iy] - cy) / R,
                         cf_peak_dist_mm=np.hypot(axis[kx] - cx, axis[ky] - cy),
                         cf_mean_central10=cf[rr <= 10].mean(), cf_mean_breast=cf[inside].mean(),
                         energy_frac_central10=(img[rr <= 10] ** 2).sum() / (img[inside] ** 2).sum(),
                         area_frac_central10=(rr <= 10).sum() / inside.sum()))
    per = pd.DataFrame(rows)
    per.to_csv(A / tag / "fig5_adipose_cf_per_reference.csv", index=False)
    return u, np.nanmean(maps, axis=0), np.nanmean(imgs, axis=0), per


if __name__ == "__main__":
    tags = sys.argv[1:] or ["baseline", "fixed"]
    f6 = {t: figure6(t) for t in tags}
    s6 = pd.concat([s for _, s in f6.values()])
    s6.to_csv(A / "fig6_fit_stats.csv", index=False)
    print(s6.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tags):
        m = f6[t][0]
        for col, c, lab in (("scr_db", BLUE, "at predicted blob"), ("scr_gt_disc_db", ORANGE, "in disc at true tumour")):
            ax.scatter(m.gt_dist_mm, m[col], s=10, color=c, alpha=0.45, edgecolors="none", label=lab)
            st = f6[t][1].query("scr == @col and x == 'gt_dist_mm'").iloc[0]
            xs = np.array([0, m.gt_dist_mm.max()])
            ax.plot(xs, st.intercept_db + st.slope_db_per_mm * xs, color=c, lw=2)
            ax.annotate(f"{st.slope_db_per_mm:+.2f} dB/mm, r = {st.pearson_r:+.2f}",
                        (xs[1], st.intercept_db + st.slope_db_per_mm * xs[1]), color=INK, fontsize=8,
                        ha="right", va="bottom", xytext=(0, 4), textcoords="offset points")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{t} pipeline (n = {len(m)})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("true tumour distance from centre (mm)")
    axes[0][0].set_ylabel("SCR (dB)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(A / "fig6_scr_vs_distance_REAL.png", dpi=200)

    have5 = [t for t in tags if (A / t / "cache/recon_adipose.npz").exists()]
    if have5:
        fig, axes = plt.subplots(1, len(have5), figsize=(3.9 * len(have5), 3.5), squeeze=False)
        per_all = []
        for ax, t in zip(axes[0], have5):
            u, cf, _, per = figure5(t)
            per_all.append(per)
            im = ax.imshow(cf, extent=[u[0], u[-1], u[0], u[-1]], origin="lower", cmap=SEQ)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(th), np.sin(th), color=INK, lw=0.8, ls="--")
            ax.set_title(f"{t}: mean CF, adipose-only (n = {len(per)})", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("x / R_shell"); ax.set_ylabel("y / R_shell"); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, label="coherence factor")
        fig.tight_layout()
        fig.savefig(A / "fig5_cf_adipose_REAL.png", dpi=200)
        per_all = pd.concat(per_all)
        per_all.to_csv(A / "fig5_adipose_cf_per_reference.csv", index=False)
        print(per_all.drop(columns=["phant_id"]).groupby("tag").agg(["mean", "median"]).T.round(3).to_string())
