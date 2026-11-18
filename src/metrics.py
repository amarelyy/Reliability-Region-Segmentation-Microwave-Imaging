"""
src/metrics.py

Localization + image-quality metrics, BI-RADS-stratified grouping helpers,
and a front-surface-bias diagnostic (checks whether the predicted-vs-GT
offset is systematically aligned with the antenna-to-tumor direction, which
would confirm front-surface bias rather than random localization error).
"""

import numpy as np
import pandas as pd


def compute_localization_error(peak_x_mm, peak_y_mm, gt_x_mm, gt_y_mm):
    return float(np.sqrt((peak_x_mm - gt_x_mm) ** 2 + (peak_y_mm - gt_y_mm) ** 2))


def compute_iou_dice(tumor_mask, grid_x_mm, grid_y_mm, gt_x_mm, gt_y_mm, gt_r_mm):
    gt_mask = (grid_x_mm - gt_x_mm) ** 2 + (grid_y_mm - gt_y_mm) ** 2 <= gt_r_mm ** 2
    intersection = np.logical_and(tumor_mask, gt_mask).sum()
    union = np.logical_or(tumor_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    denom = tumor_mask.sum() + gt_mask.sum()
    dice = (2 * intersection) / denom if denom > 0 else 0.0
    return float(iou), float(dice)


def compute_scr_cnr(img, tumor_mask, region=None):
    """SCR (dB), SMR (dB), CNR — signal region = tumor_mask, clutter = rest of `region`
    (whole image if region is None)."""
    region = np.ones_like(tumor_mask, dtype=bool) if region is None else region
    clutter = region & ~tumor_mask
    signal_power = img[tumor_mask].mean() if tumor_mask.sum() > 0 else 0.0
    clutter_power = img[clutter].mean()
    clutter_std = img[clutter].std()

    scr_db = 10 * np.log10(signal_power / (clutter_power + 1e-12)) if signal_power > 0 else 0.0
    smr_db = 10 * np.log10(signal_power / (img[region].mean() + 1e-12)) if signal_power > 0 else 0.0
    cnr = (signal_power - clutter_power) / (clutter_std + 1e-12)

    return float(scr_db), float(smr_db), float(cnr)


def compute_all_metrics(img, tumor_mask, x_axis_mm, y_axis_mm, peak_x_mm, peak_y_mm,
                         gt_x_mm, gt_y_mm, gt_r_mm, region=None):
    grid_x_mm, grid_y_mm = np.meshgrid(x_axis_mm, y_axis_mm)
    le = compute_localization_error(peak_x_mm, peak_y_mm, gt_x_mm, gt_y_mm)
    iou, dice = compute_iou_dice(tumor_mask, grid_x_mm, grid_y_mm, gt_x_mm, gt_y_mm, gt_r_mm)
    scr_db, smr_db, cnr = compute_scr_cnr(img, tumor_mask, region)
    return dict(localization_error_mm=le, iou=iou, dice=dice,
                scr_db=scr_db, smr_db=smr_db, cnr=cnr)


# ============================================================================
# BI-RADS-stratified grouping
# ============================================================================
def birads_stratified_summary(results_df, birads_col="birads",
                                metric_cols=("localization_error_mm", "iou", "dice", "scr_db")):
    """results_df must have a birads column (1-4) alongside the metric columns."""
    return results_df.groupby(birads_col)[list(metric_cols)].agg(["mean", "std", "count"])


# ============================================================================
# Front-surface-bias diagnostic
# ============================================================================
def _first_column(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError("none of the columns %s found; available: %s" % (list(names), list(df.columns)))


def front_surface_bias_diagnostic(results_df, center_cols=("shell_cx_mm", "shell_cy_mm")):
    """
    Checks whether the predicted peak sits systematically closer to the array/shell
    centre than the true tumour does — the signature of a pull towards the centre
    (front-surface bias or a central artifact) as opposed to unbiased random
    localization error.

    Accepted column names (first match wins):
      predicted peak: peak_x_mm / peak_y_mm, or pred_x_mm / pred_y_mm
      ground truth:   gt_x_mm / gt_y_mm,     or true_x_mm / true_y_mm
      centre:         `center_cols` (default shell_cx_mm / shell_cy_mm, the estimated shell
                      centre the pipeline uses); falls back to the origin (0, 0) if absent.
    Adds a column 'radial_bias_mm' = (GT distance from centre) - (peak distance from centre):
    positive means the peak sits CLOSER to the centre than the true tumour.

    Returns (results_df_with_column, summary_dict) where summary_dict has
    mean/std of radial_bias_mm and a fraction-positive statistic — a value
    consistently > 0 across most scans is evidence for a pull towards the centre
    rather than random error (which would center around 0 with no systematic sign).
    """
    df = results_df.copy()
    px = _first_column(df, ("peak_x_mm", "pred_x_mm"))
    py = _first_column(df, ("peak_y_mm", "pred_y_mm"))
    gx = _first_column(df, ("gt_x_mm", "true_x_mm"))
    gy = _first_column(df, ("gt_y_mm", "true_y_mm"))
    cx = df[center_cols[0]] if center_cols[0] in df.columns else 0.0
    cy = df[center_cols[1]] if center_cols[1] in df.columns else 0.0

    gt_dist_from_center = np.sqrt((df[gx] - cx) ** 2 + (df[gy] - cy) ** 2)
    pred_dist_from_center = np.sqrt((df[px] - cx) ** 2 + (df[py] - cy) ** 2)
    df["radial_bias_mm"] = gt_dist_from_center - pred_dist_from_center

    valid = df["radial_bias_mm"].notna()
    summary = dict(
        mean_radial_bias_mm=float(df.loc[valid, "radial_bias_mm"].mean()),
        std_radial_bias_mm=float(df.loc[valid, "radial_bias_mm"].std()),
        fraction_positive=float((df.loc[valid, "radial_bias_mm"] > 0).mean()),
        n_scans=int(valid.sum()),
    )
    return df, summary