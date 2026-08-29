"""
visualize_reconstruction.py

Generates ACTUAL reconstructed images for representative scans.
6-panel output: Raw DAS, DAS+CF, DAS+CF-debiased-zscore,
DAS-debiased-zscore, DAS-debiased-ratio, DAS-debiased+CF-zscore.

Usage:
    python visualize_reconstruction.py --n-phantoms 30 --display-margin 0.8
    python visualize_reconstruction.py --scan-idx 142
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loading import load_all_data
from src import physics
from src import signal_processing as sp
from src import beamforming as bf
from select_phantoms import build_phantom_table, stratified_select
from baseline_argmax_test import (
    build_grid, get_cf_baseline_cached, debias_cf_zscore,
    argmax_localize_and_score,
    get_das_baseline_cached, debias_das_zscore, debias_das_ratio,
)


def reconstruct_images_for_scan(scan_idx, s21, tumor_model, id_to_original_idx):
    """Minimal reconstruction pipeline — produces 6 image variants for plotting.
    
    Includes:
    - Empty-chamber calibration (emp_ref_id subtraction before ICZT)
    - DAS-magnitude debiasing (attacks coherent-summation center bias directly)
    - CF debiasing (existing z-score approach)
    - Combined DAS-debiased + CF-zscore
    """
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, _ = physics.compute_tissue_velocity(fat_frac, fib_frac)
    ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    # --- Get raw frequency-domain scan ---
    s21_idx = int(row["original_s21_idx"])
    fd_scan = s21[s21_idx]

    # --- Empty-chamber calibration (CRITICAL — was missing, see debugging item 29) ---
    emp_ref_id = row.get("emp_ref_id", None)
    calibration_applied = False
    if emp_ref_id is not None and not pd.isna(emp_ref_id) and int(emp_ref_id) in id_to_original_idx:
        emp_idx = id_to_original_idx[int(emp_ref_id)]
        fd_scan = fd_scan - s21[emp_idx]
        calibration_applied = True
    
    if not calibration_applied:
        print(f"  WARNING: scan_idx={scan_idx} — no valid emp_ref_id found, "
              f"proceeding WITHOUT calibration")

    # --- Transform to time domain ---
    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])

    # --- Build delay grid ---
    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
    grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0

    delay_grid = physics.two_medium_delay(
        geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
        grid_x_m, grid_y_m, breast_radius_mm / 1000.0, v_tissue,
    )
    delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)

    n_ant = time_signal.shape[1]

    # --- Raw DAS (no CF) ---
    raw_das = bf.das_coherent(time_signal, time_axis, delay_grid)

    # --- DAS + CF (original) ---
    das_image_raw, cf_map, das_cf_img = bf.das_coherent_cf(
        time_signal, time_axis, delay_grid)

    # --- CF debiasing (existing approach) ---
    cf_baseline_mean, cf_baseline_std = get_cf_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)
    cf_zscore = debias_cf_zscore(cf_map, cf_baseline_mean, cf_baseline_std)
    das_cf_debiased = das_image_raw * cf_zscore

    # --- DAS MAGNITUDE debiasing (FIXED — amplitude-normalized) ---
    das_baseline_mean, das_baseline_std = get_das_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)
    
    # Amplitude normalization: scale the noise baseline to match the real
    # signal's energy level before comparing. Without this, calibrated
    # signals (~0.3-0.5 magnitude) are ~100x smaller than unit-variance
    # noise baselines (~30-60), making z-scores uniformly negative →
    # clipped to 0 → flat image.
    real_energy = float(np.mean(raw_das[raw_das > 0])) if np.any(raw_das > 0) else 1.0
    baseline_energy = float(np.mean(das_baseline_mean[das_baseline_mean > 0])) if np.any(das_baseline_mean > 0) else 1.0
    amplitude_scale = real_energy / (baseline_energy + 1e-10)
    
    scaled_baseline_mean = das_baseline_mean * amplitude_scale
    scaled_baseline_std = das_baseline_std * amplitude_scale
    
    das_debiased_zscore = debias_das_zscore(raw_das, scaled_baseline_mean, scaled_baseline_std)
    das_debiased_ratio = debias_das_ratio(raw_das, das_baseline_mean)  # ratio is scale-invariant, no fix needed
    
    # --- Combined: DAS-magnitude-debiased AND CF-zscore-weighted ---
    das_combined = das_debiased_zscore * cf_zscore
    # --- Ground truth ---
    gt_x_mm, gt_y_mm = float(row["tumor_x_mm"]), float(row["tumor_y_mm"])

    return dict(
        images={
            "Raw DAS": raw_das,
            "DAS+CF": das_cf_img,
            "DAS+CF-debiased-zscore": das_cf_debiased,
            "DAS-debiased-zscore (NEW)": das_debiased_zscore,
            "DAS-debiased-ratio (NEW)": das_debiased_ratio,
            "DAS-debiased + CF-zscore (NEW)": das_combined,
        },
        axis_mm=axis_mm,
        grid_radius_mm=grid_radius_mm,
        gt_x_mm=gt_x_mm, gt_y_mm=gt_y_mm,
        phantom_id=row["phant_id"],
        breast_radius_mm=breast_radius_mm,
        calibration_applied=calibration_applied,
    )


def plot_scan(scan_idx, result, out_dir=".", display_margin=None):
    """Plots up to 6 panels in a 2x3 grid."""
    images = result["images"]
    axis_mm = result["axis_mm"]
    grid_radius_mm = result["grid_radius_mm"]
    gt_x_mm, gt_y_mm = result["gt_x_mm"], result["gt_y_mm"]
    breast_radius_mm = result["breast_radius_mm"]

    n_panels = len(images)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.array(axes).ravel()

    extent = [-grid_radius_mm, grid_radius_mm, -grid_radius_mm, grid_radius_mm]

    for idx, (ax, (name, img)) in enumerate(zip(axes, images.items())):
        le, scr, on_edge, px, py = argmax_localize_and_score(
            img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        im = ax.imshow(img, extent=extent, origin="lower", cmap="turbo", aspect="equal")
        ax.plot(gt_x_mm, gt_y_mm, "gx", markersize=14, markeredgewidth=3, label="Ground truth")
        ax.plot(px, py, "r+", markersize=14, markeredgewidth=3, label="Predicted (argmax)")
        ax.set_title(f"{name}\nLE={le:.1f}mm SCR={scr:.1f}dB", fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        if display_margin is not None:
            display_radius_mm = breast_radius_mm * display_margin
            ax.set_xlim(-display_radius_mm, display_radius_mm)
            ax.set_ylim(-display_radius_mm, display_radius_mm)

    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    cal_str = "CALIBRATED" if result.get("calibration_applied") else "UNCALIBRATED ⚠️"
    fig.suptitle(
        f"scan_idx={scan_idx} phant_id={result['phantom_id']} "
        f"breast_radius={breast_radius_mm:.1f}mm [{cal_str}]",
        fontsize=13)
    plt.tight_layout()
    out_path = f"{out_dir}/reconstruction_scan_{scan_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phantoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-viz", type=int, default=10,
                         help="Number of scans to visualize (one per distinct phantom). Default: 10.")
    parser.add_argument("--scan-idx", type=int, default=None,
                         help="Visualize one specific scan_idx.")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--display-margin", type=float, default=None,
                         help="Crop DISPLAYED view to breast_radius_mm * this value. "
                              "Does NOT change computed grid or metrics. Default: full grid.")
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]

    if args.n_phantoms > 0:
        phantom_table = build_phantom_table(tumor_model)
        selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
        selected_ids = selected["phant_id"].tolist()
        tumor_model = tumor_model[tumor_model["phant_id"].isin(selected_ids)].reset_index(drop=True)

    if args.scan_idx is not None:
        scan_indices = [args.scan_idx]
    else:
        rng = np.random.default_rng(args.seed)
        scan_indices = []
        for pid, group in tumor_model.groupby("phant_id"):
            scan_indices.append(int(rng.choice(group.index.values)))
        rng.shuffle(scan_indices)
        scan_indices = sorted(scan_indices[:args.n_viz])
        print(f"No --scan-idx given — visualizing {len(scan_indices)} scans, "
              f"one per distinct phantom: {scan_indices}")

    for scan_idx in scan_indices:
        if scan_idx >= len(tumor_model):
            print(f"Skipping scan_idx={scan_idx} — out of range "
                  f"(len={len(tumor_model)})")
            continue
        print(f"Reconstructing scan_idx={scan_idx}...")
        result = reconstruct_images_for_scan(scan_idx, s21, tumor_model, id_to_original_idx)
        plot_scan(scan_idx, result, out_dir=args.out_dir, display_margin=args.display_margin)


if __name__ == "__main__":
    main()