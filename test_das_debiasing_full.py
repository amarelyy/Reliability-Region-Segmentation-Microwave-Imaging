"""
test_das_debiasing_full.py

Tests DAS-magnitude debiasing (the "(on one image)" result from the Week 2
deck — 5.1/5.8/7.3mm LE) across the FULL 214-scan/30-phantom stratified
sample, to check whether it holds up at scale. NOTE: baseline_argmax_test.py
never actually measured this at scale — the debiased images get added to
`images` AFTER the LE/SCR-computing loop already ran, so their accuracy was
silently never aggregated in that script. This script exists to actually
answer the question.

Usage:
    python test_das_debiasing_full.py --n-phantoms 30 --n-jobs 4
"""
import argparse
import time as time_module

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_loading import load_all_data
from select_phantoms import build_phantom_table, stratified_select
from src import physics
from src import signal_processing as sp
from src import beamforming as bf
from baseline_argmax_test import (
    build_grid, get_cf_baseline_cached, debias_cf_zscore,
    get_das_baseline_cached, debias_das_zscore, debias_das_ratio,
    argmax_localize_and_score,
)

DETECTION_THRESHOLD_MM = 20.0


def reconstruct_and_score_das_variants(scan_idx, s21, tumor_model, id_to_original_idx):
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    v_tissue, _ = physics.compute_tissue_velocity(
        float(row["fat_fraction"]), float(row["fib_fraction"]))
    ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    s21_idx = int(row["original_s21_idx"])
    fd_scan = s21[s21_idx]
    emp_ref_id = row.get("emp_ref_id", None)
    if emp_ref_id is not None and not pd.isna(emp_ref_id) and int(emp_ref_id) in id_to_original_idx:
        emp_idx = id_to_original_idx[int(emp_ref_id)]
        fd_scan = fd_scan - s21[emp_idx]

    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])

    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
    grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0

    delay_grid = physics.two_medium_delay(
        geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
        grid_x_m, grid_y_m, breast_radius_mm / 1000.0, v_tissue,
    ).reshape(-1, *grid_x_mm.shape)

    n_ant = time_signal.shape[1]
    gt_x_mm, gt_y_mm = float(row["tumor_x_mm"]), float(row["tumor_y_mm"])

    raw_das = bf.das_coherent(time_signal, time_axis, delay_grid)
    das_image_raw, cf_map, das_cf_img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)

    cf_baseline_mean, cf_baseline_std = get_cf_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)
    cf_zscore = debias_cf_zscore(cf_map, cf_baseline_mean, cf_baseline_std)

    das_baseline_mean, das_baseline_std = get_das_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)
    real_energy = float(np.mean(raw_das[raw_das > 0])) if np.any(raw_das > 0) else 1.0
    baseline_energy = float(np.mean(das_baseline_mean[das_baseline_mean > 0])) if np.any(das_baseline_mean > 0) else 1.0
    amplitude_scale = real_energy / (baseline_energy + 1e-10)

    das_debiased_z = debias_das_zscore(raw_das, das_baseline_mean * amplitude_scale,
                                        das_baseline_std * amplitude_scale)
    das_debiased_r = debias_das_ratio(raw_das, das_baseline_mean)
    das_combined = das_debiased_z * cf_zscore

    images = {
        "Raw DAS": raw_das,
        "DAS+CF": das_cf_img,
        "DAS-debiased-zscore": das_debiased_z,
        "DAS-debiased-ratio": das_debiased_r,
        "DAS-debiased+CF-zscore": das_combined,
    }

    results = {}
    for name, img in images.items():
        le, scr, on_edge, px, py = argmax_localize_and_score(
            img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        results[name] = dict(localization_error_mm=le, scr_db=scr, on_edge=on_edge)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phantoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]

    phantom_table = build_phantom_table(tumor_model)
    selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
    selected_ids = selected["phant_id"].tolist()
    tumor_model = tumor_model[tumor_model["phant_id"].isin(selected_ids)].reset_index(drop=True)
    print(f"Running on {len(tumor_model)} scans across {len(selected_ids)} phantoms "
          f"(same stratified selection used throughout the project, seed={args.seed})")

    variant_names = ["Raw DAS", "DAS+CF", "DAS-debiased-zscore",
                      "DAS-debiased-ratio", "DAS-debiased+CF-zscore"]
    variant_rows = {name: [] for name in variant_names}

    def _process_one(idx):
        try:
            return ("ok", idx, reconstruct_and_score_das_variants(
                idx, s21, tumor_model, id_to_original_idx))
        except Exception as e:
            return ("fail", idx, str(e))

    t_start = time_module.time()
    outcomes = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_one)(i) for i in range(len(tumor_model))
    )

    failed = []
    for outcome in outcomes:
        if outcome[0] == "ok":
            _, idx, scan_results = outcome
            for name, metrics in scan_results.items():
                variant_rows[name].append(metrics)
        else:
            failed.append(outcome[1:])

    print(f"\n{len(outcomes) - len(failed)}/{len(tumor_model)} scans done "
          f"({time_module.time() - t_start:.0f}s elapsed)")
    if failed:
        print(f"{len(failed)} scans failed. First: {failed[0]}")

    print("\n" + "=" * 70)
    print(f"DAS-MAGNITUDE DEBIASING — FULL-SCALE RESULT (n={len(tumor_model)}, "
          f"vs. the single-image 5.1/5.8/7.3mm claim from the deck)")
    print("=" * 70)
    for name in variant_names:
        rows = variant_rows[name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print(f"\n{name} (n={len(df)})")
        print(f"  mean LE   : {df['localization_error_mm'].mean():.2f} mm")
        print(f"  median LE : {df['localization_error_mm'].median():.2f} mm")
        print(f"  mean SCR  : {df['scr_db'].mean():.1f} dB")
        print(f"  detection @20mm : {(df['localization_error_mm'] <= DETECTION_THRESHOLD_MM).mean():.1%}")


if __name__ == "__main__":
    main()