"""
test_boundary_boost.py

Tests distance-based gain compensation (Time Gain Compensation, TGC-style)
against the ring-symmetry/boundary-collapse problem — the inverse idea to
CF/DAS debiasing: instead of suppressing the center, boost the boundary,
using the ALREADY-MEASURED SCR-by-distance-ratio falloff as the calibration
curve (no new measurement needed).

Tests several gain caps side by side, since an uncapped compensation would
require ~56x boost at the 90%+ bin — plausible over-amplification of
grating-lobe noise, not yet tested at any scale.

Small default sample (--n-scans 20) for a fast first look before committing
to a full 214-scan run.

Usage:
    python test_boundary_boost.py --n-scans 20 --n-jobs 4
    python test_boundary_boost.py --n-scans 214 --n-phantoms 30 --n-jobs 4   # full run later
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
from baseline_argmax_test import build_grid, argmax_localize_and_score

DETECTION_THRESHOLD_MM = 20.0

# Calibration curve from the ALREADY-MEASURED SCR-by-distance-ratio table
# (see chat discussion — DAS+CF, n=50 run). Bin centers are approximate
# midpoints of the original bins [0,0.5], [0.5,0.75], [0.75,0.9], [0.9,1.5].
_CALIB_DIST_RATIO = np.array([0.25, 0.625, 0.825, 1.1])
_CALIB_SCR_DB = np.array([10.2, 7.5, -2.3, -24.8])
_CALIB_REFERENCE_DB = _CALIB_SCR_DB[0]   # 0-50% bin = no boost needed
_CALIB_BOOST_DB = _CALIB_REFERENCE_DB - _CALIB_SCR_DB   # [0, 2.7, 12.5, 35.0]

GAIN_CAPS_DB = [0, 10, 20, 35]   # 0 = no boost at all (sanity-check baseline),
                                   # 35 = effectively uncapped (matches max boost needed)


def build_gain_map(axis_mm, breast_radius_mm, cap_db):
    """Per-pixel gain map (linear amplitude multiplier) from distance-ratio,
    via linear interpolation of the calibration curve above, capped at
    cap_db. cap_db=0 means no boost applied anywhere (control condition)."""
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    dist_ratio = np.hypot(grid_x_mm, grid_y_mm) / breast_radius_mm

    boost_db = np.interp(dist_ratio, _CALIB_DIST_RATIO, _CALIB_BOOST_DB)
    boost_db = np.clip(boost_db, 0, cap_db)
    return 10 ** (boost_db / 20.0)   # amplitude-domain gain (matches 20*log10 SCR convention)


def reconstruct_and_score_boost_variants(scan_idx, s21, tumor_model, id_to_original_idx):
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

    gt_x_mm, gt_y_mm = float(row["tumor_x_mm"]), float(row["tumor_y_mm"])

    _, cf_map, das_cf_img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)

    results = {}
    for cap_db in GAIN_CAPS_DB:
        gain_map = build_gain_map(axis_mm, breast_radius_mm, cap_db)
        boosted_img = das_cf_img * gain_map
        le, scr, on_edge, px, py = argmax_localize_and_score(
            boosted_img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        name = f"DAS+CF+boost_cap{cap_db}dB"
        results[name] = dict(localization_error_mm=le, scr_db=scr, on_edge=on_edge)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scans", type=int, default=20,
                         help="Quick-test sample size. Use 214 for the full batch later.")
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

    rng = np.random.default_rng(args.seed)
    n_scans = min(args.n_scans, len(tumor_model))
    scan_indices = rng.choice(len(tumor_model), size=n_scans, replace=False)
    print(f"Running boundary-boost test on {n_scans} scans "
          f"(sampled from the same {len(selected_ids)}-phantom stratified set, seed={args.seed})")

    variant_names = [f"DAS+CF+boost_cap{c}dB" for c in GAIN_CAPS_DB]
    variant_rows = {name: [] for name in variant_names}

    def _process_one(idx):
        try:
            return ("ok", idx, reconstruct_and_score_boost_variants(
                int(idx), s21, tumor_model, id_to_original_idx))
        except Exception as e:
            return ("fail", idx, str(e))

    t_start = time_module.time()
    outcomes = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_one)(idx) for idx in scan_indices
    )

    failed = []
    for outcome in outcomes:
        if outcome[0] == "ok":
            _, idx, scan_results = outcome
            for name, metrics in scan_results.items():
                variant_rows[name].append(metrics)
        else:
            failed.append(outcome[1:])

    print(f"\n{len(outcomes) - len(failed)}/{n_scans} scans done "
          f"({time_module.time() - t_start:.0f}s elapsed)")
    if failed:
        print(f"{len(failed)} scans failed. First: {failed[0]}")

    print("\n" + "=" * 70)
    print(f"BOUNDARY-BOOST TEST (n={n_scans}) — cap_db=0 is the no-boost control")
    print("=" * 70)
    for name, cap_db in zip(variant_names, GAIN_CAPS_DB):
        rows = variant_rows[name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print(f"\ncap={cap_db}dB (n={len(df)})")
        print(f"  mean LE   : {df['localization_error_mm'].mean():.2f} mm")
        print(f"  median LE : {df['localization_error_mm'].median():.2f} mm")
        print(f"  mean SCR  : {df['scr_db'].mean():.1f} dB")
        print(f"  detection @20mm : {(df['localization_error_mm'] <= DETECTION_THRESHOLD_MM).mean():.1%}")


if __name__ == "__main__":
    main()