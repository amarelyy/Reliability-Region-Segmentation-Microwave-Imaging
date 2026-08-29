"""
src/pipeline.py

reconstruct_scan() — self-contained, per-scan reconstruction with physics-informed
preprocessing stack, strictly following Isabel Olaya Lopez (2024) calibration protocol.
"""

from time import perf_counter
import numpy as np
import pandas as pd

from . import physics
from . import signal_processing as sp
from . import beamforming as bf
from . import blob_detection as bd
from . import metrics as mx

DEFAULT_GRID_MARGIN_FACTOR = 1.5
DEFAULT_GRID_STEP_MM = 1.0

_DELAY_CACHE = {}

def _delay_cache_key(phant_id, use_snellius, ant_rad_mm, breast_radius_mm,
                      v_tissue, margin_factor, grid_step_mm, shell_center):
    return (phant_id, use_snellius, round(ant_rad_mm, 3), round(breast_radius_mm, 3),
            round(v_tissue, 3), margin_factor, grid_step_mm, shell_center)


def build_grid(breast_radius_mm, margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                grid_step_mm=DEFAULT_GRID_STEP_MM):
    grid_radius_mm = breast_radius_mm * margin_factor
    axis_mm = np.arange(-grid_radius_mm, grid_radius_mm + grid_step_mm, grid_step_mm)
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    return grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm


def reconstruct_scan(scan_idx, s21, tumor_model, id_to_original_idx, freq_axis=None,
                      beamformer="das", use_snellius=True, use_cf=True,
                      use_tvsvd=False, use_bandpass=True, use_depth_gain=True,
                      shell_center=(0.0, 0.0),
                      margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                      grid_step_mm=DEFAULT_GRID_STEP_MM,
                      return_diagnostics=False):
    """
    Fully self-contained per-scan reconstruction.
    """
    t_start = perf_counter()
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    
    # 1. Calculate Effective Velocity based on phantom composition
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, eps_tissue = physics.compute_effective_velocity(fat_frac, fib_frac)

    # 2. Antenna geometry setup
    ant_rad_cm = float(row.get("ant_rad", 21.5))
    ant_rad_mm = ant_rad_cm * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    # 3. Signal Pre-processing (STRICTLY FOLLOWING PAPER ISABEL)
    s21_idx = int(row["original_s21_idx"])
    
    # Get Raw Data and ensure it's a complex numpy array
    fd_raw = np.asarray(s21[s21_idx], dtype=np.complex128)
    
    # A. COMPLEX CALIBRATION (Eq. 1 in Paper: S_cal = S_adi / S_emp)
    emp_ref_id = row.get("emp_ref_id", None)
    fd_scan = fd_raw.copy()
    
    if emp_ref_id is not None and not pd.isna(emp_ref_id):
        try:
            emp_idx = id_to_original_idx.get(int(emp_ref_id), None)
            if emp_idx is not None:
                fd_empty = np.asarray(s21[emp_idx], dtype=np.complex128)
                # Perform complex division to de-embed system response
                fd_scan = sp.calibrate_s_parameters(fd_raw, fd_empty)
            else:
                print(f"[Warning] emp_ref_id {int(emp_ref_id)} not found in map for scan {scan_idx}")
        except Exception as e:
            print(f"[Warning] Calibration failed for scan {scan_idx}: {e}")

    # B. Bandpass Filter (4-6 GHz) - Isolating discriminative power
    if use_bandpass and freq_axis is not None:
        freq_axis_arr = np.asarray(freq_axis)
        fd_scan = sp.apply_bandpass_filter(fd_scan, freq_axis_arr, low_cut=4e9, high_cut=6e9)

    # C. Time Domain Transform (ICZT)
    try:
        time_signal = sp.to_time_domain(fd_scan)
    except Exception as e:
        raise RuntimeError(f"ICZT failed for scan {scan_idx}: {e}")
        
    time_axis = sp.get_time_axis(time_signal.shape[0])

    # D. Clutter Suppression (Optional TVSVD)
    if use_tvsvd:
        time_signal, n_removed = sp.apply_hybrid_tvsvd(time_signal)
    else:
        n_removed = 0

    # 4. Imaging Grid Setup
    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(
        breast_radius_mm, margin_factor, grid_step_mm)
    grid_x_m = grid_x_mm.ravel() / 1000.0
    grid_y_m = grid_y_mm.ravel() / 1000.0
    shell_center_m = (shell_center[0] / 1000.0, shell_center[1] / 1000.0)

    # 5. Delay Grid Calculation (Snellius Bistatic Precise)
    cache_key = _delay_cache_key(
        row["phant_id"], use_snellius, ant_rad_mm, breast_radius_mm, 
        v_tissue, margin_factor, grid_step_mm, shell_center
    )
    
    if cache_key in _DELAY_CACHE:
        delay_grid = _DELAY_CACHE[cache_key]
    else:
        if use_snellius:
            delay_grid = physics.snellius_bistatic_delay_precise(
                geom["ant_x"], geom["ant_y"], 
                geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue
            )
        else:
            # Fallback to simple straight-ray bistatic
            tx_pos = np.stack([geom["ant_x"], geom["ant_y"]], axis=1)
            rx_pos = np.stack([geom["ant_x_b"], geom["ant_y_b"]], axis=1)
            grid_pos = np.stack([grid_x_m, grid_y_m], axis=1)
            n_ant = len(geom["ant_x"])
            n_pix = len(grid_x_m)
            delay_grid = np.zeros((n_ant, n_pix))
            for i in range(n_ant):
                d_tx = np.linalg.norm(grid_pos - tx_pos[i], axis=1)
                d_rx = np.linalg.norm(grid_pos - rx_pos[i], axis=1)
                delay_grid[i] = (d_tx + d_rx) / v_tissue
        
        delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)
        _DELAY_CACHE[cache_key] = delay_grid

    # E. Depth Gain Compensation
    if use_depth_gain:
        time_signal = sp.apply_depth_gain(time_signal, delay_grid)
    
    # 6. Beamforming
    if beamformer == "das":
        if use_cf:
            _, cf_map, img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)
        else:
            img = bf.das_coherent(time_signal, time_axis, delay_grid)
            cf_map = None
    elif beamformer == "dmas":
        td_mag = np.abs(time_signal)
        if use_cf:
            img, cf_map = bf.dmas_cf(time_signal, td_mag, time_axis, delay_grid)
        else:
            img = bf.dmas(td_mag, time_axis, delay_grid)
            cf_map = None
    else:
        raise ValueError(f"Unknown beamformer: {beamformer!r}")

    # 7. Blob Extraction + Localization
    blob = bd.extract_blob_candidate(img, axis_mm, axis_mm)

    gt_x_mm = float(row["tumor_x_mm"])
    gt_y_mm = float(row["tumor_y_mm"])
    gt_r_mm = float(row["tumor_radius_mm"])

    computed = mx.compute_all_metrics(
        img, blob["tumor_mask"], axis_mm, axis_mm,
        blob["peak_x"], blob["peak_y"], gt_x_mm, gt_y_mm, gt_r_mm,
    )

    cf_at_peak = None
    if cf_map is not None:
        peak_iy = np.argmin(np.abs(axis_mm - blob["peak_y"]))
        peak_ix = np.argmin(np.abs(axis_mm - blob["peak_x"]))
        cf_at_peak = float(cf_map[peak_iy, peak_ix])

    runtime_sec = perf_counter() - t_start

    result = dict(
        scan_idx=scan_idx,
        phant_id=row["phant_id"],
        birads=row.get("birads", np.nan),
        beamformer=beamformer, use_snellius=use_snellius, use_cf=use_cf,
        breast_radius_mm=breast_radius_mm, grid_radius_mm=grid_radius_mm,
        tvsvd_removed=n_removed,
        peak_x_mm=blob["peak_x"], peak_y_mm=blob["peak_y"],
        gt_x_mm=gt_x_mm, gt_y_mm=gt_y_mm, gt_r_mm=gt_r_mm,
        blob_area_px=blob["blob_area_px"], blob_compactness=blob["blob_compactness"],
        cf_at_peak=cf_at_peak,
        runtime_sec=runtime_sec,
        **computed,
    )

    if return_diagnostics:
        result["diagnostics"] = dict(
            image=img, cf_map=cf_map, tumor_mask=blob["tumor_mask"],
            axis_mm=axis_mm, time_signal=time_signal,
            time_signal_filtered=time_signal, delay_grid=delay_grid,
        )

    return result