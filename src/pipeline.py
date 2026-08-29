"""
src/pipeline.py

reconstruct_scan() — self-contained, per-scan reconstruction with physics-informed
preprocessing stack.

Improvements over baseline:
1. Complex Calibration (Division by Empty Chamber) per Isabel Olaya Lopez Eq. (1).
2. Snellius-Corrected Bistatic Delay: Precise geometry for 60-degree separation.
3. Effective Velocity: Calculates v_tissue per-scan based on fat/fib fraction.
4. Bandpass Filtering: Restricts signal to 4-6 GHz for optimal dielectric contrast.
5. Depth Gain Compensation: Equalizes signal strength across depth.
"""

from time import perf_counter
import numpy as np

from . import physics
from . import signal_processing as sp
from . import beamforming as bf
from . import blob_detection as bd
from . import metrics as mx

DEFAULT_GRID_MARGIN_FACTOR = 1.5
DEFAULT_GRID_STEP_MM = 1.0

# Delay grids depend only on phantom-level geometry. 
_DELAY_CACHE = {}

def _delay_cache_key(phant_id, use_snellius, ant_rad_mm, breast_radius_mm,
                      v_tissue, margin_factor, grid_step_mm, shell_center):
    """Generate a unique key for caching delay grids."""
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
    Fully self-contained per-scan reconstruction with improved physics pipeline.
    """
    t_start = perf_counter()

    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    
    # 1. Calculate Effective Velocity based on phantom composition
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, eps_tissue = physics.compute_effective_velocity(fat_frac, fib_frac)

    # Antenna geometry setup
    ant_rad_cm = float(row.get("ant_rad", 21.5))
    ant_rad_mm = ant_rad_cm * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    # ---- Signal Pre-processing ----
    # Get raw scan index
    s21_idx = int(row["original_s21_idx"])
    fd_raw = s21[s21_idx]
    
    # A. COMPLEX CALIBRATION (CRITICAL FIX)
    # Get empty chamber reference index
    emp_ref_id = row.get("emp_ref_id", None)
    if emp_ref_id is not None and not np.isnan(emp_ref_id):
        emp_idx = id_to_original_idx.get(int(emp_ref_id), None)
        if emp_idx is not None:
            fd_empty = s21[emp_idx]
            fd_scan = sp.calibrate_s_parameters(fd_raw, fd_empty)
        else:
            fd_scan = fd_raw # Fallback if ref missing
    else:
        fd_scan = fd_raw

    # B. Bandpass Filter (4-6 GHz)
    if use_bandpass and freq_axis is not None:
        fd_scan = sp.apply_bandpass_filter(fd_scan, freq_axis, low_cut=4e9, high_cut=6e9)

    # C. Time Domain Transform
    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])

    # ---- Clutter Suppression (Optional) ----
    if use_tvsvd:
        time_signal_filtered, n_removed = sp.apply_hybrid_tvsvd(time_signal)
    else:
        time_signal_filtered, n_removed = time_signal, 0

    # ---- Imaging Grid Setup ----
    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(
        breast_radius_mm, margin_factor, grid_step_mm)
    grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0
    shell_center_m = (shell_center[0] / 1000.0, shell_center[1] / 1000.0)

    # ---- Delay Grid Calculation (Snellius Bistatic Precise) ----
    cache_key = _delay_cache_key(
        row["phant_id"], use_snellius, ant_rad_mm, breast_radius_mm, 
        v_tissue, margin_factor, grid_step_mm, shell_center
    )
    
    if cache_key in _DELAY_CACHE:
        delay_grid = _DELAY_CACHE[cache_key]
    else:
        if use_snellius:
            # USE THE NEW PRECISE BISTATIC FUNCTION
            delay_grid = physics.snellius_bistatic_delay_precise(
                geom["ant_x"], geom["ant_y"], 
                geom["ant_x_b"], geom["ant_y_b"], # Pass Rx positions!
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue
            )
        else:
            # Fallback to legacy two-medium model
            delay_grid = physics.two_medium_delay(
                geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue, shell_center=shell_center_m,
            )
        
        # Reshape once for caching and usage
        delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)
        _DELAY_CACHE[cache_key] = delay_grid

    # D. Apply Depth Gain AFTER delay grid is ready
    if use_depth_gain:
        time_signal_filtered = sp.apply_depth_gain(time_signal_filtered, delay_grid)
    
    # ---- Beamforming ----
    if beamformer == "das":
        if use_cf:
            _, cf_map, img = bf.das_coherent_cf(time_signal_filtered, time_axis, delay_grid)
        else:
            img = bf.das_coherent(time_signal_filtered, time_axis, delay_grid)
            cf_map = None
    elif beamformer == "dmas":
        td_mag = np.abs(time_signal_filtered)
        if use_cf:
            img, cf_map = bf.dmas_cf(time_signal_filtered, td_mag, time_axis, delay_grid)
        else:
            img = bf.dmas(td_mag, time_axis, delay_grid)
            cf_map = None
    else:
        raise ValueError(f"Unknown beamformer: {beamformer!r}")

    # ---- Blob Extraction + Localization ----
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
            time_signal_filtered=time_signal_filtered, delay_grid=delay_grid,
        )

    return result