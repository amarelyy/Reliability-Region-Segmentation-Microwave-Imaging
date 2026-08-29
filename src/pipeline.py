"""
src/pipeline.py

reconstruct_scan() — self-contained, per-scan reconstruction with physics-informed
preprocessing stack.

Improvements over baseline:
1. Snellius-Corrected Delay: Replaces unstable 3-layer bent-ray for homogeneous phantoms.
2. Effective Velocity: Calculates v_tissue per-scan based on fat/fib fraction.
3. Bandpass Filtering: Restricts signal to 4-6 GHz (Isabel Olaya Lopez, 2024) for optimal contrast.
4. Depth Gain Compensation: Equalizes signal strength across depth to aid deep-tumor detection.
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
# Caching by phantom identity avoids recomputing an identical delay grid ~15x over.
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


def reconstruct_scan(scan_idx, s21, tumor_model, freq_axis=None,
                      beamformer="das", use_snellius=True, use_cf=True,
                      use_tvsvd=False, use_bandpass=True, use_depth_gain=True,
                      shell_center=(0.0, 0.0),
                      margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                      grid_step_mm=DEFAULT_GRID_STEP_MM,
                      return_diagnostics=False):
    """
    Fully self-contained per-scan reconstruction with improved physics pipeline.

    Parameters
    ----------
    scan_idx : int, row index into tumor_model / first axis of s21
    s21 : complex128 array (n_scans, n_freq, n_ant)
    tumor_model : DataFrame from data_loading.build_tumor_model()
    freq_axis : array-like, frequency points in Hz (required for bandpass)
    beamformer : 'das' | 'dmas'
    use_snellius : bool — Use Snellius-corrected straight ray vs legacy two_medium
    use_cf : bool — coherence-factor weighting on/off
    use_tvsvd : bool — hybrid TVSVD clutter suppression on/off
    use_bandpass : bool — Apply 4-6 GHz filter (Isabel Olaya Lopez recommendation)
    use_depth_gain : bool — Apply exponential gain to compensate for attenuation
    shell_center : (x_mm, y_mm) offset of the phantom shell from chamber origin

    Returns a dict with images, peak location, GT, and all metrics.
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
    fd_scan = s21[scan_idx]  # (n_freq, n_ant)
    
    # A. Bandpass Filter (4-6 GHz) to isolate dielectric contrast
    if use_bandpass and freq_axis is not None:
        fd_scan = sp.apply_bandpass_filter(fd_scan, freq_axis, low_cut=4e9, high_cut=6e9)

    # B. Time Domain Transform
    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])

    # C. Depth Gain Compensation (Equalize signal strength across depth)
    # Note: We need a rough delay estimate for gain. We'll use a placeholder 
    # or compute it after grid generation. For now, we apply it post-grid.
    
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

    # ---- Delay Grid Calculation (Snellius Corrected) ----
    cache_key = _delay_cache_key(
        row["phant_id"], use_snellius, ant_rad_mm, breast_radius_mm, 
        v_tissue, margin_factor, grid_step_mm, shell_center
    )
    
    if cache_key in _DELAY_CACHE:
        delay_grid = _DELAY_CACHE[cache_key]
    else:
        if use_snellius:
            # Use our new robust model for homogeneous phantoms
            delay_grid = physics.snellius_corrected_delay(
                geom["ant_x"], geom["ant_y"], 
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue
            )
        else:
            # Fallback to legacy two-medium model if needed for comparison
            delay_grid = physics.two_medium_delay(
                geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue, shell_center=shell_center_m,
            )
        
        delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)
        _DELAY_CACHE[cache_key] = delay_grid

    # D. Apply Depth Gain AFTER delay grid is ready (for accurate distance estimation)
        # ... setelah delay_grid dihitung ...
    delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape) # Menjadi (n_ant, ny, nx)

    # 4. DEPTH GAIN COMPENSATION
    if use_depth_gain:
        # Teruskan delay_grid yang sudah direshape, fungsi apply_depth_gain akan handle flattening
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
        raise ValueError(f"Unknown beamformer: {beamformer!r} (expected 'das' or 'dmas')")

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