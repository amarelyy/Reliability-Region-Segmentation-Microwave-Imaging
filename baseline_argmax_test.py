"""
baseline_argmax_test.py (IMPROVED PHYSICS STACK - COMPLETE)

Standalone isolation test with improved physics pipeline:
1. Snellius-Corrected Straight Ray Delay Model
2. Effective Velocity based on fat/fib fraction
3. Bandpass Filter (4-6 GHz) based on Isabel Olaya Lopez (2024)
4. Depth Gain Compensation for deep-tissue equalization
"""

import argparse
import time as time_module
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import butter, sosfilt
from joblib import Parallel, delayed

from src.data_loading import load_all_data
from select_phantoms import build_phantom_table, stratified_select
from src import physics
from src import signal_processing as sp
from src import beamforming as bf

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
GRID_STEP_MM = 0.25
GRID_MARGIN_FACTOR = 1.5
SCR_ROI_RADIUS_MM = 2.0
DETECTION_THRESHOLD_MM = 20.0
CENTER_EXCLUDE_RADIUS_MM = 5.0   # radius around the geometric center (0,0) to
                                  # mask out before argmax — the antenna ring's
                                  # rotational symmetry makes this pixel
                                  # artificially bright/coherent regardless of
                                  # any real tumor (see chat discussion)

# Frequency settings for Bandpass Filter (Isabel Olaya Lopez recommendation)
FREQ_START_HZ = 1e9
FREQ_STOP_HZ = 8e9
N_FREQ_PTS = 1001 # Adjust based on your actual dataset resolution
freq_axis = np.linspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ_PTS)

CF_BASELINE_N_TRIALS = 10
CF_BASELINE_SEED = 12345
_CF_BASELINE_CACHE = {}   # phant_id -> (baseline_mean, baseline_std)
_DAS_BASELINE_CACHE = {}  # phant_id -> (das_baseline_mean, das_baseline_std)

TOP_K_CANDIDATES = 5
CANDIDATE_MIN_SEPARATION_MM = 10.0   # minimum spacing between candidate peaks
CANDIDATE_LOCAL_WINDOW_MM = 3.0       # window radius for the area/compactness feature
HESSIAN_STEP_MM = 1.0   # finite-difference step, matches expected real-tumor blob scale

# ============================================================================
# IMPROVED SIGNAL PROCESSING HELPERS
# ============================================================================

def apply_bandpass_filter(signal_fd, freqs, low_cut=4e9, high_cut=6e9):
    """
    Filter frekuensi 4-6 GHz.
    Berdasarkan paper Isabel Olaya Lopez, sebagian besar informasi tumor
    ada di sub-band ini. Di bawah itu resolusi buruk, di atas itu attenuasi tinggi.
    """
    try:
        sos = butter(4, [low_cut, high_cut], btype='band', fs=freqs[-1]*2, output='sos')
        filtered = sosfilt(sos, signal_fd, axis=0)
        return filtered
    except Exception:
        return signal_fd

def apply_depth_gain(time_signal, delay_grid, alpha_db_per_cm=0.7):
    """
    Kompensasi attenuasi berdasarkan jarak tempuh sinyal.
    Ini 'mengangkat' sinyal dari bagian dalam phantom agar seimbang dengan permukaan.
    time_signal: (n_time, n_ant)
    delay_grid: (n_ant, n_pix) -> kita ambil rata-rata delay per antena sebagai estimasi kedalaman
    """
    C_LIGHT = 3e8
    # Estimasi jarak rata-rata yang ditempuh sinyal untuk setiap antena
    avg_distance_m = np.median(delay_grid, axis=1) * C_LIGHT 
    
    # Faktor gain linear
    alpha_np = alpha_db_per_cm / 100 # konversi ke Neper/meter (approx)
    gain_linear = 10 ** (alpha_np * avg_distance_m / 20)
    
    # Reshape gain_per_ant agar bisa dikalikan dengan time_signal (n_time, n_ant)
    gain_per_ant = gain_linear[np.newaxis, :] 
    
    return time_signal * gain_per_ant

# ============================================================================
# CF & DAS DEBIASING HELPERS (Original Logic Preserved)
# ============================================================================

def compute_cf_baseline(delay_grid, time_axis, n_ant, n_trials=CF_BASELINE_N_TRIALS,
                         seed=CF_BASELINE_SEED):
    """
    Runs das_coherent_cf on N_TRIALS of pure independent random complex noise
    through the SAME delay_grid used for the real scan. Isolates exactly how
    much CF is inflated by antenna-ring geometry ALONE.
    """
    rng = np.random.default_rng(seed)
    n_time = len(time_axis)
    cf_maps = []
    for _ in range(n_trials):
        noise_signal = (rng.standard_normal((n_time, n_ant))
                         + 1j * rng.standard_normal((n_time, n_ant)))
        _, cf_map, _ = bf.das_coherent_cf(noise_signal, time_axis, delay_grid)
        cf_maps.append(cf_map)
    cf_stack = np.stack(cf_maps, axis=0)
    return cf_stack.mean(axis=0), cf_stack.std(axis=0)


def get_cf_baseline_cached(phant_id, delay_grid, time_axis, n_ant):
    """Baseline only depends on phantom geometry (delay_grid), not the
    actual scan/tumor — cache per phant_id so repeated phantoms in a sample
    don't recompute it."""
    if phant_id not in _CF_BASELINE_CACHE:
        _CF_BASELINE_CACHE[phant_id] = compute_cf_baseline(delay_grid, time_axis, n_ant)
    return _CF_BASELINE_CACHE[phant_id]


def debias_cf_subtraction(cf_map, baseline_mean):
    return np.clip(cf_map - baseline_mean, 0, None)


def debias_cf_ratio(cf_map, baseline_mean, eps=1e-6):
    return cf_map / (baseline_mean + eps)


def debias_cf_zscore(cf_map, baseline_mean, baseline_std, eps=1e-6):
    z = (cf_map - baseline_mean) / (baseline_std + eps)
    return np.clip(z, 0, None)   # negative z (below chance-level coherence) -> 0 weight

def compute_das_magnitude_baseline(delay_grid, time_axis, n_ant,
                                    n_trials=CF_BASELINE_N_TRIALS,
                                    seed=CF_BASELINE_SEED + 999):
    """
    Runs das_coherent (raw DAS, NO CF) on N trials of pure random complex
    noise through the SAME delay_grid. Captures how much raw DAS magnitude
    is inflated at each pixel by antenna-ring geometry ALONE.
    """
    rng = np.random.default_rng(seed)
    n_time = len(time_axis)
    das_maps = []
    for _ in range(n_trials):
        noise_signal = (rng.standard_normal((n_time, n_ant))
                        + 1j * rng.standard_normal((n_time, n_ant)))
        das_mag = bf.das_coherent(noise_signal, time_axis, delay_grid)
        das_maps.append(das_mag)
    das_stack = np.stack(das_maps, axis=0)
    return das_stack.mean(axis=0), das_stack.std(axis=0)


def get_das_baseline_cached(phant_id, delay_grid, time_axis, n_ant):
    """Cache per phant_id — baseline depends only on geometry."""
    key = str(phant_id)
    if key not in _DAS_BASELINE_CACHE:
        _DAS_BASELINE_CACHE[key] = \
            compute_das_magnitude_baseline(delay_grid, time_axis, n_ant)
    return _DAS_BASELINE_CACHE[key]


def debias_das_zscore(das_image_raw, baseline_mean, baseline_std, eps=1e-6):
    """
    Z-score debiasing applied directly to raw DAS magnitude.
    Removes the geometry-only coherent-addition bias at center.
    """
    z = (das_image_raw - baseline_mean) / (baseline_std + eps)
    return np.clip(z, 0, None)


def debias_das_ratio(das_image_raw, baseline_mean, eps=1e-6):
    """
    Ratio debiasing: das_image_raw / baseline_mean.
    Converts absolute magnitude to 'multiple of geometry-only expectation'.
    """
    return das_image_raw / (baseline_mean + eps)

# ============================================================================
# CANDIDATE EXTRACTION HELPERS
# ============================================================================

def find_top_k_peaks(img, axis_mm, k=TOP_K_CANDIDATES, min_separation_mm=CANDIDATE_MIN_SEPARATION_MM):
    """Greedy non-max-suppression peak finder: repeatedly take the brightest
    remaining pixel, mask out a min_separation_mm radius around it, repeat k
    times. Returns a list of (px_mm, py_mm, iy, ix) tuples, brightest first."""
    step_mm = axis_mm[1] - axis_mm[0]
    sep_px = max(1, int(round(min_separation_mm / step_mm)))
    search_img = img.copy()
    peaks = []
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(search_img), search_img.shape)
        if not np.isfinite(search_img[iy, ix]):
            break
        px, py = axis_mm[ix], axis_mm[iy]
        peaks.append((px, py, iy, ix))
        y0, y1 = max(0, iy - sep_px), min(search_img.shape[0], iy + sep_px + 1)
        x0, x1 = max(0, ix - sep_px), min(search_img.shape[1], ix + sep_px + 1)
        search_img[y0:y1, x0:x1] = -np.inf
    return peaks


HESSIAN_STEP_MM = 1.0   # finite-difference step, matches expected real-tumor blob scale
                         # (rather than pixel-level noise at the 0.25mm grid resolution)


def compute_hessian_eigenratio(img, axis_mm, iy, ix, step_mm=HESSIAN_STEP_MM):
    """
    Hessian (2nd-derivative) eigenvalue ratio at a candidate peak — the
    'blobness' feature. A round, compact reflector (real tumor) curves down
    similarly in every direction -> eigenvalues of similar magnitude -> ratio near 1.
    An elongated ridge (grating-lobe artifact) curves sharply in one
    direction, gently in the other -> ratio near 0.
    """
    step_mm_actual = axis_mm[1] - axis_mm[0]
    step_px = max(1, int(round(step_mm / step_mm_actual)))
    ny, nx = img.shape

    iy_p, iy_m = min(iy + step_px, ny - 1), max(iy - step_px, 0)
    ix_p, ix_m = min(ix + step_px, nx - 1), max(ix - step_px, 0)

    if iy_p == iy or iy_m == iy or ix_p == ix or ix_m == ix:
        return 0.0   # candidate too close to grid edge for a meaningful stencil

    center_val = img[iy, ix]
    ixx = img[iy, ix_p] - 2 * center_val + img[iy, ix_m]
    iyy = img[iy_p, ix] - 2 * center_val + img[iy_m, ix]
    ixy = (img[iy_p, ix_p] - img[iy_p, ix_m] - img[iy_m, ix_p] + img[iy_m, ix_m]) / 4.0

    hessian = np.array([[ixx, ixy], [ixy, iyy]])
    eigvals = np.abs(np.linalg.eigvalsh(hessian))
    max_eig, min_eig = eigvals.max(), eigvals.min()
    if max_eig < 1e-12:
        return 0.0
    return float(min_eig / max_eig)


def extract_candidate_features(img, cf_map, axis_mm, iy, ix, px_mm, py_mm,
                                shell_center_mm, birads, breast_radius_mm,
                                window_mm=CANDIDATE_LOCAL_WINDOW_MM,
                                scr_roi_radius_mm=SCR_ROI_RADIUS_MM):
    """9-feature vector for one candidate peak: CF-at-peak, CF-to-background
    ratio, local area/compactness, distance to shell/ring center, local SCR,
    Hessian eigenvalue ratio (blobness), BI-RADS class."""
    cf_at_peak = float(cf_map[iy, ix])
    flat_idx = iy * cf_map.shape[1] + ix
    cf_background_mean = float(np.mean(np.delete(cf_map.ravel(), flat_idx)))
    cf_background_ratio = cf_at_peak / (cf_background_mean + 1e-10)

    step_mm = axis_mm[1] - axis_mm[0]
    win_px = max(1, int(round(window_mm / step_mm)))
    y0, y1 = max(0, iy - win_px), min(img.shape[0], iy + win_px + 1)
    x0, x1 = max(0, ix - win_px), min(img.shape[1], ix + win_px + 1)
    window = img[y0:y1, x0:x1]
    peak_val = img[iy, ix]
    area_fraction = float(np.mean(window >= 0.5 * peak_val)) if peak_val > 0 else 0.0

    dist_to_shell_center = float(np.hypot(px_mm - shell_center_mm[0], py_mm - shell_center_mm[1]))
    dist_to_ring_center = float(np.hypot(px_mm, py_mm))
    dist_to_ring_center_ratio = dist_to_ring_center / breast_radius_mm if breast_radius_mm > 0 else 0.0

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    roi_mask = np.sqrt((gx_mm - px_mm) ** 2 + (gy_mm - py_mm) ** 2) <= scr_roi_radius_mm
    outside_mean = img[~roi_mask].mean() if (~roi_mask).any() else 1e-10
    local_scr = float(20 * np.log10(np.abs(peak_val) / (np.abs(outside_mean) + 1e-10))) \
        if peak_val > 0 and outside_mean > 0 else 0.0

    hessian_eigenratio = compute_hessian_eigenratio(img, axis_mm, iy, ix)

    return dict(
        cf_at_peak=cf_at_peak,
        cf_background_ratio=cf_background_ratio,
        area_fraction=area_fraction,
        dist_to_shell_center_mm=dist_to_shell_center,
        dist_to_ring_center_mm=dist_to_ring_center,
        dist_to_ring_center_ratio=dist_to_ring_center_ratio,
        local_scr_db=local_scr,
        hessian_eigenratio=hessian_eigenratio,
        birads=birads,
    )


def generate_candidates(img, cf_map, axis_mm, gt_x_mm, gt_y_mm, shell_center_mm, birads,
                         breast_radius_mm, phant_id,
                         k=TOP_K_CANDIDATES, label_threshold_mm=DETECTION_THRESHOLD_MM,
                         artifact_exclusion_radius_mm=CENTER_EXCLUDE_RADIUS_MM):
    """Top-K candidates + features + label (1 = within label_threshold_mm of
    ground truth, else 0) for one image. Candidates within the artifact
    exclusion radius are ALWAYS labeled 0."""
    peaks = find_top_k_peaks(img, axis_mm, k=k)
    candidates = []
    for px_mm, py_mm, iy, ix in peaks:
        features = extract_candidate_features(img, cf_map, axis_mm, iy, ix, px_mm, py_mm,
                                               shell_center_mm, birads, breast_radius_mm)
        le_to_gt = float(np.hypot(px_mm - gt_x_mm, py_mm - gt_y_mm))
        residual_x_mm = float(px_mm - gt_x_mm)
        residual_y_mm = float(py_mm - gt_y_mm)
        dist_to_ring_center = features["dist_to_ring_center_mm"]
        if dist_to_ring_center <= artifact_exclusion_radius_mm:
            label = 0   # known artifact zone — never a genuine detection
        else:
            label = 1 if le_to_gt <= label_threshold_mm else 0
        candidates.append(dict(phant_id=phant_id, peak_x_mm=px_mm, peak_y_mm=py_mm,
                                le_to_gt_mm=le_to_gt,
                                residual_x_mm=residual_x_mm, residual_y_mm=residual_y_mm,
                                label=label, **features))
    return candidates

def build_grid(breast_radius_mm, margin_factor=GRID_MARGIN_FACTOR, step_mm=GRID_STEP_MM):
    grid_radius_mm = breast_radius_mm * margin_factor
    axis_mm = np.arange(-grid_radius_mm, grid_radius_mm + step_mm, step_mm)
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    return grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm


def argmax_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                               roi_radius_mm=SCR_ROI_RADIUS_MM,
                               exclude_center_radius_mm=None):
    """Global argmax peak + LE + fixed-ROI SCR, matching Aurel's original
    compute_metrics() exactly (20*log10, not 10*log10; fixed 2mm ROI, not
    the phantom's actual tumor radius)."""
    search_img = img
    if exclude_center_radius_mm is not None:
        gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
        center_dist = np.sqrt(gx_mm ** 2 + gy_mm ** 2)
        search_img = np.where(center_dist <= exclude_center_radius_mm, -np.inf, img)

    peak = np.unravel_index(np.argmax(search_img), search_img.shape)
    px = axis_mm[peak[1]]
    py = axis_mm[peak[0]]
    le = np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2)

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    tumor_dist = np.sqrt((gx_mm - gt_x_mm) ** 2 + (gy_mm - gt_y_mm) ** 2)
    tumor_mask = tumor_dist <= roi_radius_mm
    clutter_mask = ~tumor_mask

    tumor_peak = img[tumor_mask].max() if tumor_mask.any() else 1e-10
    clutter_mean = img[clutter_mask].mean() if clutter_mask.any() else 1e-10

    if clutter_mean <= 0 or tumor_peak <= 0:
        scr = 0.0
    else:
        scr = 20 * np.log10(np.abs(tumor_peak) / (np.abs(clutter_mean) + 1e-10))

    on_edge = (abs(abs(px) - grid_radius_mm) < 1e-6) or (abs(abs(py) - grid_radius_mm) < 1e-6)
    return le, scr, on_edge, px, py


def weighted_centroid_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                                          roi_radius_mm=SCR_ROI_RADIUS_MM,
                                          threshold_fraction=0.5,
                                          exclude_center_radius_mm=None):
    """Alternative to pure argmax: find the peak, threshold the image, isolate
    the connected blob containing that peak, then take the intensity-weighted
    centroid of that whole blob."""
    search_img = img
    if exclude_center_radius_mm is not None:
        gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
        center_dist = np.sqrt(gx_mm ** 2 + gy_mm ** 2)
        search_img = np.where(center_dist <= exclude_center_radius_mm, -np.inf, img)

    peak = np.unravel_index(np.argmax(search_img), search_img.shape)
    peak_val = img[peak]

    threshold = threshold_fraction * peak_val
    binary_mask = img >= threshold
    labeled_mask, n_blobs = ndimage.label(binary_mask)
    blob_label = labeled_mask[peak]

    if blob_label == 0 or n_blobs == 0:
        px, py = axis_mm[peak[1]], axis_mm[peak[0]]
    else:
        blob_mask = labeled_mask == blob_label
        ys, xs = np.where(blob_mask)
        weights = img[ys, xs]
        px = float(np.average(axis_mm[xs], weights=weights))
        py = float(np.average(axis_mm[ys], weights=weights))

    le = float(np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2))
    # SCR calculation omitted for brevity in this variant, same as argmax version
    return le, 0.0, False, px, py


def largest_blob_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                                     roi_radius_mm=SCR_ROI_RADIUS_MM,
                                     percentile_threshold=90.0):
    """Largest-connected-AREA localization — picks whichever connected region
    has the most PIXELS, regardless of whether it contains the single
    brightest point."""
    threshold = np.percentile(img, percentile_threshold)
    binary_mask = img >= threshold
    labeled_mask, n_blobs = ndimage.label(binary_mask)

    if n_blobs == 0:
        peak = np.unravel_index(np.argmax(img), img.shape)
        px, py = axis_mm[peak[1]], axis_mm[peak[0]]
    else:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, n_blobs + 1))
        largest_label = int(np.argmax(sizes)) + 1
        blob_mask = labeled_mask == largest_label
        ys, xs = np.where(blob_mask)
        weights = img[ys, xs]
        px = float(np.average(axis_mm[xs], weights=weights))
        py = float(np.average(axis_mm[ys], weights=weights))

    le = float(np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2))
    return le, 0.0, False, px, py


def reconstruct_and_score_all_variants(scan_idx, s21, tumor_model, id_to_original_idx, verbose=False, use_tvsvd=False):
    """
    IMPROVED PHYSICS PIPELINE:
    1. Effective Velocity (Fat/Fib composition)
    2. Bandpass Filter (4-6 GHz) based on Isabel Olaya Lopez (2024)
    3. Snellius-Corrected Delay Grid
    4. Depth Gain Compensation
    """
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    
    # 1. EFFECTIVE VELOCITY CALCULATION
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, _ = physics.compute_effective_velocity(fat_frac, fib_frac)

    ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    s21_idx = int(row["original_s21_idx"])
    fd_scan = s21[s21_idx]
    
    # Calibration (Original Logic)
    emp_ref_id = row.get("emp_ref_id", None)
    calibration_applied = False
    if emp_ref_id is not None and not pd.isna(emp_ref_id) and int(emp_ref_id) in id_to_original_idx:
        emp_idx = id_to_original_idx[int(emp_ref_id)]
        fd_scan = fd_scan - s21[emp_idx]
        calibration_applied = True

    diag = dict(
        original_s21_idx=s21_idx,
        old_buggy_idx=scan_idx,
        idx_drift=s21_idx - scan_idx,
        calibration_applied=calibration_applied,
        phant_id=row["phant_id"],
        gt_x_mm=float(row["tumor_x_mm"]),
        gt_y_mm=float(row["tumor_y_mm"]),
        breast_radius_mm=breast_radius_mm,
        tum_in_fib=row.get("tum_in_fib", np.nan),
    )

    # 2. BANDPASS FILTER (4-6 GHz)
    # Based on Isabel Olaya Lopez: "almost all discriminative power to the four-to-six gigahertz sub-band"
    fd_scan_filtered = apply_bandpass_filter(fd_scan, freq_axis, low_cut=4e9, high_cut=6e9)

    # Time Domain Transform
    time_signal = sp.to_time_domain(fd_scan_filtered)
    
    # Optional: TVSVD
    n_sv_removed = 0
    if use_tvsvd:
        time_signal, n_sv_removed = sp.apply_hybrid_tvsvd(time_signal)

    time_axis = sp.get_time_axis(time_signal.shape[0])
    td_mag = np.abs(time_signal)

    # Grid Setup
    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
    grid_x_m = grid_x_mm.ravel() / 1000.0
    grid_y_m = grid_y_mm.ravel() / 1000.0

    # 3. SNELLIUS-CORRECTED DELAY GRID
    # Replaces two_medium_delay/bent_ray_3layer_delay for geometric precision
    delay_grid = physics.snellius_corrected_delay(
        geom["ant_x"], geom["ant_y"], 
        grid_x_m, grid_y_m,
        breast_radius_mm / 1000.0, v_tissue
    )
    delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)

    # 4. DEPTH GAIN COMPENSATION
    # Equalizes signal strength across depth to aid deep-tumor detection
    time_signal = apply_depth_gain(time_signal, delay_grid)

    gt_x_mm = float(row["tumor_x_mm"])
    gt_y_mm = float(row["tumor_y_mm"])

    # Beamforming Variants (Using improved time_signal)
    images = {
        "Raw DAS": bf.das_coherent(time_signal, time_axis, delay_grid),
        "Raw DMAS": bf.dmas(td_mag, time_axis, delay_grid),
    }
    das_image_raw, cf_map, das_cf_img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)
    images["DAS+CF"] = das_cf_img
    
    dmas_image_raw = bf.dmas(td_mag, time_axis, delay_grid)
    dmas_cf_img, _ = bf.dmas_cf(time_signal, td_mag, time_axis, delay_grid)
    images["DMAS+CF"] = dmas_cf_img

    # CF Debiasing (Original Logic)
    n_ant = time_signal.shape[1]
    baseline_mean, baseline_std = get_cf_baseline_cached(row["phant_id"], delay_grid, time_axis, n_ant)
    
    cf_sub = debias_cf_subtraction(cf_map, baseline_mean)
    cf_ratio = debias_cf_ratio(cf_map, baseline_mean)
    cf_zscore = debias_cf_zscore(cf_map, baseline_mean, baseline_std)

    images["DAS+CF-debiased-sub"] = das_image_raw * cf_sub
    images["DAS+CF-debiased-ratio"] = das_image_raw * cf_ratio
    images["DAS+CF-debiased-zscore"] = das_image_raw * cf_zscore
    images["DMAS+CF-debiased-sub"] = dmas_image_raw * cf_sub
    images["DMAS+CF-debiased-ratio"] = dmas_image_raw * cf_ratio
    images["DMAS+CF-debiased-zscore"] = dmas_image_raw * cf_zscore

    # DAS Magnitude Debiassing (Original Logic)
    das_baseline_mean, das_baseline_std = get_das_baseline_cached(row["phant_id"], delay_grid, time_axis, n_ant)
    real_energy = float(np.mean(das_image_raw[das_image_raw > 0])) if np.any(das_image_raw > 0) else 1.0
    baseline_energy = float(np.mean(das_baseline_mean[das_baseline_mean > 0])) if np.any(das_baseline_mean > 0) else 1.0
    amplitude_scale = real_energy / (baseline_energy + 1e-10)
    
    das_debiased_z = debias_das_zscore(das_image_raw, das_baseline_mean * amplitude_scale, das_baseline_std * amplitude_scale)
    das_debiased_r = debias_das_ratio(das_image_raw, das_baseline_mean)
    das_combined = das_debiased_z * cf_zscore

    images["DAS-debiased-zscore"] = das_debiased_z
    images["DAS-debiased-ratio"] = das_debiased_r
    images["DAS-debiased+CF-zscore"] = das_combined

    # Scoring and Candidate Generation
    CENTROID_TEST_VARIANTS = ("Raw DAS", "DAS+CF", "DAS+CF-debiased-zscore")
    results = {}
    for name, img in images.items():
        le, scr, on_edge, px, py = argmax_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        results[name] = dict(localization_error_mm=le, scr_db=scr, on_edge=on_edge)

        le_m, scr_m, on_edge_m, px_m, py_m = argmax_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm, exclude_center_radius_mm=CENTER_EXCLUDE_RADIUS_MM)
        results[f"{name} (masked)"] = dict(localization_error_mm=le_m, scr_db=scr_m, on_edge=on_edge_m)

        if name in CENTROID_TEST_VARIANTS:
            le_c, scr_c, on_edge_c, px_c, py_c = weighted_centroid_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
            results[f"{name} (centroid)"] = dict(localization_error_mm=le_c, scr_db=scr_c, on_edge=on_edge_c)
            
            le_b, scr_b, on_edge_b, px_b, py_b = largest_blob_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
            results[f"{name} (largest-blob)"] = dict(localization_error_mm=le_b, scr_db=scr_b, on_edge=on_edge_b)

    # Candidate Generation for SVM
    birads = row.get("birads", np.nan)
    candidates = generate_candidates(images["DAS+CF"], cf_map, axis_mm, gt_x_mm, gt_y_mm, shell_center_mm=(0.0, 0.0), birads=birads, breast_radius_mm=breast_radius_mm, phant_id=row["phant_id"])
    candidates_k10 = generate_candidates(images["DAS+CF"], cf_map, axis_mm, gt_x_mm, gt_y_mm, shell_center_mm=(0.0, 0.0), birads=birads, breast_radius_mm=breast_radius_mm, phant_id=row["phant_id"], k=10)

    return results, diag, candidates, candidates_k10

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scans", type=int, default=None,
                         help="Limit to a random sample of N scans (smoke test).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for phantom selection AND scan sampling.")
    parser.add_argument("--n-phantoms", type=int, default=30,
                         help="Number of phantoms to restrict to, via select_phantoms.py's "
                              "stratified (size x BI-RADS) selection.")
    parser.add_argument("--one-per-phantom", action="store_true",
                         help="Pick exactly ONE scan per selected phantom (randomly, seeded).")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Parallel worker processes (joblib). 1=serial, -1=all cores.")
    parser.add_argument("--use-tvsvd", action="store_true",
                         help="Apply Ursula's hybrid TVSVD filtering before beamforming.")
    parser.add_argument("--save-candidates", type=str, default=None,
                         help="If given a path, saves the full k=10 candidate dataset to CSV.")
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]

    # Restrict to the stratified phantom subset
    if args.n_phantoms > 0:
        phantom_table = build_phantom_table(tumor_model)
        selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
        selected_ids = selected["phant_id"].tolist()
        tumor_model = tumor_model[tumor_model["phant_id"].isin(selected_ids)].reset_index(drop=True)
        print(f"\nRestricted to {len(selected_ids)} phantoms (seed={args.seed}, "
              f"stratified by size x BI-RADS):")
        print(f"  {selected_ids}")
        print(f"  -> {len(tumor_model)} scans available within these phantoms")
    else:
        print(f"\nUsing all {tumor_model['phant_id'].nunique()} available phantoms "
              f"({len(tumor_model)} scans) — no phantom restriction.")

    # RANDOM sample logic
    rng = np.random.default_rng(args.seed)
    if args.one_per_phantom:
        scan_indices = []
        for pid, group in tumor_model.groupby("phant_id"):
            scan_indices.append(int(rng.choice(group.index.values)))
        scan_indices = np.array(sorted(scan_indices))
        print(f"\n--one-per-phantom: selected exactly 1 scan from each of "
              f"{len(scan_indices)} distinct phantoms")
    elif args.n_scans is not None and args.n_scans < len(tumor_model):
        scan_indices = rng.choice(len(tumor_model), size=args.n_scans, replace=False)
    else:
        scan_indices = np.arange(len(tumor_model))
    
    n_scans = len(scan_indices)
    print(f"\nRunning improved physics pipeline on {n_scans} scans (seed={args.seed})\n")

    base_names = ("Raw DAS", "Raw DMAS", "DAS+CF", "DMAS+CF",
                  "DAS+CF-debiased-sub", "DAS+CF-debiased-ratio", "DAS+CF-debiased-zscore",
                  "DMAS+CF-debiased-sub", "DMAS+CF-debiased-ratio", "DMAS+CF-debiased-zscore")
    variant_rows = {name: [] for name in base_names}
    for name in base_names:
        variant_rows[f"{name} (masked)"] = []
    for name in ("Raw DAS", "DAS+CF", "DAS+CF-debiased-zscore"):
        variant_rows[f"{name} (centroid)"] = []
        variant_rows[f"{name} (largest-blob)"] = []
    
    diag_rows = []
    all_candidates = []
    all_candidates_k10 = []
    failed = []
    t_start = time_module.time()

    def _process_one(i, idx):
        """Runs in a worker process."""
        try:
            scan_results, diag, candidates, candidates_k10 = reconstruct_and_score_all_variants(
                idx, s21, tumor_model, id_to_original_idx, verbose=(i < 3), use_tvsvd=args.use_tvsvd)
            return ("ok", idx, scan_results, diag, candidates, candidates_k10)
        except Exception as e:
            return ("fail", idx, str(e))

    print(f"Running with --n-jobs={args.n_jobs} "
          f"({'parallel' if args.n_jobs != 1 else 'serial'})...")
    outcomes = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_one)(i, int(idx)) for i, idx in enumerate(scan_indices)
    )

    for outcome in outcomes:
        if outcome[0] == "ok":
            _, idx, scan_results, diag, candidates, candidates_k10 = outcome
            for name, metrics in scan_results.items():
                variant_rows[name].append(metrics)
            diag["scan_idx"] = idx
            diag_rows.append(diag)
            for c in candidates:
                c["scan_idx"] = idx
            all_candidates.extend(candidates)
            for c in candidates_k10:
                c["scan_idx"] = idx
            all_candidates_k10.extend(candidates_k10)
        else:
            _, idx, err = outcome
            failed.append((idx, err))

    print(f"  {len(outcomes) - len(failed)}/{n_scans} scans done "
          f"({time_module.time() - t_start:.0f}s elapsed)")

    if failed:
        print(f"\n{len(failed)} scans failed. First failure: {failed[0]}")

    # ========================================================================
    # DIAGNOSTICS & SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: index alignment fix + calibration engagement")
    print("=" * 70)
    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df) > 0:
        n_drifted = (diag_df["idx_drift"] != 0).sum()
        print(f"Scans where original_s21_idx differs from the old buggy index : "
              f"{n_drifted}/{len(diag_df)}")
        print(f"  mean |drift|  : {diag_df['idx_drift'].abs().mean():.1f}")
        n_calibrated = diag_df["calibration_applied"].sum()
        print(f"Scans where empty-chamber calibration was actually applied : "
              f"{n_calibrated}/{len(diag_df)}")

    print("\n" + "=" * 70)
    print("BASELINE ARGMAX TEST — SUMMARY (Improved Physics Stack)")
    print("=" * 70)
    for name, rows in variant_rows.items():
        if not rows:
            print(f"{name}: no successful scans")
            continue
        df = pd.DataFrame(rows)
        mean_le = df["localization_error_mm"].mean()
        median_le = df["localization_error_mm"].median()
        detection_rate = (df["localization_error_mm"] <= DETECTION_THRESHOLD_MM).mean()
        edge_fraction = df["on_edge"].mean()
        print(f"\n{name}  (n={len(df)})")
        print(f"  mean LE   : {mean_le:.2f} mm")
        print(f"  median LE : {median_le:.2f} mm")
        print(f"  detection @20mm : {detection_rate:.1%}")
        print(f"  on-edge fraction : {edge_fraction:.1%}")

    total_elapsed = time_module.time() - t_start
    print(f"\nTotal runtime: {total_elapsed:.0f}s")

    # SVM Candidate Dataset Summary
    print("\n" + "=" * 70)
    print("SVM CANDIDATE DATASET SUMMARY (DAS+CF image, top-{} peaks/scan)".format(TOP_K_CANDIDATES))
    print("=" * 70)
    cand_df = pd.DataFrame(all_candidates)
    if len(cand_df) > 0:
        n_pos = int((cand_df["label"] == 1).sum())
        n_neg = int((cand_df["label"] == 0).sum())
        print(f"Total candidates : {len(cand_df)}  (from {n_scans} scans, "
              f"{len(cand_df) / n_scans:.1f} avg/scan)")
        print(f"Positive (label=1, real tumor)     : {n_pos}  ({n_pos / len(cand_df):.1%})")
        print(f"Negative (label=0, false positive) : {n_neg}  ({n_neg / len(cand_df):.1%})")
        
        feature_cols = ["cf_at_peak", "cf_background_ratio", "area_fraction",
                         "dist_to_shell_center_mm", "dist_to_ring_center_mm",
                         "dist_to_ring_center_ratio", "local_scr_db", "hessian_eigenratio"]
        print("\nFeature ranges (min / mean / max), split by label:")
        for col in feature_cols:
            pos_vals = cand_df.loc[cand_df["label"] == 1, col]
            neg_vals = cand_df.loc[cand_df["label"] == 0, col]
            print(f"  {col}:")
            if len(pos_vals) > 0:
                print(f"    label=1 : {pos_vals.min():.3f} / {pos_vals.mean():.3f} / {pos_vals.max():.3f}")
            print(f"    label=0 : {neg_vals.min():.3f} / {neg_vals.mean():.3f} / {neg_vals.max():.3f}")

    # Save candidates if requested
    if args.save_candidates:
        cand_df_k10_save = pd.DataFrame(all_candidates_k10)
        if len(cand_df_k10_save) > 0:
            cand_df_k10_save.to_csv(args.save_candidates, index=False)
            print(f"\nSaved {len(cand_df_k10_save)} candidates to {args.save_candidates}")
        else:
            print(f"\nNo candidates to save — skipping --save-candidates={args.save_candidates}")

if __name__ == "__main__":
    main()