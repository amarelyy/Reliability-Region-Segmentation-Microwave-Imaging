"""
src/signal_processing.py

Frequency->time domain transform (ICZT, 1-8GHz per UM-BMID spec) and 
Ursula's hybrid TVSVD clutter suppression.
Includes safety wrappers for calibration and bandpass filtering.
"""

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, sosfilt

try:
    from umbmid.sigproc import iczt
    ICZT_AVAILABLE = True
except ImportError:
    ICZT_AVAILABLE = False

# Confirmed against UM-BMID Gen-2 docs
FREQ_START_HZ = 1e9
FREQ_STOP_HZ = 8e9
TIME_START_S = 0.0
TIME_STOP_S = 6e-9
N_TIME_PTS = 1024
C_LIGHT = 3e8

def calibrate_s_parameters(s_raw, s_empty):
    """
    Perform complex division calibration as per Eq. (1) in Isabel Olaya Lopez.
    S_cal = S_adi / S_emp
    Includes safety checks to prevent float/scalar errors.
    """
    # Ensure inputs are numpy arrays
    s_raw = np.asarray(s_raw, dtype=np.complex128)
    s_empty = np.asarray(s_empty, dtype=np.complex128)
    
    # Avoid division by zero
    s_empty_safe = np.where(np.abs(s_empty) < 1e-12, 1e-12, s_empty)
    return s_raw / s_empty_safe

def apply_bandpass_filter(signal_fd, freqs, low_cut=4e9, high_cut=6e9):
    """
    Filter frekuensi 4-6 GHz based on Isabel Olaya Lopez findings.
    """
    try:
        signal_fd = np.asarray(signal_fd, dtype=np.complex128)
        freqs = np.asarray(freqs)
        sos = butter(4, [low_cut, high_cut], btype='band', fs=freqs[-1]*2, output='sos')
        filtered = sosfilt(sos, signal_fd, axis=0)
        return filtered
    except Exception as e:
        print(f"[Warning] Bandpass filter failed: {e}. Returning original signal.")
        return signal_fd

def to_time_domain(fd_signal, window_alpha=0.25, n_time_pts=N_TIME_PTS):
    """
    Safe wrapper for ICZT. Ensures input is a proper 2D complex array.
    fd_signal: (n_freq, n_ant) complex frequency-domain signal.
    Returns (n_time_pts, n_ant) complex time-domain signal.
    """
    if not ICZT_AVAILABLE:
        raise ImportError("umbmid.sigproc.iczt not found.")
        
    # CRITICAL SAFETY CHECKS
    fd_signal = np.asarray(fd_signal, dtype=np.complex128)
    
    if fd_signal.ndim != 2:
        raise ValueError(f"Expected 2D array (n_freq, n_ant), got shape {fd_signal.shape}")
        
    if not np.isfinite(fd_signal).all():
        print("[Warning] Non-finite values detected in fd_signal. Cleaning...")
        fd_signal = np.nan_to_num(fd_signal, nan=0.0, posinf=1e12, neginf=-1e12)

    # Apply Tukey window to reduce spectral leakage
    window = tukey(fd_signal.shape[0], alpha=window_alpha)
    fd_windowed = fd_signal * window[:, None]
    
    # Call the ORIGINAL iczt function from umbmid
    try:
        return iczt(fd_windowed, ini_t=TIME_START_S, fin_t=TIME_STOP_S,
                    n_time_pts=n_time_pts, ini_f=FREQ_START_HZ, fin_f=FREQ_STOP_HZ)
    except Exception as e:
        raise RuntimeError(f"ICZT execution failed: {e}")

def get_time_axis(n_time_pts=N_TIME_PTS):
    return np.linspace(TIME_START_S, TIME_STOP_S, n_time_pts)


# ============================================================================
# Hybrid TVSVD (Ursula's Phase 5.5 approach - KEPT INTACT)
# ============================================================================
def select_tvsvd_rank_adaptive(S, min_rank=1, max_energy=0.98):
    """Kneedle elbow detection on cumulative-energy curve."""
    energy = S ** 2
    cum = np.cumsum(energy) / np.sum(energy)
    n = len(cum)
    if n < 3:
        return max(min_rank, int(np.argmax(cum >= 0.90)))

    x_norm = np.arange(n) / (n - 1)
    p1 = np.array([x_norm[0], cum[0]])
    p2 = np.array([x_norm[-1], cum[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return max(min_rank, int(np.argmax(cum >= 0.90)))

    line_unit = line_vec / line_len
    pts = np.stack([x_norm, cum], axis=1) - p1
    proj_len = pts @ line_unit
    proj_pts = np.outer(proj_len, line_unit)
    dist = np.linalg.norm(pts - proj_pts, axis=1)

    knee = max(min_rank, int(np.argmax(dist)))
    hard_cap = int(np.argmax(cum >= max_energy))
    return min(knee, hard_cap) if hard_cap > 0 else knee


def select_tvsvd_rank_hybrid(S, Vt, energy_lower=0.01, flatness_thresh=0.85,
                              max_energy=0.98):
    """
    A component is removed only if it is BOTH energetic AND spatially uniform.
    """
    knee = select_tvsvd_rank_adaptive(S, max_energy=max_energy)
    energy_frac = (S ** 2) / np.sum(S ** 2)

    remove_mask = np.zeros(len(S), dtype=bool)
    for k in range(knee):
        row = Vt[k]
        flatness = np.abs(np.mean(row)) / (np.sqrt(np.mean(row ** 2)) + 1e-12)
        if energy_frac[k] > energy_lower and flatness > flatness_thresh:
            remove_mask[k] = True
    return remove_mask


def apply_hybrid_tvsvd(time_signal):
    """time_signal: (n_time, n_ant) complex. Returns (filtered_signal, n_removed)."""
    U, S, Vt = np.linalg.svd(time_signal, full_matrices=False)
    remove_mask = select_tvsvd_rank_hybrid(S, Vt)
    S_filtered = S.copy()
    S_filtered[remove_mask] = 0.0
    filtered = U @ np.diag(S_filtered) @ Vt
    return filtered, int(remove_mask.sum())

def apply_depth_gain(time_signal, delay_grid, alpha_db_per_cm=0.7):
    """
    Kompensasi attenuasi berdasarkan jarak tempuh sinyal.
    Handles both 2D and 3D delay grids safely.
    """
    # Flatten delay grid if needed
    if delay_grid.ndim > 2:
        n_ant = delay_grid.shape[0]
        delay_flat = delay_grid.reshape(n_ant, -1)
    else:
        delay_flat = delay_grid

    avg_distance_m = np.median(delay_flat, axis=1) 
    
    alpha_np = alpha_db_per_cm / 100 
    gain_linear = 10 ** (alpha_np * avg_distance_m / 20)
    
    # Broadcast to (n_time, n_ant)
    return time_signal * gain_linear[np.newaxis, :]