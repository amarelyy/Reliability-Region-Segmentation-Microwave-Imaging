"""
src/signal_processing.py

Advanced signal processing stack for UM-BMID Gen-2:
1. Complex Calibration (Division by Empty Chamber)
2. Bandpass Filtering (4-6 GHz based on Isabel Olaya Lopez)
3. ICZT Time-Domain Transform
4. Depth Gain Compensation
5. Hybrid TVSVD Clutter Suppression
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
    """
    # Avoid division by zero
    s_empty_safe = np.where(np.abs(s_empty) < 1e-12, 1e-12, s_empty)
    return s_raw / s_empty_safe

def apply_bandpass_filter(signal_fd, freqs, low_cut=4e9, high_cut=6e9):
    """
    Filter frekuensi 4-6 GHz.
    Berdasarkan paper Isabel Olaya Lopez, sebagian besar informasi tumor
    ada di sub-band ini.
    """
    try:
        # fs harus setidaknya 2x frekuensi tertinggi (Nyquist)
        sos = butter(4, [low_cut, high_cut], btype='band', fs=freqs[-1]*2, output='sos')
        filtered = sosfilt(sos, signal_fd, axis=0)
        return filtered
    except Exception:
        return signal_fd

def apply_depth_gain(time_signal, delay_grid, alpha_db_per_cm=0.7):
    """
    Kompensasi attenuasi berdasarkan jarak tempuh sinyal.
    time_signal: (n_time, n_ant)
    delay_grid: (n_ant, n_pix) ATAU (n_ant, n_y, n_x)
    """
    # Flatten delay grid jika masih dalam bentuk 2D spasial
    if delay_grid.ndim > 2:
        n_ant = delay_grid.shape[0]
        delay_flat = delay_grid.reshape(n_ant, -1)
    else:
        delay_flat = delay_grid

    # Estimasi jarak rata-rata per antena
    avg_distance_m = np.median(delay_flat, axis=1) 
    
    alpha_np = alpha_db_per_cm / 100 
    gain_linear = 10 ** (alpha_np * avg_distance_m / 20)
    
    # Broadcast ke time_signal (n_time, n_ant)
    return time_signal * gain_linear[np.newaxis, :]

def to_time_domain(fd_signal, window_alpha=0.25, n_time_pts=N_TIME_PTS):
    """
    fd_signal: (n_freq, n_ant) complex frequency-domain signal for one scan.
    Returns (n_time_pts, n_ant) complex time-domain signal via ICZT.
    """
    if not ICZT_AVAILABLE:
        raise ImportError(
            "umbmid.sigproc.iczt not importable — copy the umbmid/ package "
            "into the repo root (see README) before running the pipeline."
        )
    window = tukey(fd_signal.shape[0], alpha=window_alpha)
    fd_windowed = fd_signal * window[:, None]
    return iczt(fd_windowed, ini_t=TIME_START_S, fin_t=TIME_STOP_S,
                n_time_pts=n_time_pts, ini_f=FREQ_START_HZ, fin_f=FREQ_STOP_HZ)


def get_time_axis(n_time_pts=N_TIME_PTS):
    return np.linspace(TIME_START_S, TIME_STOP_S, n_time_pts)


# ============================================================================
# Hybrid TVSVD (Ursula's Phase 5.5 approach)
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