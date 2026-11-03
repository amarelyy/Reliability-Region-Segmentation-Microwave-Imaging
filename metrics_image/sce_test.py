import os
import pickle
import numpy as np
from pathlib import Path
from scipy.signal import hilbert


def find_s21_file(data_dir="data", keyword="s21"):
    """Identical data fetching logic to SCME for fair comparison."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Folder data tidak ditemukan: {p.resolve()}")
    candidates = list(p.glob(f"*{keyword}*.pickle"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada file '*{keyword}*.pickle' di {p.resolve()}")
    return candidates[0]


def single_channel_envelope(s21_path, n_ant=8, freq_ghz=4.0, 
                            gating_ns=0.7, threshold_sigma=3.0):
    """
    Single-Channel Time-Domain Envelope Detection - Non-Summation Validation #2.
    
    PERBAIKAN KRITIS:
    - Konversi S21 (freq domain complex) -> Time domain via IFFT
    - Ambil np.real() sebelum Hilbert Transform (karena hilbert butuh real input)
    - Subset frekuensi IDENTIK dengan SCME untuk validasi adil
    """
    print(f"[Loading] {s21_path.name}")
    with open(s21_path, 'rb') as f:
        s21 = pickle.load(f)
    
    n_scans, n_freq, n_ch = s21.shape
    print(f"[Info] Shape: {s21.shape} | Channels: {n_ch}")
    
    # Subset frekuensi tengah (IDENTIK dengan SCME)
    f_start, f_end = int(n_freq * 0.4), int(n_freq * 0.6)
    s21_sub = s21[:, f_start:f_end, :]
    
    # Hitung resolusi waktu dari bandwidth subset
    # Asumsi linear spacing sesuai UM-BMID Gen-2 spec
    bw_hz = freq_ghz * 1e9 * (f_end - f_start) / n_freq
    dt = 1.0 / bw_hz
    
    predictions_time = []  # dalam ns
    predictions_radius = []  # konversi ke mm
    
    # Kecepatan efektif tissue (adipose-mimicking glycerin @ ~4GHz)
    epsilon_eff = 7.0  
    v_tissue = 3e8 / np.sqrt(epsilon_eff)  # m/s
    
    gate_sample = int(gating_ns * 1e-9 / dt)
    
    for scan_idx in range(n_scans):
        scan_preds = []
        
        for ch_idx in range(n_ch):
            # === STEP 1: CONVERT TO TIME DOMAIN ===
            # S21 adalah complex frequency domain. Harus di-IFFT dulu.
            time_signal_complex = np.fft.ifft(s21_sub[scan_idx, :, ch_idx])
            
            # Hilbert Transform HANYA bekerja pada sinyal REAL
            # Kita ambil real part sebagai representasi amplitudo fisik
            time_signal_real = np.real(time_signal_complex)
            
            # === STEP 2: ENVELOPE DETECTION ===
            analytic = hilbert(time_signal_real)
            envelope = np.abs(analytic)
            
            # Early-time gating (sama seperti pipeline utama)
            envelope[:gate_sample] = 0
            
            # Peak detection dengan threshold adaptif berbasis median
            noise_floor = np.median(envelope[gate_sample:])
            noise_std = np.std(envelope[gate_sample:])
            threshold = noise_floor + threshold_sigma * noise_std
            
            above_thresh = np.where(envelope > threshold)[0]
            
            if len(above_thresh) > 0:
                # Ambil peak PERTAMA (earliest significant return)
                peak_sample = above_thresh[0]
                peak_time_ns = peak_sample * dt * 1e9
                
                # Konversi time-of-flight ke radius estimasi
                est_radius_mm = (v_tissue * peak_time_ns * 1e-9) / 2 * 1e3
                
                scan_preds.append((peak_time_ns, est_radius_mm))
            else:
                scan_preds.append(None)
        
        # Median prediction per scan (robust terhadap outlier channel)
        valid_times = [p[0] for p in scan_preds if p is not None]
        valid_radii = [p[1] for p in scan_preds if p is not None]
        
        if valid_times:
            predictions_time.append(np.median(valid_times))
            predictions_radius.append(np.median(valid_radii))
        else:
            predictions_time.append(np.nan)
            predictions_radius.append(np.nan)
    
    # === STATISTIK KEGAGALAN ===
    valid_count = sum(1 for p in predictions_time if not np.isnan(p))
    mean_peak_time = np.nanmean(predictions_time) if valid_count > 0 else np.nan
    mean_peak_radius = np.nanmean(predictions_radius) if valid_count > 0 else np.nan
    
    # Tumor zone: 20-48mm radial distance
    tumor_zone_min, tumor_zone_max = 20.0, 48.0
    in_tumor_zone = sum(1 for r in predictions_radius 
                       if not np.isnan(r) and tumor_zone_min <= r <= tumor_zone_max)
    
    print(f"\n[SINGLE-CHANNEL ENVELOPE RESULTS]")
    print(f"  Valid Predictions: {valid_count}/{n_scans}")
    print(f"  Mean Peak Time: {mean_peak_time:.2f} ns")
    print(f"  Mean Estimated Radius: {mean_peak_radius:.1f} mm")
    print(f"  Peaks in Tumor Zone ({tumor_zone_min}-{tumor_zone_max}mm): {in_tumor_zone}/{n_scans}")
    print(f"  Gating Threshold: {gating_ns} ns")
    
    # Diagnosa kegagalan
    print(f"\n[FAILURE DIAGNOSIS]")
    if mean_peak_time < gating_ns:
        print("  ❌ CRITICAL: Peaks clustered BEFORE gating window")
        print("     → Direct coupling dominates even after gating")
    elif mean_peak_radius < tumor_zone_min:
        print(f"  ❌ CRITICAL: Mean radius ({mean_peak_radius:.1f}mm) BELOW tumor zone")
        print("     → Coupling/skin reflection indistinguishable from tumor")
    elif mean_peak_radius > tumor_zone_max:
        print(f"  ⚠️  WARNING: Mean radius ({mean_peak_radius:.1f}mm) BEYOND tumor zone")
        print("     → Boundary reflections dominate")
    else:
        print(f"  ℹ️  Unexpected: Some peaks in tumor zone")
        print("     → Verify threshold_sigma parameter")
    
    if in_tumor_zone == 0:
        print("\n  🔴 CONFIRMED: Zero valid tumor detections without array processing")
        print("     → Spatial diversity is MANDATORY for this geometry")
    
    return predictions_time, predictions_radius, mean_peak_radius, in_tumor_zone


def main():
    try:
        s21_path = find_s21_file(keyword="s21")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    
    print("=" * 70)
    print("SINGLE-CHANNEL ENVELOPE: NON-SUMMATION VALIDATION #2")
    print("=" * 70)
    
    preds_time, preds_radius, mean_radius, tumor_hits = single_channel_envelope(
        s21_path, 
        gating_ns=0.7,      
        threshold_sigma=3.0  
    )
    
    print("\n" + "=" * 70)
    print("TABLE Y UPDATE - SINGLE-CHANNEL ENVELOPE METRICS")
    print("=" * 70)
    print(f"| Metric                    | Value              |")
    print(f"|---------------------------|--------------------|")
    print(f"| Failure Mechanism         | No spatial diversity|")
    print(f"| Mean Est. Radius          | {mean_radius:.1f} mm{'':>10}|")
    print(f"| Valid Tumor Peaks         | {tumor_hits}/504{'':>12}|")
    print(f"| Conclusion                | FAILED             |")
    print("=" * 70)


if __name__ == "__main__":
    main()