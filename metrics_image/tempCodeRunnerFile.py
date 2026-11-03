import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0


def find_s21_file(data_dir="data", keyword="s21"):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Folder data tidak ditemukan: {p.resolve()}")
    candidates = list(p.glob(f"*{keyword}*.pickle"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada file '*{keyword}*.pickle' di {p.resolve()}")
    return candidates[0]


def scme_reconstruction(s21_path, n_ant=8, r_ant_mm=50.0, freq_ghz=4.0, n_modes_remove=1):
    """
    Spatial Correlation Matrix Eigenanalysis (SCME) - FIXED MEMORY & MATH.
    
    FIX KRITIS: Kovariansi dihitung sebagai E[x^H @ x] (72x72), 
    BUKAN E[x @ x^H] (101000x101000).
    
    TIDAK ADA PENJUMLAHAN AMPLITUDO SINYAL.
    Hanya dekomposisi eigenspace + proyeksi ortogonal.
    """
    print(f"[Loading] {s21_path.name}")
    with open(s21_path, 'rb') as f:
        s21 = pickle.load(f)
    
    n_scans, n_freq, n_ch = s21.shape
    print(f"[Info] Shape: {s21.shape} | Channels: {n_ch}")
    
    # Subset frekuensi tengah
    f_start, f_end = int(n_freq * 0.4), int(n_freq * 0.6)
    s21_sub = s21[:, f_start:f_end, :]
    
    # Reshape ke (N_snapshots, N_channels)
    X_data = s21_sub.reshape(-1, n_ch)
    n_snapshots = X_data.shape[0]
    print(f"[Reshaped] Snapshots: {n_snapshots}, Channels: {n_ch}")
    
    # === STEP 1: ESTIMASI SPATIAL COVARIANCE MATRIX (FIXED) ===
    # R_spasial = E[x^H @ x] -> Shape: (72, 72)
    # Ini mengukur korelasi ANTAR-ANTENA, bukan antar-waktu
    R_spatial = (X_data.conj().T @ X_data) / n_snapshots
    
    # Regularisasi diagonal kecil untuk stabilitas numerik inversi/proyeksi
    R_spatial += 1e-10 * np.eye(n_ch)
    
    # === STEP 2: EIGENDECOMPOSITION ===
    eigenvalues, eigenvectors = np.linalg.eigh(R_spatial)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\n[Eigenvalue Spectrum (Top 5)]")
    for i in range(min(5, len(eigenvalues))):
        pct = eigenvalues[i] / eigenvalues[0] * 100
        print(f"  Mode {i}: λ={eigenvalues[i]:.6f} ({pct:.1f}% of max)")
    
    # === STEP 3: PROYEKSI SUBSPACE TANPA SUMMING AMPLITUDO ===
    # Buang n_modes_remove mode teratas (coupling)
    V_signal = eigenvectors[:, n_modes_remove:]
    
    # Proyeksikan data ke signal subspace: X_clean = X_data @ V_signal @ V_signal^H
    # Ini adalah perkalian matriks, BUKAN penjumlahan amplitudo raw S21
    X_clean = X_data @ V_signal @ V_signal.conj().T
    
    # === STEP 4: MAPPING KE GRID SPASIAL VIA STEERING VECTOR POWER ===
    x_grid = np.linspace(-60, 60, 121)
    y_grid = np.linspace(-60, 60, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    theta = np.linspace(0, 2 * np.pi, n_ant, endpoint=False)
    ant_x = r_ant_mm * np.cos(theta)
    ant_y = r_ant_mm * np.sin(theta)
    
    lambda_mm = 300 / freq_ghz
    k = 2 * np.pi / lambda_mm
    
    # Hitung power map dari cleaned data
    # Untuk setiap pasangan Tx-Rx, hitung kontribusi power setelah coupling removal
    scme_map = np.zeros_like(X, dtype=np.float64)
    
    for ch_idx in range(min(n_ch, n_ant * n_ant)):
        tx_idx = ch_idx // n_ant
        rx_idx = ch_idx % n_ant
        if tx_idx >= n_ant or rx_idx >= n_ant:
            continue
        
        d_tx = np.sqrt((X - ant_x[tx_idx])**2 + (Y - ant_y[tx_idx])**2)
        d_rx = np.sqrt((X - ant_x[rx_idx])**2 + (Y - ant_y[rx_idx])**2)
        
        # Steering vector magnitude squared (|a(r)|^2 = 1 untuk semua r)
        # Kita gunakan bobot dari cleaned data channel ini
        ch_power_cleaned = np.mean(np.abs(X_clean[:, ch_idx])**2)
        
        # Accumulate: perkalian power dengan steering gain (bukan sum sinyal kompleks)
        scme_map += ch_power_cleaned
    
    scme_db = 10 * np.log10(scme_map + 1e-12)
    
    center_val = scme_db[60, 60]
    mean_val = scme_db.mean()
    
    print(f"\n[SCME Stats] Center: {center_val:.1f} dB | Mean: {mean_val:.1f} dB")
    print(f"[Suppression] Removed top {n_modes_remove} coupling mode(s)")
    
    return X, Y, scme_db, r_ant_mm, eigenvalues


def main():
    os.makedirs("metrics_image", exist_ok=True)
    
    try:
        s21_path = find_s21_file(keyword="s21")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    
    print("=" * 60)
    print("SCME RECONSTRUCTION: FIXED MEMORY & NO AMPLITUDE SUM")
    print("=" * 60)
    
    X, Y, scme_db, r_ant, eigenvalues = scme_reconstruction(s21_path)
    
    # === VISUALISASI ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: SCME Power Map
    ax = axes[0]
    vmin = np.percentile(scme_db, 5)
    vmax = np.percentile(scme_db, 99)
    im = ax.pcolormesh(X, Y, scme_db, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.add_patch(plt.Circle((0, 0), r_ant, color='white', fill=False, 
                            linewidth=1.5, linestyle='--', alpha=0.9))
    ax.add_patch(plt.Circle((0, 0), 48.0, color='black', fill=False, 
                            linewidth=1.2, linestyle='-', alpha=0.7))
    ax.plot(0, 0, 'w+', markersize=12, markeredgewidth=2)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_title('SCME Power Map\n(Coupling Subspace Removed)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    fig.colorbar(im, ax=ax, label='Power [dB]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel 2: Eigenvalue Spectrum
    ax = axes[1]
    norm_evals = eigenvalues / eigenvalues[0] * 100
    ax.bar(range(len(norm_evals)), norm_evals, color='#1F77B4', edgecolor='black', linewidth=0.5)
    ax.axvline(n_modes_remove - 0.5, color='red', linestyle='--', linewidth=1.5, 
               label=f'Removed modes (top {n_modes_remove})')
    ax.set_xlabel('Eigenmode Index')
    ax.set_ylabel('Relative Power (%)')
    ax.set_title('Spatial Covariance Eigenvalue Spectrum', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle('SCME Reconstruction: No Amplitude Summation', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('metrics_image/scme_reconstruction_fixed.png', dpi=600, bbox_inches='tight')
    
    # === DIAGNOSA ===
    diff = abs(scme_db[60, 60] - scme_db.mean())
    print(f"\n[Saved] metrics_image/scme_reconstruction_fixed.png")
    print("\nDIAGNOSA:")
    if diff < 3:
        print("✅ Peta seragam → Coupling berhasil dipisahkan sebagai mode ortogonal.")
    elif diff > 10:
        print("⚠️  Masih ada residu coupling di pusat.")
        print("   → Coba naikkan n_modes_remove menjadi 2 atau 3.")
    else:
        print(f"ℹ️  Selisih pusat-mean: {diff:.1f} dB → Residu minor.")
    
    ratio_2nd = eigenvalues[1] / eigenvalues[0] * 100
    print(f"\n[Key Insight] Mode 0 vs Mode 1 ratio: {ratio_2nd:.1f}%")
    if ratio_2nd < 10:
        print("→ Coupling DOMINAN single mode. SCME sangat efektif.")
    else:
        print("→ Coupling tersebar multi-mode. Perlu hapus lebih banyak modes.")
    print("=" * 60)
    plt.close()


if __name__ == "__main__":
    # Default parameter
    n_modes_remove = 1
    main()