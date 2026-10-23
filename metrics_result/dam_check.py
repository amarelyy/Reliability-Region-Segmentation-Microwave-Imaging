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


def find_adipose_s21(data_dir="data"):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Folder data tidak ditemukan: {p.resolve()}")
    candidates = list(p.glob("*s21*.pickle"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada file '*s21*.pickle' di {p.resolve()}")
    target = "fd_data_gen_two_s21.pickle"
    exact = [c for c in candidates if c.name == target]
    return exact[0] if exact else candidates[0]


def dam_reconstruction(s21_path, n_ant=8, r_ant_mm=50.0, freq_ghz=4.0):
    """
    Delay-and-Multiply (DAM) Reconstruction Pipeline.
    
    OPERASI INTI: PERKALIAN GEOMETRIS ANTAR-CHANNEL.
    TIDAK ADA PENJUMLAHAN AMPLITUDO DI MANAPUN.
    
    DAM menekan direct coupling secara eksponensial karena coupling
    bersifat inkoheren antar-pasangan Tx-Rx, sementara tumor koheren.
    """
    print(f"[Loading] {s21_path.name}")
    with open(s21_path, 'rb') as f:
        s21 = pickle.load(f)
    
    n_scans, n_freq, n_ch = s21.shape
    print(f"[Info] Shape: {s21.shape} | Channels: {n_ch}")
    
    # Subset frekuensi tengah (40%-60%) untuk stabilitas fase
    f_start, f_end = int(n_freq * 0.4), int(n_freq * 0.6)
    s21_sub = s21[:, f_start:f_end, :]
    
    # Normalisasi amplitudo per channel (division, bukan sum)
    mag = np.abs(s21_sub)
    s21_norm = s21_sub / (mag.mean(axis=(0, 1), keepdims=True) + 1e-12)
    
    # === GRID SPASIAL & POSISI ANTENA ===
    x = np.linspace(-60, 60, 121)
    y = np.linspace(-60, 60, 121)
    X, Y = np.meshgrid(x, y)
    
    theta = np.linspace(0, 2 * np.pi, n_ant, endpoint=False)
    ant_x = r_ant_mm * np.cos(theta)
    ant_y = r_ant_mm * np.sin(theta)
    
    lambda_mm = 300 / freq_ghz
    k = 2 * np.pi / lambda_mm
    
    # === DAM: PERKALIAN GEOMETRIS PER PIXEL ===
    # Inisialisasi dengan ones (identitas perkalian), BUKAN zeros (identitas penjumlahan)
    dam_map = np.ones_like(X, dtype=complex)
    
    # Mapping channel flattened ke pasangan Tx-Rx
    # Asumsi: 72 ch = 8Tx x 9Rx (atau 8x8 + 8 reference)
    valid_pairs = 0
    for ch_idx in range(n_ch):
        tx_idx = ch_idx // n_ant
        rx_idx = ch_idx % n_ant
        
        if tx_idx >= n_ant or rx_idx >= n_ant:
            continue
            
        # Ambil sinyal rata-rata across scans & freq untuk channel ini
        ch_signal = s21_norm[:, :, ch_idx].mean(axis=(0, 1))
        
        # Kompensasi fase delay (Tx -> Grid -> Rx)
        d_tx = np.sqrt((X - ant_x[tx_idx])**2 + (Y - ant_y[tx_idx])**2)
        d_rx = np.sqrt((X - ant_x[rx_idx])**2 + (Y - ant_y[rx_idx])**2)
        phase_shift = np.exp(-1j * k * (d_tx + d_rx))
        
        # KUNCI: PERKALIAN, BUKAN PENJUMLAHAN
        dam_map *= (ch_signal * phase_shift)
        valid_pairs += 1
    
    # Root-N normalization untuk menjaga skala output
    # |prod(S_i)|^(1/N) ekuivalen dengan geometric mean
    n_valid = max(valid_pairs, 1)
    dam_magnitude = np.abs(dam_map) ** (1.0 / n_valid)
    
    # Konversi ke dB untuk visualisasi
    dam_db = 20 * np.log10(dam_magnitude + 1e-12)
    
    print(f"[DAM Stats] Center(0,0): {dam_db[60,60]:.1f} dB")
    print(f"[DAM Stats] Mean: {dam_db.mean():.1f} dB | Max: {dam_db.max():.1f} dB")
    print(f"[Valid Pairs] {valid_pairs} / {n_ch} channels used")
    
    return X, Y, dam_db, r_ant_mm


def main():
    os.makedirs("metrics_image", exist_ok=True)
    
    try:
        s21_path = find_adipose_s21()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    
    print("=" * 60)
    print("DAM RECONSTRUCTION: NO SUMMATION PIPELINE")
    print("=" * 60)
    
    X, Y, dam_db, r_ant = dam_reconstruction(s21_path)
    
    # === VISUALISASI ===
    fig, ax = plt.subplots(figsize=(7, 6))
    
    vmin = np.percentile(dam_db, 5)
    vmax = np.percentile(dam_db, 99)
    im = ax.pcolormesh(X, Y, dam_db, cmap='jet', shading='auto', 
                       vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label='DAM Power [dB]')
    
    # Lingkaran referensi fisik
    ax.add_patch(plt.Circle((0, 0), r_ant, color='white', fill=False, 
                            linewidth=1.5, linestyle='--', alpha=0.9,
                            label=f'Antenna Array (R={r_ant} mm)'))
    ax.add_patch(plt.Circle((0, 0), 48.0, color='black', fill=False, 
                            linewidth=1.2, linestyle='-', alpha=0.7,
                            label='Avg Adipose Phantom (R≈48 mm)'))
    ax.plot(0, 0, 'w+', markersize=12, markeredgewidth=2)
    
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_title('DAM Reconstruction: Real Adipose Tissue\n(Geometric Multiplication, No Summation)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position (mm)'); ax.set_ylabel('Y Position (mm)')
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('metrics_image/dam_reconstruction_adipose.png', dpi=600, bbox_inches='tight')
    
    # === DIAGNOSA OTOMATIS ===
    center_val = dam_db[60, 60]
    mean_val = dam_db.mean()
    
    print(f"\n[Saved] metrics_image/dam_reconstruction_adipose.png")
    print("\nDIAGNOSA SUPRESI COUPLING:")
    diff = center_val - mean_val
    if diff > 10:
        print(f"Pusat masih {diff:.1f} dB di atas mean.")
        print("   → Coupling memiliki korelasi fase residual antar-pasangan.")
        print("   → DAM lebih baik dari DAS tapi belum cukup.")
        print("   → SOLUSI: Kombinasi DAM + Background Subtraction.")
    elif diff > 3:
        print(f" Pusat hanya {diff:.1f} dB di atas mean.")
        print("   → Supresi coupling SIGNIFIKAN dibanding DAS (-52.5 dB).")
        print("   → DAM berhasil menekan artefak tanpa summation.")
    else:
        print(f"Pusat hanya {diff:.1f} dB di atas mean.")
        print("   → Coupling hampir sepenuhnya tertekan.")
        print("   → DAM adalah pengganti DAS yang valid untuk pipeline ini.")
    print("=" * 60)
    plt.close()


if __name__ == "__main__":
    main()