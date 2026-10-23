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
    """Mencari file S21 adiposa secara otomatis untuk menghindari error path/spasi."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Folder data tidak ditemukan: {p.resolve()}")
        
    candidates = list(p.glob("*adi*.pickle"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"Tidak ada file '*adi*.pickle' di folder {p.resolve()}")
        
    target_name = "fd_data_s21_adi (1).pickle"
    exact_match = [c for c in candidates if c.name == target_name]
    
    if exact_match:
        return exact_match[0]
    else:
        print(f"[Warning] '{target_name}' tidak ditemukan persis. Menggunakan: {candidates[0].name}")
        return candidates[0]

def compute_das_adipose(s21_path, n_ant=8, r_ant_mm=50.0, freq_ghz=4.0):
    """
    Menghitung peta energi DAS dari data adiposa murni.
    Shape input: (N_scans, N_freq, 72_channels_flattened)
    Asumsi mapping: Channel ke-i merepresentasikan pasangan Tx-Rx tertentu.
    Untuk forensik coupling, kita gunakan pendekatan 'Synthetic Focus' 
    dengan mengasumsikan 72 ch berasal dari kombinasi antena circular.
    """
    print(f"[Loading] Membaca data adiposa: {s21_path.name}")
    with open(s21_path, 'rb') as f:
        s21 = pickle.load(f)
        
    n_scans, n_freq, n_ch = s21.shape
    print(f"[Info] Raw shape: {s21.shape}")
    
    # Subset frekuensi broadband (30%-70%) untuk resolusi spasial maksimal
    f_start, f_end = int(n_freq * 0.3), int(n_freq * 0.7)
    s21_sub = s21[:, f_start:f_end, :]
    
    # Virtual Beamforming Grid (-60mm s/d 60mm)
    x = np.linspace(-60, 60, 121)
    y = np.linspace(-60, 60, 121)
    X, Y = np.meshgrid(x, y)
    
    # Posisi antena circular array
    theta = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    ant_x = r_ant_mm * np.cos(theta)
    ant_y = r_ant_mm * np.sin(theta)
    
    lambda_mm = 300 / freq_ghz
    k = 2 * np.pi / lambda_mm
    
    # DAS Power Map (Complex accumulation)
    das_map = np.zeros_like(X, dtype=complex)
    
    # Iterasi per channel flattened
    # Catatan: Tanpa metadata mapping Tx-Rx eksplisit, kita asumsikan 
    # channel terdistribusi merata antar pasangan antena.
    # Ini cukup valid untuk mendeteksi DIRECT COUPLING di pusat.
    for ch_idx in range(n_ch):
        # Ambil data rata-rata across scans & freq untuk channel ini
        ch_signal = s21_sub[:, :, ch_idx].mean(axis=(0, 1))
        
        # Tentukan indeks Tx dan Rx berdasarkan channel index
        # Asumsi sederhana: Tx = ch_idx // n_ant, Rx = ch_idx % n_ant
        tx_idx = ch_idx // n_ant
        rx_idx = ch_idx % n_ant
        
        # Handle overflow jika n_ch > n_ant*n_ant
        if tx_idx >= n_ant or rx_idx >= n_ant:
            continue
            
        # Hitung delay total (Tx -> Grid -> Rx)
        d_tx = np.sqrt((X - ant_x[tx_idx])**2 + (Y - ant_y[tx_idx])**2)
        d_rx = np.sqrt((X - ant_x[rx_idx])**2 + (Y - ant_y[rx_idx])**2)
        
        # Kompensasi fase
        phase_shift = np.exp(-1j * k * (d_tx + d_rx))
        
        # Akumulasi DAS
        das_map += ch_signal * phase_shift
        
    # Konversi ke dB scale
    das_power_db = 20 * np.log10(np.abs(das_map) + 1e-12)
    return X, Y, das_power_db, r_ant_mm

def main():
    os.makedirs("metrics_image", exist_ok=True)
    
    try:
        s21_path = find_adipose_s21()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}"); return
        
    print("="*60)
    print("DAS FORENSICS: REAL ADIPOSE TISSUE ENERGY MAP")
    print("="*60)
    
    X, Y, das_db, r_ant = compute_das_adipose(s21_path)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot heatmap DAS dalam dB
    vmin = np.percentile(das_db, 5)  # Clip noise floor bawah
    vmax = np.percentile(das_db, 99) # Clip outlier atas
    im = ax.pcolormesh(X, Y, das_db, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label='DAS Power [dB]')
    
    # GAMBAR LINGKARAN PHANTOM DAN ANTENA
    ant_circle = plt.Circle((0, 0), r_ant, color='white', fill=False, 
                            linewidth=1.5, linestyle='--', alpha=0.9, 
                            label=f'Antenna Array (R={r_ant} mm)')
    phantom_circle = plt.Circle((0, 0), 48.0, color='black', fill=False, 
                                linewidth=1.2, linestyle='-', alpha=0.7, 
                                label='Avg Adipose Phantom Boundary (R≈48 mm)')
    ax.add_patch(ant_circle)
    ax.add_patch(phantom_circle)
    
    # Crosshair pusat
    ax.axhline(0, color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(0, color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax.plot(0, 0, 'w+', markersize=12, markeredgewidth=2)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_title('DAS Energy Map: Real Adipose Tissue\n(Forensic Check for Central Coupling Artifact)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('metrics_image/das_adipose_forensics.png', dpi=600, bbox_inches='tight')
    
    # Print statistik kunci untuk diagnosa
    center_val = das_db[60, 60] # Nilai tepat di tengah grid 121x121
    mean_val = das_db.mean()
    max_val = das_db.max()
    
    print(f"\n[Saved] metrics_image/das_adipose_forensics.png")
    print(f"[DAS Stats] Center(0,0): {center_val:.1f} dB | Mean: {mean_val:.1f} dB | Max: {max_val:.1f} dB")
    print("\nDIAGNOSA:")
    if center_val > mean_val + 10:
        print("⚠️  PUNCAK ENERGI TERDETEKSI DI PUSAT!")
        print("   → Central artifact disebabkan oleh DIRECT COUPLING HARDWARE.")
        print("   → CF gagal mendeteksi ini karena menormalisasi energi tinggi tsb.")
        print("   → WAJIB lakukan coupling subtraction sebelum beamforming tumor.")
    else:
        print("✅ Tidak ada puncak energi signifikan di pusat.")
        print("   → Sistem relatif bersih dari direct coupling dominan.")
        print("   → Artefak pada data tumor mungkin berasal dari fokus gelombang/clutter tissue.")
    print("="*60)
    plt.close()

if __name__ == "__main__":
    main()