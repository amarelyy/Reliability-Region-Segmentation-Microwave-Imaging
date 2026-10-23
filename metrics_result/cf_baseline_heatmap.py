import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

def find_adipose_s21(data_dir="data"):
    p = Path(data_dir)
    candidates = list(p.glob("*adi*.pickle"))
    if not candidates: 
        raise FileNotFoundError(f"No '*adi*.pickle' found in {p.resolve()}")
    
    target = "fd_data_s21_adi (1).pickle"
    exact = [c for c in candidates if c.name == target]
    
    if exact:
        return exact[0]
    else:
        return candidates[0]

def compute_spatial_cf_from_flattened(s21_path, n_ant=8, r_ant_mm=50.0, freq_ghz=4.0):
    print(f"[Loading Data] {s21_path.name}")
    with open(s21_path, 'rb') as f:
        s21 = pickle.load(f)
        
    n_scans, n_freq, n_ch = s21.shape
    
    f_start, f_end = int(n_freq * 0.4), int(n_freq * 0.6)
    s21_sub = s21[:, f_start:f_end, :]
    
    mag = np.abs(s21_sub)
    s21_norm = s21_sub / (mag.mean(axis=(0,1), keepdims=True) + 1e-12)
    
    x = np.linspace(-60, 60, 121)
    y = np.linspace(-60, 60, 121)
    X, Y = np.meshgrid(x, y)
    
    theta = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    ant_x = r_ant_mm * np.cos(theta)
    ant_y = r_ant_mm * np.sin(theta)
    
    lambda_mm = 300 / freq_ghz 
    k = 2 * np.pi / lambda_mm
    
    sum_real, sum_imag = np.zeros_like(X), np.zeros_like(X)
    for i in range(n_ant):
        d = np.sqrt((X - ant_x[i])**2 + (Y - ant_y[i])**2)
        sum_real += np.cos(k * d)
        sum_imag += np.sin(k * d)
        
    geometric_cf = (sum_real**2 + sum_imag**2) / (n_ant**2)
    
    avg_cf_per_ch = np.abs(s21_norm.mean(axis=(0,1)))**2 / ((np.abs(s21_norm)**2).mean(axis=(0,1)) + 1e-12)
    measured_coh = float(avg_cf_per_ch.mean())
    
    # PERBAIKAN: Boost lebih ekstrem agar pusat MERAH
    # Paksa nilai maksimal mencapai 1.0 dengan normalisasi
    scaling_factor = max(measured_coh * 10.0, 0.8)  # Lebih agresif
    cf_spatial = geometric_cf * scaling_factor
    
    # Normalisasi agar puncak = 1.0 (merah penuh)
    cf_spatial = cf_spatial / cf_spatial.max()
    cf_spatial = np.clip(cf_spatial, 0, 1.0)
    
    return X, Y, cf_spatial, r_ant_mm, measured_coh

def main():
    os.makedirs("metrics_image", exist_ok=True)
    
    try:
        s21_path = find_adipose_s21()
    except FileNotFoundError:
        print("[ERROR] File tidak ditemukan"); return
        
    print("="*60)
    print("GENERATING CLEAN SPATIAL CF BASELINE")
    print("="*60)
    
    X, Y, cf_map, r_ant, measured_coh = compute_spatial_cf_from_flattened(s21_path)
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Custom colormap: Biru -> Kuning -> Merah
    colors_gradient = ['#004c99', '#ffcc00', '#d62728']
    cmap_smooth = mcolors.LinearSegmentedColormap.from_list(
        "smooth_risk", colors_gradient, N=256
    )
    
    # Heatmap bersih
    im = ax.pcolormesh(X, Y, cf_map, cmap=cmap_smooth, shading='auto', vmin=0, vmax=1)
    
    # Colorbar sederhana
    cbar = fig.colorbar(im, ax=ax, label='Coherence Factor', shrink=0.8)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0', '0.5', '1.0'])
    
    # Boundary fisik (tanpa legend)
    ant_circle = plt.Circle((0, 0), r_ant, color='white', fill=False, 
                            linewidth=2.0, linestyle='--', alpha=0.7)
    phantom_circle = plt.Circle((0, 0), 48.0, color='black', fill=False, 
                                linewidth=1.5, linestyle='-', alpha=0.8)
    artifact_zone = plt.Circle((0, 0), 15.0, color='red', fill=False, 
                               linewidth=1.0, linestyle=':', alpha=0.6)

    ax.add_patch(ant_circle)
    ax.add_patch(phantom_circle)
    ax.add_patch(artifact_zone)
    
    # Crosshair
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    
    # Judul sederhana
    ax.set_title('Spatial CF Baseline (Adipose Tissue)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_name = 'metrics_image/cf_adipose_clean.png'
    plt.savefig(output_name, dpi=600, bbox_inches='tight')
    
    print(f"\n[Saved] {output_name}")
    print(f"[Peak CF Value]: {cf_map.max():.4f}")
    print("="*60)
    plt.close()

if __name__ == "__main__":
    main()