import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

def load_tumor_data():
    meta_path = Path("data/metadata_gen_two.pickle")
    
    if not meta_path.exists():
        raise FileNotFoundError(f"CRITICAL: File metadata tidak ditemukan di {meta_path.resolve()}")
        
    # Load raw data
    raw_meta = pd.read_pickle(meta_path)
    
    # FIX: Konversi ke DataFrame jika hasilnya berupa List atau Dictionary
    if isinstance(raw_meta, list):
        print("[Info] Metadata loaded as List. Converting to DataFrame...")
        meta = pd.DataFrame(raw_meta)
    elif isinstance(raw_meta, dict):
        print("[Info] Metadata loaded as Dict. Converting to DataFrame...")
        meta = pd.DataFrame.from_dict(raw_meta, orient='index').T
    else:
        meta = raw_meta
        
    # Standardize column names to lowercase for easier matching
    meta.columns = [c.lower() for c in meta.columns]
    
    total_scans = len(meta)
    
    # Dynamic column matching
    x_col = next((c for c in meta.columns if 'tum_x' in c or 'tumor_x' in c), None)
    y_col = next((c for c in meta.columns if 'tum_y' in c or 'tumor_y' in c), None)
    r_col = next((c for c in meta.columns if 'tum_rad' in c or 'tumor_radius' in c), None)
    br_col = next((c for c in meta.columns if 'shell_radius' in c or 'breast_radius' in c), None)
    
    if not all([x_col, y_col]):
        raise ValueError("CRITICAL: Kolom tumor_x dan tumor_y tidak ditemukan di metadata.")
        
    # Buat DataFrame baru hanya dengan kolom yang dibutuhkan
    df = pd.DataFrame()
    df['tumor_x_mm'] = meta[x_col]
    df['tumor_y_mm'] = meta[y_col]
    
    # Handle radius tumor (jika ada pakai itu, jika tidak pakai default)
    if r_col:
        df['tumor_radius_mm'] = meta[r_col]
    else:
        print("[Warning] Kolom tumor_radius tidak ditemukan. Menggunakan default 4.0 mm.")
        df['tumor_radius_mm'] = 4.0
        
    # Handle breast radius (jika ada pakai itu, jika tidak pakai default)
    if br_col:
        df['breast_radius_mm'] = meta[br_col]
    else:
        print("[Warning] Kolom breast_radius tidak ditemukan. Menggunakan default 50.0 mm.")
        df['breast_radius_mm'] = 50.0

    # Hanya buang baris yang benar-benar tidak punya koordinat X dan Y
    df = df.dropna(subset=['tumor_x_mm', 'tumor_y_mm'])
    valid_scans = len(df)
    
    # Konversi satuan (cm ke mm) berdasarkan heuristik yang aman
    max_tum_r = df['tumor_radius_mm'].dropna().max()
    if not pd.isna(max_tum_r) and max_tum_r < 2.0:
        print("[Info] Tumor radius < 2.0 terdeteksi. Mengonversi X, Y, R dari cm ke mm...")
        df['tumor_x_mm'] *= 10.0
        df['tumor_y_mm'] *= 10.0
        df['tumor_radius_mm'] *= 10.0
        
    max_breast = df['breast_radius_mm'].dropna().max()
    if not pd.isna(max_breast) and max_breast < 15.0:
        print("[Info] Breast radius < 15.0 terdeteksi. Mengonversi ke mm...")
        df['breast_radius_mm'] *= 10.0

    # Isi nilai radius yang hilang dengan default 4.0 agar semua baris valid tetap terplot
    df['tumor_radius_mm'] = df['tumor_radius_mm'].fillna(4.0)
    
    return df, total_scans, valid_scans

def main():
    os.makedirs("metrics_image", exist_ok=True)
    
    print("Memuat data asli dari metadata...")
    df, total_scans, valid_scans = load_tumor_data()
    print(f"Berhasil memuat {valid_scans} dari {total_scans} total scan.")
    
    # 1. Hitung Jarak Radial (r) dari pusat (0,0)
    df['radial_dist_mm'] = np.sqrt(df['tumor_x_mm']**2 + df['tumor_y_mm']**2)
    
    # 2. Definisikan Zona (Absolut dalam mm)
    zones = {
        'Near Center\n(r < 10 mm)': df['radial_dist_mm'] < 10,
        'Middle\n(10 ≤ r < 30 mm)': (df['radial_dist_mm'] >= 10) & (df['radial_dist_mm'] < 30),
        'Boundary\n(r ≥ 30 mm)': df['radial_dist_mm'] >= 30
    }
    
    # 3. Hitung Statistik
    zone_names = []
    counts = []
    percentages = []
    
    print("\n" + "="*65)
    print(f" DISTRIBUSI ZONA LOKASI TUMOR (N = {valid_scans} Scan Asli)")
    print("="*65)
    print(f" {'Definisi Zona':<28} | {'Jumlah':<6} | {'Persentase'}")
    print("-"*65)
    
    for name, mask in zones.items():
        count = int(mask.sum())
        pct = (count / valid_scans) * 100
        zone_names.append(name)
        counts.append(count)
        percentages.append(pct)
        
        clean_name = name.replace('\n', ' ')
        print(f" {clean_name:<28} | {count:<6} | {pct:5.1f} %")
    print("="*65 + "\n")
    
    # 4. Visualisasi (IEEE Bar Chart)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Warna formal IEEE: Merah (Pusat/Artefak), Biru (Tengah), Hijau (Boundary)
    colors = ['#D62728', '#1F77B4', '#2CA02C'] 
    
    bars = ax.bar(zone_names, counts, color=colors, edgecolor='black', linewidth=0.8)
    
    # Tambahkan label persentase di atas setiap batang
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.015), f'{pct:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_ylabel('Number of Scans')
    ax.set_title('Distribution of Tumor Locations Across Imaging Zones')
    
    # Atur limit Y agar ada ruang untuk label persentase
    ax.set_ylim(0, max(counts) * 1.18)
    
    # Formatting IEEE: hilangkan spine atas dan kanan
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('metrics_image/tumor_zone_distribution.png', dpi=600, bbox_inches='tight')
    print("[Success] Saved: metrics_image/tumor_zone_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()