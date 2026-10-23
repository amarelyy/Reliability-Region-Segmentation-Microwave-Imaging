import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

def load_and_analyze_phantom_sizes():
    db_path = Path("data/phantom_database.csv")
    if not db_path.exists(): raise FileNotFoundError(f"DB not found at {db_path}")
        
    db = pd.read_csv(db_path)
    r_col = next((c for c in db.columns if any(k in c.lower() for k in ['shell_r', 'breast_r', 'radius'])), None)
    id_col = next((c for c in db.columns if 'phantom_id' in c.lower() or 'id' == c.lower()), None)
    
    phantom_sizes = db[[id_col, r_col]].copy()
    phantom_sizes.columns = ['phant_id', 'radius_mm']
    phantom_sizes['phant_id'] = phantom_sizes['phant_id'].astype(str).str.strip().str.upper()
    phantom_sizes['radius_mm'] = pd.to_numeric(phantom_sizes['radius_mm'], errors='coerce')
    phantom_sizes = phantom_sizes.dropna(subset=['radius_mm'])
    
    size_dict = dict(zip(phantom_sizes['phant_id'], phantom_sizes['radius_mm']))
    
    # Tentukan kategori berdasarkan quartile
    radii = sorted(size_dict.values())
    q33 = np.quantile(radii, 0.33)
    q66 = np.quantile(radii, 0.66)
    
    categories = {
        f'Small (R < {q33:.0f} mm)': lambda r: r < q33,
        f'Medium ({q33:.0f} ≤ R < {q66:.0f} mm)': lambda r: (r >= q33) & (r < q66),
        f'Large (R ≥ {q66:.0f} mm)': lambda r: r >= q66
    }
    
    print(f"[DB Info] Loaded {len(size_dict)} unique phantoms. Quartiles: Q33={q33:.1f}, Q66={q66:.1f}")
    return size_dict, categories

def robust_map_phant_id(meta_id, size_dict):
    """Mencocokkan ID metadata dengan DB menggunakan substring matching"""
    meta_id_clean = str(meta_id).strip().upper()
    
    # 1. Exact match dulu
    if meta_id_clean in size_dict: return size_dict[meta_id_clean]
    
    # 2. Substring match (cari apakah meta_id ada di dalam key DB atau sebaliknya)
    for db_id, radius in size_dict.items():
        if meta_id_clean in db_id or db_id in meta_id_clean:
            return radius
            
    return None

def load_tumor_data(phantom_size_dict):
    meta_path = Path("data/metadata_gen_two.pickle")
    raw_meta = pd.read_pickle(meta_path)
    meta = pd.DataFrame(raw_meta) if isinstance(raw_meta, list) else raw_meta
    meta.columns = [c.lower() for c in meta.columns]
    
    x_col = next((c for c in meta.columns if 'tum_x' in c), None)
    y_col = next((c for c in meta.columns if 'tum_y' in c), None)
    r_col = next((c for c in meta.columns if 'tum_rad' in c), None)
    id_col = next((c for c in meta.columns if 'phant_id' in c), None)
    
    df = pd.DataFrame({
        'tumor_x_mm': meta[x_col],
        'tumor_y_mm': meta[y_col],
        'phant_id_raw': meta[id_col].astype(str).str.strip(),
        'tumor_radius_mm': meta[r_col] if r_col else 4.0
    })
    
    df = df.dropna(subset=['tumor_x_mm', 'tumor_y_mm'])
    
    # Apply robust mapping row by row
    mapped_radii = df['phant_id_raw'].apply(lambda x: robust_map_phant_id(x, phantom_size_dict))
    n_mapped = mapped_radii.notna().sum()
    n_fallback = len(df) - n_mapped
    
    print(f"[Mapping Report] Exact/Substring Matched: {n_mapped} | Fallback Median: {n_fallback}")
    if n_fallback > 0:
        unmapped_samples = df.loc[mapped_radii.isna(), 'phant_id_raw'].unique()[:3]
        print(f"  Sample Unmapped IDs: {list(unmapped_samples)}")
        
    median_r = float(np.median(list(phantom_size_dict.values())))
    df['breast_radius_mm'] = mapped_radii.fillna(median_r)

    # Unit Correction
    if df['tumor_x_mm'].abs().max() < 10 and df['breast_radius_mm'].max() > 20:
        df['tumor_x_mm'] *= 10.0
        df['tumor_y_mm'] *= 10.0
        if df['tumor_radius_mm'].max() < 2.0: df['tumor_radius_mm'] *= 10.0
            
    df['tumor_radius_mm'] = df['tumor_radius_mm'].fillna(4.0)
    df['radial_dist_mm'] = np.sqrt(df['tumor_x_mm']**2 + df['tumor_y_mm']**2)
    return df, n_mapped, n_fallback

def main():
    os.makedirs("metrics_image", exist_ok=True)
    phantom_size_dict, categories = load_and_analyze_phantom_sizes()
    df, n_mapped, n_fallback = load_tumor_data(phantom_size_dict)
    
    bins = np.arange(0, 61, 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ['#D62728', '#1F77B4', '#2CA02C']
    
    for i, (title, filter_fn) in enumerate(categories.items()):
        ax = axes[i]
        subset = df[df['breast_radius_mm'].apply(filter_fn)]
        n_total = len(subset)
        
        # Hitung berapa banyak di subset ini yang benar-benar mapped vs fallback
        # (Kita asumsikan fallback tersebar proporsional, atau bisa dihitung eksak jika perlu)
        
        if n_total == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue
            
        counts, _ = np.histogram(subset['radial_dist_mm'], bins=bins)
        percentages = (counts / n_total) * 100
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        ax.bar(bin_centers, percentages, width=2.0, color=colors[i], 
               edgecolor='black', linewidth=0.5, alpha=0.85)
        
        avg_r = subset['breast_radius_mm'].mean()
        
        # GARIS VERTIKAL DASHED PER ZONA (Near/Medium/Far)
        near_limit = avg_r * 0.20
        far_limit = avg_r * 0.60
        
        ax.axvline(near_limit, color='gray', linestyle=':', linewidth=1.5, label='Near/Mid Boundary (20% R)')
        ax.axvline(far_limit, color='gray', linestyle='-.', linewidth=1.5, label='Mid/Far Boundary (60% R)')
        ax.axvline(avg_r, color='black', linestyle='--', linewidth=1.2, label=f'Avg Phantom Boundary')
        
        peak_idx = np.argmax(percentages)
        if percentages[peak_idx] > 0:
            ax.annotate(f'Peak: {percentages[peak_idx]:.1f}%',
                xy=(bin_centers[peak_idx], percentages[peak_idx]),
                xytext=(bin_centers[peak_idx] + 4, percentages[peak_idx] + 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=9, fontweight='bold')
                
        # Update title dengan info mapping
        map_status = f"Mapped:{n_mapped}" if i==0 else "" # Tampilkan sekali saja di subplot pertama
        ax.set_title(f"{title}\n(N={n_total})", fontsize=11)
        ax.set_xlabel('Radial Distance (mm)')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(fontsize=8, loc='upper left')
        
    axes[0].set_ylabel('Frequency (%)')
    plt.suptitle(f'Stratified Tumor Radial Distribution ({n_mapped} Mapped, {n_fallback} Fallback)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('metrics_image/stratified_radial_histogram_zones.png', dpi=600, bbox_inches='tight')
    
    print(f"[Saved] metrics_image/stratified_radial_histogram_zones.png")
    plt.close()

if __name__ == "__main__":
    main()