import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# Mapping Radius Fisik dari src/data_loading.py
ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM = {
    'A1': 22.9, 'A2': 48.5, 'A3': 56.9, 'A11': 33.1, 'A12': 39.0,
    'A13': 43.8, 'A14': 45.4, 'A15': 56.2, 'A16': 57.0,
}

def load_and_process_data():
    meta_path = Path("data/metadata_gen_two.pickle")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
    raw_meta = pd.read_pickle(meta_path)
    if isinstance(raw_meta, list):
        meta = pd.DataFrame(raw_meta)
    else:
        meta = raw_meta
        
    meta.columns = [c.lower() for c in meta.columns]
    
    x_col = next((c for c in meta.columns if 'tum_x' in c), None)
    y_col = next((c for c in meta.columns if 'tum_y' in c), None)
    r_col = next((c for c in meta.columns if 'tum_rad' in c), None)
    id_col = next((c for c in meta.columns if 'phant_id' in c), None)
    
    if not all([x_col, y_col, id_col]):
        raise ValueError(f"Missing essential columns.")

    df = pd.DataFrame()
    df['tumor_x_mm'] = meta[x_col]
    df['tumor_y_mm'] = meta[y_col]
    df['phant_id'] = meta[id_col].astype(str).str.strip()
    df['tumor_radius_mm'] = meta[r_col] if r_col else 4.0
    
    df = df.dropna(subset=['tumor_x_mm', 'tumor_y_mm'])
    
    # Map PhantID to Breast Radius
    df['breast_radius_mm'] = df['phant_id'].map(ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM)
    missing_r = df['breast_radius_mm'].isna().sum()
    if missing_r > 0:
        print(f"[Warning] {missing_r} scans have unknown PhantID. Assigning default R=50mm.")
        df['breast_radius_mm'] = df['breast_radius_mm'].fillna(50.0)

    # Unit Correction (cm to mm)
    if df['tumor_x_mm'].abs().max() < 10 and df['breast_radius_mm'].max() > 20:
        print("[Correction] Converting coordinates from cm to mm...")
        df['tumor_x_mm'] *= 10.0
        df['tumor_y_mm'] *= 10.0
        if df['tumor_radius_mm'].max() < 2.0:
            df['tumor_radius_mm'] *= 10.0
            
    df['tumor_radius_mm'] = df['tumor_radius_mm'].fillna(4.0)
    
    return df

def main():
    os.makedirs("metrics_image", exist_ok=True)
    df = load_and_process_data()
    print(f"[Info] Processing {len(df)} valid scans.")
    
    # STRATEGI BARU: Bagi menjadi 3 grup ukuran agar tidak overload
    # Small: < 35mm (A1, A11)
    # Medium: 35 - 48mm (A12, A13, A14)
    # Large: >= 48mm (A2, A3, A15, A16)
    groups = {
        'Small Phantoms (R < 35 mm)': df[df['breast_radius_mm'] < 35],
        'Medium Phantoms (35 ≤ R < 48 mm)': df[(df['breast_radius_mm'] >= 35) & (df['breast_radius_mm'] < 48)],
        'Large Phantoms (R ≥ 48 mm)': df[df['breast_radius_mm'] >= 48]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (title, subset) in enumerate(groups.items()):
        ax = axes[i]
        n_scans = len(subset)
        
        if n_scans == 0:
            ax.text(0.5, 0.5, f'No Data\n({title})', ha='center', va='center', fontsize=12, color='gray')
            ax.set_aspect('equal')
            continue
            
        avg_r = subset['breast_radius_mm'].mean()
        
        # Draw boundary
        boundary = patches.Circle((0, 0), avg_r, color='black', fill=False, linewidth=1.2, linestyle='-')
        ax.add_patch(boundary)
        
        # Plot tumors with JITTER to prevent solid blob overlap
        # Jitter sebesar ±1.5mm cukup untuk memisahkan lingkaran tanpa merusak akurasi posisi
        jitter = np.random.uniform(-1.5, 1.5, size=n_scans)
        x_jittered = subset['tumor_x_mm'].values + jitter
        y_jittered = subset['tumor_y_mm'].values + jitter
        
        for j in range(n_scans):
            t_circle = patches.Circle(
                (x_jittered[j], y_jittered[j]), 
                subset['tumor_radius_mm'].values[j], 
                color='#1f77b4', alpha=0.4, edgecolor='none'
            )
            ax.add_patch(t_circle)
            
        limit = avg_r * 1.15
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\n(N={n_scans})", fontsize=11)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.suptitle('Physical Tumor Distribution by Phantom Size Category', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('metrics_image/size_grouped_distribution_fixed.png', dpi=600, bbox_inches='tight')
    print("[Saved] metrics_image/size_grouped_distribution_fixed.png")
    plt.close()

if __name__ == "__main__":
    main()