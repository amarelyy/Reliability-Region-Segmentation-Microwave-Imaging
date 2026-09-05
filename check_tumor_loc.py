"""
check_tumor_location.py

Analisis distribusi lokasi tumor relatif terhadap batas phantom.
Output: statistik jarak tumor dari pusat dan dari boundary.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
META_PATH = DATA_DIR / "metadata_gen_two.pickle"
PHANTOM_CSV_PATH = DATA_DIR / "phantom_database.csv"


def main():
    # Load data
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    meta_df = pd.DataFrame(meta)
    
    phantom_db = pd.read_csv(PHANTOM_CSV_PATH)
    
    # Merge metadata dengan phantom database
    merged = meta_df.merge(
        phantom_db, 
        left_on="phant_id", 
        right_on="phantom_id", 
        how="left"
    )
    
    # Filter hanya scan yang punya tumor (tum_rad tidak NaN)
    tumor_col = None
    for col in merged.columns:
        if "tum_rad" in col.lower() or "tumor_radius" in col.lower():
            tumor_col = col
            break
    
    if tumor_col is None:
        print("ERROR: Kolom radius tumor tidak ditemukan.")
        print(f"Available columns: {merged.columns.tolist()}")
        return
    
    tumor_scans = merged[merged[tumor_col].notna()].copy()
    print(f"Total scans dengan tumor: {len(tumor_scans)}")
    
    if len(tumor_scans) == 0:
        print("WARNING: Tidak ada scan dengan tumor valid.")
        return
    
    # Identifikasi kolom koordinat tumor
    tum_x_col = None
    tum_y_col = None
    for col in tumor_scans.columns:
        if "tum_x" in col.lower() or "tumor_x" in col.lower():
            tum_x_col = col
        if "tum_y" in col.lower() or "tumor_y" in col.lower():
            tum_y_col = col
    
    if tum_x_col is None or tum_y_col is None:
        print("ERROR: Kolom koordinat tumor (x/y) tidak ditemukan.")
        print(f"Available columns: {tumor_scans.columns.tolist()}")
        return
    
    # Hitung jarak tumor dari pusat phantom
    tumor_scans["tum_dist_from_center_mm"] = np.sqrt(
        tumor_scans[tum_x_col] ** 2 + tumor_scans[tum_y_col] ** 2
    )
    
    # Identifikasi kolom radius phantom
    radius_col = None
    for col in tumor_scans.columns:
        if "shell_radius" in col.lower() or "breast_radius" in col.lower():
            radius_col = col
            break
    
    if radius_col is None:
        print("WARNING: Kolom radius phantom tidak ditemukan.")
        print(f"Available columns: {tumor_scans.columns.tolist()}")
        print("Skipping boundary distance calculation.")
    else:
        # Hitung jarak tumor dari batas luar phantom
        tumor_scans["tum_dist_from_boundary_mm"] = (
            tumor_scans[radius_col] - tumor_scans["tum_dist_from_center_mm"]
        )
        
        print("\n" + "=" * 60)
        print("STATISTIK JARAK TUMOR DARI PUSAT PHANTOM (mm)")
        print("=" * 60)
        print(tumor_scans["tum_dist_from_center_mm"].describe())
        
        print("\n" + "=" * 60)
        print("STATISTIK JARAK TUMOR DARI BATAS LUAR PHANTOM (mm)")
        print("=" * 60)
        print(tumor_scans["tum_dist_from_boundary_mm"].describe())
        
        # Persentase tumor dekat pusat (< 10 mm dari center)
        near_center_pct = (
            (tumor_scans["tum_dist_from_center_mm"] < 10).mean() * 100
        )
        print(f"\nPersentase tumor dekat pusat (< 10 mm): {near_center_pct:.1f}%")
        
        # Persentase tumor dekat boundary (< 10 mm dari kulit)
        near_boundary_pct = (
            (tumor_scans["tum_dist_from_boundary_mm"] < 10).mean() * 100
        )
        print(f"Persentase tumor dekat boundary (< 10 mm): {near_boundary_pct:.1f}%")
        
        # Persentase tumor di zona tengah (10-30 mm dari center)
        mid_zone_pct = (
            (
                (tumor_scans["tum_dist_from_center_mm"] >= 10) &
                (tumor_scans["tum_dist_from_center_mm"] < 30)
            ).mean() * 100
        )
        print(f"Persentase tumor di zona tengah (10-30 mm): {mid_zone_pct:.1f}%")
    
    # Cek kolom tum_in_fib kalau ada
    fib_col = None
    for col in tumor_scans.columns:
        if "tum_in_fib" in col.lower():
            fib_col = col
            break
    
    if fib_col is not None:
        print("\n" + "=" * 60)
        print("DISTRIBUSI TUMOR IN FIBROGLANDULAR")
        print("=" * 60)
        print(tumor_scans[fib_col].value_counts(dropna=False))


if __name__ == "__main__":
    main()