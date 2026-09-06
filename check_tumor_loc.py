"""
check_tumor_loc.py — Analisis distribusi lokasi tumor dari metadata pickle.
Menjelaskan kenapa DAS+CF gagal: tumor terlalu dekat pusat, 
di mana sinyal artifact terakumulasi akibat koherensi geometri array sirkular.
"""
import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
META_PATH = DATA_DIR / "metadata_gen_two.pickle"


def main():
    # Load langsung dari pickle, tanpa CSV
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    print(f"Total entries in metadata: {len(meta)}")
    print(f"Keys per entry: {list(meta[0].keys()) if len(meta) > 0 else 'EMPTY'}")

    # Identifikasi kolom secara dinamis
    sample = meta[0] if len(meta) > 0 else {}
    tum_x_col = tum_y_col = tum_r_col = radius_col = None

    for key in sample.keys():
        kl = key.lower()
        if "tum_x" in kl or "tumor_x" in kl:
            tum_x_col = key
        elif "tum_y" in kl or "tumor_y" in kl:
            tum_y_col = key
        elif "tum_rad" in kl or "tumor_rad" in kl or "tumor_radius" in kl:
            tum_r_col = key
        elif "shell_radius" in kl or "breast_radius" in kl:
            radius_col = key

    missing = []
    if tum_x_col is None:
        missing.append("tumor_x")
    if tum_y_col is None:
        missing.append("tumor_y")
    if missing:
        print(f"\nERROR: Kolom tidak ditemukan: {missing}")
        print(f"Available keys: {list(sample.keys())}")
        return

    print(f"\nUsing columns: x={tum_x_col}, y={tum_y_col}, "
          f"r={tum_r_col}, breast_radius={radius_col}")

    # Ekstrak arrays
    tum_x = np.array([e.get(tum_x_col, np.nan) for e in meta], dtype=float)
    tum_y = np.array([e.get(tum_y_col, np.nan) for e in meta], dtype=float)

    # Filter hanya yang punya tumor valid
    valid = ~(np.isnan(tum_x) | np.isnan(tum_y))
    tum_x = tum_x[valid]
    tum_y = tum_y[valid]
    n_tumor = len(tum_x)

    if n_tumor == 0:
        print("WARNING: Tidak ada scan dengan tumor valid.")
        return

    # Konversi cm ke mm jika perlu (cek magnitude)
    if np.nanmedian(np.sqrt(tum_x**2 + tum_y**2)) < 5:
        print("\n[INFO] Koordinat terdeteksi dalam cm, konversi ke mm")
        tum_x *= 10.0
        tum_y *= 10.0

    # Hitung jarak dari pusat
    dist_center = np.sqrt(tum_x**2 + tum_y**2)

    # Breast radius
    if radius_col is not None:
        breast_r = np.array([e.get(radius_col, np.nan) for e in meta], dtype=float)
        breast_r = breast_r[valid]
        # Konversi cm ke mm jika perlu
        if np.nanmedian(breast_r) < 10:
            breast_r *= 10.0
        valid_r = ~np.isnan(breast_r)
        dist_boundary = breast_r[valid_r] - dist_center[valid_r]
    else:
        breast_r = None
        dist_boundary = None

    # === OUTPUT ===
    print("\n" + "=" * 70)
    print("DISTRIBUSI LOKASI TUMOR RELATIF TERHADAP PUSAT ARRAY")
    print("=" * 70)
    print(f"Total tumor-bearing scans: {n_tumor}")
    print(f"\nJarak dari pusat (mm):")
    print(f"  Min:    {dist_center.min():.1f}")
    print(f"  Mean:   {dist_center.mean():.1f}")
    print(f"  Median: {np.median(dist_center):.1f}")
    print(f"  Max:    {dist_center.max():.1f}")
    print(f"  Std:    {dist_center.std():.1f}")

    if dist_boundary is not None:
        print(f"\nJarak dari boundary luar (mm):")
        print(f"  Min:    {dist_boundary.min():.1f}")
        print(f"  Mean:   {dist_boundary.mean():.1f}")
        print(f"  Median: {np.median(dist_boundary):.1f}")
        print(f"  Max:    {dist_boundary.max():.1f}")

    # Zona distribusi
    zones = [
        ("Sangat dekat pusat (< 5 mm)", dist_center < 5),
        ("Dekat pusat (5-10 mm)", (dist_center >= 5) & (dist_center < 10)),
        ("Zona tengah (10-20 mm)", (dist_center >= 10) & (dist_center < 20)),
        ("Zona luar (20-30 mm)", (dist_center >= 20) & (dist_center < 30)),
        ("Dekat boundary (> 30 mm)", dist_center >= 30),
    ]

    print(f"\nDistribusi per zona:")
    for label, mask in zones:
        pct = mask.sum() / n_tumor * 100
        print(f"  {label:<35}: {mask.sum():>4} ({pct:5.1f}%)")

    near_center = (dist_center < 10).sum()
    print(f"\n⚠️  {(near_center/n_tumor*100):.1f}% tumor berada < 10 mm dari pusat")
    print(f"   Di zona ini, DAS+CF tidak dapat membedakan tumor dari central")
    print(f"   ring artifact karena akumulasi koheren sinyal pada geometri")
    print(f"   array sirkular 72 elemen.")

    if dist_boundary is not None:
        near_bound = (dist_boundary < 10).sum()
        n_valid_r = valid_r.sum()
        print(f"\n⚠️  {(near_bound/n_valid_r*100):.1f}% tumor berada < 10 mm dari boundary")
        print(f"   Di zona ini, signal-to-clutter ratio terdegradasi karena")
        print(f"   proximity effect pada batas phantom.")


if __name__ == "__main__":
    main()