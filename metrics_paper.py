import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# === CENTRAL CONCENTRATION STATS ===
df = pd.read_csv("rrs_mbi/results/metrics_raw.csv")
cc = df["central_concentration"].dropna().to_numpy()
print(f"X (evaluated scans) = {len(cc)}")
print(f"MEAN CC = {cc.mean():.4f}")
print(f"MEDIAN CC = {np.median(cc):.4f}")
print(f"SD CC = {cc.std():.4f}")

# === BOUNDARY PROXIMITY ===
meta_path = Path("data/metadata_gen_two.pickle")
if meta_path.exists():
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    tum_x = np.array([e.get("tum_x", np.nan) for e in meta], dtype=float)
    tum_y = np.array([e.get("tum_y", np.nan) for e in meta], dtype=float)
    valid = ~(np.isnan(tum_x) | np.isnan(tum_y))
    tum_x, tum_y = tum_x[valid], tum_y[valid]
    
    dist = np.sqrt(tum_x**2 + tum_y**2)
    if np.median(dist) < 5:
        tum_x *= 10
        tum_y *= 10
        dist *= 10
    
    breast_radius_mm = 45.0
    boundary_zone_mm = 10.0
    dist_from_boundary = breast_radius_mm - dist
    within_10mm = (dist_from_boundary <= boundary_zone_mm).sum()
    pct_within = within_10mm / len(dist) * 100
    
    print(f"\nBreast radius assumed: {breast_radius_mm} mm")
    print(f"%WITHIN 10MM OF BOUNDARY = {pct_within :.1f}% (n={within_10mm} / {len(dist)})")