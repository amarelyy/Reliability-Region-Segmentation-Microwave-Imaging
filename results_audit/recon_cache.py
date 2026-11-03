"""Reconstruct every scan once (evaluator settings) and cache image + CF map to disk.

Usage:
  python results_audit/recon_cache.py <out.npz>              # 504 tumour scans
  python results_audit/recon_cache.py <out.npz> --adipose    # adipose-only reference scans (adi_ref_id)

The code under test is whatever `src` is first on sys.path (set PYTHONPATH to use another copy).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not any(Path(p, "src", "pipeline.py").exists() for p in sys.path if p):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan
import src.physics as _ph

out = Path(sys.argv[1])
adipose = "--adipose" in sys.argv
print("using src from:", Path(_ph.__file__).parent)

DATA = Path(__file__).resolve().parent.parent / "data"   # read-only
d = load_all_data(DATA / "fd_data_gen_two_s21.pickle", DATA / "metadata_gen_two.pickle",
                  DATA / "phantom_database.csv")
s21, tm, id_map = d["s21"], d["tumor_model"], d["id_to_original_idx"]

if adipose:
    # One reconstruction per adipose reference scan; same empty-chamber subtraction,
    # breast radius from the phantom, velocity from fat only (no fibroglandular insert).
    tm = tm.drop_duplicates("adi_ref_id").reset_index(drop=True).copy()
    tm["original_s21_idx"] = tm["adi_ref_id"].astype(int).map(id_map)
    tm = tm[tm["original_s21_idx"].notna()].reset_index(drop=True)
    tm["original_s21_idx"] = tm["original_s21_idx"].astype(int)
    tm["fat_fraction"], tm["fib_fraction"] = 1.0, 0.0
if "--adi-sub" in sys.argv:
    # Sensitivity analysis: subtract the adipose-only reference scan instead of the empty chamber
    # (the pipeline's subtraction step reads emp_ref_id; only the data passed to it changes).
    tm = tm.copy()
    tm["emp_ref_id"] = tm["adi_ref_id"]

import inspect
_kw = {}
if "ref_col" in inspect.signature(reconstruct_scan).parameters:   # final pipeline only
    # adipose-only scans are themselves the adipose reference -> subtract the empty chamber
    _kw["ref_col"] = "emp_ref_id" if adipose else "adi_ref_id"

store = {}
rows = []
for i in range(len(tm)):
    if i % 50 == 0:
        print(f"  {i + 1}/{len(tm)}", flush=True)
    r = reconstruct_scan(i, s21, tm, id_map, freq_axis=d["freq_axis"], use_tvsvd=True,
                         use_rms_norm=True, bandpass_mode="full", gate_ns=0.70,
                         return_diagnostics=True, **_kw)
    g = r.pop("diagnostics")
    store[f"img_{i}"] = g["image"].astype(np.float32)
    store[f"cf_{i}"] = g["cf_map"].astype(np.float32)
    store[f"mask_{i}"] = g["tumor_mask"]
    store[f"axis_{i}"] = g["axis_mm"]
    row = tm.iloc[i]
    rows.append(dict(i=i, phant_id=row["phant_id"], breast_radius_mm=row["breast_radius_mm"],
                     adi_ref_id=row["adi_ref_id"], tumor_x_mm=row["tumor_x_mm"],
                     tumor_y_mm=row["tumor_y_mm"], tumor_radius_mm=row["tumor_radius_mm"],
                     peak_x_mm=r["peak_x_mm"], peak_y_mm=r["peak_y_mm"], scr_db=r["scr_db"],
                     localization_error_mm=r["localization_error_mm"],
                     shell_cx_mm=r.get("shell_cx_mm", 0.0), shell_cy_mm=r.get("shell_cy_mm", 0.0)))

out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, **store)
pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
print("saved", out, len(rows))
