"""
debug_data_structure.py
Script diagnostik untuk membongkar struktur data s21 dan metadata
guna menemukan penyebab error 'float object does not support item assignment'.
"""
import numpy as np
import pandas as pd
from src.data_loading import load_all_data

print("="*70)
print("MEMULAI DIAGNOSIS STRUKTUR DATA")
print("="*70)

d = load_all_data()
s21 = d["s21"]
tumor_model = d["tumor_model"]
id_map = d["id_to_original_idx"]

print(f"\n1. INFO ARRAY S21 GLOBAL:")
print(f"   Type: {type(s21)}")
print(f"   Shape: {s21.shape}")
print(f"   Dtype: {s21.dtype}")

print(f"\n2. SAMPLING 5 SCAN PERTAMA DARI TUMOR_MODEL:")
for i in range(min(5, len(tumor_model))):
    row = tumor_model.iloc[i]
    s21_idx = int(row["original_s21_idx"])
    emp_id = row.get("emp_ref_id", None)
    
    print(f"\n--- Scan Index: {i} | Original S21 Idx: {s21_idx} ---")
    
    # Cek Raw Data
    raw_slice = s21[s21_idx]
    print(f"   s21[{s21_idx}] type: {type(raw_slice)}")
    print(f"   s21[{s21_idx}] shape: {getattr(raw_slice, 'shape', 'NO SHAPE')}")
    print(f"   s21[{s21_idx}] dtype: {getattr(raw_slice, 'dtype', 'NO DTYPE')}")
    print(f"   Is ndarray? {isinstance(raw_slice, np.ndarray)}")
    
    # Cek Empty Reference
    if emp_id is not None and not pd.isna(emp_id):
        emp_idx = id_map.get(int(emp_id), None)
        print(f"   Emp Ref ID: {int(emp_id)} -> Mapped Idx: {emp_idx}")
        if emp_idx is not None:
            emp_slice = s21[emp_idx]
            print(f"   s21[{emp_idx}] (empty) type: {type(emp_slice)}")
            print(f"   s21[{emp_idx}] (empty) shape: {getattr(emp_slice, 'shape', 'NO SHAPE')}")
        else:
            print(f"EMP REF IDX NOT FOUND IN MAP!")
    else:
        print(f"NO EMP_REF_ID FOR THIS SCAN")

print("\n" + "="*70)
print("DIAGNOSIS SELESAI. Paste output ini jika masih error.")
print("="*70)