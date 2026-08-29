"""
debug_pipeline.py
Memanggil reconstruct_scan secara langsung TANPA Joblib 
untuk memunculkan FULL TRACEBACK dari error yang tersembunyi.
"""
import traceback
import numpy as np
from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

print("Loading data...")
d = load_all_data()
s21 = d["s21"]
tumor_model = d["tumor_model"]
id_to_original_idx = d["id_to_original_idx"]

# Buat freq_axis dummy untuk bandpass
freq_axis = np.linspace(1e9, 8e9, 1001)

print("\nMencoba menjalankan reconstruct_scan untuk scan_idx = 0...")
print("="*70)

try:
    # Panggil langsung tanpa joblib
    result = reconstruct_scan(
        scan_idx=0, 
        s21=s21, 
        tumor_model=tumor_model, 
        id_to_original_idx=id_to_original_idx,
        freq_axis=freq_axis,
        beamformer="das", 
        use_snellius=True, 
        use_cf=False,       # Matikan CF dulu biar simpel
        use_tvsvd=False, 
        use_bandpass=False, # Matikan bandpass dulu biar simpel
        use_depth_gain=False
    )
    print("SUKSES! Hasil LE:", result.get("localization_error_mm", "N/A"))
    
except Exception as e:
    print("CRASH TERJADI! Ini adalah FULL TRACEBACK-nya:")
    print("="*70)
    traceback.print_exc()
    print("="*70)