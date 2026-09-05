"""
coords_check.py (Advanced Sanity Check & Confusion Matrix)

Analisis mendalam:
1. Koordinat Prediksi vs Ground Truth
2. Confusion Matrix: TP, Fortuitous (Akurat tapi Low Rel), TN, FP (Misleading)
3. Statistik Jarak Tumor (Pusat & Boundary)
4. Efek Pipeline Baru (TVSVD ON)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

# Konfigurasi Threshold
ERROR_THRESH_MM = 15.0
REL_HIGH_THRESH = 0.70
REL_LOW_THRESH = 0.30

def main():
    print("=" * 80)
    print("RRS-MBI ADVANCED SANITY CHECK & CONFUSION MATRIX")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/4] Loading data...")
    d = load_all_data()
    s21 = d["s21"]
    tumor_model = d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]
    freq_axis = d["freq_axis"]

    n_scans = len(tumor_model)
    print(f"Total tumor-bearing scans: {n_scans}")

    # 2. Inisialisasi Storage
    results_list = []
    stats = {"TP": 0, "Fortuitous": 0, "TN": 0, "FP": 0}
    
    # Pilih sample acak untuk display detail (biar tidak terlalu panjang outputnya)
    rng = np.random.default_rng(42)
    display_indices = set(rng.choice(n_scans, size=min(10, n_scans), replace=False))

    print(f"\n[2/4] Running reconstruction with NEW pipeline config (TVSVD=ON)...")
    print(f"      (Note: use_rms_norm not yet implemented in pipeline.py, skipping for now)")
    
    # 3. Loop Evaluasi
    for idx in range(n_scans):
        try:
            # --- PANGGILAN PIPELINE DENGAN PARAMETER BARU ---
            result = reconstruct_scan(
                scan_idx=int(idx),
                s21=s21,
                tumor_model=tumor_model,
                id_to_original_idx=id_to_original_idx,
                freq_axis=freq_axis,
                use_tvsvd=True,          # <-- NYALAKAN TVSVD
                use_rms_norm=True,      # <-- Belum ada di pipeline.py, set False dulu
                bandpass_mode="full",    # <-- Default full band
                gate_ns=0.70,              # <-- Default gate
                return_diagnostics=False
            )
            
               # ... (di dalam loop for idx in range(n_scans):) ...
            
            # Ekstrak Metrik
            gt_x = result["gt_x_mm"]
            gt_y = result["gt_y_mm"]
            pred_x = result["peak_x_mm"]
            pred_y = result["peak_y_mm"]
            error = result["localization_error_mm"]
            
            # --- PERBAIKAN LOGIKA RELIABILITY ---
            rel = result.get("reliability_score", np.nan)
            
            # FALLBACK: Kalau 'reliability_score' gak ada, coba hitung manual dari D dan B
            if np.isnan(rel):
                # Cek berbagai kemungkinan nama key dari metrics.py
                D = result.get("peak_dominance", result.get("D", np.nan))
                B = result.get("boundary_risk", result.get("B", np.nan))
                if not np.isnan(D) and not np.isnan(B):
                    rel = min(1.0, D / 5.0) * (1.0 - B)
            # --------------------------------------
            
            # Hitung Jarak
            gt_dist_center = np.sqrt(gt_x**2 + gt_y**2)
            pred_dist_center = np.sqrt(pred_x**2 + pred_y**2)
            
            # Simpan ke list untuk statistik agregat
            results_list.append({
                "scan_idx": idx,
                "phant_id": result["phant_id"],
                "gt_x": gt_x, "gt_y": gt_y,
                "pred_x": pred_x, "pred_y": pred_y,
                "error": error,
                "reliability": rel,
                "gt_dist_center": gt_dist_center,
                "pred_dist_center": pred_dist_center
            })
            
            # Capture keys dari scan pertama buat debugging kalau error
            if idx == 0:
                first_scan_keys = list(result.keys())

            # 4. Logika Confusion Matrix
            is_accurate = error <= ERROR_THRESH_MM
            is_high_rel = rel >= REL_HIGH_THRESH
            is_low_rel = rel < REL_LOW_THRESH
            
            category = ""
            if is_accurate and is_high_rel:
                stats["TP"] += 1
                category = "TP (True Positive)"
            elif is_accurate and is_low_rel:
                stats["Fortuitous"] += 1
                category = "FORTUITOUS (Accurate but Unreliable)"
            elif not is_accurate and is_low_rel:
                stats["TN"] += 1
                category = "TN (True Negative)"
            elif not is_accurate and is_high_rel:
                stats["FP"] += 1
                category = "FP (False Positive / Misleading)"
            else:
                # Medium confidence cases
                pass 
            
            # Print Detail untuk Sample Acak
            if idx in display_indices:
                print(f"Scan {idx:>3} [{result['phant_id']:>6}] | "
                      f"GT:({gt_x:5.1f}, {gt_y:5.1f}) d={gt_dist_center:4.1f} | "
                      f"Pred:({pred_x:5.1f}, {pred_y:5.1f}) d={pred_dist_center:4.1f} | "
                      f"Err:{error:5.1f} | Rel:{rel:.3f} | {category}")

        except Exception as e:
            print(f"Scan {idx} ERROR: {e}")

        # 5. Agregasi Statistik
    print("\n" + "=" * 80)
    print("[3/4] AGGREGATE STATISTICS")
    print("=" * 80)
    
    df = pd.DataFrame(results_list)
    
    if len(df) > 0:
        print(f"\n--- Localization Performance ---")
        print(f"Mean Error: {df['error'].mean():.2f} mm")
        print(f"Median Error: {df['error'].median():.2f} mm")
        print(f"Detection Rate (@{ERROR_THRESH_MM}mm): {(df['error'] <= ERROR_THRESH_MM).mean()*100:.1f}%")
        
        print(f"\n--- Geometric Distribution ---")
        print(f"Mean GT Distance from Center: {df['gt_dist_center'].mean():.2f} mm")
        print(f"Mean Pred Distance from Center: {df['pred_dist_center'].mean():.2f} mm")
        print(f"Max GT Distance: {df['gt_dist_center'].max():.2f} mm")
        print(f"Min GT Distance: {df['gt_dist_center'].min():.2f} mm")
        
        print(f"\n--- Confusion Matrix (Reliability vs Accuracy) ---")
        total_classified = sum(stats.values())
        
        # Fungsi pembantu anti-crash
        def safe_pct(num, den):
            return f"{num/den*100:.1f}%" if den > 0 else "N/A"

        print(f"True Positives (Accurate + High Rel):      {stats['TP']:>4} ({safe_pct(stats['TP'], total_classified)})")
        print(f"Fortuitous     (Accurate + Low Rel):       {stats['Fortuitous']:>4} ({safe_pct(stats['Fortuitous'], total_classified)})")
        print(f"True Negatives (Inaccurate + Low Rel):     {stats['TN']:>4} ({safe_pct(stats['TN'], total_classified)})")
        print(f"False Positives(Inaccurate + High Rel):    {stats['FP']:>4} ({safe_pct(stats['FP'], total_classified)})")
        print(f"Medium/Unclassified:                       {n_scans - total_classified:>4}")
        
        # --- DEBUG INFO FALLBACK ---
        if total_classified == 0:
            print("\n[WARNING] Reliability score is NaN for ALL scans!")
            print("Metrics.py might be returning D and B under different key names.")
            print(f"Here are all the keys returned by reconstruct_scan for the first scan:")
            print(first_scan_keys)
        else:
            print("\n[INTERPRETASI]")
            if stats['Fortuitous'] > 0:
                print(f"-> Ditemukan {stats['Fortuitous']} scan yang AKURAT tapi ditandai LOW RELIABILITY.")
                print("   Ini memvalidasi bahwa RRS-MBI mendeteksi 'kebetulan geometris'.")
            if stats['FP'] > 0:
                print(f"-> WARNING: {stats['FP']} scan TIDAK AKURAT tapi ditandai HIGH RELIABILITY.")
                
    print("\n[4/4] Done.")

if __name__ == "__main__":
    main()