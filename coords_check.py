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
import rrs_mbi.metrics as rmx 

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

# Konfigurasi Threshold
ERROR_THRESH_MM = 15.0
REL_HIGH_THRESH = 0.40
REL_LOW_THRESH = 0.15

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
    
        # 3. Loop Evaluasi
    for idx in range(n_scans):
        try:
            # --- PANGGILAN PIPELINE ---
            # PENTING: return_diagnostics MUSTI TRUE biar bisa ambil gambar mentah
            result = reconstruct_scan(
                scan_idx=int(idx),
                s21=s21,
                tumor_model=tumor_model,
                id_to_original_idx=id_to_original_idx,
                freq_axis=freq_axis,
                use_tvsvd=True,          # NYALAKAN TVSVD
                use_rms_norm=True,      # Ablation step 1: TVSVD only
                bandpass_mode="full",  # Ablation step 2: Bandpass 4-6 GHz
                gate_ns=0.5,             # Ablation step 1: Gate standar dulu
                return_diagnostics=True  # <-- WAJIB TRUE
            )
            
            # Ekstrak Koordinat & Error
            gt_x = result["gt_x_mm"]
            gt_y = result["gt_y_mm"]
            pred_x = result["peak_x_mm"]
            pred_y = result["peak_y_mm"]
            error = result["localization_error_mm"]
            
            # --- HITUNG RELIABILITY DARI GAMBAR MENTAH ---
            img_raw = result["diagnostics"]["image"]
            axis_mm = result["diagnostics"]["axis_mm"]
            cf_val = result.get("cf_at_peak", None)
            rel_dict = rmx.evaluate_scan_reliability(img_raw, axis_mm=axis_mm, 
                                                    cf_at_peak=cf_val,
                                                    saturation_radius_mm=5.0)
                        
            rel = rel_dict.get("reliability_score", np.nan)
            D = rel_dict.get("peak_dominance", np.nan)
            B = rel_dict.get("boundary_risk", np.nan)
            # ---------------------------------------------
            
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
                "D": D, "B": B,
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
    
    from scipy.stats import spearmanr, pearsonr

    rel = df["reliability"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float)
    mask = ~(np.isnan(rel) | np.isnan(err))

    rho, p_rho = spearmanr(rel[mask], err[mask])
    r, p_r = pearsonr(rel[mask], err[mask])

    print(f"\n--- RELIABILITY vs ERROR CORRELATION (PENENTU) ---")
    print(f"Spearman rho: {rho:+.3f}  (p={p_rho:.4f})")
    print(f"Pearson   r : {r:+.3f}  (p={p_r:.4f})")
    print(f"N valid     : {mask.sum()}")
    print()
    print("INTERPRETASI:")
    print("  rho NEGATIF signifikan → reliability tinggi = error rendah → abstrak SUPPORTED")
    print("  rho ~ 0 atau POSITIF  → reliability TIDAK prediktif → abstrak BELUM didukung")

    # === SIGNAL FIDELITY ANALYSIS ===
    print(f"\n--- SIGNAL FIDELITY ANALYSIS ---")
    rel_arr = df["reliability"].to_numpy(dtype=float)
    err_arr = df["error"].to_numpy(dtype=float)
    valid = ~(np.isnan(rel_arr) | np.isnan(err_arr))
    rel_v = rel_arr[valid]
    err_v = err_arr[valid]

    # 1. Point-biserial: Reliability vs Near-GT (binary)
    near_gt = (err_v <= ERROR_THRESH_MM).astype(int)
    if 0 < near_gt.sum() < len(near_gt):
        from scipy.stats import pointbiserialr
        r_pb, p_pb = pointbiserialr(rel_v, near_gt)
        print(f"Point-biserial (Rel vs Near-GT): r={r_pb:+.3f}, p={p_pb:.4f}")
        if r_pb > 0 and p_pb < 0.05:
            print("  ✅ High reliability correlates with prediction near GT")
        else:
            print("  ⚠️  No significant correlation")

    # 2. Mann-Whitney U: Error high-rel vs low-rel (non-parametric)
    median_rel = np.nanmedian(rel_v)
    high_mask = rel_v >= median_rel
    low_mask = rel_v < median_rel
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        from scipy.stats import mannwhitneyu
        err_high = err_v[high_mask]
        err_low = err_v[low_mask]
        try:
            u_stat, u_p = mannwhitneyu(err_high, err_low, alternative='less')
            print(f"\nMann-Whitney U (high-rel error < low-rel error):")
            print(f"  Mean error high-rel (n={high_mask.sum()}): {np.mean(err_high):.2f} mm")
            print(f"  Mean error low-rel  (n={low_mask.sum()}):  {np.mean(err_low):.2f} mm")
            print(f"  U={u_stat:.0f}, p={u_p:.4f}")
            if u_p < 0.05:
                print(f"  ✅ SIGNIFICANT — high-reliability scans have LOWER error!")
                print(f"     This directly supports the abstract claim.")
            else:
                print(f"  ❌ Not significant (p={u_p:.4f})")
        except Exception as e:
            print(f"  Mann-Whitney failed: {e}")

    # 3. Variance reduction
    if high_mask.sum() > 0:
        var_full = np.var(err_v)
        var_high = np.var(err_high)
        reduction = (1 - var_high / var_full) * 100 if var_full > 0 else 0
        print(f"\nVariance Reduction (high-rel subset):")
        print(f"  Full dataset variance:  {var_full:.2f}")
        print(f"  High-rel variance:    {var_high:.2f}")
        print(f"  Reduction:            {reduction:.1f}%")
    # === END SIGNAL FIDELITY ===

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