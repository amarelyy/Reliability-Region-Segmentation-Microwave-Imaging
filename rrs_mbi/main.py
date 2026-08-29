"""
rrs_mbi/main.py
Entry point untuk menjalankan seluruh pipeline RRS-MBI.
"""

import sys
from pathlib import Path

# Memastikan root folder project bisa diakses oleh Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rrs_mbi.evaluator import run_evaluation, analyze_thresholds, plot_selective_classification_curve
from rrs_mbi.visualizer import generate_all_visuals

def main():
    print("=" * 70)
    print("RRS-MBI EVALUATION PIPELINE (STRICT MODE)")
    print("=" * 70)
    
    # 1. Jalankan Evaluasi (Menghitung metrik & error untuk semua scan)
    print("\n[1/3] Running full dataset evaluation...")
    results_df = run_evaluation()
    
    # 2. Simpan Hasil Mentah
    results_dir = Path("rrs_mbi/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / "metrics_raw.csv", index=False)
    print(f"✅ Raw metrics saved to {results_dir / 'metrics_raw.csv'}")
    
    # 3. Analisis Threshold (Selective Classification)
    print("\n[2/3] Analyzing selective classification thresholds...")
    analysis_df = analyze_thresholds(results_df)
    analysis_df.to_csv(results_dir / "threshold_analysis.csv", index=False)
    
    # Cetak ringkasan tabel untuk LKT
    print("\n" + "=" * 75)
    print("SUMMARY: Localization Error vs. Data Retention")
    print("=" * 75)
    print(f"{'Scans Kept':<12} | {'Threshold':<10} | {'Mean Error (mm)':<16} | {'95% CI'}")
    print("-" * 75)
    for _, row in analysis_df.iterrows():
        ci_str = f"[{row['ci_lower']:.2f} - {row['ci_upper']:.2f}]"
        print(f"{row['scans_kept']:<12} | {row['actual_threshold']:<10.3f} | {row['mean_error']:<16.2f} | {ci_str}")
    print("=" * 75)
    
    # 4. Generate Visualisasi
    print("\n[3/3] Generating publication-ready figures...")
    plot_selective_classification_curve(analysis_df)
    generate_all_visuals(results_df)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Check the '{results_dir}' folder for your LKT figures and CSV data.")

if __name__ == "__main__":
    main()