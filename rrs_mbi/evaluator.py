"""
rrs_mbi/evaluator.py
Strict Evaluation Pipeline for RRS-MBI.
Synced with src/pipeline.py architecture.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan
from rrs_mbi.metrics import evaluate_scan_reliability

def compute_localization_error(pred_coords: Tuple[float, float], true_coords: Tuple[float, float]) -> float:
    return float(np.sqrt((pred_coords[0] - true_coords[0])**2 + (pred_coords[1] - true_coords[1])**2))

def bootstrap_confidence_interval(errors: np.ndarray, n_bootstraps: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    if len(errors) == 0:
        return 0.0, 0.0
    errors_clean = np.array(errors, dtype=np.float64).ravel()
    bootstrap_means = [float(np.mean(np.random.choice(errors_clean, size=len(errors_clean), replace=True))) for _ in range(n_bootstraps)]
    alpha = 1.0 - confidence
    lower_bound = float(np.percentile(bootstrap_means, 100 * (alpha / 2)))
    upper_bound = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return lower_bound, upper_bound

def run_evaluation() -> pd.DataFrame:
    print("Loading data...")
    data = load_all_data()  # type: ignore
    
    s21 = data["s21"]
    tumor_model = data["tumor_model"]
    id_to_original_idx = data["id_to_original_idx"]
    
    # Kita gunakan indeks posisional (0 sampai N-1) untuk .iloc[]
    total_scans = len(tumor_model)
    
    results: List[Dict[str, Any]] = []
    skipped_scans: List[int] = []
    
    print(f"Evaluating {total_scans} scans...")
    
    for i in range(total_scans):
        if i % 50 == 0:
            print(f"  Processing scan {i+1}/{total_scans}...")
            
        try:
            # 1. Reconstruct (Kirim indeks posisional 'i' dan minta diagnostics untuk mendapatkan gambar)
            res = reconstruct_scan(
                i, s21, tumor_model, id_to_original_idx, 
                return_diagnostics=True
            )  # type: ignore
            
            # 2. Ekstrak Gambar untuk Reliability Metrics
            img = res["diagnostics"]["image"]
            metrics = evaluate_scan_reliability(img)
            
            # 3. Ekstrak Prediksi & Ground Truth (SUDAH DIHITUNG OLEH PIPELINE dalam mm!)
            pred_x_mm = float(res.get("peak_x_mm", np.nan))
            pred_y_mm = float(res.get("peak_y_mm", np.nan))
            true_x_mm = float(res.get("gt_x_mm", np.nan))
            true_y_mm = float(res.get("gt_y_mm", np.nan))
            
            # Skip jika ini scan healthy (tidak ada tumor / ground truth NaN)
            if np.isnan(true_x_mm) or np.isnan(true_y_mm):
                skipped_scans.append(i)
                continue
                
            # 4. Hitung Error Lokalisasi (dalam mm)
            error = compute_localization_error((pred_x_mm, pred_y_mm), (true_x_mm, true_y_mm))
            
            # Ambil ID asli untuk logging
            scan_id = res.get("phant_id", tumor_model.iloc[i].get("id", i))
            
            results.append({
                "scan_id": str(scan_id),
                "scan_idx": i,  # Simpan indeks posisional untuk visualizer nanti
                "reliability_score": metrics["reliability_score"],
                "peak_dominance": metrics["peak_dominance"],
                "boundary_risk": metrics["boundary_risk"],
                "localization_error": error,
                "pred_x_mm": pred_x_mm,
                "pred_y_mm": pred_y_mm,
                "true_x_mm": true_x_mm,
                "true_y_mm": true_y_mm
            })
            
        except Exception as e:
            skipped_scans.append(i)
            continue
            
    if skipped_scans:
        print(f"\n⚠️ Skipped {len(skipped_scans)} scans (healthy scans or processing errors).")
        
    if not results:
        raise ValueError("No valid scans processed.")
        
    return pd.DataFrame(results)

def analyze_thresholds(df: pd.DataFrame, num_thresholds: int = 10) -> pd.DataFrame:
    df_sorted = df.sort_values(by="reliability_score", ascending=False).reset_index(drop=True)
    total_scans = len(df_sorted)
    
    analysis_results = []
    
    for keep_ratio in np.linspace(1.0, 0.1, num_thresholds):
        n_keep = max(1, int(total_scans * keep_ratio))
        subset = df_sorted.iloc[:n_keep]
        
        errors = subset["localization_error"].dropna().to_numpy(dtype=np.float64)
        if len(errors) == 0:
            continue
            
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        ci_lower, ci_upper = bootstrap_confidence_interval(errors)
        actual_threshold = float(subset["reliability_score"].iloc[-1])
        
        analysis_results.append({
            "keep_ratio": keep_ratio,
            "scans_kept": n_keep,
            "actual_threshold": actual_threshold,
            "mean_error": mean_err,
            "std_error": std_err,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        
    return pd.DataFrame(analysis_results)

def plot_selective_classification_curve(analysis_df: pd.DataFrame, save_path: str = "rrs_mbi/results/selective_curve.png") -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    ax1.fill_between(analysis_df["scans_kept"], analysis_df["ci_lower"], analysis_df["ci_upper"], color="#1f77b4", alpha=0.2, label="95% CI")
    ax1.plot(analysis_df["scans_kept"], analysis_df["mean_error"], color="#1f77b4", marker="o", linewidth=2, label="Mean Error")
    ax1.set_xlabel("Scans Kept (Sorted by Reliability)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Localization Error (mm)", color="#1f77b4", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(analysis_df["scans_kept"], analysis_df["actual_threshold"], color="#d62728", marker="s", linestyle="--", linewidth=2, label="Threshold")
    ax2.set_ylabel("Min Reliability Threshold", color="#d62728", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    
    plt.title("Selective Classification Improves Accuracy", fontsize=14, fontweight="bold", pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to {save_path}")
    plt.close()

def main() -> None:
    print("=" * 70)
    print("RRS-MBI EVALUATION PIPELINE")
    print("=" * 70)
    
    results_df = run_evaluation()
    results_dir = Path("rrs_mbi/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / "metrics_raw.csv", index=False)
    
    analysis_df = analyze_thresholds(results_df)
    analysis_df.to_csv(results_dir / "threshold_analysis.csv", index=False)
    
    print("\n" + "=" * 70)
    print(f"{'Scans Kept':<12} | {'Threshold':<10} | {'Mean Error (mm)':<16} | {'95% CI'}")
    print("-" * 70)
    for _, row in analysis_df.iterrows():
        print(f"{row['scans_kept']:<12} | {row['actual_threshold']:<10.3f} | {row['mean_error']:<16.2f} | [{row['ci_lower']:.2f} - {row['ci_upper']:.2f}]")
    print("=" * 70)
    
    plot_selective_classification_curve(analysis_df)
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()