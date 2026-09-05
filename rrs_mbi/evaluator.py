"""
rrs_mbi/evaluator.py
Complete RRS-MBI Evaluation Pipeline with Size Estimation & Full Metrics.
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


def compute_localization_error(pred: Tuple[float, float], true: Tuple[float, float]) -> float:
    return float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))


def bootstrap_ci(errors: np.ndarray, n_boot: int = 1000, conf: float = 0.95) -> Tuple[float, float]:
    if len(errors) == 0:
        return 0.0, 0.0
    e = np.asarray(errors, dtype=np.float64).ravel()
    means = [
        float(np.mean(np.random.choice(e, size=len(e), replace=True)))
        for _ in range(n_boot)
    ]
    alpha = 1.0 - conf
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_evaluation() -> pd.DataFrame:
    print("Loading data...")
    data = load_all_data()  # type: ignore

    s21 = data["s21"]
    tumor_model = data["tumor_model"]
    id_to_original_idx = data["id_to_original_idx"]
    freq_axis = data["freq_axis"]

    # ==================================================================
    # DIAGNOSTIC: Cek ketersediaan Ground Truth
    # ==================================================================
    total_raw = len(tumor_model)
    gt_cols_candidates = [
        ["tumor_x_mm", "tumor_y_mm", "tumor_radius_mm"],
        ["tum_x", "tum_y", "tum_rad"],
    ]
    available_gt: list[str] = []
    for cols in gt_cols_candidates:
        if all(c in tumor_model.columns for c in cols):
            available_gt = cols
            break

    if available_gt:
        valid_mask = tumor_model[available_gt].notna().all(axis=1)
        n_valid = int(valid_mask.sum())
        print(f"\nGT Diagnostic:")
        print(f"   Total tumor_model rows: {total_raw}")
        print(f"   Valid GT columns found: {available_gt}")
        print(f"   Rows with valid GT: {n_valid}")
        print(f"   Rows without GT (NaN): {total_raw - n_valid}")
    else:
        print(f"\nWARNING: No GT columns found! Available: {tumor_model.columns.tolist()}")

    # Sample check
    try:
        sample = reconstruct_scan(
            0, s21, tumor_model, id_to_original_idx,
            freq_axis=freq_axis, use_tvsvd=True, use_rms_norm=True,
            bandpass_mode="full", gate_ns=0.70,
            return_diagnostics=True
        )
        print(
            f"   Sample gt_x_mm={sample.get('gt_x_mm')}, "
            f"gt_y_mm={sample.get('gt_y_mm')}, "
            f"gt_r_mm={sample.get('gt_r_mm')}"
        )
    except Exception as e:
        print(f"   Sample reconstruct failed: {e}")
    print()

    # ==================================================================
    # MAIN EVALUATION LOOP
    # ==================================================================
    total_scans = len(tumor_model)
    results: List[Dict[str, Any]] = []
    skipped: List[int] = []

    print(f"Evaluating {total_scans} scans...")
    for i in range(total_scans):
        if i % 50 == 0:
            print(f"  Processing {i + 1}/{total_scans}...")

        try:
            res = reconstruct_scan(
                i, s21, tumor_model, id_to_original_idx,
                freq_axis=freq_axis,
                use_tvsvd=True,
                use_rms_norm=True,
                bandpass_mode="full",
                gate_ns=0.70,
                return_diagnostics=True
            )

            img = res["diagnostics"]["image"]
            axis_mm = res["diagnostics"]["axis_mm"]
            rel_metrics = evaluate_scan_reliability(img)

            # Ground Truth
            gt_x = float(res.get("gt_x_mm", np.nan))
            gt_y = float(res.get("gt_y_mm", np.nan))
            gt_r = float(res.get("gt_r_mm", np.nan))

            if np.isnan(gt_x) or np.isnan(gt_y):
                skipped.append(i)
                continue

            # Prediction
            pred_x = float(res.get("peak_x_mm", np.nan))
            pred_y = float(res.get("peak_y_mm", np.nan))
            loc_error = compute_localization_error((pred_x, pred_y), (gt_x, gt_y))

            # Size Estimation
            blob_area_px = float(res.get("blob_area_px", np.nan))
            pixel_size_mm = (
                abs(float(axis_mm[1]) - float(axis_mm[0]))
                if len(axis_mm) > 1
                else 1.0
            )
            if not np.isnan(blob_area_px) and blob_area_px > 0:
                predicted_r_mm = float(np.sqrt(blob_area_px / np.pi) * pixel_size_mm)
            else:
                predicted_r_mm = float("nan")

            if not np.isnan(predicted_r_mm) and not np.isnan(gt_r):
                size_error = float(abs(predicted_r_mm - gt_r))
            else:
                size_error = float("nan")

            scan_id = res.get("phant_id", str(i))

            results.append({
                "scan_id": str(scan_id),
                "scan_idx": i,
                # Reliability
                "reliability_score": rel_metrics["reliability_score"],
                "peak_dominance": rel_metrics["peak_dominance"],
                "boundary_risk": rel_metrics["boundary_risk"],
                # Localization
                "localization_error": loc_error,
                "pred_x_mm": pred_x,
                "pred_y_mm": pred_y,
                "true_x_mm": gt_x,
                "true_y_mm": gt_y,
                "true_r_mm": gt_r,
                # Size
                "predicted_r_mm": predicted_r_mm,
                "size_error_mm": size_error,
                # Pipeline metrics
                "blob_area_px": blob_area_px,
                "blob_compactness": float(res.get("blob_compactness", np.nan)),
                "cf_at_peak": float(res.get("cf_at_peak", np.nan)),
                "scr_db": float(res.get("scr_db", np.nan)),
                "smr_db": float(res.get("smr_db", np.nan)),
                "cnr": float(res.get("cnr", np.nan)),
                # Pipeline config traceability  <-- DI SINI
                "tvsvd_removed": int(res.get("tvsvd_removed", 0)),
                "gate_ns": float(res.get("gate_ns", np.nan)),
                "use_rms_norm": bool(res.get("use_rms_norm", False)),
                "bandpass_mode": res.get("bandpass_mode", "full"),
            })

        except Exception as e:
            skipped.append(i)
            continue

    print(f"\nSkipped {len(skipped)} scans (no GT or error). Evaluated: {len(results)}")

    if not results:
        raise ValueError("No valid scans processed.")

    return pd.DataFrame(results)


def analyze_thresholds(df: pd.DataFrame, num_thresholds: int = 10) -> pd.DataFrame:
    df_sorted = df.sort_values(by="reliability_score", ascending=False).reset_index(drop=True)
    total = len(df_sorted)
    rows: list[dict[str, Any]] = []

    for ratio in np.linspace(1.0, 0.1, num_thresholds):
        n_keep = max(1, int(total * ratio))
        subset = df_sorted.iloc[:n_keep]

        errs = subset["localization_error"].dropna().to_numpy(dtype=np.float64)
        if len(errs) == 0:
            continue

        ci_lo, ci_hi = bootstrap_ci(errs)

        loc_detected = int(np.sum(errs <= 15.0))
        loc_det_rate = loc_detected / len(errs) * 100

        size_errs = (
            subset["size_error_mm"].dropna().to_numpy(dtype=np.float64)
            if "size_error_mm" in subset.columns
            else np.array([], dtype=np.float64)
        )
        if len(size_errs) > 0:
            size_detected = int(np.sum(size_errs <= 5.0))
            size_det_rate = size_detected / len(size_errs) * 100
            mean_size_err = float(np.mean(size_errs))
        else:
            size_detected = 0
            size_det_rate = float("nan")
            mean_size_err = float("nan")

        rows.append({
            "keep_ratio": round(float(ratio), 2),
            "scans_kept": n_keep,
            "actual_threshold": float(subset["reliability_score"].iloc[-1]),
            "mean_error_mm": float(np.mean(errs)),
            "std_error_mm": float(np.std(errs)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "loc_detection_rate_pct": round(loc_det_rate, 1),
            "mean_size_error_mm": round(mean_size_err, 2) if not np.isnan(mean_size_err) else float("nan"),
            "size_detection_rate_pct": round(size_det_rate, 1) if not np.isnan(size_det_rate) else float("nan"),
        })

    return pd.DataFrame(rows)


def plot_selective_classification_curve(
    analysis_df: pd.DataFrame,
    save_path: str = "rrs_mbi/results/selective_curve.png",
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.fill_between(
        analysis_df["scans_kept"],
        analysis_df["ci_lower"],
        analysis_df["ci_upper"],
        color="#1f77b4",
        alpha=0.2,
        label="95% CI",
    )
    ax1.plot(
        analysis_df["scans_kept"],
        analysis_df["mean_error_mm"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Mean Error",
    )
    ax1.set_xlabel("Scans Kept (Sorted by Reliability)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Localization Error (mm)", color="#1f77b4", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        analysis_df["scans_kept"],
        analysis_df["actual_threshold"],
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    ax2.set_ylabel("Min Reliability Threshold", color="#d62728", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Selective Classification Improves Accuracy", fontsize=14, fontweight="bold", pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def main() -> None:
    print("=" * 70)
    print("RRS-MBI COMPLETE EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Evaluate
    df = run_evaluation()
    out_dir = Path("rrs_mbi/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics_raw.csv", index=False)

    # 2. Threshold Analysis
    thresh_df = analyze_thresholds(df)
    thresh_df.to_csv(out_dir / "threshold_analysis.csv", index=False)

    # 3. Print Complete Summary
    print("\n" + "=" * 70)
    print("COMPLETE SUMMARY")
    print("=" * 70)
    print(f"Scans evaluated: {len(df)}")

    print(f"\n--- Localization ---")
    print(f"  Mean Error:       {df['localization_error'].mean():.2f} mm")
    print(f"  Std Error:        {df['localization_error'].std():.2f} mm")
    det_count = int((df["localization_error"] <= 15.0).sum())
    print(f"  Detection Rate (<=15mm): {det_count / len(df) * 100:.1f}% ({det_count}/{len(df)})")

    print(f"\n--- Size Estimation ---")
    size_err = df["size_error_mm"].dropna()
    if len(size_err) > 0:
        print(f"  Mean Size Error:  {size_err.mean():.2f} mm")
        size_det = int((size_err <= 5.0).sum())
        print(f"  Size Detection (<=5mm): {size_det / len(size_err) * 100:.1f}% ({size_det}/{len(size_err)})")
    else:
        print(f"  Mean Size Error:  N/A")
        print(f"  Size Detection:   N/A")

    print(f"\n--- Signal Quality ---")
    scr = df["scr_db"].dropna()
    cnr = df["cnr"].dropna()
    cf = df["cf_at_peak"].dropna()
    print(f"  Mean SCR (dB):    {scr.mean():.2f}" if len(scr) > 0 else "  Mean SCR (dB):    N/A")
    print(f"  Mean CNR:         {cnr.mean():.2f}" if len(cnr) > 0 else "  Mean CNR:         N/A")
    print(f"  Mean CF at Peak:  {cf.mean():.4f}" if len(cf) > 0 else "  Mean CF at Peak:  N/A")

    print(f"\n--- Reliability ---")
    print(f"  Mean Score:       {df['reliability_score'].mean():.4f}")
    print(f"  Mean Dominance:   {df['peak_dominance'].mean():.2f}")
    print(f"  Mean Boundary Risk: {df['boundary_risk'].mean():.4f}")

    print(f"\n--- Threshold Analysis ---")
    print(f"{'Kept':<8} | {'Thresh':<10} | {'MeanErr':<10} | {'Std':<8} | {'DetRate':<8} | {'95% CI'}")
    print("-" * 75)
    for _, r in thresh_df.iterrows():
        print(
            f"{r['scans_kept']:<8} | {r['actual_threshold']:<10.4f} | "
            f"{r['mean_error_mm']:<10.2f} | {r['std_error_mm']:<8.2f} | "
            f"{r['loc_detection_rate_pct']:<7.1f}% | "
            f"[{r['ci_lower']:.2f}-{r['ci_upper']:.2f}]"
        )
        
    acc = df["localization_error"] <= 15.0
    high = df["reliability_score"] >= 0.70
    low = df["reliability_score"] < 0.30
    print(f"\n--- Confusion Matrix ---")
    print(f"  TP (accurate + high rel):        {int((acc & high).sum())}")
    print(f"  Fortuitous (accurate + low rel): {int((acc & low).sum())}")
    print(f"  TN (inaccurate + low rel):       {int((~acc & low).sum())}")
    print(f"  FP (inaccurate + high rel):      {int((~acc & high).sum())}")
    
    print("=" * 70)

    # 4. Plot
    plot_selective_classification_curve(thresh_df)
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()