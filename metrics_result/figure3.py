"""
metrics_result/fig3_case_studies.py

Generates Figure 3: 2x2 case study grid comparing high-reliability
vs low-reliability reconstructions with GT and prediction overlays.

Input:  rrs_mbi/results/metrics_raw.csv
        src/data_loading.load_all_data()
        src/pipeline.reconstruct_scan()
Output: metrics_result/fig3_case_studies.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"
OUTPUT_PATH = ROOT / "metrics_result" / "fig3_case_studies.png"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run 'python -m rrs_mbi.main' first."
        )

    df = pd.read_csv(CSV_PATH)

    required = ["reliability_score", "scan_idx", "true_x_mm", "true_y_mm", "pred_x_mm", "pred_y_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    # Select top 2 highest and bottom 2 lowest reliability scans
    df_sorted = df.sort_values("reliability_score", ascending=False).reset_index(drop=True)
    high_cases = df_sorted.head(2)
    low_cases = df_sorted.tail(2).iloc[::-1]  # Reverse so worst is last

    from src.data_loading import load_all_data
    from src.pipeline import reconstruct_scan

    data = load_all_data()  # type: ignore
    s21 = data["s21"]
    tumor_model = data["tumor_model"]
    id_to_original_idx = data["id_to_original_idx"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    case_list = [
        (high_cases.iloc[0], "High"),
        (high_cases.iloc[1], "High"),
        (low_cases.iloc[0], "Low"),
        (low_cases.iloc[1], "Low"),
    ]

    for col_idx, (row, label) in enumerate(case_list):
        scan_idx = int(row["scan_idx"])
        rel_score = float(row["reliability_score"])
        gt_x = float(row["true_x_mm"])
        gt_y = float(row["true_y_mm"])
        pred_x = float(row["pred_x_mm"])
        pred_y = float(row["pred_y_mm"])
        loc_err = float(row.get("localization_error", np.nan))

        try:
            res = reconstruct_scan(
                scan_idx, s21, tumor_model, id_to_original_idx, return_diagnostics=True
            )  # type: ignore
            img = res["diagnostics"]["image"]
            axis_mm = res["diagnostics"]["axis_mm"]
        except Exception as e:
            print(f"  Failed to reconstruct scan {scan_idx}: {e}")
            continue

        # Normalize image to [0, 1]
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max - img_min > 1e-8:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)

        extent = [float(axis_mm[0]), float(axis_mm[-1]), float(axis_mm[0]), float(axis_mm[-1])]

        # Left subplot: full image
        ax_img = axes[0, col_idx] if col_idx < 2 else axes[1, col_idx - 2]
        ax_img.imshow(img_norm, extent=extent, origin="lower", cmap="hot", aspect="equal")
        ax_img.plot(gt_x, gt_y, "g*", markersize=12, markeredgecolor="white", markeredgewidth=1.5, label="GT")
        ax_img.plot(pred_x, pred_y, "bx", markersize=10, markeredgewidth=2.5, label="Pred")
        ax_img.set_title(f"{label} Reliability\nScore={rel_score:.4f}", fontsize=11, fontweight="bold")
        ax_img.set_xlabel("x (mm)", fontsize=9)
        ax_img.set_ylabel("y (mm)", fontsize=9)
        ax_img.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax_img.grid(True, alpha=0.2, color="white")

        # Right subplot: zoomed view around center
        ax_zoom = axes[0, col_idx + 2] if col_idx < 2 else axes[1, col_idx]  
        # Actually let's use a simpler 2x4 layout: row 0 = high, row 1 = low, cols 0-1 = full, cols 2-3 = zoom
        # Reorganize: use 2 rows x 4 cols where each case gets 2 columns
        
    plt.close()

    # Redo with cleaner layout: 2 rows (high/low) x 4 cols (case1_full, case1_zoom, case2_full, case2_zoom)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for row_idx, (cases, label) in enumerate([(high_cases, "High"), (low_cases, "Low")]):
        for case_idx in range(2):
            if case_idx >= len(cases):
                continue
            row = cases.iloc[case_idx]
            scan_idx = int(row["scan_idx"])
            rel_score = float(row["reliability_score"])
            gt_x = float(row["true_x_mm"])
            gt_y = float(row["true_y_mm"])
            pred_x = float(row["pred_x_mm"])
            pred_y = float(row["pred_y_mm"])
            loc_err = float(row.get("localization_error", np.nan))
            peak_dom = float(row.get("peak_dominance", np.nan))
            bound_risk = float(row.get("boundary_risk", np.nan))

            try:
                res = reconstruct_scan(
                    scan_idx, s21, tumor_model, id_to_original_idx, return_diagnostics=True
                )  # type: ignore
                img = res["diagnostics"]["image"]
                axis_mm = res["diagnostics"]["axis_mm"]
            except Exception as e:
                print(f"  Failed scan {scan_idx}: {e}")
                continue

            img_min = float(np.min(img))
            img_max = float(np.max(img))
            if img_max - img_min > 1e-8:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img)

            extent = [float(axis_mm[0]), float(axis_mm[-1]), float(axis_mm[0]), float(axis_mm[-1])]
            scan_id = str(row.get("scan_id", f"idx{scan_idx}"))

            # Full image
            col_full = case_idx * 2
            ax_f = axes[row_idx, col_full]
            ax_f.imshow(img_norm, extent=extent, origin="lower", cmap="hot", aspect="equal")
            ax_f.plot(gt_x, gt_y, "g*", markersize=14, markeredgecolor="white", markeredgewidth=1.5)
            ax_f.plot(pred_x, pred_y, "bx", markersize=12, markeredgewidth=2.5)
            ax_f.set_title(
                f"{label} | {scan_id}\nRel={rel_score:.3f} | Dom={peak_dom:.1f} | Risk={bound_risk:.3f}",
                fontsize=10, fontweight="bold"
            )
            ax_f.set_xlabel("x (mm)", fontsize=9)
            if col_full == 0:
                ax_f.set_ylabel("y (mm)", fontsize=9)
            ax_f.grid(True, alpha=0.15, color="white")

            # Zoomed view (±25 mm around center)
            col_zoom = case_idx * 2 + 1
            ax_z = axes[row_idx, col_zoom]
            zoom_range = 25.0
            ax_z.imshow(img_norm, extent=extent, origin="lower", cmap="hot", aspect="equal")
            ax_z.set_xlim(-zoom_range, zoom_range)
            ax_z.set_ylim(-zoom_range, zoom_range)
            ax_z.plot(gt_x, gt_y, "g*", markersize=16, markeredgecolor="white", markeredgewidth=1.5, label="Ground Truth")
            ax_z.plot(pred_x, pred_y, "bx", markersize=14, markeredgewidth=2.5, label="Prediction")
            err_str = f"Err={loc_err:.1f}mm" if not np.isnan(loc_err) else "Err=N/A"
            ax_z.set_title(f"Zoomed ±{zoom_range:.0f}mm | {err_str}", fontsize=10, fontweight="bold")
            ax_z.set_xlabel("x (mm)", fontsize=9)
            if col_zoom == 1:
                ax_z.set_ylabel("y (mm)", fontsize=9)
            ax_z.legend(loc="upper right", fontsize=8, framealpha=0.85)
            ax_z.grid(True, alpha=0.2, color="white")

    # Row labels
    axes[0, 0].annotate("HIGH CONFIDENCE", xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=13, fontweight="bold", color="#2ca02c",
                        rotation=90, va="center", ha="center")
    axes[1, 0].annotate("LOW CONFIDENCE", xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=13, fontweight="bold", color="#d62728",
                        rotation=90, va="center", ha="center")

    plt.suptitle(
        "Qualitative Comparison: High vs Low Reliability Reconstructions",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved to: {OUTPUT_PATH}")
    print(f"\nCases selected:")
    for _, r in high_cases.iterrows():
        print(f"  HIGH: {r['scan_id']} | Rel={r['reliability_score']:.4f} | Err={r.get('localization_error', np.nan):.1f}mm")
    for _, r in low_cases.iterrows():
        print(f"  LOW:  {r['scan_id']} | Rel={r['reliability_score']:.6f} | Err={r.get('localization_error', np.nan):.1f}mm")


if __name__ == "__main__":
    main()