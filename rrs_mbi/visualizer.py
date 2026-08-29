"""
rrs_mbi/visualizer.py
Strict Visualization for RRS-MBI.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import reconstruct_scan
from src.data_loading import load_all_data

def plot_reliability_vs_error_scatter(df: pd.DataFrame, save_path: str = "rrs_mbi/results/scatter_reliability_error.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(df['reliability_score'], df['localization_error'], alpha=0.6, edgecolors='w', linewidth=0.5,
                c=df['reliability_score'], cmap='RdYlGn_r', label='Scans')
    
    x_vals = df['reliability_score'].to_numpy(dtype=np.float64)
    y_vals = df['localization_error'].to_numpy(dtype=np.float64)
    
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    plt.plot(x_vals, p(x_vals), "r--", linewidth=2, label=f'Trend (y = {z[0]:.2f}x + {z[1]:.2f})')
    
    plt.xlabel('Reliability Score', fontsize=12, fontweight='bold')
    plt.ylabel('Localization Error (mm)', fontsize=12, fontweight='bold')
    plt.title('Correlation: Higher Reliability = Lower Error', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Scatter plot saved to {save_path}")
    plt.close()

def plot_case_studies(df: pd.DataFrame, save_path: str = "rrs_mbi/results/case_studies.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    valid_df = df.dropna(subset=['true_x_mm', 'true_y_mm', 'pred_x_mm', 'pred_y_mm'])
    if valid_df.empty:
        print("⚠️ WARNING: No valid scans for case studies.")
        return
        
    high_rel_row = valid_df.loc[valid_df['reliability_score'].idxmax()]
    low_rel_row = valid_df.loc[valid_df['reliability_score'].idxmin()]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    rows_to_plot = [
        (high_rel_row, "High Reliability Scan", axes[0, 0]),
        (high_rel_row, "High Reliability (GT vs Pred)", axes[0, 1]),
        (low_rel_row, "Low Reliability Scan (Artifact-Prone)", axes[1, 0]),
        (low_rel_row, "Low Reliability (GT vs Pred)", axes[1, 1])
    ]
    
    data = load_all_data()  # type: ignore
    s21 = data["s21"]
    tumor_model = data["tumor_model"]
    id_to_original_idx = data["id_to_original_idx"]

    for i, (row, title, ax) in enumerate(rows_to_plot):
        # Gunakan scan_idx (indeks posisional) yang disimpan di CSV
        scan_idx = int(row['scan_idx']) 
        
        # Reconstruct dengan diagnostics
        res = reconstruct_scan(
            scan_idx, s21, tumor_model, id_to_original_idx, 
            return_diagnostics=True
        )  # type: ignore
        
        img = res["diagnostics"]["image"]
        axis_mm = res["diagnostics"]["axis_mm"] # Sumbu fisik dalam mm
        
        # Plot gambar dengan extent agar sumbu X dan Y dalam satuan mm!
        im = ax.imshow(img, cmap='magma', aspect='equal', 
                       extent=[axis_mm[0], axis_mm[-1], axis_mm[0], axis_mm[-1]], origin='lower')
        
        # Plot Ground Truth dan Prediksi (Koordinat sudah dalam mm, jadi akan pas sempurna!)
        ax.plot(row['true_x_mm'], row['true_y_mm'], 'g*', markersize=15, label='Ground Truth')
        ax.plot(row['pred_x_mm'], row['pred_y_mm'], 'bx', markersize=12, markeredgewidth=2, label='Prediction')
        
        ax.set_title(f"{title}\nScore: {row['reliability_score']:.3f} | Error: {row['localization_error']:.2f}mm", 
                     fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        
        if i == 0:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', rotation=270, labelpad=15)

    plt.suptitle('Qualitative Comparison: High vs. Low Reliability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Case studies plot saved to {save_path}")
    plt.close()

def generate_all_visuals(df: pd.DataFrame):
    print("Generating LKT-ready visualizations...")
    plot_reliability_vs_error_scatter(df)
    plot_case_studies(df)
    print("✅ Visualization generation complete.")