"""
metrics_result/figure_case_fullfield.py

Poster-ready qualitative figure for RRS-MBI. Fixes case_studies.png:
  1. "Black field + central dot" -> plot RAW DAS magnitude (use_cf=False)
     with PERCENTILE contrast stretch, so clutter/noise structure is
     visible across the whole field (like the old 4-panel figure).
  2. "High vs Low look identical" -> pick CONSISTENT cases (score agrees
     with outcome): high score + small error vs low score + large error.
Trust is communicated ONLY via metrics in the title (Score | Error).

Usage:
    python metrics_result/figure_case_fullfield.py
    python metrics_result/figure_case_fullfield.py --image cf --cmap turbo
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CSV_PATH = ROOT / "rrs_mbi" / "results" / "metrics_raw.csv"
OUT_PATH = ROOT / "metrics_result" / "fig_case_fullfield.png"


def pick_case(df: pd.DataFrame, want: str) -> pd.Series:
    """Pick a CONSISTENT case: reliability score agrees with the outcome."""
    ok = df.dropna(subset=["reliability_score", "localization_error",
                           "true_x_mm", "true_y_mm",
                           "pred_x_mm", "pred_y_mm"]).copy()
    if want == "high":
        cand = ok[(ok["reliability_score"] >= 0.7) &
                  (ok["localization_error"] <= 10.0)]
        cand = cand.sort_values("reliability_score", ascending=False)
    else:
        cand = ok[(ok["reliability_score"] <= 0.3) &
                  (ok["localization_error"] >= 25.0)]
        cand = cand.sort_values("reliability_score", ascending=True)
    if cand.empty:  # fallback: extreme score only
        cand = ok.sort_values("reliability_score", ascending=(want == "low"))
    return cand.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", choices=["raw", "cf"], default="raw",
                    help="raw = DAS magnitude (clutter visible, default); "
                         "cf = CF-weighted (suppressed clutter)")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--vmin-pct", type=float, default=2.0)
    ap.add_argument("--vmax-pct", type=float, default=99.5)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)

    from src.data_loading import load_all_data
    from src.pipeline import reconstruct_scan

    data = load_all_data()
    s21, tumor_model, idmap = (data["s21"], data["tumor_model"],
                               data["id_to_original_idx"])

    cases = [
        (pick_case(df, "high"), "MEDIUM RELIABILITY"),
        (pick_case(df, "low"),  "LOW RELIABILITY"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, (row, label) in zip(axes, cases):
        scan_idx = int(row["scan_idx"])
        res = reconstruct_scan(
            scan_idx, s21, tumor_model, idmap,
            beamformer="das",
            use_cf=(args.image == "cf"),   # KEY FIX 1: no CF -> clutter visible
            return_diagnostics=True,
        )
        img = res["diagnostics"]["image"]
        axis_mm = res["diagnostics"]["axis_mm"]

        # KEY FIX 2: percentile stretch, NOT min-max autoscale
        v0 = np.percentile(img, args.vmin_pct)
        v1 = np.percentile(img, args.vmax_pct)

        ext = [float(axis_mm[0]), float(axis_mm[-1]),
               float(axis_mm[0]), float(axis_mm[-1])]
        im = ax.imshow(img, extent=ext, origin="lower", cmap=args.cmap,
                       vmin=v0, vmax=v1, aspect="equal")

        ax.plot(row["true_x_mm"], row["true_y_mm"], "g*", markersize=16,
                markeredgecolor="white", markeredgewidth=1.2,
                label="Ground truth")
        ax.plot(row["pred_x_mm"], row["pred_y_mm"], "bx", markersize=12,
                markeredgewidth=2.5, label="Prediction")

        color = "#2ca02c" if "MEDIUM" in label else "#d62728"
        ax.set_title(
            f"{label}\nScore: {row['reliability_score']:.3f} | "
            f"Error: {row['localization_error']:.1f} mm",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")

    # plt.suptitle("Full-field reconstruction: seeing where the noise lives",
    #              fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for row, label in cases:
        print(f"  {label}: scan_idx={int(row['scan_idx'])} "
              f"score={row['reliability_score']:.3f} "
              f"err={row['localization_error']:.1f}mm")


if __name__ == "__main__":
    main()