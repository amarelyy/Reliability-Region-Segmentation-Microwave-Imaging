"""
compute_phantom_baselines.py

Computes the CF noise-baseline (mean, std) for every phantom, WITHOUT
loading S21, running ICZT, or beamforming any real scan — the baseline
depends only on phantom geometry (breast_radius_mm, ant_rad, tissue
velocity), all available from tumor_model metadata alone. Reuses
compute_cf_baseline() from baseline_argmax_test.py exactly as-is.

Exists so trust-map/reliability work (reliability.py) never has to re-run
the ~30min full reconstruction pipeline — run this once (~1-2 min for 30
phantoms), then iterate on reliability.py freely.

Usage:
    python compute_phantom_baselines.py --n-phantoms 30
    # -> phantom_baselines.pkl : {phant_id: dict(baseline_mean, baseline_std,
    #                                              axis_mm, breast_radius_mm)}
"""
import argparse
import pickle

from src.data_loading import load_all_data
from select_phantoms import build_phantom_table, stratified_select
from src import physics
from src import signal_processing as sp
from baseline_argmax_test import build_grid, compute_cf_baseline

N_ANT = 72


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phantoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="phantom_baselines.pkl")
    args = parser.parse_args()

    print("Loading tumor_model metadata only (no S21 needed)...")
    d = load_all_data()
    tumor_model = d["tumor_model"]

    phantom_table = build_phantom_table(tumor_model)
    if args.n_phantoms > 0:
        selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
        selected_ids = selected["phant_id"].tolist()
    else:
        selected_ids = phantom_table["phant_id"].tolist()

    meta_by_phantom = (
        tumor_model[tumor_model["phant_id"].isin(selected_ids)]
        .groupby("phant_id").first()
    )

    time_axis = sp.get_time_axis()  # constant — no scan needed
    baselines = {}

    for phant_id, row in meta_by_phantom.iterrows():
        breast_radius_mm = float(row["breast_radius_mm"])
        v_tissue, _ = physics.compute_tissue_velocity(
            float(row["fat_fraction"]), float(row["fib_fraction"]))
        ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
        geom = physics.get_antenna_geometry(ant_rad_mm)

        grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
        grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0

        delay_grid = physics.two_medium_delay(
            geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
            grid_x_m, grid_y_m, breast_radius_mm / 1000.0, v_tissue,
        ).reshape(-1, *grid_x_mm.shape)

        baseline_mean, baseline_std = compute_cf_baseline(delay_grid, time_axis, N_ANT)
        baselines[phant_id] = dict(baseline_mean=baseline_mean, baseline_std=baseline_std,
                                    axis_mm=axis_mm, breast_radius_mm=breast_radius_mm)
        print(f"  {phant_id}: done (radius={breast_radius_mm:.1f}mm)")

    with open(args.out, "wb") as f:
        pickle.dump(baselines, f)
    print(f"\nSaved {len(baselines)} phantom baselines to {args.out}")


if __name__ == "__main__":
    main()