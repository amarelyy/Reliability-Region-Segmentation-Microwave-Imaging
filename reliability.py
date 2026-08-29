"""
reliability.py

Builds and evaluates the reliability layer on TOP OF an already-saved
svm_candidates.csv and phantom_baselines.pkl. No reconstruction happens
here — everything runs in seconds.

CHANGE LOG (see chat discussion):
  - artifact_z direction FLIPPED: diagnostics showed label=0 (false
    positive) has HIGHER artifact_z than label=1 (real tumor) — same
    wrong-direction pattern as cf_at_peak/cf_background_ratio/local_scr_db
    in the original SVM feature set. High z now means LOWER trust, not
    higher.
  - dist_ratio-based boundary component REMOVED from the trust map: 99.2%
    of candidates sit below dist_ratio 0.75 because peaks are found via
    local-maxima search, and real boundary-tumor signal is too weak to
    ever form a peak out there in the first place (consistent with the
    -24.8dB boundary-collapse finding) — so this component was checking a
    region candidates structurally don't occupy, not measuring anything.
  - REPLACED with peak dominance: top-1 vs top-2 candidate SCR gap
    (approximated via local_scr_db, since raw peak intensity isn't stored
    per candidate) as a second, independent, inference-safe signal that
    doesn't require knowing the true tumor location.

Trust is now computed at the SCAN level (not per-candidate) — it describes
confidence in the scan's reported answer (the argmax candidate), which
matches how the reliability label R is itself defined.

Usage:
    python reliability.py --diagnose-only     # print diagnostics, stop
    python reliability.py                     # full run, once thresholds
                                               # below are trusted
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DETECTION_THRESHOLD_MM = 20.0

# --- artifact_z threshold (FLIPPED direction) ---
# Must be re-derived from print_argmax_artifact_z_diagnostics()'s R=1
# argmax-level p90 (scan-level population), NOT the per-candidate/all-ranks
# label=1 p90 used previously — that was a brightness-biased subpopulation
# (rank-#0 candidates are systematically higher artifact_z than the general
# pool just by virtue of being the peak). Placeholder until rerun.
ARTIFACT_Z_ARTIFACT_RISK = 110.59

# --- peak dominance thresholds: TBD, printed as diagnostics below before
# being wired into the trust map at all ---
DOMINANCE_HIGH_DB = None   # set after inspecting print_dominance_diagnostics() output
DOMINANCE_LOW_DB = None


def trust_bin_at_point(peak_x_mm, peak_y_mm, cf_at_peak, breast_radius_mm, baseline_entry):
    """Per-candidate artifact_z only (dist_ratio component removed — see
    module docstring). Scan-level trust combines this with dominance in
    assign_scan_trust(), not here."""
    axis_mm = baseline_entry["axis_mm"]
    ix = int(np.argmin(np.abs(axis_mm - peak_x_mm)))
    iy = int(np.argmin(np.abs(axis_mm - peak_y_mm)))

    b_mean = baseline_entry["baseline_mean"][iy, ix]
    b_std = baseline_entry["baseline_std"][iy, ix]
    artifact_z = (cf_at_peak - b_mean) / (b_std + 1e-6)

    dist_ratio = float(np.hypot(peak_x_mm, peak_y_mm)) / breast_radius_mm
    return artifact_z, dist_ratio


def tag_candidates_with_trust(cand_df, baselines, phantom_table):
    """Adds artifact_z, dist_ratio_calc per candidate. trust_bin itself is
    no longer assigned here — see assign_scan_trust()."""
    radius_lookup = dict(zip(phantom_table["phant_id"], phantom_table["breast_radius_mm"]))

    artifact_zs, dist_ratios = [], []
    for _, row in cand_df.iterrows():
        phant_id = row["phant_id"]
        if phant_id not in baselines or phant_id not in radius_lookup:
            artifact_zs.append(np.nan); dist_ratios.append(np.nan)
            continue
        breast_radius_mm = radius_lookup[phant_id]
        z, dr = trust_bin_at_point(
            row["peak_x_mm"], row["peak_y_mm"], row["cf_at_peak"],
            breast_radius_mm, baselines[phant_id])
        artifact_zs.append(z); dist_ratios.append(dr)

    cand_df = cand_df.copy()
    cand_df["artifact_z"] = artifact_zs
    cand_df["dist_ratio_calc"] = dist_ratios
    return cand_df


def print_trust_diagnostics(cand_df):
    """artifact_z-only diagnostics (dist_ratio component already
    diagnosed and removed — see module docstring)."""
    print("\n" + "=" * 70)
    print("ARTIFACT_Z DIAGNOSTICS")
    print("=" * 70)
    print("\n  artifact_z by label:")
    for label, group in cand_df.groupby("label"):
        print(f"    label={label} (n={len(group)}): "
              f"mean={group['artifact_z'].mean():.2f}  median={group['artifact_z'].median():.2f}  "
              f"p10={group['artifact_z'].quantile(.1):.2f}  p90={group['artifact_z'].quantile(.9):.2f}")
    print(f"\n  Current ARTIFACT_Z_ARTIFACT_RISK={ARTIFACT_Z_ARTIFACT_RISK}  "
          f"(label=1's p90 — candidates above this look more like label=0's population)")
    print(f"  Fraction of label=1 candidates ABOVE this cutoff (false positives we'd wrongly flag): "
          f"{(cand_df.loc[cand_df['label']==1,'artifact_z'] > ARTIFACT_Z_ARTIFACT_RISK).mean():.1%}")
    print(f"  Fraction of label=0 candidates ABOVE this cutoff (true positives for the flag): "
          f"{(cand_df.loc[cand_df['label']==0,'artifact_z'] > ARTIFACT_Z_ARTIFACT_RISK).mean():.1%}")
    print("=" * 70 + "\n")


def compute_peak_dominance(cand_df):
    """Per scan: local_scr_db gap between the top-ranked candidate (#0,
    brightest peak) and the runner-up (#1). Approximates how much the
    reconstruction 'commits' to one location vs. leaving several
    similar-strength peaks competing. Uses local_scr_db as a proxy for
    peak magnitude (raw peak intensity isn't stored per candidate — see
    module docstring). Requires >=2 candidates for a scan; returns NaN
    otherwise (shouldn't happen with k=10, but guarded anyway)."""
    rows = []
    for scan_idx, group in cand_df.groupby("scan_idx"):
        group = group.sort_index()  # restore original brightest-first order
        if len(group) < 2:
            dominance_db = np.nan
        else:
            dominance_db = float(group.iloc[0]["local_scr_db"] - group.iloc[1]["local_scr_db"])
        rows.append(dict(scan_idx=scan_idx, dominance_db=dominance_db))
    return pd.DataFrame(rows)


def print_dominance_diagnostics(scan_df):
    """Run BEFORE wiring dominance into the trust map. Checks whether
    dominance_db actually separates R=1 (recall+argmax-LE success) from
    R=0 scans, and shows real percentiles to set thresholds from — same
    discipline as the artifact_z diagnostic that caught the direction bug."""
    print("\n" + "=" * 70)
    print("PEAK DOMINANCE DIAGNOSTICS (standalone — not yet in the trust map)")
    print("=" * 70)
    valid = scan_df.dropna(subset=["dominance_db"])
    print(f"\n  dominance_db distribution (n={len(valid)} scans):")
    print(f"    {valid['dominance_db'].describe(percentiles=[.1, .25, .5, .75, .9])}")

    print("\n  dominance_db by R (does it separate reliable from unreliable scans?):")
    for r_val, group in valid.groupby("R"):
        print(f"    R={r_val} (n={len(group)}): "
              f"mean={group['dominance_db'].mean():.2f}dB  median={group['dominance_db'].median():.2f}dB  "
              f"p10={group['dominance_db'].quantile(.1):.2f}  p90={group['dominance_db'].quantile(.9):.2f}")

    if valid["R"].nunique() == 2:
        auc = roc_auc_score(valid["R"], valid["dominance_db"])
        print(f"\n  AUC(dominance_db -> R) = {auc:.3f}"
              f"{'  <-- WRONG DIRECTION, would need flipping' if auc < 0.5 else ''}")
    print("=" * 70 + "\n")


def print_argmax_artifact_z_diagnostics(scan_df):
    """artifact_z recalibration at the SCAN level (argmax candidates only),
    stratified by R — NOT the per-candidate/all-ranks population used
    before, which is a different, brightness-biased subpopulation (see
    chat discussion: rank-#0 candidates are systematically higher
    artifact_z than the general pool just by virtue of being the peak)."""
    print("\n" + "=" * 70)
    print("ARGMAX-LEVEL artifact_z DIAGNOSTICS (for scan-level threshold)")
    print("=" * 70)
    valid = scan_df.dropna(subset=["argmax_artifact_z"])
    print(f"\n  argmax_artifact_z by R (n={len(valid)} scans):")
    for r_val, group in valid.groupby("R"):
        print(f"    R={r_val} (n={len(group)}): "
              f"mean={group['argmax_artifact_z'].mean():.2f}  "
              f"median={group['argmax_artifact_z'].median():.2f}  "
              f"p10={group['argmax_artifact_z'].quantile(.1):.2f}  "
              f"p90={group['argmax_artifact_z'].quantile(.9):.2f}")
    print("=" * 70 + "\n")


def compute_scan_level_outcomes(cand_df):
    """R defined from OUTCOME (recall + argmax LE), independent of any
    trust signal — unchanged from before."""
    rows = []
    for scan_idx, group in cand_df.groupby("scan_idx"):
        group = group.sort_index()
        argmax_row = group.iloc[0]
        recall_hit = bool(group["label"].max() == 1)
        argmax_le = float(argmax_row["le_to_gt_mm"])
        R = int(recall_hit and argmax_le <= DETECTION_THRESHOLD_MM)

        rows.append(dict(
            scan_idx=scan_idx, phant_id=argmax_row["phant_id"], R=R,
            argmax_le_mm=argmax_le, recall_hit=recall_hit,
            argmax_artifact_z=argmax_row["artifact_z"],
            top10_max_artifact_z=float(group["artifact_z"].max()),
        ))
    return pd.DataFrame(rows)


def assign_scan_trust(scan_df, dominance_high_db, dominance_low_db):
    """Combines artifact_z (flipped direction) and dominance_db into a
    scan-level trust_bin. Thresholds are passed in explicitly rather than
    hardcoded, since dominance thresholds are only known after inspecting
    print_dominance_diagnostics() output for THIS run."""
    def _bin(row):
        if row["argmax_artifact_z"] > ARTIFACT_Z_ARTIFACT_RISK:
            return "LOW"
        if pd.isna(row["dominance_db"]):
            return "MEDIUM"
        if row["dominance_db"] >= dominance_high_db:
            return "HIGH"
        if row["dominance_db"] <= dominance_low_db:
            return "LOW"
        return "MEDIUM"

    scan_df = scan_df.copy()
    scan_df["trust_bin"] = scan_df.apply(_bin, axis=1)
    return scan_df


def experiment_a_auc(scan_df):
    valid = scan_df.dropna(subset=["argmax_artifact_z"])
    if valid["R"].nunique() < 2:
        print("  Not enough R variation for AUC — skipping.")
        return
    # flipped: higher artifact_z now predicted to mean LOWER reliability
    auc = roc_auc_score(valid["R"], -valid["argmax_artifact_z"])
    print(f"  AUC(-argmax_artifact_z -> R) = {auc:.3f}")


def experiment_b_stratified(scan_df):
    print("\n  Mean argmax LE by trust bin:")
    for bin_name, group in scan_df.groupby("trust_bin", dropna=False):
        print(f"    {bin_name}: n={len(group)}  mean LE={group['argmax_le_mm'].mean():.2f}mm  "
              f"R-rate={group['R'].mean():.1%}")


def bootstrap_ci(scan_df, n_boot=2000, seed=42):
    """
    Bootstrap CI on two things that matter for the abstract claim:
      1. HIGH vs LOW mean-LE gap
      2. HIGH vs pooled-baseline R-rate gap
    Resamples SCANS with replacement, independently within each group being
    compared (standard two-sample bootstrap) — not resampling candidates or
    mixing across bins, since the claim is specifically about the gap
    between bins as currently assigned.

    Reports the 95% percentile CI on each gap, plus the fraction of
    bootstrap replicates where the gap goes the "wrong" way (a rough,
    non-parametric read on how solid the direction is at this sample size).
    """
    rng = np.random.default_rng(seed)

    high = scan_df.loc[scan_df["trust_bin"] == "HIGH"]
    low = scan_df.loc[scan_df["trust_bin"] == "LOW"]
    all_scans = scan_df

    print("\n" + "=" * 70)
    print(f"BOOTSTRAP CI (n_boot={n_boot}, resampling scans with replacement)")
    print("=" * 70)
    print(f"  HIGH: n={len(high)}   LOW: n={len(low)}   ALL: n={len(all_scans)}")

    if len(high) < 5 or len(low) < 5:
        print("  WARNING: fewer than 5 scans in HIGH or LOW — bootstrap CI will be "
              "very wide and not very informative. Report with that caveat if used.")

    # --- LE gap: mean(LOW LE) - mean(HIGH LE), positive = HIGH does better ---
    le_gaps = np.empty(n_boot)
    for b in range(n_boot):
        h_sample = rng.choice(high["argmax_le_mm"].values, size=len(high), replace=True)
        l_sample = rng.choice(low["argmax_le_mm"].values, size=len(low), replace=True)
        le_gaps[b] = l_sample.mean() - h_sample.mean()

    le_gap_point = low["argmax_le_mm"].mean() - high["argmax_le_mm"].mean()
    le_ci_lo, le_ci_hi = np.percentile(le_gaps, [2.5, 97.5])
    le_wrong_direction = float((le_gaps <= 0).mean())

    print(f"\n  LE gap (LOW mean - HIGH mean, positive = HIGH localizes better):")
    print(f"    point estimate: {le_gap_point:+.2f}mm")
    print(f"    95% CI: [{le_ci_lo:+.2f}, {le_ci_hi:+.2f}]mm")
    print(f"    fraction of bootstrap replicates where HIGH does NOT do better: "
          f"{le_wrong_direction:.1%}")

    # --- R-rate gap: HIGH R-rate - pooled-baseline R-rate ---
    baseline_rate = all_scans["R"].mean()
    r_gaps = np.empty(n_boot)
    for b in range(n_boot):
        h_sample = rng.choice(high["R"].values, size=len(high), replace=True)
        r_gaps[b] = h_sample.mean() - baseline_rate

    r_gap_point = high["R"].mean() - baseline_rate
    r_ci_lo, r_ci_hi = np.percentile(r_gaps, [2.5, 97.5])
    r_wrong_direction = float((r_gaps <= 0).mean())

    print(f"\n  R-rate gap (HIGH R-rate - pooled baseline {baseline_rate:.1%}):")
    print(f"    point estimate: {r_gap_point:+.1%}")
    print(f"    95% CI: [{r_ci_lo:+.1%}, {r_ci_hi:+.1%}]")
    print(f"    fraction of bootstrap replicates where HIGH does NOT beat baseline: "
          f"{r_wrong_direction:.1%}")

    print("\n  Reading this: if either CI comfortably excludes 0 AND the "
          "'wrong direction' fraction is low (<5%), that gap is solid enough to "
          "state as a real finding. If the CI straddles 0, or wrong-direction is "
          "high, report the point estimate with an explicit 'not yet statistically "
          "robust at this sample size' caveat rather than dropping the finding — "
          "the direction is still informative, just not provably reliable yet.")
    print("=" * 70 + "\n")

    return dict(le_gap_point=le_gap_point, le_ci=(le_ci_lo, le_ci_hi),
                r_gap_point=r_gap_point, r_ci=(r_ci_lo, r_ci_hi))


def train_reliability_model(scan_df):
    feature_cols = ["argmax_artifact_z", "dominance_db"]
    df = scan_df.dropna(subset=feature_cols + ["R"])
    if df["phant_id"].nunique() < 3:
        print("  Not enough distinct phantoms for GroupKFold — skipping.")
        return

    X = df[feature_cols].values
    y = df["R"].values
    groups = df["phant_id"].values
    n_splits = min(5, df["phant_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(df))
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    if len(np.unique(y)) == 2:
        print(f"  Reliability model out-of-fold AUC: {roc_auc_score(y, oof):.3f}")


def compute_coverage_curve(scan_df, score_col="dominance_db", n_thresholds=11):
    """
    Selective-prediction curve: sweep score_col as a confidence threshold,
    at each level report coverage (fraction of scans accepted), mean LE
    among accepted scans, and R-rate among accepted scans. Uses
    dominance_db by default since it was the stronger standalone
    continuous predictor (AUC 0.634 vs artifact_z's 0.622) — see chat
    discussion on why the combined logistic model didn't beat it alone.

    Thresholds are evenly spaced across score_col's own observed
    percentile range (not fixed values), so the curve is data-driven
    rather than guessed, same principle as the trust-bin recalibration.
    """
    valid = scan_df.dropna(subset=[score_col, "argmax_le_mm", "R"])
    thresholds = np.percentile(valid[score_col], np.linspace(0, 90, n_thresholds))

    rows = []
    for t in thresholds:
        accepted = valid[valid[score_col] >= t]
        if len(accepted) == 0:
            continue
        rows.append(dict(
            threshold=t,
            coverage=len(accepted) / len(valid),
            n_accepted=len(accepted),
            mean_le_mm=accepted["argmax_le_mm"].mean(),
            r_rate=accepted["R"].mean(),
        ))
    return pd.DataFrame(rows)


def print_coverage_curve(curve_df):
    print("\n" + "=" * 70)
    print("SELECTIVE-PREDICTION COVERAGE CURVE")
    print("=" * 70)
    print(f"{'coverage':>10}  {'n':>5}  {'mean_le_mm':>12}  {'r_rate':>8}  {'threshold':>10}")
    for _, row in curve_df.iterrows():
        print(f"{row['coverage']:>9.1%}  {row['n_accepted']:>5}  "
              f"{row['mean_le_mm']:>12.2f}  {row['r_rate']:>7.1%}  {row['threshold']:>10.2f}")
    print("\n  Reading this: as coverage drops (moving down the table), mean LE should\n"
          "  fall and R-rate should rise if the confidence score is doing real work.\n"
          "  A flat or non-monotonic curve means the score isn't actually selective.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="svm_candidates.csv")
    parser.add_argument("--baselines", type=str, default="phantom_baselines.pkl")
    parser.add_argument("--diagnose-only", action="store_true",
                         help="Print artifact_z + dominance diagnostics only, stop "
                              "before assigning trust bins or running experiments.")
    parser.add_argument("--dominance-high-db", type=float, default=None,
                         help="Required once --diagnose-only is dropped — read off "
                              "the printed dominance_db percentiles by R first.")
    parser.add_argument("--dominance-low-db", type=float, default=None)
    args = parser.parse_args()

    print(f"Loading {args.candidates} and {args.baselines}...")
    cand_df = pd.read_csv(args.candidates)
    with open(args.baselines, "rb") as f:
        baselines = pickle.load(f)

    phantom_table = pd.DataFrame(
        [{"phant_id": pid, "breast_radius_mm": b["breast_radius_mm"]}
         for pid, b in baselines.items()])

    print("Tagging candidates with artifact_z...")
    cand_df = tag_candidates_with_trust(cand_df, baselines, phantom_table)
    print_trust_diagnostics(cand_df)

    print("Computing non-circular scan-level outcome R...")
    scan_df = compute_scan_level_outcomes(cand_df)
    print(f"  {len(scan_df)} scans, R=1 rate: {scan_df['R'].mean():.1%}")

    print_argmax_artifact_z_diagnostics(scan_df)

    print("Computing peak dominance (top-1 vs top-2 local_scr_db gap)...")
    dom_df = compute_peak_dominance(cand_df)
    scan_df = scan_df.merge(dom_df, on="scan_idx", how="left")
    print_dominance_diagnostics(scan_df)

    if args.diagnose_only:
        print("--diagnose-only set — stopping here. Pick dominance thresholds from "
              "the printout above, then rerun with --dominance-high-db/--dominance-low-db.")
        return

    if args.dominance_high_db is None or args.dominance_low_db is None:
        print("ERROR: --dominance-high-db and --dominance-low-db are required once "
              "--diagnose-only is dropped. Run with --diagnose-only first.")
        return

    scan_df = assign_scan_trust(scan_df, args.dominance_high_db, args.dominance_low_db)

    print("\nExperiment A (AUC, flipped direction):")
    experiment_a_auc(scan_df)

    print("\nExperiment B (stratified LE by trust_bin):")
    experiment_b_stratified(scan_df)

    print("\nBootstrap CI on HIGH-vs-LOW / HIGH-vs-baseline gaps:")
    bootstrap_ci(scan_df)

    print("\nReliability model (logistic regression, GroupKFold by phant_id):")
    train_reliability_model(scan_df)

    print("\nSelective-prediction coverage curve (dominance_db):")
    curve_df = compute_coverage_curve(scan_df, score_col="dominance_db")
    print_coverage_curve(curve_df)
    curve_df.to_csv("coverage_curve.csv", index=False)
    print("Saved coverage_curve.csv for plotting later.")


if __name__ == "__main__":
    main()