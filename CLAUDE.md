# Project context: RRS-MBI audit

## Reading order (do this at the start of every session)
1. **`RESEARCH_CONTEXT.md`**: the research itself.
   - background, aims and the core argument
   - dataset details and pipeline
   - the RRS-MBI score and the evaluation protocol
   - key findings as reported
   - which script produces which report element
2. **This file (`CLAUDE.md`)**: rules, the bugs in short, the values to reproduce, and the step-by-step plan.
3. **`RRS_code_audit.md`**: full details of every bug and the evidence for it.
4. **`fixes.patch`** and **`sim.py`**: the proposed fixes and the synthetic test.
5. **`CHAT_CONTEXT.md`** (optional background): examiner-review findings carried over from the claude.ai chats.

Keep these files in sync. When a step in the plan below is finished, add a short dated note under **Progress log** at the bottom of this file, with results saved in `results_audit/`. That way the next session can continue where this one stopped.

## What this repo is
- The code behind rr's internship report (Laporan Kerja Praktik) on **microwave breast imaging**:
  - dataset: UM-BMID Gen-2 (bistatic S21)
  - reconstruction: DAS + Coherence Factor
  - proposed per-scan reliability score: **RRS-MBI**, R = D_norm · CF_norm · (1 − CC) · G
- Working branch: `fix/preprocessing`.
- An audit found bugs that change the report's numbers. The job now: **reproduce the report numbers, apply the fixes, re-run, compare, and rebuild the figures from real data.**

## Files you must read
- `RRS_code_audit.md`: the full list of problems, ordered by impact on the paper.
- `fixes.patch`: the proposed fixes. Apply with `git apply fixes.patch`.
- `sim.py`: a synthetic Gen-2 test. It simulates tumours with the documented geometry and runs the repo pipeline.

## Rules
- Use simple language. Explain each step before doing it. rr sometimes writes in Indonesian.
- Make **minimal, incremental** code changes. No full rewrites, no refactors.
- Ask rr before any code change that is not in `fixes.patch`.
- **Never modify or delete anything in `data/`.**
- Ignore: the SVM code (`train_svm.py`, `baseline_argmax_test.py`, `metrics_result/hasil_svm.py`, `svmvsargmax.py`), `reliability.py`, `debug_*.py`, `tempCodeRunnerFile.py`, `check_*.py`, and the old images.
- Save every result as CSV under `results_audit/` (subfolders `baseline/` and `fixed/`).
- Never present mock or synthetic data as real results. Label any synthetic result as synthetic.

## Data (in `data/`, not in git)
- `fd_data_gen_two_s21.pickle`: S21, about (n_scans, 1001 freqs, 72 positions), complex
- `metadata_gen_two.pickle`
- `phantom_database.csv`

## UM-BMID Gen-2 geometry (from `UM-BMID/docs/scan_geometry.md`)
- 72 positions, 5° steps, **clockwise**, spanning 0–355°.
- Port A starts at **−130°**. Port B starts at −70° (+60° offset).
- `ant_rad` (cm) is measured from the antenna SMA connector.
- Phase-centre correction (Rodriguez-Herrera), **in metres**: `r = 0.97*(r_m − 0.106) + 0.148`. For 21.5 cm this gives 0.2537 m.
- 1–8 GHz, 1001 points. ICZT to 0–6 ns, 1024 samples.
- Tumour coordinates `tum_x`/`tum_y` are in cm, relative to the chamber centre.

## Key bugs (details in `RRS_code_audit.md`)
1. **Start angle:** `physics.get_antenna_geometry` starts at 0° instead of −130°, so images are rotated about 130°.
2. **Radius units:** `get_corrected_ant_radius_m` applies the metre formula to mm, giving 0.2086 m instead of 0.2537 m.
3. **Snellius delay:** `snellius_bistatic_delay_precise` uses `np.sum(grid_pos**2)` (all pixels) in the receive leg, giving 70–600 ps delay errors.
   - Bugs 1–3 together: the pipeline fails even on ideal synthetic data (mean LE 38 mm). Fixed, it localizes perfectly.
4. **Mock figure:** `metrics_result/signal_death.py` (report Figure 6) plots mock random data.
5. **Synthetic figure:** `metrics_result/cf_baseline_heatmap.py` (report Figure 5) is a synthetic 8-antenna pattern, not a beamformed map.
6. **U-Net double sigmoid:** `unet/evaluate_unet.py` applies sigmoid twice, so every pixel is predicted as tumour. This alone produces Dice 0.024, precision 0.012 and recall 1.0. Training is also split by row order, not by phantom (leakage).
7. **R_sat mismatch:** `rrs_mbi/evaluator.py` uses `saturation_radius_mm=5.0`; the report says 20 mm.
8. **Broken calls:** `ablation_runner.py` and the U-Net dataset pass `use_bandpass=`, which raises `TypeError`.
9. **Defaults:** the default gate is 0.5 ns (the report says 0.7). Depth gain is on by default and has no effect because seconds are used as metres.

## Report values to reproduce (baseline, 504 tumour-bearing scans)
| Metric | Report value |
|---|---|
| Mean / median LE | 31.84 / 31.53 mm (SD 13.28) |
| Detection @15 mm | 12.3 % (62/504) |
| Mean predicted / ground-truth distance from centre | 10.87 / 30.64 mm |
| Phantom-level mean error | 32.42 mm (σ 7.45); 31/66 phantoms with ≥1 detection |
| Mean CC | 0.4043 (median 0.3209, SD 0.2598) |
| Mean CF at peak | 0.3746 |
| R | mean 0.162, range 0.002–0.41; 0 scans ≥0.70, 59 in 0.30–0.70, 445 <0.30 |
| Top-10 % retention (n = 50) | LE 29.49 mm, threshold 0.317, detection 16.0 % |
| Top-20 % retention | LE 30.97 mm |
| Permutation p (top-50) | 0.098 |
| Bootstrap gap CI | [−1.50, +6.03] mm |
| Spearman / Pearson (scan level) | −0.004 / −0.018 |
| Spearman / Pearson (phantom level) | −0.022 / −0.076 |
| U-Net | Dice 0.024, precision 0.012, recall 1.000 |
| Ablation (DAS+CF) | mean LE 30.7, median 31.2, SCR 16.4 dB |

## Plan
1. **Baseline.**
   - Create branch `audit/baseline` and leave the code unchanged.
   - Run `python -m rrs_mbi.evaluator` once with `saturation_radius_mm=5.0` and once with `20.0`.
   - Say which run reproduces the table above.
2. **Data checks.** Print:
   - how many scans are missing `ant_rad` and its min/max
   - `adi_x`/`adi_y` (the shell centre is currently assumed to be 0,0)
   - whether the CSV shell radii match `ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM`
3. **Fixed run.**
   - Create branch `audit/fixed` and apply `fixes.patch`.
   - Run `python sim.py` first; it should localize the synthetic tumours.
   - Re-run the evaluator.
   - Build a side-by-side table (report | baseline | fixed), including:
     - the confusion matrix for the 33rd/67th percentile tiers and for fixed 0.30/0.70
     - a chi-square test on the percentile tiers
4. **Rebuild the figures from real data.**
   - Figure 6: SCR against tumour distance, measured at the predicted blob AND in a disc around the ground truth. Report the slope and Pearson r.
   - Figure 5: real CF map from the adipose-only reference scans (`adi_ref_id`).
   - "% of tumours within 10 mm of the boundary" using each phantom's own radius.
5. **U-Net.**
   - Cache the reconstructions to disk first.
   - Retrain with the phantom-grouped split.
   - Report `evaluate_unet_heldout()` on held-out phantoms only.
6. **Ablation.**
   - Run `ablation_runner.py` on the fixed branch.
   - There is no GIBR (bent-ray) code in the repo. Do not recreate it.

## Expected final output
- A table: report value | baseline rerun | fixed rerun.
- A short list of which report claims still hold, and which change or must be removed. In particular, decide whether "the limit is physical, not algorithmic" still holds after the fixes.

## Progress log
<!-- Add one line per finished step: date — step — result file — 1-line takeaway -->
- 2026-10-28: audit written (`RRS_code_audit.md`), fixes drafted (`fixes.patch`), synthetic test (`sim.py`). The real-data reruns have not been started.
- 2026-10-28 — Step 1 baseline (branch `audit/baseline`, code unchanged; driver `results_audit/run_evaluator.py` overrides R_sat from outside) — `results_audit/baseline/comparison_report_vs_baseline.csv`, `sat5/`, `sat20/`, logs, original report CSVs backed up in `report_original/` (they are gitignored) — **R_sat = 5.0 reproduces the report bit-for-bit** (metrics_raw.csv identical); R_sat = 20 does not (R mean 0.085, p = 0.43, top-10 % LE 31.5). So the report's R values used 5 mm, not the 20 mm stated in the text. Report's "R range 0.002–0.41" is wrong for sat5 (max is 0.639).
- 2026-10-28 — NOTE: rr asked to do steps 2–6 on `audit/baseline` itself (no `audit/fixed` branch). `fixes.patch` is applied to the working tree of `audit/baseline` (uncommitted). Baseline code for later comparisons was extracted with `git archive HEAD src` into the scratchpad.
- 2026-10-28 — Step 2 data checks — `results_audit/baseline/data_checks_log.txt`, `data_check_shell_radius.csv` — ant_rad present for all 504 (21/22/23 cm, not 21.5); adi_x/adi_y non-zero in 202/504 scans (up to 10 mm); CSV shell radii match the table for all 504; 64/504 tumour centres lie outside their own shell radius, 3/504 outside the image grid.
- 2026-10-28 — Step 3 fixed run — `results_audit/fixed/comparison_report_baseline_fixed.csv`, `confusion_matrices.csv`, `chi_square_tiers.csv`, `sim_results_SYNTHETIC_*.csv` — sim.py: fixed code localizes synthetic tumours perfectly (SYNTHETIC). Real data with fixes.patch: mean LE 49.9 mm, det 5.4 % (worse); cause = blob extraction picks the sharply focused shell ring (60 % of blobs outside R). R now correlates with LE (scan ρ −0.29, phantom ρ −0.53, perm p < 1e-4).
- 2026-10-28 — Sensitivity (not in plan, results_audit only) — `results_audit/sensitivity_summary.csv` — argmax inside the shell: fixed 31.9 mm, fixed + adipose-reference subtraction 31.6 mm, baseline 32.1 mm; predicted points stay ~9 mm from centre after the fixes, so the central bias is NOT caused by bugs 1–3. Orientation scan (`fixed/orientation_scan.csv`): no rotation/mirror helps.
- 2026-10-28 — Step 4 figures — `results_audit/fig6_scr_vs_distance_REAL.png`, `fig6_fit_stats.csv`, `fig5_cf_adipose_REAL.png`, `fig5_adipose_cf_per_reference.csv` — SCR at predicted blob has no distance trend (baseline +0.04, fixed −0.00 dB/mm); SCR in GT disc: baseline −0.32 dB/mm r −0.55 (0 dB at 32 mm, min −24.7 dB), fixed −0.14 dB/mm r −0.32 (0 dB at 20 mm). Adipose-only (141 refs): image peak at r/R < 0.25 in 62–65 %, CF within 10 mm ≈ 3× breast mean, in both pipelines. Tumours within 10 mm of own boundary: 35.9 % (report 38.3 % used fixed 45 mm).
- 2026-10-28 — Step 5 U-Net — `results_audit/fixed/unet/` (checkpoint overwritten; original backed up as `results_audit/baseline/unet_reliability_ORIGINAL.pth`) — held-out 14 phantoms / 103 scans: Dice 0.085, precision 0.081, recall 0.136 (all-ones Dice 0.045); mask-centroid LE 37.1 mm, det@15 19.4 %. Single seed.
- 2026-10-28 — Step 6 ablation — `results_audit/fixed/ablation/` — runs now (0 failures). Raw DAS 53.4, DAS+CF 49.9, DAS+CF+4–6 GHz 40.9, +depth gain 40.9 (identical: depth gain still inert, unit bug not fixed by patch), DMAS full stack 43.1 mm. Variants differ from the report's table; no GIBR.
- 2026-10-28 — Q1 tumours outside shell — `results_audit/q1_tumour_outside_radius.txt`, `shell_estimates.csv` — causes: (a) shells not centred at chamber origin: echo fit puts A1/A11 ~13–15 mm towards −y (not in adi_x/adi_y); with echo centre, out-of-shell tumours drop 64→31 (A11 18→2); (b) radius is the antenna-plane radius, many A1 tumours sit 15–35 mm above that plane (3D vs 2D). Echo-based absolute radius is biased ~15 % low (timing), so only the centre is used.
- 2026-10-28 — FINAL fixes applied (rr approved "terapin semua"; beyond fixes.patch): `src/shell_geometry.py` (echo-estimated shell centre), `src/pipeline.py` (ref_col="adi_ref_id" adipose-reference subtraction, shell-centred delay model, grid widened by offset, image/CF restricted to the breast disc, SCR clutter inside breast only), `src/metrics.py` (SCR region), `src/beamforming.py` (antenna Tukey window off: α 0.2→0), `src/signal_processing.py` (depth gain units; still off), `rrs_mbi/metrics.py` (D, G, CF at the SAME predicted point; CC/G about the shell centre), `rrs_mbi/evaluator.py` (trivial "predict centre" reference, phantom-cluster bootstrap with re-selection, chi-square, inference CSV; boundary_risk NaN column removed).
- 2026-10-28 — FINAL run (R_sat 20, params frozen at report values, no re-tuning) — `results_audit/final/` (`sat20/`, `comparison_report_baseline_patch_final.csv`, `pipeline_vs_trivial.txt`, `rrs_component_diagnostics.txt`, `LE_by_size_and_distance.csv`, `diag_prediction_vs_fib.csv`), figures `results_audit/fig5_cf_adipose_REAL.png`, `fig6_scr_vs_distance_REAL.png` (baseline vs final) — LE 30.87 mm, det 16.7 % (84/504); trivial centre guess 29.22 mm / 12.5 %; pipeline − trivial LE +1.66 mm (cluster CI +0.17…+3.06) → corrected DAS+CF does not beat "predict the centre". Predictions ~11 mm from centre at every true distance. RRS-MBI ranks INVERSELY: ρ(R, LE) +0.19 (CI +0.08…+0.31), top-20 % LE 4 mm worse; D_norm and 1−CC inverted, CF and G uninformative. fib_ref_id usable for only 1/504 scans, fib-insert clutter could not be tested.
- 2026-10-28 — Branch `fix/logic-refactor` (from audit/baseline working tree; origin/fix/preprocessing had no new commits). Removed tracked: `unet/`, ablation_runner.py, SVM files, reliability.py, debug_*/check_*/tempCodeRunnerFile.py (32 files, recoverable from git). Added `bandpass_mode=(lo, hi)` to pipeline, `rrs_mbi/candidate_metrics.py`, `rrs_mbi/validate_candidates.py`.
- 2026-10-28 — RRS-MBI v2 candidate test — `results_audit/logic_refactor/` (`auroc.csv`, `risk_coverage.csv`, `coverage_gaps.csv`, `comp_B_fold_coefficients.csv`, `analyze.log`) — AUROC for success (LE≤15): sub-aperture half-split distance 0.585 [0.509, 0.665], sub-band distance 0.602 [0.514, 0.682]; artifact_prior 0.534, peak_margin 0.492, CF 0.476, interleave 0.524 (≈0 for 90 % of scans) all CI∋0.5; old R 0.434 (inverted). Composite B (logistic, phantom-grouped 5-fold OOF) 0.609 [0.542, 0.671]; its LE gap at 20 % coverage +1.45 mm, CI∋0. Composite A (rank mean, no fitting) AUROC 0.559 CI∋0.5 but LE gap +5.1 mm [1.6, 7.0] at 20 % — accepted scans have more central tumours (trivial LE 25.9) and pipeline ≈ trivial inside them. 6 candidates tested → multiplicity: the two lower bounds (0.509/0.514) would not survive correction. Verdict: consistency metrics carry weak signal; not enough for a "fail-safe" claim.
- 2026-11-03 — U-Net re-run on the FINAL pipeline (branch `fix/logic-refactor`; `unet/` restored from `fix/preprocessing`, U-Net hunks of `fixes.patch` applied; driver `results_audit/unet_final.py`) — `results_audit/final/unet/` (`summary_by_seed.csv`, `summary_over_seeds.csv`, `heldout_per_scan_all_seeds.csv`, `training_log_all_seeds.csv`, `official_heldout_seed42.csv`, `run_log.txt`; cache `results_audit/final/cache/unet_inputs_final.npz`) — 504/504 scans cached, 0 skipped; 5 seeds, each a different phantom-grouped split (14 held-out phantoms each, fixed 20 epochs, no checkpoint selection). Held-out mean±SD over seeds: Dice 0.172±0.044, precision 0.178, recall 0.256, empty masks 21 % (seed 46: 60 %), mask-centroid LE 28.4±2.2 mm, det@15 32.8±8.7 %. Same scans: centre-disc Dice 0.034, all-ones Dice 0.032, "predict centre" LE 31.9 mm / det 9.2 %, final DAS+CF LE 32.4 mm / det 14.6 %. Seed 42 = official checkpoint; `evaluate_unet_heldout()` matches the cached eval (Dice 0.170). Replaces the earlier single-seed Step 5 (Dice 0.085, older pipeline). No paired CI yet; held-out sets overlap across seeds. No CNN code in repo; ablation_runner.py removed on this branch.
