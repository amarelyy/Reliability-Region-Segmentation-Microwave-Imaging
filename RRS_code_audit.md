# Code audit: `fix/preprocessing` branch (RRS-MBI)

**Scope.** This audit covers only the code that produces numbers or figures in the report: `src/`, `rrs_mbi/`, `unet/`, `ablation_runner.py`, and the figure scripts in `metrics_result/` and `metrics_image/` that feed report figures. The SVM, `reliability.py` (the older candidate-based layer) and debug/temp files are excluded.

**What was run.** The dataset files (`data/*.pickle`, `phantom_database.csv`) are not in the repo, and the sandbox cannot download them. Every bug below was therefore confirmed in one of two ways:

- by running the repo's own functions on synthetic inputs, or
- by reading the code directly.

The real metrics (31.84 mm, R distribution, Table 9, and so on) can be re-run once the three data files are attached.

Items are ordered by how much they affect the paper.

---

## A. Problems that change the main results

### A1. Antenna start angle is 0° instead of −130° (`src/physics.py`, `get_antenna_geometry`)
- **The bug.** UM-BMID Gen-2 starts Port A at **−130°** and Port B at −70° (`UM-BMID/docs/scan_geometry.md`). The code starts the array at **0°**, so every reconstructed image is rotated about 130° relative to the ground-truth tumour coordinates.
- **Measured effect.** On synthetic Gen-2 data, this bug alone raises the mean LE of a clean point target from ≈7 mm to ≈25 mm.

### A2. Rodriguez-Herrera radius correction applied in the wrong units (`get_corrected_ant_radius_m`)
- **The bug.** The UM-BMID formula `0.97·(r − 0.106) + 0.148` is in **metres**. The code feeds it millimetres and divides by 1000 afterwards.
- **Result.** For `ant_rad` = 21.5 cm the code places the antennas at **0.2086 m**; the correct value is **0.2537 m**, so the array is 45 mm too small.
- **Measured effect.** This alone gives a mean LE of ≈36 mm on synthetic data.
- **Side effect on the paper.** The direct Port A→B path at the wrong radius is 0.70 ns, which is probably why the 0.7 ns gate "matched". At the correct radius the direct path arrives at about 0.85 ns, while the skin echo arrives at about 1.4 ns.

### A3. Snellius receive leg computed with a scalar instead of per pixel (`snellius_bistatic_delay_precise`)
- **The bug.** `c2 = np.sum(grid_pos**2) - R**2` sums over **all pixels**, so the receive-leg air/tissue split is wrong.
- **Size of the error.** Compared with an exact two-medium model, delays inside the breast are off by **70–600 ps** (mean −270 ps ≈ 13 mm of range in tissue), and the error depends on pixel position, so it distorts the focus.
- **Second problem.** Both legs also mis-handle pixels *outside* the breast circle when the ray passes through the breast.
- **Measured effect.** This alone gives a mean LE of ≈36 mm on synthetic data.

**Combined effect (synthetic, 16 tumours, homogeneous breast R = 45 mm, with a skin reflection):**

| Pipeline | Mean LE | Detection @15 mm |
|---|---|---|
| Repo as-is | **38.2 mm** | 12.5 % |
| Radius + angle fixed | 10.6 mm | 75 % |
| All three fixed | ≈0 mm | 100 % |

The synthetic test is idealised: it uses the same straight-ray model for simulation and reconstruction, so the "all fixed" row is a best case. It still shows that **the repo pipeline fails even on ideal data**. The ~31 mm error and 12 % detection in the paper therefore cannot be attributed to physics until these bugs are fixed and the dataset is re-run.

**Consequence for the report.** The central claim that "the limit is physical, not algorithmic" (Sections 3.5, 4.1, 4.2) is not supported by this code. The baseline, the CC/G values, Table 9, the R distribution, and every number derived from them have to be regenerated.

### A4. Figure 6 ("Signal Death Curve") is generated from mock data (`metrics_result/signal_death.py`)
- **What the script does.** It raises `FileNotFoundError("Mocking data for demonstration.")` on purpose, then plots 150 random points from the formula `scr = 10 − (r−15)/35·34 + noise`.
- **What that means.** The slope **−0.96 dB/mm** and the **≈26 mm** zero crossing are properties of that formula, not of the dataset.
- **Required action.** Remove the figure and every claim based on it (Section 4.1.4, the −0.96 dB/mm in the abstract and intro, the "r ≈ −0.76", and the 26 mm crossing), or replace it with SCR computed from real reconstructions.

### A5. Figure 5 ("Spatial CF baseline, adipose") is not a beamformed map (`metrics_result/cf_baseline_heatmap.py`)
- **How the map is built.**
  - It is the interference pattern of **8 virtual antennas at r = 50 mm at a single 4 GHz frequency**: `Σ exp(jk·d)` over the antennas.
  - The adipose data only contribute one scalar, which is then cancelled by normalising the peak to 1.
  - The 50 mm array circle, the 48 mm phantom circle and the 15 mm "artifact zone" are hard-coded circles.
- **Why the centre is red.** Any symmetric ring of sources peaks at its centre by construction, so the map cannot "prove" anything about the data.
- **The underlying misreading.** The script treats the 72 S21 columns as "flattened Tx-Rx channels of 8 antennas". They are actually 72 rotation positions.
- **Required action.** The text in 3.5/4.1.3 saying it was "virtually beamformed from real tumour-free adipose S21" must be removed or the figure rebuilt. To rebuild it, run `reconstruct_scan` on the adipose-only reference scans (`adi_ref_id`).

### A6. U-Net "collapse" is an evaluation bug (`unet/evaluate_unet.py`)
- **Double sigmoid.**
  - `UNet2Ch.forward` already ends in `Sigmoid`, and the evaluator applies `torch.sigmoid` **again**.
  - sigmoid(sigmoid(x)) is always > 0.5, so after `> 0.5` **every pixel is predicted as tumour, whatever the model learned**.
  - That gives recall = 1.000, precision = tumour-area fraction ≈ 0.012, and Dice ≈ 0.024. These are exactly the numbers in Table 9 of the report.
- **Other problems in the evaluation.**
  - Inputs are not resized to 128×128 and are normalised jointly, not per channel, so they differ from training.
  - The 2-level U-Net only accepts grids divisible by 4. Of the fallback radii, only phantoms A3/A16 (grid 172) qualify; the rest throw an exception and are silently skipped.
  - Evaluation runs on **all** scans, including the training scans.
- **Training problems.**
  - `train.py` splits 80/20 **by row order**, not by phantom, so the same phantoms appear in train and validation (leakage).
  - The dataset class calls `reconstruct_scan(use_bandpass=True)`, which raises `TypeError` in this branch. The checkpoint was therefore trained with an older pipeline.
- **Consequence.** Section 4.2.2, Table 9 and the "U-Net Dice 0.024" in the conclusion are evidence of the bug, not of physical inseparability.

### A7. Reported `R_sat` = 20 mm but the evaluator uses 5 mm (`rrs_mbi/evaluator.py`)
`run_evaluation()` calls `evaluate_scan_reliability(..., saturation_radius_mm=5.0)`. The report (3.7.4, 4.3.1) says 5 mm was rejected and replaced by 20 mm. Either the reported R values were computed with 5 mm, or the final code was never committed. Please confirm which.

---

## B. Data leakage and validity

1. **Tuning on the evaluation set.** `R_sat`, the CF range [0.1, 0.8] and the D/5 clip were chosen by looking at the same 504 scans used to report Table 9, so the selective-classification result is in-sample. A phantom-grouped split or a held-out set of phantoms is needed.
2. **U-Net split not grouped by phantom** (A6).
3. **Oracle priors in reconstruction.** Each scan uses the *true* phantom composition (`fib_volume`/`shell_volume` → ε_eff) and the true shell radius from `phantom_database.csv`. This is acceptable for a phantom study, but the report should say so: R is "ground-truth-free" only with respect to the tumour, not the phantom.
4. **Two different points per scan.** `G` and `D` use the **global argmax** of the image, while LE and `cf_at_peak` use the **blob centroid**, so they can describe different points. The report describes them as one "predicted peak".
5. **Unseeded bootstrap.** The per-row CIs in Table 9 (`bootstrap_ci`) use `np.random.choice` without a seed, so they are not reproducible.
6. **"38.3 % of tumours near the boundary"** (`metrics_paper.py`) assumes **one breast radius of 45 mm for every phantom**. The code's own table gives antenna-plane radii from 22.9 to 57.0 mm, so the per-phantom radius must be used.

---

## C. Pipeline mismatches with the report

| Item | Report | Code (this branch) |
|---|---|---|
| Early-time gate | 0.7 ns | default **0.5 ns** (`pipeline.py`, `beamforming.py`); 0.70 only when the evaluator passes it |
| Depth-gain compensation | "rejected" | `use_depth_gain=True` by default and active in the main evaluation, but inert (see below) |
| Depth gain "numerically inert" | explained physically | **unit bug**: `np.median(delay_grid)` is in *seconds* and is used as metres, giving a gain of 1.000000000001 |
| Ablation table (DAS/DMAS ± CF, GIBR) | Table x | `ablation_runner.py` passes `use_bandpass=` → `TypeError` for **every scan**; no GIBR code exists on either branch. The table cannot be reproduced from this repo |
| SCME (Table x) | "peak at 51.0 mm" | `scme_map += scalar` adds the same constant to every pixel, so the map is spatially flat and its "peak" is meaningless; it also uses 8 virtual antennas at r = 50 mm |
| Single-channel envelope (71 mm, 0/1301) | Table x | no code in the repo |
| Risk-matrix figure | uncaptioned | hard-coded illustrative array, not computed from data |
| `ant_rad` default | — | 21.5 cm used whenever metadata lacks `ant_rad`; check that every scan has it |
| Adipose shell centre (`adi_x`, `adi_y`) | — | ignored (`shell_center=(0,0)`); check whether Gen-2 metadata has non-zero values |

## D. Memory and runtime

- `_DELAY_CACHE` in `pipeline.py` is an unbounded global dict, holding about 17 MB per phantom grid (≈1.1 GB for 66 phantoms) in every process. It is now capped at 8 entries.
- The U-Net dataset re-runs the full reconstruction for every sample in every epoch (20 × 403 reconstructions), with no caching. Caching the reconstructed images to disk once would be faster.
- With `joblib n_jobs > 1`, each worker gets its own copy of the ~1.2 GB S21 array plus its own delay cache.

---

## Fixes applied (see `fixes.patch`, apply with `git apply fixes.patch`)

1. `physics.py`: start angle −130°, radius correction in metres, and an exact clipped-chord two-medium delay for both legs. Verified against an independent implementation: max error 0.000 ps over the full grid.
2. `pipeline.py` / `beamforming.py`: default gate 0.70 ns, depth gain off by default, bounded delay cache.
3. `rrs_mbi/evaluator.py`: `saturation_radius_mm=20.0` (matching the report) and a seeded bootstrap.
4. `ablation_runner.py`, `unet/dataset/microwave_dataset.py`: `use_bandpass` → `bandpass_mode`, so they run again.
5. `unet/training/train.py`: train/val split grouped by phantom; validation phantom IDs are saved.
6. `unet/evaluate_unet.py`: no double sigmoid; new `evaluate_unet_heldout()` uses the training preprocessing on held-out phantoms only. The U-Net must be **retrained** after the geometry fix.

## Next step (needs the data)
Run `python -m rrs_mbi.evaluator` with the patched code on the 504 scans and compare against the report: LE, detection rate, CC, R distribution, Table 9, permutation p and the confusion matrix. Then regenerate Figures 5–9 from real outputs.
