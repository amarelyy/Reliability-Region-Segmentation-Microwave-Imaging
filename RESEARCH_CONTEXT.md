# Research context: RRS-MBI (Reliability-Region Segmentation for Microwave Breast Imaging)

> Read this file first, then `CLAUDE.md` (rules and task plan), then `RRS_code_audit.md` (the bugs).
> This file explains **what the research is, why it exists, what the report claims, and which code produces each claim**, so that any change to the code can be judged by its effect on the paper.

---

## 1. Who and what
- **Author:** rr, a 2nd-year Information Engineering student (DTETI, Universitas Gadjah Mada).
- **Output:** an internship report (*Laporan Kerja Praktik*) written during an internship at the BREATHE Laboratory, DTETI UGM (July–September 2026), supervised by Ir. Ridwan Wicaksono, S.T., M.Eng., Ph.D.
- **Topic:** radar-based microwave breast imaging (MBI) on a public phantom dataset, plus a new **per-scan reliability score** that tells the user when a reconstructed image should not be trusted.
- **Status:** the report is drafted. A code audit (see `RRS_code_audit.md`) found bugs that may change the main conclusion, so the numbers must be regenerated before the report is final.

## 2. Background (why this matters)
- Breast cancer is the most common cancer in women: about 2.3 million new cases and 670,000 deaths in 2022 (WHO).
- Mammography uses ionizing radiation and works poorly in dense breasts. MRI is expensive and rarely available in rural areas.
- **Microwave imaging** uses the dielectric contrast between healthy and malignant tissue (permittivity and conductivity; Gabriel 1996, Lazebnik 2007). Malignant tissue has higher permittivity.
- The hardware is low-power, non-ionizing, cheap and portable, which makes it attractive for point-of-care or rural use.
- **Radar-based MBI:**
  1. Antennas around the breast send ultra-wideband signals and record the echoes.
  2. A beamformer (delay-and-sum, DAS) time-aligns the echoes for every image pixel and adds them up.
  3. A tumour appears as a bright spot.
  4. The **Coherence Factor (CF)** weights each pixel by how much the channels agree in phase (Li & Li 2003; Klemm et al. 2008 for MBI).
- **The gap this work addresses:** MBI research focuses on making images more accurate *on average*. Clinical use also needs to know **whether one particular scan can be trusted**. No per-scan reliability estimate exists for radar-based MBI.

## 3. Research question and aims
**Question:** given a finished reconstruction, can we tell whether it is trustworthy **without knowing where the tumour really is**?

**Aims (Section 1.2 of the report):**
1. **Find the origin of failure.** Show that localization failures come from the circular acquisition geometry and the coherent summation, not from the choice of peak-selection algorithm.
2. **Build RRS-MBI**, a per-scan score R ∈ [0, 1] computed only from the reconstructed image and the known antenna coordinates.
3. **Validate it.** Scans with higher R should have lower localization error. Keeping only high-R scans (selective classification) should improve accuracy and detection.

**Framing:** shift the goal from "improve localization" to "know when localization failed". Scans with low R are flagged for **re-scanning**, which is safe because microwave exposure is negligible (Bhargava & Rattanadecho 2022). The intended use is **longitudinal monitoring** of patients who already have a diagnosis, not primary screening.

### The core argument (must stay intact in the report)
The report rests on a causal chain:
1. **Physical/geometric limits** of a circular bistatic array make reconstructions fail in two structured ways.
2. These failures arise **before** peak selection, so no post-hoc algorithm (SVM, U-Net, smarter peak picking) can recover the lost tumour information (data-processing inequality).
3. **Therefore** the right response is a **reliability triage** (RRS-MBI) that flags untrustworthy scans instead of trying to repair them.

> ⚠️ The audit shows that step 1 is currently **not proven**. Three geometry bugs make the pipeline fail even on ideal synthetic data, the U-Net "failure" is an evaluation bug, and Figures 5 and 6 are not built from real data. After the fixes, re-check whether the failures are still there. If they largely disappear, the triage framing must be reworded (e.g. "residual failures after correct reconstruction"). Do not delete the argument; report honestly what the corrected data show.

## 4. Dataset: UM-BMID Gen-2
- **Source:** University of Manitoba Breast Microwave Imaging Dataset. Cite Reimer, Krenkevich & Pistorius, *EuCAP 2020*, doi:10.23919/EuCAP48036.2020.9135659. Repo: github.com/UManitoba-BMS/UM-BMID.
- **Phantoms:**
  - 3D-printed, MRI-derived breast shapes from UW-Madison.
  - PC-CPE shells with a 2 mm wall.
  - **Adipose shell** filled with glycerin (εr ≈ 7 at 2 GHz). The code uses ε_fat = 7.0.
  - **Fibroglandular shell** filled with 30 % Triton X-100 (εr ≈ 45). The code uses ε_fib = 45.0.
  - **Tumours** are saline-filled glass spheres with radius 5, 10 or 15 mm, at known (x, y).
  - 9 adipose shells (A1, A2, A3, A11–A16) × fibroglandular shells (F1–F5, F11–F14) give **66 phantom combinations** (IDs like `A3F4`).
  - Adipose radius at the antenna plane is 22.9–57.0 mm (`ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM` in `src/data_loading.py`). Table 1 of the report gives 12.0–16.3 cm, which is the maximum shell **diameter**, not the radius at the antenna plane.
- **Acquisition:**
  - A rotating module with 2 horn antennas moves **clockwise** through 72 positions in 5° steps (0–355°).
  - A VNA records S11 and S21 at 1001 points from 1 to 8 GHz.
  - Gen-2 is **bistatic**: Port A starts at −130°, Port B at −70° (+60°).
  - `ant_rad` (cm) is measured from the SMA connector. The phase-centre correction is `0.97·(r_m − 0.106) + 0.148` **in metres**.
- **Coordinates:** origin at the chamber centre. Tumour `tum_x`, `tum_y`, `tum_rad` are in **cm** (the code converts to mm). The adipose shell centre is `adi_x`, `adi_y`.
- **References:** each scan has `emp_ref_id` (empty-chamber scan) and `adi_ref_id` (adipose-only scan).
- **Used in this work:** S21 only, **504 tumour-bearing scans** across 66 phantoms (about 7–8 scans per phantom, so scans are **not independent**). Tumour-free scans are not evaluated.
- **Ground-truth distribution** (report Table 6): 0.2 % < 5 mm, 1.8 % 5–10 mm, 12.9 % 10–20 mm, 28.0 % 20–30 mm, 57.1 % > 30 mm from the centre. The mean tumour distance is 30.64 mm.
- **Local files** (not in git): `data/fd_data_gen_two_s21.pickle`, `data/metadata_gen_two.pickle`, `data/phantom_database.csv` (phantom volumes, shell radius, density class; merged by `phant_id`).

## 5. Reconstruction pipeline (as described in the report, Section 3.4)
Code entry point: `src/pipeline.py::reconstruct_scan()`

| # | Step | Report setting | Code |
|---|---|---|---|
| 1 | Empty-chamber subtraction | `fd_scan = fd_raw − fd_empty` | `pipeline.py` |
| 2 | Frequency Tukey window | α = 0.25 over 1–8 GHz | `signal_processing.to_time_domain` |
| 3 | ICZT to time domain | 0–6 ns, 1024 samples (UM-BMID toolkit) | `umbmid/sigproc.py` |
| 4 | Per-channel RMS normalization | divide each position by its RMS | `apply_rms_norm` |
| 5 | Hybrid TVSVD clutter removal | remove singular component if energy > 1 % AND flatness > 0.85; cumulative cap 0.98 | `apply_hybrid_tvsvd` |
| 6 | Effective velocity | ε_eff = f_fat·7 + f_fib·45 (linear mix); v = c/√ε_eff | `physics.compute_effective_velocity` |
| 7 | Antenna geometry | 72 positions, 60° bistatic, Rodriguez-Herrera radius | `physics.get_antenna_geometry` ⚠️ bugs 1–2 |
| 8 | Delay grid | straight-ray two-medium (air/tissue) "Snellius" model, grid = 1.5 × breast radius, 1 mm step, cached per phantom | `physics.snellius_bistatic_delay_precise` ⚠️ bug 3 |
| 9 | Early-time gate | zero samples before 0.7 ns | `beamforming._gate_early_samples` (default 0.5 ⚠️) |
| 10 | Antenna Tukey window | α = 0.2 across positions | `beamforming.get_antenna_window` |
| 11 | DAS + CF | image = \|Σ w·s\| · CF, with CF = \|Σ w s\|² / (Σw² · Σ\|s\|²) | `beamforming.das_coherent_cf` |
| 12 | Blob extraction | multi-scale LoG → Otsu (90th-percentile fallback) → largest component | `blob_detection.py` |
| 13 | Localization | intensity-weighted centroid of the blob | `blob_detection.weighted_centroid` |
| 14 | Metrics | LE, SCR (blob vs rest), CF at the predicted point | `src/metrics.py` |

## 6. Failure modes claimed in the report (Sections 3.5, 4.1)
1. **Ring-symmetry coherence artifact.**
   - The array centre is equally far from every antenna, so any signal common to all channels adds in phase there.
   - This produces false high CF at (0, 0), and peaks get pulled toward the centre.
   - Reported evidence: mean CC 0.4043, mean CF at peak 0.3746, mean predicted distance 10.87 mm against ground truth 30.64 mm, and Figure 5 (CF map).
   - ⚠️ Figure 5 is synthetic (`metrics_result/cf_baseline_heatmap.py`). Bug 1 (wrong start angle) can also create fake central focusing.
2. **Boundary signal collapse.**
   - Tumours near the boundary give SCR below 0 dB, with a claimed slope of −0.96 dB/mm and a 0 dB crossing at about 26 mm.
   - Also claimed: 38.3 % of tumours are affected, 99.2 % of candidate peaks lie at normalized radius < 0.75, and the collapse reaches −24.8 dB.
   - ⚠️ Figure 6 / the slope come from **mock data** (`metrics_result/signal_death.py`). The 38.3 % assumes a 45 mm radius for every phantom. The −24.8 dB has no traceable source.
   - The report's physical explanation was also corrected: a boundary tumour is *closer* to the nearest antennas. The plausible causes are fewer coherently contributing positions and the echo overlapping with the skin reflection.

## 7. Mitigation atlas (Section 3.6): tried and failed or kept
- **Kept:** Hybrid TVSVD, per-channel RMS normalization, early-time gate.
- **Rejected:**
  - Depth-gain compensation at 0.7 dB/cm: "inert", but actually a seconds-vs-metres unit bug.
  - 4–6 GHz sub-band mask: implemented but not used.
  - SVM peak ranking: 37.7 mm against 32.2 mm for argmax; AUC 0.536. Not in this repo, ignore.
  - Blob-area size estimation: worse than a constant guess.
- **Ablation table:** DAS+CF 30.7 mm, DMAS+CF 30.8, raw DMAS 33.6, raw DAS 41.2, and all GIBR (bent-ray) variants 62–69 mm.
  - ⚠️ `ablation_runner.py` crashes in this branch, and no GIBR code exists. The table cannot be reproduced yet.
- **Non-summation checks:** SCME (peak at 51 mm, 0/504) and single-channel envelope (71 mm, 0/1301).
  - ⚠️ The SCME map in `metrics_image/scme_test.py` is spatially constant. No envelope code exists.
- **U-Net (Section 4.2.2):** 2-channel input (DAS magnitude + CF), 128×128, BCE + Dice loss. Reported Dice 0.024, precision 0.012, recall 1.0.
  - ⚠️ These numbers are produced by a double-sigmoid bug in `unet/evaluate_unet.py`.

## 8. RRS-MBI score (Section 3.7)
Code: `rrs_mbi/metrics.py::evaluate_scan_reliability()`, called from `rrs_mbi/evaluator.py`

- **Breast mask:** pixels above the 15th percentile of positive intensities, holes filled, largest component kept.
- **Peak dominance D** = (I_peak − μ_clutter) / σ_clutter.
  - Clutter = mask eroded 3×, excluding 5 px around the peak.
  - D_norm = min(1, max(0, D)/5).
- **Central concentration CC** = energy within 10 mm of the centre ÷ energy in the mask. It enters the score as (1 − CC).
- **Geometric plausibility G** = min(1, d / R_sat).
  - d = distance of the **global argmax** from the centre.
  - The report says R_sat = 20 mm; ⚠️ the evaluator passes 5.0.
- **CF_norm** = clip((CF − 0.1)/0.7, 0.1, 1.0), with CF sampled at the **blob centroid**.
- **R** = D_norm · CF_norm · (1 − CC) · G. This is a conjunctive product, used as a **ranking**, not a probability.
- **Design note:** R_sat, the CF range and the D/5 clip were tuned on the same 504 scans used for evaluation, so results are in-sample.

## 9. Evaluation protocol (Section 3.8)
- **LE:** Euclidean distance between the predicted and true tumour centre (mm).
- **Detection:** LE ≤ 15 mm. The SVM comparison used 20 mm.
- **Selective classification** (Chow 1970; Geifman & El-Yaniv 2017):
  - Sort scans by R and keep the top 100 % … 10 %.
  - At each level report mean LE, SD, detection rate and a 1000× bootstrap 95 % CI.
- **Permutation test:** top-50 mean LE against 10,000 random subsets of 50, one-sided.
- **Bootstrap of the gap** (full-set mean − top-50 mean), 10,000×.
- **Triage tiers:** the report says 33rd/67th percentiles. Table 4 shows fixed bands 0.30/0.70.
  - The confusion-matrix counts are printed by `evaluator.main()` but are **not in the report yet**.
  - A chi-square test is promised but not reported.
- **Phantom level:** Spearman/Pearson between per-phantom mean R and mean LE (66 phantoms).
- **Significance:** α = 0.05. p between 0.05 and 0.10 counts as a "trend".

## 10. Key findings as currently reported (to be re-verified)
| Finding | Reported value |
|---|---|
| DAS+CF baseline | mean LE 31.84 ± 13.28 mm, median 31.53, detection 12.3 % (62/504) |
| Predicted vs true distance from centre | 10.87 vs 30.64 mm (a trivial "always predict centre" guess would score 30.64 mm) |
| Phantom level | mean error 32.42 mm (σ 7.45); 31/66 phantoms with ≥1 detection |
| R distribution | mean 0.162, range 0.002–0.41 (histogram shows up to ~0.63); 88.3 % < 0.30, 11.7 % 0.30–0.70, 0 % ≥ 0.70 |
| Correlation R vs LE | scan ρ = −0.004 (p 0.924), r = −0.018; phantom ρ = −0.022, r = −0.076 |
| Selective classification | flat (31.5–31.9 mm) from 100 % to 30 % retention; 30.97 mm at 20 %; **29.49 mm at 10 %** (threshold 0.317), detection 16.0 % |
| Significance | permutation p = 0.098 (a trend, not significant); gap CI [−1.50, +6.03] mm includes 0 |
| Conclusion drawn | failures are physical, ML cannot fix them, and RRS-MBI gives modest, trend-level triage benefit |

## 11. Where each report element comes from
| Report element | Script | Status |
|---|---|---|
| Table 5, Table 8, Table 9, Figure 8, Figure 9, correlations, permutation, confusion matrix | `rrs_mbi/evaluator.py` → `rrs_mbi/results/metrics_raw.csv` | real data, affected by bugs 1–3 and 7 |
| Bootstrap gap CI | `bootstrap_vali.py` (reads `metrics_raw.csv`) | real |
| CC stats, "38.3 %" | `metrics_paper.py` | 38.3 % uses a fixed 45 mm radius |
| R histogram with 0.30/0.70 bands | `metrics_result/all_scan_distribution.py`, `sebaran_thresh_seg.py` | real |
| Table 6 tumour distribution | `metrics_result/tumor_dist.py`, `tumor_loc_dist.py` | metadata |
| Figure 5 CF baseline | `metrics_result/cf_baseline_heatmap.py` | ⚠️ synthetic pattern |
| Figure 6 signal death | `metrics_result/signal_death.py` | ⚠️ mock data |
| Ablation table | `ablation_runner.py` | ⚠️ crashes; no GIBR code |
| SCME table | `metrics_image/scme_test.py` | ⚠️ spatially constant map |
| U-Net table | `unet/evaluate_unet.py` | ⚠️ double sigmoid |
| Risk matrix figure | `metrics_result/RISKMAP.py` | illustrative, hard-coded |
| SVM (Table 7, Figure 7) | not in repo | ignore |

## 12. What "done" looks like
1. The report numbers are reproduced from the original code (which confirms the pipeline that made them).
2. The fixes are applied and the same numbers regenerated (report | baseline | fixed table).
3. Figures 5 and 6 are rebuilt from real reconstructions, or removed.
4. The U-Net is retrained and evaluated correctly on held-out phantoms.
5. There is a clear verdict on the core argument: do the two failure modes survive correct geometry, and does RRS-MBI still rank scans usefully? rr will then update the report text from these results.

## Glossary
| Term | Meaning |
|---|---|
| DAS | delay-and-sum beamforming |
| DMAS | delay-multiply-and-sum beamforming |
| CF | coherence factor |
| ICZT | inverse chirp Z-transform |
| TVSVD | the custom SVD-based clutter filter |
| LE | localization error |
| SCR | signal-to-clutter ratio |
| CC | central concentration |
| G | geometric plausibility |
| GIBR | geometry-informed bent ray |
| SCME | spatial correlation matrix eigenanalysis |
| BI-RADS | breast density class |
| KP | Kerja Praktik (internship) |
