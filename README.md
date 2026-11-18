# Microwave Breast Imaging on UM-BMID Gen-2: DAS + CF Reconstruction, Evaluation and U-Net

Radar-based microwave breast imaging on the UM-BMID Gen-2 phantom dataset (bistatic S21, 1-8 GHz,
72 rotation positions). The repository contains the reconstruction pipeline (delay-and-sum with a
Coherence Factor), its evaluation against ground-truth tumour positions, a display-only visual
enhancement for the reconstruction images, and a U-Net segmentation experiment.

## 1. Environment
```
py -3.11 -m venv venv_ai
.\venv_ai\Scripts\activate
pip install -r requirements.txt
```

## 2. Data
`data/` is not tracked by git. It must contain these UM-BMID Gen-2 files:
- `fd_data_gen_two_s21.pickle`
- `metadata_gen_two.pickle`
- `phantom_database.csv`

504 tumour-bearing scans from 66 phantoms are used (about 7.6 scans per phantom, so scans of one
phantom are not independent: all confidence intervals resample phantoms).

## 3. Repository layout
| Path | Content |
|---|---|
| `src/` | Reconstruction pipeline: data loading, signal processing, geometry and delay model, DAS + CF, blob detection, metrics |
| `umbmid/` | Vendored UM-BMID toolkit (the ICZT is used) |
| `reconstruction/` | Image generation, visual enhancement, evaluation tables (see section 4) |
| `rrs_mbi/` | Evaluation protocol (`evaluator.py`: bootstrap, trivial baseline) and the RRS reliability score (`metrics.py`) |
| `unet/` | 2-channel U-Net: model, dataset, training, held-out evaluation |
| `results_unet_fixed/` | Phantom-grouped 5-fold CV for the U-Net and a unit test for the sigmoid bug |
| `baseline_argmax_test.py`, `select_phantoms.py`, `train_svm.py` | Legacy SVM peak-ranking experiment (not re-validated on the corrected pipeline) |
| `metrics_result/` | Data audits (`check_missing_metadata.py`, `scan_utils_audit.py`) |
| `research_notes/`, `reports/` | Literature and research notes |

The branch `old-version` keeps the tracked code and figures from before the clean-up (git-ignored CSV
outputs are not part of any branch).

## 4. Pipelines
### A. Image generation (`src/pipeline.py`, `reconstruct_scan`)
Subtract the adipose-only reference scan, Tukey window + ICZT (0-6 ns, 1024 samples), RMS normalisation
per position, hybrid TVSVD, then delay-and-sum x CF (power 1) over 72 positions using a two-medium
straight-ray delay model (antenna start angle -130 deg, clockwise, 60 deg bistatic offset, corrected
antenna radius, shell centre estimated from the adipose echo, radius from `phantom_database.csv`).
The image is masked to the breast disc. The peak is picked by LoG (sigma 2-5) -> Otsu -> largest blob ->
weighted centroid. LE, Dice, IoU, SCR, SMR, CNR and CF at the peak are computed from that blob.

### B. Visual enhancement (`reconstruction/`)
Display only; it never feeds back into the peak or any metric. Normalise by the in-breast maximum, MER
(`(I/max)^p`, p = 2), gamma (`x^g`, g = 0.7), then render with the turbo colormap and bicubic
interpolation, with the true tumour circle and the pipeline peak overlaid. Parameters are in
`reconstruction/config.json`.

### C. U-Net (`unet/`)
Input: DAS magnitude and CF map (per-channel min-max, 128 x 128). Training and evaluation split by
phantom (the held-out phantom list is saved to `unet/checkpoints/val_phantoms.npy`). The model already
ends in a Sigmoid, so its output is used as a probability.

## 5. Commands (run from the repository root)
```
# Evaluate all 504 scans, write tables to reconstruction/results/, render the 10 best images
python reconstruction/reconstruct_images.py --all --n-jobs 4          # add --reuse to skip recomputation

# Best / worst / median figures (needs the per-scan table from --all) -> reconstruction/images/paper/
python reconstruction/reconstruct_images.py --paper

# Specific or random scans, gamma comparison
python reconstruction/reconstruct_images.py --scans 44 220 328
python reconstruction/reconstruct_images.py --n 5 --seed 42
python reconstruction/reconstruct_images.py --scans 44 220 328 --gamma-sweep

# Pipeline evaluation protocol (localisation error, RRS score, bootstrap)
python -m rrs_mbi.evaluator

# U-Net
python -m unet.training.train                    # train (phantom-grouped split)
python -m unet.evaluate_unet                     # held-out evaluation
python results_unet_fixed/run_unet_cv.py         # phantom-grouped 5-fold CV (builds its input cache on first run)
python results_unet_fixed/test_sigmoid_once.py   # unit test: sigmoid applied once
```

## 6. Outputs
- `reconstruction/results/evaluation_tables.md`: summary tables (tracked). The per-scan CSVs next to it are
  git-ignored (`*.csv` is in `.gitignore`).
- `reconstruction/images/`: top-10 images and `steps/`; `reconstruction/images/paper/`: best / worst / median.
- Cached arrays (`*.npz`, `*.npy`) are git-ignored.

## 7. Notes
- The corrected pipeline does not localise better than always predicting the shell centre
  (see the "centre guess" rows in the evaluation tables), so every localisation error is reported next
  to that baseline.
- Images chosen by ground truth (top-10, best / worst / median) are illustrations, not representative samples.
