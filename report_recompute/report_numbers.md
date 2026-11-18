# Recomputed report numbers (current pipeline, 504 scans, 66 phantoms)

Estimate [95% CI]: phantom-cluster percentile bootstrap (2000 resamples, seed 42), same routine as reconstruction/results/evaluation_ci.md.

## 1. SCR evaluated at the ground-truth location vs true tumour distance

Definition: image = DAS x CF (breast-masked, same as the main results); signal = disc of the TRUE tumour radius (min 1 mm) centred on the ground truth; clutter = all other pixels of the breast disc (radius R, shell centre); SCR = 10 log10(mean signal / mean clutter). Distance = true tumour distance from the estimated shell centre.

| True distance (mm) | n | phantoms | mean SCR at GT (dB) [95% CI] | mean legacy SCR (dB, 20log10 peak-in-2mm / mean; valid n) | mean blob SCR (dB) |
|---|---|---|---|---|---|
| 0-10 | 29 | 16 | 4.15 [2.65, 6.22] | 12.33 (n=29) | 9.63 |
| 10-20 | 73 | 26 | 1.96 [0.69, 3.70] | 4.75 (n=73) | 10.13 |
| 20-30 | 166 | 52 | -0.07 [-0.91, 0.72] | 2.88 (n=159) | 10.44 |
| 30-40 | 142 | 52 | -1.46 [-2.18, -0.69] | -0.72 (n=137) | 10.47 |
| >40 | 86 | 33 | -2.22 [-3.10, -1.41] | -1.71 (n=80) | 11.10 |

- OLS fit: SCR = 3.89 + (-0.144) x distance, Pearson r = -0.38; zero crossing (OLS) = 27.0 mm.
- Zero crossing from 5-mm binned means (linear interpolation) = 21.5 mm.
- SCR at the mean tumour distance (29.22 mm): OLS -0.32 dB; mean of scans within +-2.5 mm = -0.03 dB (n=75).
- Excluded from item 1: 8 scans whose ground-truth disc lies entirely outside the breast disc (true distance > R); the legacy 2-mm-ROI column additionally drops scans whose ROI has no breast pixel.
- Scans with SCR at GT < 0 dB: 52.2%; min -20.1 dB, max 13.2 dB.

## 3. Central concentration of the DAS+CF image

Main definition (rrs_mbi/metrics.py `calculate_central_concentration`, as called by rrs_mbi/evaluator.py): CC = sum(I^2 over mask pixels within 10 mm of the centre) / sum(I^2 over the mask); energy = intensity squared of the final DAS x CF image; mask = `extract_breast_mask` (pixels above the 15th percentile of positive pixels, holes filled, largest connected component); centre = estimated shell centre. Variants: centre = array centre (0, 0); mask = geometric breast disc (radius R around the shell centre).

| Definition | Mean | Median | SD |
|---|---|---|---|
| Main (image mask, shell centre) | 0.3643 | 0.3222 | 0.2171 |
| Array centre (0,0), image mask | 0.3254 | 0.2963 | 0.2114 |
| Breast-disc mask, shell centre | 0.3643 | 0.3221 | 0.2171 |
