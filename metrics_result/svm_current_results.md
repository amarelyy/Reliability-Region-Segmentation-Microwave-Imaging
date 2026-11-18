# SVM candidate verifier on the current pipeline

504 scans, 66 phantoms, 3043 candidates (718 positive = centroid within 20 mm of the tumour). Estimate [95% CI], phantom-cluster bootstrap (2000, seed 42); SVM = OOF, 5-fold GroupKFold by phantom.

## Table 1 - LE (mm) and detection (%)

| Method | Mean LE | Median LE | det@15 | det@20 |
|---|---|---|---|---|
| Pipeline peak | 30.87 [28.38, 33.36] | 31.20 | 16.7 [10.9, 22.6] | 26.0 [18.8, 33.5] |
| Plain argmax | 30.66 [28.50, 32.87] | 31.07 | 14.9 [10.2, 19.8] | 23.4 [17.0, 30.0] |
| Centre guess | 29.22 [27.14, 31.27] | 29.33 | 12.5 [7.3, 18.3] | 20.2 [13.9, 27.0] |
| Nearest-centre cand. | 29.14 [26.95, 31.25] | 29.52 | 14.3 [9.1, 20.2] | 23.4 [17.3, 29.8] |
| SVM C1 all 4 | 33.12 [29.94, 36.70] | 30.74 | 21.6 [14.6, 28.3] | 31.7 [23.5, 39.7] |
| SVM C2 no position (-dist_ratio) | 30.98 [28.40, 33.58] | 30.09 | 17.7 [11.9, 23.6] | 25.8 [19.6, 32.2] |
| SVM C3 position only (dist_ratio) | 34.38 [31.24, 37.68] | 30.88 | 19.4 [13.3, 25.9] | 31.3 [24.0, 38.6] |
| SVM C4 no area (-area_frac) | 33.77 [30.80, 36.91] | 31.17 | 18.5 [12.6, 24.5] | 29.4 [22.0, 36.9] |
| Oracle | 17.41 [15.84, 19.24] | 15.14 | 49.6 [42.6, 56.1] | 66.5 [59.7, 72.6] |

## Table 2 - SVM minus baseline (mean LE in mm, det@20 in pp)

| SVM | vs | Mean LE diff | det@20 diff |
|---|---|---|---|
| C1 all 4 | Pipeline peak | 2.25 [0.38, 4.27] | 5.8 [0.6, 10.4] |
| C1 all 4 | Centre guess | 3.91 [1.31, 6.53] | 11.5 [4.1, 18.9] |
| C2 no position (-dist_ratio) | Pipeline peak | 0.10 [-0.83, 1.14] | -0.2 [-3.4, 3.0] |
| C2 no position (-dist_ratio) | Centre guess | 1.76 [0.32, 3.21] | 5.6 [0.6, 10.5] |
| C3 position only (dist_ratio) | Pipeline peak | 3.51 [1.48, 5.66] | 5.4 [-1.6, 11.9] |
| C3 position only (dist_ratio) | Centre guess | 5.16 [2.44, 7.85] | 11.1 [2.1, 20.0] |
| C4 no area (-area_frac) | Pipeline peak | 2.90 [1.04, 4.89] | 3.4 [-3.2, 9.0] |
| C4 no area (-area_frac) | Centre guess | 4.56 [1.87, 7.15] | 9.1 [0.2, 17.6] |

## Table 3 - AUC (OOF) and standardised SVM weights

| Combination | AUC fold-mean | AUC overall | f_cf_mean | f_peak_rel | f_dist_ratio | f_area_frac |
|---|---|---|---|---|---|---|
| C1 all 4 | 0.578 | 0.585 | 0.55 | -0.27 | 0.62 | 0.42 |
| C2 no position (-dist_ratio) | 0.562 | 0.576 | 0.43 | -0.44 | - | 0.72 |
| C3 position only (dist_ratio) | 0.550 | 0.550 | - | - | 0.57 | - |
| C4 no area (-area_frac) | 0.562 | 0.571 | 0.48 | 0.17 | 0.72 | - |
