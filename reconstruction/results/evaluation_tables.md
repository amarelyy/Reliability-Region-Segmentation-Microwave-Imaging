Scans evaluated: 504, phantoms: 66. Original pipeline image and peak (enhancement is display only).

| Localisation error (mm) | min | p25 | median | mean | mean_CI95 | p75 | max | sd | mode_5mm | det@10 (%) | det@15 (%) | det@20 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline peak | 0.38 | 19.49 | 31.20 | 30.87 | [28.4, 33.4] | 41.31 | 87.04 | 14.63 | 35-40 (n=67) | 9.52 | 16.67 | 25.99 |
| Centre guess (trivial baseline) | 3.02 | 21.73 | 29.33 | 29.22 | [27.1, 31.3] | 36.87 | 58.70 | 11.30 | 25-30 (n=97) | 5.75 | 12.50 | 20.24 |

Paired mean LE difference, pipeline peak - centre guess (phantom-cluster bootstrap 95% CI; negative = imaging better): +1.66 mm [+0.17, +3.06]; pipeline better in 44.6% of scans.

| Metric | mean | median | min | max | sd |
|---|---|---|---|---|---|
| SCR (dB) | 10.448 | 10.475 | 4.835 | 16.866 | 2.238 |
| SMR (dB) | 9.538 | 9.692 | 4.038 | 15.486 | 2.152 |
| CNR | 7.026 | 5.926 | 1.375 | 29.722 | 4.380 |
| Dice | 0.040 | 0.000 | 0.000 | 0.695 | 0.103 |
| IoU | 0.024 | 0.000 | 0.000 | 0.532 | 0.064 |
| Blob area (px, 1 mm/px) | 131.440 | 113.000 | 44.000 | 395.000 | 58.160 |
| Size error (mm) | 4.316 | 3.950 | 0.017 | 10.971 | 3.140 |
| CF at peak | 0.387 | 0.395 | 0.002 | 0.902 | 0.205 |
| Peak distance from centre (mm) | 11.132 | 10.016 | 0.379 | 52.811 | 7.967 |
| GT distance from centre (mm) | 29.216 | 29.326 | 3.016 | 58.705 | 11.301 |
| Peak margin (cleanliness, 1 = clean) | 0.178 | 0.328 | -1.000 | 0.962 | 0.598 |
| GT inside blob (% scans) | 2.778 |  |  |  |  |
| Size error <= 5 mm (% scans) | 67.460 |  |  |  |  |

| Tumour radius (mm) | n | LE | LE_centre_guess | det15 | dice | scr_db |
|---|---|---|---|---|---|---|
| 5.0 | 159 | 32.75 | 30.59 | 13.84 | 0.02 | 10.73 |
| 10.0 | 200 | 31.75 | 29.85 | 12.50 | 0.03 | 10.48 |
| 15.0 | 145 | 27.61 | 26.84 | 25.52 | 0.07 | 10.09 |

| True distance from centre (mm) | n | LE | LE_centre_guess | det15 | dice | scr_db |
|---|---|---|---|---|---|---|
| 0-10 | 29 | 12.53 | 7.40 | 68.97 | 0.19 | 9.63 |
| 10-20 | 73 | 15.71 | 15.18 | 50.68 | 0.12 | 10.13 |
| 20-30 | 166 | 26.47 | 25.37 | 12.05 | 0.03 | 10.44 |
| 30-40 | 147 | 37.33 | 34.89 | 4.76 | 0.01 | 10.41 |
| >40 | 89 | 46.84 | 45.63 | 0.00 | 0.00 | 11.03 |

| BI-RADS | n | LE | LE_centre_guess | det15 | dice | scr_db |
|---|---|---|---|---|---|---|
| 1.0 | 112 | 28.05 | 30.37 | 23.21 | 0.06 | 10.90 |
| 2.0 | 115 | 33.41 | 29.46 | 7.83 | 0.02 | 10.84 |
| 3.0 | 84 | 31.70 | 28.90 | 15.48 | 0.04 | 10.18 |
| 4.0 | 121 | 35.39 | 31.64 | 9.09 | 0.02 | 10.46 |

