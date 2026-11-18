Scans evaluated: 504, phantoms: 66. Original pipeline image and peak (enhancement is display only).

| Localisation error (mm) | min | p25 | median | mean | mean_CI95 | p75 | max | sd | mode_5mm | det@10 (%) | det@15 (%) | det@20 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline peak | 0.38 | 19.49 | 31.20 | 30.87 | [28.4, 33.4] | 41.31 | 87.04 | 14.63 | 35-40 (n=67) | 9.52 | 16.67 | 25.99 |
| Centre guess (trivial baseline) | 3.02 | 21.73 | 29.33 | 29.22 | [27.1, 31.3] | 36.87 | 58.70 | 11.30 | 25-30 (n=97) | 5.75 | 12.50 | 20.24 |

| Image quality and segmentation (pipeline blob) | mean | median | min | max | sd |
|---|---|---|---|---|---|
| SCR (dB) | 10.45 | 10.48 | 4.84 | 16.87 | 2.24 |
| SMR (dB) | 9.54 | 9.69 | 4.04 | 15.49 | 2.15 |
| CNR | 7.03 | 5.93 | 1.38 | 29.72 | 4.38 |
| Dice | 0.04 | 0.00 | 0.00 | 0.69 | 0.10 |
| IoU | 0.02 | 0.00 | 0.00 | 0.53 | 0.06 |
| Blob area (px, 1 mm/px) | 131.44 | 113.00 | 44.00 | 395.00 | 58.16 |
| Size error (mm) | 4.32 | 3.95 | 0.02 | 10.97 | 3.14 |
| CF at peak | 0.39 | 0.40 | 0.00 | 0.90 | 0.20 |
| Peak distance from centre (mm) | 11.13 | 10.02 | 0.38 | 52.81 | 7.97 |
| GT distance from centre (mm) | 29.22 | 29.33 | 3.02 | 58.70 | 11.30 |
| Peak margin (cleanliness, 1 = clean) | 0.18 | 0.33 | -1.00 | 0.96 | 0.60 |

| Blob | GT inside blob (% scans) | Size error <= 5 mm (% scans) | Dice = 0 (% scans) |
|---|---|---|---|
| pipeline blob | 2.78 | 67.46 | 79.17 |

| Paired mean LE | Difference (mm) | 95% CI (phantom-cluster bootstrap) | Pipeline better (% scans) |
|---|---|---|---|
| pipeline - centre guess | +1.66 | [+0.17, +3.06] | 44.64 |
Positive means the pipeline is worse than the centre guess on average.

| True distance from centre (mm) | n | mean LE (mm) | mean centre-guess LE (mm) | det@15 (%) | mean Dice | mean IoU | mean SCR (dB) | mean SMR (dB) |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 29 | 12.53 | 7.40 | 68.97 | 0.19 | 0.11 | 9.63 | 8.36 |
| 10-20 | 73 | 15.71 | 15.18 | 50.68 | 0.12 | 0.07 | 10.13 | 9.01 |
| 20-30 | 166 | 26.47 | 25.37 | 12.05 | 0.03 | 0.02 | 10.44 | 9.51 |
| 30-40 | 147 | 37.33 | 34.89 | 4.76 | 0.01 | 0.01 | 10.41 | 9.61 |
| >40 | 89 | 46.84 | 45.63 | 0.00 | 0.00 | 0.00 | 11.03 | 10.29 |

| Tumour radius (mm) | n | mean LE (mm) | mean centre-guess LE (mm) | det@15 (%) | mean Dice | mean IoU | mean SCR (dB) | mean SMR (dB) |
|---|---|---|---|---|---|---|---|---|
| 5 | 159 | 32.75 | 30.59 | 13.84 | 0.02 | 0.01 | 10.73 | 9.75 |
| 10 | 200 | 31.75 | 29.85 | 12.50 | 0.03 | 0.02 | 10.48 | 9.57 |
| 15 | 145 | 27.61 | 26.84 | 25.52 | 0.07 | 0.04 | 10.09 | 9.26 |

| BI-RADS | n | mean LE (mm) | mean centre-guess LE (mm) | det@15 (%) | mean Dice | mean IoU | mean SCR (dB) | mean SMR (dB) |
|---|---|---|---|---|---|---|---|---|
| 1 | 112 | 28.05 | 30.37 | 23.21 | 0.06 | 0.04 | 10.90 | 10.00 |
| 2 | 115 | 33.41 | 29.46 | 7.83 | 0.02 | 0.01 | 10.84 | 10.00 |
| 3 | 84 | 31.70 | 28.90 | 15.48 | 0.04 | 0.02 | 10.18 | 9.37 |
| 4 | 121 | 35.39 | 31.64 | 9.09 | 0.02 | 0.01 | 10.46 | 9.71 |
72 scans have no BI-RADS and are excluded (n = 432).

