import pandas as pd, numpy as np
df = pd.read_csv("rrs_mbi/results/metrics_raw.csv")

# Check 1: Constant-predictor baseline
const = df["true_r_mm"].mean()
base = (np.abs(const - df["true_r_mm"]) <= 5).mean()
print(f"constant-predictor size detection: {base:.1%}  vs model 70.0%")
print(f"predicted_r_mm  std: {df['predicted_r_mm'].std():.2f} mm")
print(f"true_r_mm       std: {df['true_r_mm'].std():.2f} mm")

# Check 2: Does sizing depend on finding the tumour?
hit = df["localization_error"] <= 15
print(f"size det | localized:  {(df.loc[hit,'size_error_mm']<=5).mean():.1%}  (n={hit.sum()})")
print(f"size det | missed:     {(df.loc[~hit,'size_error_mm']<=5).mean():.1%}  (n={(~hit).sum()})")