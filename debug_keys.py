import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

d = load_all_data()
r = reconstruct_scan(0, d["s21"], d["tumor_model"], d["id_to_original_idx"],
                     freq_axis=d["freq_axis"], use_tvsvd=True,
                     return_diagnostics=False)
print(sorted(r.keys()))