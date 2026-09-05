import sys
import inspect
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import rrs_mbi.metrics as rmx

for n in dir(rmx):
    if not n.startswith("_") and callable(getattr(rmx, n)):
        try:
            print(f"{n}{inspect.signature(getattr(rmx, n))}")
        except Exception:
            print(n)