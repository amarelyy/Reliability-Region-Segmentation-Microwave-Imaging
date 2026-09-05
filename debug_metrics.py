import sys
import inspect
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.metrics as mx

for n in dir(mx):
    if not n.startswith("_") and callable(getattr(mx, n)):
        try:
            print(f"{n}{inspect.signature(getattr(mx, n))}")
        except Exception:
            print(n)