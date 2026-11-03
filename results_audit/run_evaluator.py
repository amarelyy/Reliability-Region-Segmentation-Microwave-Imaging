"""Audit driver: run rrs_mbi.evaluator.main() unchanged, but
(a) override saturation_radius_mm from outside, and
(b) write outputs to results_audit/<tag>/sat<X>/ instead of rrs_mbi/results/.

Usage: python results_audit/run_evaluator.py <tag> <saturation_radius_mm>
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

tag, sat = sys.argv[1], float(sys.argv[2])
out = REPO / "results_audit" / tag / f"sat{sat:g}"
out.mkdir(parents=True, exist_ok=True)

import rrs_mbi.evaluator as ev

_orig = ev.evaluate_scan_reliability
def _patched(*a, **k):
    k["saturation_radius_mm"] = sat
    return _orig(*a, **k)
ev.evaluate_scan_reliability = _patched

os.chdir(out)  # main() writes to the relative path "rrs_mbi/results"
ev.main()
