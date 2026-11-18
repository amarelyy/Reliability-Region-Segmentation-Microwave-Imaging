"""Unit test: default d_saturation=5.0 / d_mode='linear' reproduces the old formula.
Inputs are HAND-MADE test images (synthetic; not results). Run: python results_dnorm/test_dnorm_default_equivalence.py
"""
import subprocess, sys, types
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rrs_mbi.metrics as new

# the committed (pre-change) module, loaded from git HEAD
src = subprocess.run(["git", "show", "HEAD:rrs_mbi/metrics.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
old = types.ModuleType("old_metrics"); exec(compile(src, "old_metrics", "exec"), old.__dict__)

axis = np.linspace(-45, 45, 91)
Y, X = np.meshgrid(axis, axis, indexing="ij")
rng = np.random.default_rng(0)
imgs = []
for (px, py, amp, noise) in [(20, -10, 5.0, 0.2), (3, 2, 1.0, 0.5), (-30, 25, 50.0, 0.1), (0, 0, 0.3, 0.3)]:
    im = np.abs(rng.normal(1.0, noise, X.shape)) + amp * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / 20.0)
    imgs.append(im)
def same(u, v):
    return u.keys() == v.keys() and all(np.array_equal(u[k], v[k], equal_nan=True) for k in u)


n = 0
for im in imgs:
    for cf in [None, 0.05, 0.3, 0.9]:
        for sat in [5.0, 20.0]:
            a = old.evaluate_scan_reliability(im, axis_mm=axis, cf_at_peak=cf, saturation_radius_mm=sat)
            b = new.evaluate_scan_reliability(im, axis_mm=axis, cf_at_peak=cf, saturation_radius_mm=sat)
            c = new.evaluate_scan_reliability(im, axis_mm=axis, cf_at_peak=cf, saturation_radius_mm=sat,
                                              d_saturation=5.0, d_mode="linear")
            assert same(a, b) and same(a, c), (a, b, c)
            ref = min(1.0, a["peak_dominance"] / 5.0) * (1 - a["central_concentration"]) * a["geo_plausibility"] * a["cf_factor"]
            assert abs(ref - a["reliability_score"]) < 1e-15
            n += 1
# hand-made D values
f = new.normalise_dominance
assert f(2.5) == 0.5 and f(5.0) == 1.0 and f(80.0) == 1.0 and f(0.0) == 0.0
assert f(10.0, 20.0, "linear") == 0.5
assert f(0.0, 9.0, "log") == 0.0 and f(9.0, 9.0, "log") == 1.0 and f(99.0, 9.0, "log") == 1.0
assert abs(f(2.0, 8.0, "log") - np.log(3) / np.log(9)) < 1e-15
try:
    f(1.0, 5.0, "sqrt"); raise SystemExit("bad d_mode not rejected")
except ValueError:
    pass
print(f"PASS: defaults identical to HEAD on {n} hand-made cases (all returned keys identical, NaN-aware); linear/log formulas checked")
