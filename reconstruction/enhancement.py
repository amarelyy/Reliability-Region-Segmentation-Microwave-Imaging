"""
reconstruction/enhancement.py

Visual enhancement for the reconstruction images. DISPLAY ONLY: nothing here touches the pipeline's peak
or any metric. Definitions follow rr's enhancement code (enhancement_gallery.py in the older Bent-Ray repo):

  raw -> percentile clip 5-99 -> MER (power p) -> MER + gamma

  MER(img, p)    = (img / max(img)) ** p
  gamma(img, g)  = (img / max(img)) ** g

Parameter values come from reconstruction/config.json (mer_power 2.0, gamma 0.7, net exponent 1.4).
NOTE: with p = 2.0 and gamma = 0.5 the two steps cancel, (x**2)**0.5 == x for x in [0, 1], so that pair is
just the linearly normalised image.
"""

import numpy as np


def _region(img, region):
    return np.ones(img.shape, bool) if region is None else region


def raw_normalised(img, region=None):
    """No enhancement: scale to [0, 1] (in-region maximum)."""
    r = _region(img, region)
    return np.where(r, img / (img[r].max() + 1e-12), 0.0)


def percentile_clip(img, low=5, high=99, region=None):
    """Window [low-th, high-th percentile] of the in-region pixels -> [0, 1]."""
    r = _region(img, region)
    n = raw_normalised(img, region)
    lo, hi = np.percentile(n[r], [low, high])
    return np.where(r, np.clip((n - lo) / (hi - lo + 1e-12), 0.0, 1.0), 0.0)


def mer(img, power=2.0, region=None):
    return raw_normalised(img, region) ** power


def gamma_correct(img, gamma=0.7):
    return (img / (img.max() + 1e-12)) ** gamma
