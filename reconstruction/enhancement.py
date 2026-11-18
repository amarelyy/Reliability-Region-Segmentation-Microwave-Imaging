"""
reconstruction/enhancement.py

Enhancement + peak localisation steps from rr's tuning slides. Definitions are taken from rr's own
code (amarelyy/Coherence-Weighted-Bent-Ray-..., enhancement_gallery.py, visualize_final.py and
src/blob_detection.py):

  raw -> percentile clip 5-99 -> MER (power 2.0) -> MER + gamma (0.5)
      -> multi-scale LoG (sigma 3-7) -> Otsu -> largest blob -> weighted centroid (bias_alpha 0.3)

  MER(img, p)    = (img / max(img)) ** p
  gamma(img, g)  = (img / max(img)) ** g
  weighted centroid with bias correction (alpha = 0.3):
      p = (1 - alpha) * intensity-weighted centroid of the blob + alpha * geometric centre of the blob
      (geometric centre = plain mean of the blob's pixel coordinates; alpha 0 = pure intensity).

NOTE: with p = 2.0 and gamma = 0.5 the two steps cancel: (x**2)**0.5 == x for x in [0, 1], so
"MER + gamma(0.5)" is numerically the linearly normalised image. That is how rr's code behaves; it
is reproduced as is.
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


def gamma_correct(img, gamma=0.5):
    return (img / (img.max() + 1e-12)) ** gamma


def weighted_centroid_biased(mask, img, x_axis, y_axis, bias_alpha=0.3):
    """Intensity-weighted centroid of the blob, blended with the blob's geometric centre."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        p = (float(x_axis[ix]), float(y_axis[iy]))
        return p, p
    w = img[ys, xs]
    iw = (float(np.average(x_axis[xs], weights=w)), float(np.average(y_axis[ys], weights=w)))
    geo = (float(np.mean(x_axis[xs])), float(np.mean(y_axis[ys])))
    biased = ((1 - bias_alpha) * iw[0] + bias_alpha * geo[0], (1 - bias_alpha) * iw[1] + bias_alpha * geo[1])
    return iw, biased
