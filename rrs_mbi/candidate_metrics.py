"""
rrs_mbi/candidate_metrics.py

Candidate ground-truth-free reliability metrics for RRS-MBI v2. Every metric is evaluated
at the SAME predicted point p that the localization error refers to.

Physical rationale (and the PRE-SPECIFIED direction, fixed before any evaluation):
- sa_half_dist (lower = more reliable): a real scatterer focuses at the same place when the
  image is formed from either half of the circular aperture; view-dependent artifacts move.
- sa_interleave_dist (lower): same check with even vs odd positions (two full-circle apertures).
- subband_dist (lower): a real scatterer's location does not depend on frequency;
  ringing/sidelobe artifacts do.
- artifact_prior (lower): normalised intensity of the tumour-free (adipose-only) image of the
  same phantom at p. High = p is where this geometry produces focus without any tumour.
  Data-driven replacement of the fixed "centre is bad" prior G.
- peak_margin (higher): 1 - (strongest competing peak > 10 mm away) / (peak at p). Low = ambiguous.
"""

import numpy as np

from src import beamforming as bf

DIRECTION = {  # +1: higher = more reliable, -1: lower = more reliable
    "sa_half_dist": -1, "sa_interleave_dist": -1, "subband_dist": -1,
    "artifact_prior": -1, "peak_margin": +1, "cf_at_peak": +1,
}
COMPETING_PEAK_EXCLUSION_MM = 10.0


def breast_disc(axis_mm, center_xy, radius_mm):
    gx, gy = np.meshgrid(axis_mm, axis_mm)
    return np.hypot(gx - center_xy[0], gy - center_xy[1]) <= radius_mm


def argmax_xy(img, axis_mm, region):
    iy, ix = np.unravel_index(np.argmax(np.where(region, img, -np.inf)), img.shape)
    return float(axis_mm[ix]), float(axis_mm[iy])


def subaperture_image(time_signal, time_axis, delay_grid, idx, gate_ns):
    """DAS x CF image from the antenna positions `idx` only."""
    idx = np.asarray(idx)
    _, _, img = bf.das_coherent_cf(time_signal[:, idx], time_axis, delay_grid[idx], gate_ns=gate_ns)
    return img


def subaperture_distances(time_signal, time_axis, delay_grid, axis_mm, region, gate_ns, n_ant=72):
    """Mean peak distance between complementary half apertures (two cuts, 90 deg apart), and
    between interleaved (even/odd) apertures."""
    k = np.arange(n_ant)
    pts = lambda idx: argmax_xy(subaperture_image(time_signal, time_axis, delay_grid, idx, gate_ns),
                                axis_mm, region)
    half = []
    for start in (0, n_ant // 4):
        a = (k - start) % n_ant < n_ant // 2
        p1, p2 = pts(k[a]), pts(k[~a])
        half.append(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    e, o = pts(k[::2]), pts(k[1::2])
    return float(np.mean(half)), float(np.hypot(e[0] - o[0], e[1] - o[1]))


def value_near(img, axis_mm, xy, radius_mm=0.0):
    gx, gy = np.meshgrid(axis_mm, axis_mm)
    # at least 0.75 grid steps so the nearest pixel is always included (diagonal = 0.707 step)
    near = np.hypot(gx - xy[0], gy - xy[1]) <= max(radius_mm, 0.75 * abs(axis_mm[1] - axis_mm[0]))
    return float(img[near].max()) if near.any() else float("nan")


def peak_margin(img, axis_mm, region, p_xy, exclusion_mm=COMPETING_PEAK_EXCLUSION_MM):
    gx, gy = np.meshgrid(axis_mm, axis_mm)
    i1 = value_near(img, axis_mm, p_xy, radius_mm=3.0)
    far = region & (np.hypot(gx - p_xy[0], gy - p_xy[1]) > exclusion_mm)
    if not far.any() or not i1 > 0:
        return float("nan")
    return float(np.clip(1.0 - img[far].max() / i1, -1.0, 1.0))


def artifact_prior(ref_img, axis_mm, region, p_xy):
    """Tumour-free image of the same phantom/geometry, normalised to its in-breast max, at p."""
    m = float(np.max(np.where(region, ref_img, 0.0)))
    return value_near(ref_img, axis_mm, p_xy) / m if m > 0 else float("nan")
