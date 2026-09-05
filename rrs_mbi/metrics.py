"""
rrs_mbi/metrics.py
Heuristic Reliability Metrics for Microwave Breast Imaging.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Optional


def extract_breast_mask(image: np.ndarray, threshold_percentile: float = 15.0) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros_like(image, dtype=bool)
        
    valid_pixels = image[image > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=bool)
        
    threshold = float(np.percentile(valid_pixels, threshold_percentile))
    mask = (image > threshold).astype(bool)
    
    mask = ndimage.binary_fill_holes(mask).astype(bool)  # type: ignore
    labeled_mask, num_features = ndimage.label(mask)  # type: ignore
    
    if num_features == 0:
        return np.zeros_like(image, dtype=bool)
        
    counts = np.bincount(labeled_mask.flat)
    counts[0] = 0
    largest_component = int(np.argmax(counts))
    
    return (labeled_mask == largest_component).astype(bool)


def calculate_peak_dominance(image: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0

    peak_val = float(np.max(image))
    peak_idx = int(np.argmax(image))
    peak_coords = np.unravel_index(peak_idx, image.shape)
    
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    exclusion_zone = ((y - peak_coords[0])**2 + (x - peak_coords[1])**2) <= 25
    
    internal_mask = ndimage.binary_erosion(mask.astype(bool), iterations=3).astype(bool)  # type: ignore
    clutter_mask = np.logical_and(internal_mask, np.logical_not(exclusion_zone))
    clutter_pixels = image[clutter_mask]
    
    if len(clutter_pixels) < 10:
        clutter_pixels = image[mask]
        
    mean_clutter = float(np.mean(clutter_pixels))
    std_clutter = float(np.std(clutter_pixels))
    
    if std_clutter < 1e-6:
        return 10.0
        
    scr = (peak_val - mean_clutter) / std_clutter
    return float(max(0.0, scr))


def calculate_boundary_artifact_risk(image: np.ndarray, mask: np.ndarray, boundary_ratio: float = 0.15) -> float:
    if not np.any(mask):
        return 1.0

    dist_map = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)  # type: ignore
    max_dist = float(np.max(dist_map))
    
    if max_dist == 0:
        return 1.0

    boundary_threshold = max_dist * (1.0 - boundary_ratio)
    boundary_mask = np.logical_and(mask, np.less(dist_map, boundary_threshold))
    
    energy_boundary = float(np.sum(image[boundary_mask]**2)) + 1e-10
    energy_total = float(np.sum(image[mask]**2)) + 1e-10
    
    return float(min(1.0, energy_boundary / energy_total))


def calculate_geometric_plausibility(image: np.ndarray, axis_mm: Optional[np.ndarray],
                                     saturation_radius_mm: float = 5.0) -> float:
    """
    NEW — Geometric-plausibility prior.

    The circular antenna array produces a known coherent artifact at the array
    centre. A reconstruction peak sitting on/near that centre is therefore more
    likely to be artifact than a true target, while a peak displaced away from
    the centre is more plausible. This factor down-weights central peaks and
    saturates to 1.0 once the peak is beyond `saturation_radius_mm`.

    Uses only general imaging-geometry knowledge (NOT per-scan ground truth).
    """
    if axis_mm is None:
        return 1.0

    peak_iy, peak_ix = np.unravel_index(np.argmax(image), image.shape)
    peak_x = float(axis_mm[peak_ix])
    peak_y = float(axis_mm[peak_iy])
    pred_dist = float(np.hypot(peak_x, peak_y))

    return float(min(1.0, pred_dist / saturation_radius_mm)**2)


def evaluate_scan_reliability(image: np.ndarray, axis_mm=None, cf_at_peak=None,
                              saturation_radius_mm: float = 5.0) -> Dict[str, float]:
    mask = extract_breast_mask(image)
    dominance = calculate_peak_dominance(image, mask)
    risk = calculate_boundary_artifact_risk(image, mask)
    geo = calculate_geometric_plausibility(image, axis_mm, saturation_radius_mm)

    # CF-at-peak component (if available)
    cf_factor = 1.0
    if cf_at_peak is not None and not np.isnan(cf_at_peak):
        cf_factor = float(np.clip(cf_at_peak, 0.0, 1.0))

    norm_dominance = min(1.0, dominance / 5.0)
    reliability = float(norm_dominance * (1.0 - risk) * geo * cf_factor)

    return {
        "peak_dominance": float(dominance),
        "boundary_risk": float(risk),
        "geo_plausibility": float(geo),
        "cf_at_peak": float(cf_at_peak) if cf_at_peak is not None else float("nan"),
        "reliability_score": reliability
    }