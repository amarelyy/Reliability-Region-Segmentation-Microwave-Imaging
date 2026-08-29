"""
rrs_mbi/metrics.py
Heuristic Reliability Metrics for Microwave Breast Imaging.
"""

import numpy as np
from scipy import ndimage
from typing import Dict

def extract_breast_mask(image: np.ndarray, threshold_percentile: float = 15.0) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros_like(image, dtype=bool)
        
    valid_pixels = image[image > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=bool)
        
    threshold = float(np.percentile(valid_pixels, threshold_percentile))
    mask = (image > threshold).astype(bool)
    
    # Pylance fix: SciPy stubs incorrectly suggest this can return None. 
    # We know it returns an ndarray, so we safely ignore the warning.
    mask = ndimage.binary_fill_holes(mask).astype(bool)  # type: ignore
    
    # Pylance fix: SciPy stubs incorrectly infer this return type as non-iterable.
    # We know it returns a tuple (ndarray, int), so we safely ignore the warning.
    labeled_mask, num_features = ndimage.label(mask)  # type: ignore
    
    if num_features == 0:
        return np.zeros_like(image, dtype=bool)
        
    counts = np.bincount(labeled_mask.flat)
    counts[0] = 0  # Abaikan background (label 0)
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
    
    # Pylance fix: Same SciPy stub issue as above
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

    # Pylance fix: Force explicit casting to satisfy Pylance's strict ndarray check
    dist_map = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)  # type: ignore
    max_dist = float(np.max(dist_map))
    
    if max_dist == 0:
        return 1.0

    boundary_threshold = max_dist * (1.0 - boundary_ratio)
    
    # Use np.less to avoid Pylance operator overloading inference issues
    boundary_mask = np.logical_and(mask, np.less(dist_map, boundary_threshold))
    
    energy_boundary = float(np.sum(image[boundary_mask]**2)) + 1e-10
    energy_total = float(np.sum(image[mask]**2)) + 1e-10
    
    return float(min(1.0, energy_boundary / energy_total))


def evaluate_scan_reliability(image: np.ndarray) -> Dict[str, float]:
    mask = extract_breast_mask(image)
    dominance = calculate_peak_dominance(image, mask)
    risk = calculate_boundary_artifact_risk(image, mask)
    
    norm_dominance = min(1.0, dominance / 5.0)
    reliability = float(norm_dominance * (1.0 - risk))
    
    return {
        "peak_dominance": float(dominance),
        "boundary_risk": float(risk),
        "reliability_score": reliability
    }