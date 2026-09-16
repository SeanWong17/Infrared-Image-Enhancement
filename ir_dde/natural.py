"""Conservative display mapping with bounded contrast and residual gain."""

from dataclasses import asdict

import cv2
import numpy as np

from .filters import guided_filter
from .stats import compute_scene_stats, local_variance, robust_normalize


def enhance_natural(image, config, return_debug=False):
    src, lo, hi = robust_normalize(image, config.input_percentile_low, config.input_percentile_high)
    residual = src - cv2.GaussianBlur(src, (3, 3), 0.8)
    # Estimate residual noise only in the least variable quarter of the image.
    variance = local_variance(src, 3)
    flat = residual[variance <= np.percentile(variance, 25)]
    sigma = max(float(1.4826 * np.median(np.abs(flat - np.median(flat)))), 1e-6)
    base = guided_filter(src, 3, max(sigma * sigma * 2, 1e-6))
    hist, edges = np.histogram(np.clip(base, 0, 1), 1024, (0, 1))
    pdf = hist.astype(np.float64) / base.size
    # Redistribute clipped probability uniformly, bounding the CDF slope to < 5.
    pdf = np.minimum(pdf, 4.0 / 1024)
    pdf += (1 - pdf.sum()) / 1024
    mapped = np.interp(base, edges, np.r_[0, np.cumsum(pdf)])
    tone = 0.8 * mapped + 0.2 * base
    detail = src - base
    controlled = np.sign(detail) * np.maximum(np.abs(detail) - sigma, 0)
    controlled *= 1.5 * config.d2br
    fused = tone + controlled
    output = np.rint(np.clip(fused, 0, 1) * 255).astype(np.uint8)
    if not return_debug:
        return output
    return output, {
        "natural": True, "normalized": src, "base": base, "base_display": tone,
        "detail": detail, "detail_control": controlled, "fused": fused,
        "local_variance": variance, "noise_sigma": sigma, "scene_gain": 1.0,
        "norm_min": lo, "norm_max": hi, "stats": asdict(compute_scene_stats(src)),
    }
