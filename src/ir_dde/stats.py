from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneStats:
    robust_min: float
    robust_max: float
    robust_range: float
    std: float
    entropy: float
    occupied_bins: int
    occupied_ratio: float
    log_mean: float
    highlight_ratio: float


def robust_normalize(image: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[np.ndarray, float, float]:
    src = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(src, low_percentile))
    hi = float(np.percentile(src, high_percentile))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    normalized = np.clip((src - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32), lo, hi


def compute_scene_stats(image: np.ndarray, bins: int = 256) -> SceneStats:
    src = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    hist, _ = np.histogram(src, bins=bins, range=(0.0, 1.0))
    probabilities = hist.astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        probabilities = np.ones_like(probabilities)
        total = probabilities.sum()
    probabilities /= total
    active = probabilities > 1e-4
    entropy = float(-(probabilities[active] * np.log2(probabilities[active])).sum())
    occupied_bins = int(active.sum())

    return SceneStats(
        robust_min=float(np.percentile(src, 0.5)),
        robust_max=float(np.percentile(src, 99.5)),
        robust_range=float(np.percentile(src, 99.5) - np.percentile(src, 0.5)),
        std=float(src.std()),
        entropy=entropy,
        occupied_bins=occupied_bins,
        occupied_ratio=float(occupied_bins / bins),
        log_mean=float(np.exp(np.mean(np.log(src + 1e-4)))),
        highlight_ratio=float(np.mean(src > 0.9)),
    )


def local_variance(image: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    mean = cv2.boxFilter(src, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(src * src, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return variance.astype(np.float32)


def estimate_noise(detail: np.ndarray) -> float:
    src = np.asarray(detail, dtype=np.float32)
    median = float(np.median(src))
    mad = float(np.median(np.abs(src - median)))
    robust_sigma = 1.4826 * mad
    std_fallback = float(src.std()) * 0.25
    return max(1e-6, robust_sigma, std_fallback)
