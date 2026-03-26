import math

import cv2
import numpy as np

from .stats import robust_normalize


def to_u8(image: np.ndarray) -> np.ndarray:
    src = np.asarray(image)
    if src.dtype == np.uint8:
        return src
    if np.issubdtype(src.dtype, np.floating):
        return np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8)
    normalized, _, _ = robust_normalize(src, 0.1, 99.9)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def entropy(image: np.ndarray) -> float:
    src = to_u8(image)
    hist = cv2.calcHist([src], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    active = probabilities > 0
    return float(-(probabilities[active] * np.log2(probabilities[active])).sum())


def average_gradient(image: np.ndarray) -> float:
    src = to_u8(image).astype(np.float32)
    grad_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return float(magnitude.mean())


def rms_contrast(image: np.ndarray) -> float:
    src = to_u8(image).astype(np.float32) / 255.0
    return float(src.std())


def laplacian_variance(image: np.ndarray) -> float:
    src = to_u8(image).astype(np.float32)
    lap = cv2.Laplacian(src, cv2.CV_32F)
    return float(lap.var())


def eme(image: np.ndarray, block_size: int = 16) -> float:
    src = to_u8(image).astype(np.float32) + 1.0
    height, width = src.shape[:2]
    scores: list[float] = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = src[y : y + block_size, x : x + block_size]
            if block.size == 0:
                continue
            block_min = float(block.min())
            block_max = float(block.max())
            scores.append(20.0 * math.log10(block_max / block_min))
    return float(np.mean(scores)) if scores else 0.0


def linear_baseline(raw_image: np.ndarray, low_percentile: float = 0.1, high_percentile: float = 99.9) -> np.ndarray:
    normalized, _, _ = robust_normalize(raw_image, low_percentile, high_percentile)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def evaluate_display_image(image: np.ndarray) -> dict[str, float]:
    return {
        "entropy": entropy(image),
        "avg_gradient": average_gradient(image),
        "rms_contrast": rms_contrast(image),
        "eme": eme(image),
        "laplacian_var": laplacian_variance(image),
    }


def evaluate_against_linear_baseline(raw_image: np.ndarray, enhanced_image: np.ndarray) -> dict[str, float]:
    baseline = linear_baseline(raw_image)
    baseline_metrics = evaluate_display_image(baseline)
    enhanced_metrics = evaluate_display_image(enhanced_image)
    results = {
        "baseline_entropy": baseline_metrics["entropy"],
        "baseline_avg_gradient": baseline_metrics["avg_gradient"],
        "baseline_rms_contrast": baseline_metrics["rms_contrast"],
        "baseline_eme": baseline_metrics["eme"],
        "baseline_laplacian_var": baseline_metrics["laplacian_var"],
        "enhanced_entropy": enhanced_metrics["entropy"],
        "enhanced_avg_gradient": enhanced_metrics["avg_gradient"],
        "enhanced_rms_contrast": enhanced_metrics["rms_contrast"],
        "enhanced_eme": enhanced_metrics["eme"],
        "enhanced_laplacian_var": enhanced_metrics["laplacian_var"],
    }
    results["entropy_gain"] = enhanced_metrics["entropy"] - baseline_metrics["entropy"]
    results["avg_gradient_gain"] = enhanced_metrics["avg_gradient"] - baseline_metrics["avg_gradient"]
    results["rms_contrast_gain"] = enhanced_metrics["rms_contrast"] - baseline_metrics["rms_contrast"]
    results["eme_gain"] = enhanced_metrics["eme"] - baseline_metrics["eme"]
    results["laplacian_var_gain"] = enhanced_metrics["laplacian_var"] - baseline_metrics["laplacian_var"]
    return results
