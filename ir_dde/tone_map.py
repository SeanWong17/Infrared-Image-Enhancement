import cv2
import numpy as np

from .stats import SceneStats


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    src = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-src))


def adaptive_log_compression(image: np.ndarray, stats: SceneStats, base_strength: float) -> np.ndarray:
    src = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    dynamic_factor = np.clip(stats.occupied_ratio / 0.55, 0.0, 1.0)
    darkness_factor = np.clip((0.45 - stats.log_mean) / 0.45, 0.0, 1.0)
    strength = float(base_strength) * (1.0 + 0.6 * dynamic_factor + 0.4 * darkness_factor)
    if strength < 1e-6:
        return src
    return np.log1p(strength * src) / np.log1p(strength)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    src = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    src_u8 = np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return clahe.apply(src_u8).astype(np.float32) / 255.0


def plateau_histogram_equalization(image: np.ndarray, plateau_ratio: float, bins: int = 65536) -> np.ndarray:
    src = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    quantized = np.clip(src * (bins - 1), 0.0, bins - 1).astype(np.uint16)
    hist, _ = np.histogram(quantized.ravel(), bins=bins, range=(0, bins))
    threshold = src.size * float(plateau_ratio)
    clipped = np.minimum(hist, threshold)
    cdf = np.cumsum(clipped)
    span = float(cdf[-1] - cdf[0])
    if span < 1e-9:
        return src
    lut = ((cdf - cdf[0]) / span).astype(np.float32)
    return lut[quantized]


def percentile_remap(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(src, low_percentile))
    hi = float(np.percentile(src, high_percentile))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((src - lo) / (hi - lo), 0.0, 1.0)


def soft_knee(image: np.ndarray, gain: float, pivot: float) -> np.ndarray:
    src = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    transformed = sigmoid((src - float(pivot)) * float(gain))
    lo = float(transformed.min())
    hi = float(transformed.max())
    return np.clip((transformed - lo) / (hi - lo + 1e-6), 0.0, 1.0)
