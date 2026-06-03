import cv2
import numpy as np


def enhance_frame_legacy(
    image: np.ndarray,
    plateau_ratio: float = 0.001,
    detail_sigma_mult: float = 2.0,
    detail_max: float = 25.0,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 25.0,
    bilateral_sigma_space: float = 80.0,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("enhance_frame_legacy only supports single-channel input")

    original_f32 = image.astype(np.float32)
    height, width = original_f32.shape

    base_layer_bf = cv2.bilateralFilter(original_f32, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
    base_layer_gauss = cv2.GaussianBlur(base_layer_bf, (3, 3), 1)
    detail_layer = original_f32 - base_layer_gauss

    hist_input = base_layer_bf.astype(np.uint16)
    threshold = height * width * plateau_ratio
    hist, _ = np.histogram(hist_input.flatten(), bins=65536, range=[0, 65536])
    clipped_hist = np.copy(hist)
    clipped_hist[clipped_hist > threshold] = threshold
    cdf = np.cumsum(clipped_hist)
    cdf_min = float(cdf.min())
    cdf_max = float(cdf.max())
    if cdf_max - cdf_min < 1e-9:
        base_processed = hist_input.astype(np.float32)
    else:
        cdf_normalized = ((cdf - cdf_min) * 255.0) / (cdf_max - cdf_min)
        lut = cdf_normalized.astype(np.uint8)
        base_processed = lut[hist_input]

    sigma_r = np.std(detail_layer) * detail_sigma_mult
    detail_clipped = np.clip(detail_layer, -sigma_r, sigma_r)

    base_normalized = cv2.normalize(base_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    detail_normalized = cv2.normalize(detail_clipped, None, 0, detail_max, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    final_f32 = base_normalized + detail_normalized
    final_u8 = cv2.normalize(final_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return final_u8
