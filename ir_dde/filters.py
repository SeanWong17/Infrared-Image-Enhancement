import cv2
import numpy as np


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_float32(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def guided_filter(image: np.ndarray, radius: int, eps: float, guide: np.ndarray | None = None) -> np.ndarray:
    src = to_float32(image)
    ref = src if guide is None else to_float32(guide)
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)

    mean_ref = cv2.boxFilter(ref, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_src = cv2.boxFilter(src, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ref = cv2.boxFilter(ref * ref, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ref_src = cv2.boxFilter(ref * src, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    var_ref = corr_ref - mean_ref * mean_ref
    cov_ref_src = corr_ref_src - mean_ref * mean_src

    a = cov_ref_src / (var_ref + float(eps))
    b = mean_src - a * mean_ref

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    return mean_a * ref + mean_b


def difference_of_gaussians(image: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    src = to_float32(image)
    blur_small = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small, borderType=cv2.BORDER_REFLECT)
    blur_large = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma_large, sigmaY=sigma_large, borderType=cv2.BORDER_REFLECT)
    return blur_small - blur_large
