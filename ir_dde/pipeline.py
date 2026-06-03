from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import OpenDDEV3Config
from .filters import difference_of_gaussians, ensure_grayscale, guided_filter
from .legacy import enhance_frame_legacy
from .stats import SceneStats, compute_scene_stats, estimate_noise, local_variance, robust_normalize
from .tone_map import (
    adaptive_log_compression,
    apply_clahe,
    percentile_remap,
    plateau_histogram_equalization,
    sigmoid,
    soft_knee,
)


def _scene_detail_gain(stats: SceneStats) -> float:
    low_contrast_boost = np.clip((0.18 - stats.std) / 0.18, 0.0, 1.0)
    dark_scene_boost = np.clip((0.42 - stats.log_mean) / 0.42, 0.0, 1.0)
    clutter_penalty = np.clip((stats.occupied_ratio - 0.28) / 0.5, 0.0, 1.0)
    highlight_penalty = np.clip(stats.highlight_ratio / 0.15, 0.0, 1.0)
    gain = 1.0 + 0.45 * low_contrast_boost + 0.15 * dark_scene_boost - 0.20 * clutter_penalty - 0.10 * highlight_penalty
    return float(np.clip(gain, 0.75, 1.35))


def _hotspot_mask(base: np.ndarray, stats: SceneStats, config: OpenDDEV3Config) -> np.ndarray:
    center = float(np.percentile(base, config.hotspot_percentile))
    spread = float(np.percentile(base, 95.0) - np.percentile(base, 60.0))
    spread = max(spread, 1e-3)
    strength = float(config.hotspot_protect) * np.clip(0.5 + stats.highlight_ratio * 4.0, 0.5, 1.0)
    response = sigmoid(config.hotspot_gate_steepness * ((base - center) / spread))
    return 1.0 - strength * response


def enhance_frame(
    image: np.ndarray,
    config: OpenDDEV3Config | None = None,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | float | dict[str, float]]]:
    cfg = config or OpenDDEV3Config()
    gray = ensure_grayscale(image)

    if cfg.legacy_mode:
        output_u8 = enhance_frame_legacy(
            gray,
            plateau_ratio=cfg.plateau_ratio,
            detail_sigma_mult=cfg.legacy_detail_sigma_mult,
            detail_max=cfg.legacy_detail_max,
            bilateral_d=cfg.legacy_bilateral_d,
            bilateral_sigma_color=cfg.legacy_bilateral_sigma_color,
            bilateral_sigma_space=cfg.legacy_bilateral_sigma_space,
        )
        if not return_debug:
            return output_u8
        normalized = output_u8.astype(np.float32) / 255.0
        stats = compute_scene_stats(normalized)
        return output_u8, {
            "normalized": normalized,
            "stats": asdict(stats),
            "noise_sigma": 0.0,
            "scene_gain": 1.0,
            "norm_min": float(gray.min()),
            "norm_max": float(gray.max()),
            "legacy": True,
        }

    normalized, norm_lo, norm_hi = robust_normalize(gray, cfg.input_percentile_low, cfg.input_percentile_high)
    stats = compute_scene_stats(normalized)

    base_fine = guided_filter(normalized, cfg.guided_radius_fine, cfg.guided_eps_fine)
    base_coarse = guided_filter(base_fine, cfg.guided_radius_coarse, cfg.guided_eps_coarse)
    detail_fine = normalized - base_fine
    detail_mid = base_fine - base_coarse

    detail = cfg.detail_mix_fine * detail_fine + cfg.detail_mix_mid * detail_mid
    dog = None
    if cfg.use_dog:
        dog = difference_of_gaussians(normalized, cfg.dog_sigma_small, cfg.dog_sigma_large)
        detail = detail + cfg.detail_mix_dog * dog

    variance = local_variance(normalized, cfg.edge_window_radius)
    edge_eps = max(float(np.percentile(variance, 75) * 0.5), 1e-6)
    edge_confidence = variance / (variance + edge_eps)

    noise_sigma = estimate_noise(detail_fine)
    clip_tau = cfg.noise_clip_scale * noise_sigma * (1.0 + cfg.noise_clip_edge_boost * edge_confidence)
    clip_tau = np.maximum(clip_tau, 1e-4)
    detail_clip = np.clip(detail, -clip_tau, clip_tau)
    detail_scaled = (detail_clip / clip_tau) * cfg.detail_amplitude

    local_gain = cfg.detail_gain_min + (cfg.detail_gain_max - cfg.detail_gain_min) * np.power(edge_confidence, cfg.detail_gain_gamma)
    scene_gain = _scene_detail_gain(stats)
    detail_threshold = max(cfg.detail_threshold_scale * noise_sigma, 1e-4)
    amplitude_gate = sigmoid(cfg.amplitude_gate_steepness * ((np.abs(detail) - detail_threshold) / detail_threshold))
    spatial_gate = sigmoid(cfg.spatial_gate_steepness * (edge_confidence - cfg.spatial_threshold))
    detail_control = amplitude_gate * spatial_gate * local_gain * scene_gain * detail_scaled

    if cfg.base_method == "plateau_he":
        base_display = plateau_histogram_equalization(base_coarse, cfg.plateau_ratio)
    elif cfg.base_method == "log_clahe":
        base_global = adaptive_log_compression(base_coarse, stats, cfg.base_log_strength)
        if cfg.base_local_contrast_mix > 0:
            base_local = apply_clahe(base_global, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
            base_display = (1.0 - cfg.base_local_contrast_mix) * base_global + cfg.base_local_contrast_mix * base_local
        else:
            base_display = base_global
    else:
        raise ValueError(f"Unknown base_method: {cfg.base_method!r}")

    hotspot_mask = _hotspot_mask(base_coarse, stats, cfg)
    fused = base_display + cfg.d2br * hotspot_mask * detail_control
    output = percentile_remap(fused, cfg.output_percentile_low, cfg.output_percentile_high)
    if cfg.use_soft_knee:
        output = soft_knee(output, cfg.soft_knee_gain, cfg.soft_knee_pivot)

    output_u8 = np.clip(output * 255.0, 0.0, 255.0).astype(np.uint8)
    if not return_debug:
        return output_u8

    debug = {
        "normalized": normalized,
        "base_fine": base_fine,
        "base_coarse": base_coarse,
        "detail": detail,
        "detail_control": detail_control,
        "edge_confidence": edge_confidence,
        "hotspot_mask": hotspot_mask,
        "noise_sigma": noise_sigma,
        "norm_min": norm_lo,
        "norm_max": norm_hi,
        "scene_gain": scene_gain,
        "stats": asdict(stats),
    }
    if dog is not None:
        debug["dog"] = dog
    return output_u8, debug


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Unable to save image: {output_path}")


def enhance_image_file(input_path: str | Path, output_path: str | Path, config: OpenDDEV3Config | None = None) -> dict[str, float]:
    image = load_image(input_path)
    output, debug = enhance_frame(image, config=config, return_debug=True)
    save_image(output_path, output)
    stats = debug["stats"]
    return {
        "noise_sigma": float(debug["noise_sigma"]),
        "scene_gain": float(debug["scene_gain"]),
        "std": float(stats["std"]),
        "entropy": float(stats["entropy"]),
        "occupied_ratio": float(stats["occupied_ratio"]),
    }


def iter_image_files(input_dir: str | Path, exts: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
    for path in sorted(Path(input_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def batch_enhance(
    input_dir: str | Path,
    output_dir: str | Path,
    config: OpenDDEV3Config | None = None,
    out_ext: str = ".png",
) -> tuple[int, int]:
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    files = list(iter_image_files(source_dir, exts=(".tif", ".tiff")))
    if not files:
        return 0, 0

    ok_count = 0
    fail_count = 0
    for input_path in files:
        rel = input_path.relative_to(source_dir).with_suffix(out_ext)
        output_path = target_dir / rel
        try:
            enhance_image_file(input_path, output_path, config=config)
            ok_count += 1
        except Exception:
            fail_count += 1
    return ok_count, fail_count
