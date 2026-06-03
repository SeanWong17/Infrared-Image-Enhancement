from .config import OpenDDEV3Config


def get_preset(name: str) -> OpenDDEV3Config:
    preset = (name or "balanced").lower()
    base = OpenDDEV3Config()

    if preset == "balanced":
        return base
    if preset == "detail_plus":
        return base.with_updates(
            detail_amplitude=0.22,
            detail_gain_max=1.4,
            d2br=1.15,
            base_local_contrast_mix=0.24,
        )
    if preset == "noise_safe":
        return base.with_updates(
            detail_amplitude=0.14,
            detail_gain_max=0.95,
            detail_threshold_scale=1.8,
            spatial_threshold=0.22,
            base_local_contrast_mix=0.12,
            d2br=0.9,
        )
    if preset == "hot_scene":
        return base.with_updates(
            hotspot_protect=0.55,
            base_log_strength=7.5,
            detail_amplitude=0.16,
            output_percentile_high=99.2,
        )
    if preset == "radiometric_safe":
        return base.with_updates(
            detail_amplitude=0.10,
            detail_gain_max=0.75,
            base_local_contrast_mix=0.08,
            d2br=0.75,
            output_percentile_low=1.0,
            output_percentile_high=99.0,
        )
    if preset == "legacy":
        return base.with_updates(
            legacy_mode=True,
            plateau_ratio=0.001,
            legacy_detail_sigma_mult=2.0,
            legacy_detail_max=25.0,
        )
    raise ValueError(f"Unknown preset: {name}")
