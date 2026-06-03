from dataclasses import dataclass, replace


@dataclass(frozen=True)
class OpenDDEV3Config:
    input_percentile_low: float = 0.1
    input_percentile_high: float = 99.9

    guided_radius_fine: int = 6
    guided_eps_fine: float = 1e-4
    guided_radius_coarse: int = 18
    guided_eps_coarse: float = 4e-4
    edge_window_radius: int = 3

    detail_mix_fine: float = 0.7
    detail_mix_mid: float = 0.3
    use_dog: bool = False
    dog_sigma_small: float = 0.8
    dog_sigma_large: float = 1.6
    detail_mix_dog: float = 0.15

    noise_clip_scale: float = 3.0
    noise_clip_edge_boost: float = 1.0
    detail_amplitude: float = 0.18
    detail_gain_min: float = 0.15
    detail_gain_max: float = 1.20
    detail_gain_gamma: float = 0.8
    detail_threshold_scale: float = 1.4
    spatial_threshold: float = 0.15
    amplitude_gate_steepness: float = 10.0
    spatial_gate_steepness: float = 10.0
    d2br: float = 1.0

    base_log_strength: float = 6.0
    base_local_contrast_mix: float = 0.20
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    base_method: str = "log_clahe"
    plateau_ratio: float = 0.001

    legacy_mode: bool = False
    legacy_detail_sigma_mult: float = 2.0
    legacy_detail_max: float = 25.0
    legacy_bilateral_d: int = 9
    legacy_bilateral_sigma_color: float = 25.0
    legacy_bilateral_sigma_space: float = 80.0

    hotspot_protect: float = 0.35
    hotspot_percentile: float = 90.0
    hotspot_gate_steepness: float = 8.0

    output_percentile_low: float = 0.5
    output_percentile_high: float = 99.5
    use_soft_knee: bool = False
    soft_knee_gain: float = 6.0
    soft_knee_pivot: float = 0.5

    def with_updates(self, **kwargs: float) -> "OpenDDEV3Config":
        return replace(self, **kwargs)
