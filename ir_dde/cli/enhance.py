import argparse
from pathlib import Path

from ir_dde import OpenDDEV3Config, enhance_image_file, get_preset


def build_config(args: argparse.Namespace) -> OpenDDEV3Config:
    config = get_preset(args.preset)
    overrides = {
        "d2br": args.d2br,
        "detail_gain_max": args.detail_gain_max,
        "detail_amplitude": args.detail_amplitude,
        "detail_threshold_scale": args.detail_threshold_scale,
        "spatial_threshold": args.spatial_threshold,
        "base_local_contrast_mix": args.base_local_contrast_mix,
        "hotspot_protect": args.hotspot_protect,
        "output_percentile_low": args.output_p_lo,
        "output_percentile_high": args.output_p_hi,
        "base_method": args.base_method,
        "plateau_ratio": args.plateau_ratio,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.use_dog:
        overrides["use_dog"] = True
    return config.with_updates(**overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 Open DDE v3-like 管线增强单张红外图像。")
    parser.add_argument("-i", "--input", required=True, help="输入红外图像路径。")
    parser.add_argument("-o", "--output", required=True, help="输出图像路径。")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=["balanced", "detail_plus", "noise_safe", "hot_scene", "radiometric_safe", "legacy"],
        help="调参预设。legacy 复现早期双边+平台直方图均衡的视觉风格。",
    )
    parser.add_argument("--d2br", type=float, default=None, help="细节与背景的融合比例。")
    parser.add_argument("--detail_gain_max", type=float, default=None, help="局部细节增益上限。")
    parser.add_argument("--detail_amplitude", type=float, default=None, help="限幅后的有符号细节幅度。")
    parser.add_argument("--detail_threshold_scale", type=float, default=None, help="相对噪声估计的细节门限倍率。")
    parser.add_argument("--spatial_threshold", type=float, default=None, help="触发增强所需的最小边缘置信度。")
    parser.add_argument("--base_local_contrast_mix", type=float, default=None, help="基础层中局部对比度分支的混合比例。")
    parser.add_argument("--hotspot_protect", type=float, default=None, help="高亮热点保护强度。")
    parser.add_argument("--output_p_lo", type=float, default=None, help="输出重映射的低百分位。")
    parser.add_argument("--output_p_hi", type=float, default=None, help="输出重映射的高百分位。")
    parser.add_argument("--base_method", default=None, choices=["log_clahe", "plateau_he"], help="基础层处理方式。")
    parser.add_argument("--plateau_ratio", type=float, default=None, help="平台直方图均衡的平台高度比例 (仅 base_method=plateau_he 时生效)。")
    parser.add_argument("--use_dog", action="store_true", help="在引导滤波分解基础上附加 DoG 边缘增强。")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件 '{args.input}' 不存在。请检查路径。")
        return 1

    config = build_config(args)
    metrics = enhance_image_file(input_path, args.output, config=config)
    print("--- Open DDE v3-like 增强完成 ---")
    print(f"✅ 结果已保存至: {Path(args.output).resolve()}")
    print(
        "   "
        f"scene_gain={metrics['scene_gain']:.3f}, "
        f"noise_sigma={metrics['noise_sigma']:.5f}, "
        f"occupied_ratio={metrics['occupied_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
