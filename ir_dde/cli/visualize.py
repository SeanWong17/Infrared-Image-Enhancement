import argparse

import cv2
import numpy as np

from ir_dde import enhance_frame, get_preset, linear_baseline
from ir_dde.pipeline import load_image, save_image


def render_map(image: np.ndarray, signed: bool = False) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    if signed:
        scale = max(float(np.percentile(np.abs(src), 99.0)), 1e-6)
        normalized = np.clip((src / scale) * 0.5 + 0.5, 0.0, 1.0)
    else:
        lo = float(np.percentile(src, 1.0))
        hi = float(np.percentile(src, 99.0))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        normalized = np.clip((src - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def tile_with_label(image: np.ndarray, label: str, target_size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    canvas = cv2.resize(canvas, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    return canvas


def build_panel(raw: np.ndarray, enhanced: np.ndarray, debug: dict[str, np.ndarray | float | dict[str, float]]) -> np.ndarray:
    baseline = linear_baseline(raw)
    tiles = [
        tile_with_label(baseline, "linear_baseline", (360, 288)),
        tile_with_label(enhanced, "open_dde_v3", (360, 288)),
        tile_with_label(render_map(debug["base_fine"]), "base_fine", (360, 288)),
        tile_with_label(render_map(debug["base_coarse"]), "base_coarse", (360, 288)),
        tile_with_label(render_map(debug["detail"], signed=True), "detail", (360, 288)),
        tile_with_label(render_map(debug["detail_control"], signed=True), "detail_control", (360, 288)),
        tile_with_label(render_map(debug["edge_confidence"]), "edge_confidence", (360, 288)),
        tile_with_label(render_map(debug["hotspot_mask"]), "hotspot_mask", (360, 288)),
    ]
    top = np.hstack(tiles[:4])
    bottom = np.hstack(tiles[4:])
    return np.vstack([top, bottom])


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the Open DDE v3-like pipeline and intermediate maps.")
    parser.add_argument("-i", "--input", default="examples/single/original_16bit.tif", help="Input infrared image path.")
    parser.add_argument("-o", "--output", default="comparisons/pipeline_panel.png", help="Panel output path.")
    parser.add_argument("--enhanced_out", default="", help="Optional path to save the enhanced image itself.")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=["balanced", "detail_plus", "noise_safe", "hot_scene", "radiometric_safe"],
        help="Tuning preset.",
    )
    args = parser.parse_args()

    raw = load_image(args.input)
    config = get_preset(args.preset)
    enhanced, debug = enhance_frame(raw, config=config, return_debug=True)

    panel = build_panel(raw, enhanced, debug)
    save_image(args.output, panel)
    if args.enhanced_out:
        save_image(args.enhanced_out, enhanced)

    print(f"✅ Saved pipeline panel to {args.output}")
    if args.enhanced_out:
        print(f"✅ Saved enhanced output to {args.enhanced_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
