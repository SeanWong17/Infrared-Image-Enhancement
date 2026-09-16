"""Reproduce review experiments without modifying production presets."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from ir_dde import enhance_frame, get_preset, linear_baseline
from ir_dde.metrics import evaluate_display_image
from ir_dde.cli.evaluate import find_raw_matches


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "review"


def methods():
    configs = {name: get_preset(name) for name in (
        "balanced", "legacy", "detail_plus", "noise_safe", "hot_scene", "radiometric_safe", "natural"
    )}
    configs["v3_plateau"] = get_preset("balanced").with_updates(base_method="plateau_he")
    configs["v3_plateau_low_detail"] = configs["v3_plateau"].with_updates(detail_amplitude=0.04)
    configs["v3_no_detail"] = get_preset("balanced").with_updates(d2br=0.0)
    result = {"linear": linear_baseline}
    result["he"] = lambda raw: cv2.equalizeHist(linear_baseline(raw))
    result["clahe"] = lambda raw: cv2.createCLAHE(2.0, (8, 8)).apply(linear_baseline(raw))
    for name, cfg in configs.items():
        result[name] = lambda raw, cfg=cfg: enhance_frame(raw, cfg)
    return result


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tile(image, label):
    scale = min(400 / image.shape[1], 320 / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((350, 400), np.uint8)
    canvas[30:30 + resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    funcs = methods()
    seen, manifest, rows = set(), [], []
    paths = sorted((ROOT / "examples/single").glob("*.tif*")) + sorted((ROOT / "examples/batch/raw").glob("*.tif*"))
    for path in paths:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        manifest.append({"path": str(path.relative_to(ROOT)), "shape": raw.shape, "dtype": str(raw.dtype), "min": int(raw.min()), "max": int(raw.max()), "pixel_sha256": digest})
        outputs = {}
        for name, fn in funcs.items():
            fn(raw)
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                output = fn(raw)
                timings.append((time.perf_counter() - start) * 1000)
            outputs[name] = output
            folder = OUT / name
            folder.mkdir(exist_ok=True)
            cv2.imwrite(str(folder / (path.stem + ".png")), output)
            rows.append({"image": path.stem, "method": name, "median_ms": float(np.median(timings)), **evaluate_display_image(output), "saturated_percent": float(np.mean((output == 0) | (output == 255)) * 100)})
        selected = ["linear", "legacy", "balanced", "v3_plateau", "v3_plateau_low_detail", "clahe"]
        tiles = [tile(outputs[name], name) for name in selected]
        cv2.imwrite(str(OUT / (path.stem + "_comparison.png")), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    write_csv("real_metrics.csv", rows)
    summary = []
    for name in funcs:
        subset = [row for row in rows if row["method"] == name]
        summary.append({"method": name, **{key: float(np.mean([row[key] for row in subset])) for key in rows[0] if key not in ("image", "method")}})
    write_csv("summary.csv", summary)

    # Fixed scene geometry and independent noise draws reveal instability and CNR.
    clean = np.full((256, 320), 10000.0, np.float32)
    clean[:, 160:] = 11000
    clean[96:128, 64:96] += 50
    clean[30:40, 250:260] = 18000
    target = np.s_[100:124, 68:92]
    background = np.s_[100:124, 108:132]
    flat = np.s_[40:80, 40:120]
    synthetic = []
    for noise in (0, 2, 10, 30):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            raw = np.clip(clean + rng.normal(0, noise, clean.shape), 0, 65535).astype(np.uint16)
            for name, fn in funcs.items():
                output = fn(raw).astype(np.float32)
                contrast = float(output[target].mean() - output[background].mean())
                denominator = float(np.sqrt((output[target].var() + output[background].var()) / 2))
                synthetic.append({"noise_dn": noise, "seed": seed, "method": name, "target_contrast": contrast, "cnr": contrast / max(denominator, 1e-6), "flat_std": float(output[flat].std()), "hotspot_mean": float(output[32:38, 252:258].mean())})
    write_csv("synthetic.csv", synthetic)

    probes = {}
    for size in (1, 2, 4, 8):
        raw = np.full((256, 256), 10000, np.uint16)
        raw[100:100 + size, 100:100 + size] = 20000
        probes[f"hotspot_{size}x{size}"] = {name: int(fn(raw).max()) for name, fn in funcs.items()}
    raw = cv2.imread(str(ROOT / "examples/single/zenmuse_xtr_pure.tiff"), -1)
    normalized = (raw.astype(np.float32) - raw.min()) / float(raw.max() - raw.min())
    probes["float_vs_uint16_mae"] = {}
    for name, fn in funcs.items():
        try:
            probes["float_vs_uint16_mae"][name] = float(np.abs(fn(raw).astype(float) - fn(normalized).astype(float)).mean())
        except ValueError as error:
            probes["float_vs_uint16_mae"][name] = str(error)
    # This recreates the evaluator's ambiguous match with two nested scenes.
    collision = OUT / "matching_probe"
    for side, ext in (("raw", ".tiff"), ("enhanced", ".png")):
        for scene in ("a", "b"):
            folder = collision / side / scene
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / ("frame" + ext)), np.zeros((4, 4), np.uint8))
    probes["matching_pairs"] = [(str(a.relative_to(collision)), str(b.relative_to(collision))) for a, b in find_raw_matches(collision / "raw", collision / "enhanced")]
    (OUT / "probes.json").write_text(json.dumps(probes, indent=2))
    (OUT / "manifest.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "opencv_threads": 1, "timing_repeats": 3, "images": manifest}, indent=2))
    print(json.dumps({"unique_images": len(manifest), "methods": len(funcs), "summary": summary, "probes": probes}, indent=2))


if __name__ == "__main__":
    main()
