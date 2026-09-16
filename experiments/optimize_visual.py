"""Generate full-resolution comparisons and fixed crops for visual selection."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from ir_dde import enhance_frame, get_preset
from ir_dde.filters import guided_filter
from ir_dde.stats import robust_normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/optimization"


def candidate(raw, limit=4.0, mix=0.8):
    src, _, _ = robust_normalize(raw, 0.1, 99.9)
    residual = src - cv2.GaussianBlur(src, (3, 3), 0.8)
    variance = cv2.blur(src * src, (7, 7)) - cv2.blur(src, (7, 7)) ** 2
    flat = residual[variance <= np.percentile(variance, 25)]
    sigma = max(float(1.4826 * np.median(np.abs(flat - np.median(flat)))), 1e-6)
    base = guided_filter(src, 3, max(sigma * sigma * 2, 1e-6))
    hist, edges = np.histogram(base, 1024, (0, 1))
    pdf = hist.astype(float) / base.size
    pdf = np.minimum(pdf, limit / 1024)
    pdf += (1 - pdf.sum()) / 1024
    cdf = np.r_[0, np.cumsum(pdf)]
    mapped = np.interp(base, edges, cdf)
    tone = mix * mapped + (1 - mix) * base
    detail = src - base
    detail = np.sign(detail) * np.maximum(np.abs(detail) - sigma, 0)
    result = np.clip(tone + 1.5 * detail, 0, 1)
    return np.rint(result * 255).astype(np.uint8)


def panel(images, labels):
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        tiles.append(tile)
    return np.hstack(tiles)


def main():
    cv2.setNumThreads(1)
    funcs = {
        "legacy": lambda x: enhance_frame(x, get_preset("legacy")),
        "balanced": lambda x: enhance_frame(x, get_preset("balanced")),
        "bounded4": lambda x: candidate(x, 4),
        "bounded8": lambda x: candidate(x, 8),
        "natural": lambda x: enhance_frame(x, get_preset("natural")),
    }
    seen = set()
    for path in sorted((ROOT / "examples/single").glob("*.tif*")) + sorted((ROOT / "examples/batch/raw").glob("*.tif*")):
        raw = cv2.imread(str(path), -1)
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        images = []
        for name, fn in funcs.items():
            out = fn(raw)
            folder = OUT / name
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / (path.stem + ".png")), out)
            images.append(out)
        cv2.imwrite(str(OUT / (path.stem + "_full.png")), panel(images, funcs))
        crops = [x[256:512, 160:480] for x in images]
        cv2.imwrite(str(OUT / (path.stem + "_crop.png")), panel(crops, funcs))
    print(f"Rendered {len(seen)} scenes to {OUT}")


if __name__ == "__main__":
    main()
