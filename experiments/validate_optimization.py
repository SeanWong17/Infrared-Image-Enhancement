"""Compare production natural with the visually reviewed prototype and baseline."""

import csv
import hashlib
import json
import time

import cv2
import numpy as np

from ir_dde import enhance_frame, get_preset
from experiments.optimize_visual import ROOT, OUT, candidate


def main():
    cv2.setNumThreads(1)
    seen, rows = set(), []
    for path in sorted((ROOT / "examples").glob("**/*.tif*")):
        raw = cv2.imread(str(path), -1)
        if raw.dtype != np.uint16:
            continue
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        natural = enhance_frame(raw, get_preset("natural"))
        delta = np.abs(natural.astype(float) - candidate(raw).astype(float))
        for name in ("legacy", "balanced", "natural"):
            times = []
            for _ in range(4):
                start = time.perf_counter()
                output = enhance_frame(raw, get_preset(name))
                times.append((time.perf_counter() - start) * 1000)
            rows.append({"image": path.stem, "method": name, "ms": float(np.median(times[1:])), "prototype_mae": float(delta.mean()) if name == "natural" else "", "saturation_pct": float(np.mean((output == 0) | (output == 255)) * 100)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"scenes": len(seen), "max_prototype_mae": max(r["prototype_mae"] for r in rows if r["method"] == "natural"), "mean_ms": {name: np.mean([r["ms"] for r in rows if r["method"] == name]) for name in ("legacy", "balanced", "natural")}}, indent=2))


if __name__ == "__main__":
    main()
