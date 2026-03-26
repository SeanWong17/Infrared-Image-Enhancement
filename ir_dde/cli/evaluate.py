import argparse
import csv
from pathlib import Path

import cv2

from ir_dde import evaluate_against_linear_baseline


def find_raw_matches(raw_dir: Path, enhanced_dir: Path) -> list[tuple[Path, Path]]:
    enhanced_by_stem = {}
    for path in enhanced_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            enhanced_by_stem[path.stem] = path

    pairs: list[tuple[Path, Path]] = []
    for raw_path in sorted(raw_dir.rglob("*")):
        if raw_path.is_file() and raw_path.suffix.lower() in {".tif", ".tiff"}:
            candidate = enhanced_by_stem.get(raw_path.stem)
            if candidate is not None:
                pairs.append((raw_path, candidate))
    return pairs


def evaluate_folder(raw_dir: Path, enhanced_dir: Path) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for raw_path, enhanced_path in find_raw_matches(raw_dir, enhanced_dir):
        raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_UNCHANGED)
        if raw is None or enhanced is None:
            continue
        metrics = evaluate_against_linear_baseline(raw, enhanced)
        metrics["name"] = raw_path.stem
        results.append(metrics)
    return results


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name"] + [key for key in rows[0].keys() if key != "name"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_keys = [key for key in rows[0].keys() if key != "name"]
    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        summary[key] = sum(values) / len(values)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="将增强结果与鲁棒线性拉伸基线进行评估比较。")
    parser.add_argument("--raw_dir", default="examples/batch/raw", help="包含原始 TIFF 图像的目录。")
    parser.add_argument("--enhanced_dir", required=True, help="包含增强结果的目录。")
    parser.add_argument("--csv", default="reports/eval_metrics.csv", help="CSV 输出路径。")
    args = parser.parse_args()

    rows = evaluate_folder(Path(args.raw_dir), Path(args.enhanced_dir))
    if not rows:
        print("⚠️ 未找到匹配的原图与增强结果对。")
        return 1

    write_csv(rows, Path(args.csv))
    summary = summarize(rows)
    print(f"✅ 已为 {len(rows)} 组图像写出评估结果: {args.csv}")
    print(
        "   "
        f"entropy_gain={summary['entropy_gain']:.3f}, "
        f"avg_gradient_gain={summary['avg_gradient_gain']:.3f}, "
        f"eme_gain={summary['eme_gain']:.3f}, "
        f"laplacian_var_gain={summary['laplacian_var_gain']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
