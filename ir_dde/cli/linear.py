import argparse
from pathlib import Path

import cv2
import numpy as np


def linear_normalize_to_u8(image: np.ndarray) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    lo = float(src.min())
    hi = float(src.max())
    if hi - lo < 1e-6:
        return np.zeros_like(src, dtype=np.uint8)
    normalized = np.clip((src - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def iter_tiff_files(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            yield path


def convert_folder(input_dir: Path, output_dir: Path, out_ext: str) -> tuple[int, int]:
    files = list(iter_tiff_files(input_dir))
    if not files:
        return 0, 0

    ok_count = 0
    fail_count = 0
    for index, input_path in enumerate(files, 1):
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            fail_count += 1
            print(f"[{index}/{len(files)}] ❌ 无法读取: {input_path}")
            continue
        if image.ndim == 3 and image.shape[2] in (3, 4):
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, code)

        output = linear_normalize_to_u8(image)
        output_path = output_dir / input_path.relative_to(input_dir).with_suffix(out_ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), output)
        if not ok:
            fail_count += 1
            print(f"[{index}/{len(files)}] ❌ 保存失败: {output_path}")
            continue

        ok_count += 1
        print(f"[{index}/{len(files)}] ✅ {input_path.name} -> {output_path.name}")
    return ok_count, fail_count


def main() -> int:
    parser = argparse.ArgumentParser(description="从热红外 TIFF 文件生成线性拉伸的 8 位基线图像。")
    parser.add_argument("-i", "--input_dir", default="examples/batch/raw", help="包含 TIFF 文件的输入目录。")
    parser.add_argument("-o", "--output_dir", default="examples/batch/linear", help="线性拉伸结果输出目录。")
    parser.add_argument("--out_ext", default=".png", choices=[".png", ".jpg", ".bmp"], help="输出图像扩展名。")
    args = parser.parse_args()

    ok_count, fail_count = convert_folder(Path(args.input_dir), Path(args.output_dir), args.out_ext)
    total = ok_count + fail_count
    if total == 0:
        print(f"⚠️ 未找到TIF文件: {args.input_dir}")
        return 0

    print(f"🎉 完成：成功 {ok_count} / 失败 {fail_count} / 总计 {total}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
