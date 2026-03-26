import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ir_dde import batch_enhance, get_preset


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Open DDE v3-like infrared image enhancement.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing TIFF files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory.")
    parser.add_argument("--out_ext", default=".jpg", choices=[".jpg", ".png", ".bmp", ".tif"], help="Output format.")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=["balanced", "detail_plus", "noise_safe", "hot_scene", "radiometric_safe"],
        help="Tuning preset.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    config = get_preset(args.preset)

    print(f"📂 开始处理: {args.input_dir}")
    print(f"📁 输出目录: {os.path.abspath(args.output_dir)}")
    print(f"⚙️  预设: {args.preset}")
    print("-" * 60)

    start = time.time()
    ok_count, fail_count = batch_enhance(args.input_dir, args.output_dir, config=config, out_ext=args.out_ext)
    elapsed = time.time() - start
    total = ok_count + fail_count

    if total == 0:
        print(f"⚠️ 未找到TIF文件: {args.input_dir}")
        return 0

    print("-" * 60)
    print(f"🎉 完成：成功 {ok_count} / 失败 {fail_count} / 总计 {total}")
    print(f"⏱️ 耗时：{elapsed:.2f} 秒")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
