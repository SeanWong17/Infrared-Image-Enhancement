import argparse
import time
from pathlib import Path

from ir_dde import batch_enhance, get_preset


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch enhance infrared TIFF files with the Open DDE v3-like pipeline.")
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

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = get_preset(args.preset)

    print(f"📂 开始处理: {input_dir}")
    print(f"📁 输出目录: {output_dir.resolve()}")
    print(f"⚙️  预设: {args.preset}")
    print("-" * 60)

    start = time.time()
    ok_count, fail_count = batch_enhance(input_dir, output_dir, config=config, out_ext=args.out_ext)
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
