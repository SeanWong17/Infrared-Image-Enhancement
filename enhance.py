import cv2
import numpy as np
import os
import argparse

def apply_decomposition_enhancement(input_path, output_path):
    """
    一种基于“基础层-细节层”分解的红外图像增强算法。

    Args:
        input_path (str): 输入的16位TIF图像文件路径。
        output_path (str): 处理后要保存的8位图像文件路径。
    """
    print("--- 开始运行'基础层-细节层'分解增强算法 ---")

    # 0. 加载图像并转换为支持的浮点数类型
    print(f"步骤 0: 加载图像 '{os.path.basename(input_path)}' 并准备数据...")
    img_16bit = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_16bit is None:
        print(f"错误: 无法读取图像文件 '{input_path}'。请确保文件存在且格式正确。")
        return
    
    if img_16bit.dtype != 'uint16':
        print(f"警告: 输入图像不是16位 (dtype is {img_16bit.dtype})。算法可能无法达到预期效果。")

    original_img_f32 = img_16bit.astype(np.float32)
    height, width = original_img_f32.shape

    # --- 阶段 1: 分解图像 ---
    print("阶段 1.1: 应用双边滤波获取基础层...")
    # d: 邻域直径, sigmaColor: 颜色空间标准差, sigmaSpace: 坐标空间标准差
    base_layer_bf = cv2.bilateralFilter(original_img_f32, d=9, sigmaColor=25, sigmaSpace=80)

    print("阶段 1.2: 通过高斯模糊和相减提取细节层...")
    # 对基础层再做一次轻微高斯模糊，以获得更纯净的细节
    base_layer_gauss = cv2.GaussianBlur(base_layer_bf, (3, 3), 1)
    detail_layer = original_img_f32 - base_layer_gauss

    # --- 阶段 2: 独立处理图层 ---
    print("阶段 2.1: 对基础层应用平台直方图均衡...")
    # 将基础层转回16位整型以计算直方图
    hist_input = base_layer_bf.astype('uint16')
    
    # 计算一个裁剪阈值，忽略出现次数极少的像素值 (平台高度)
    threshold = height * width * 0.0001
    hist, _ = np.histogram(hist_input.flatten(), bins=65536, range=[0, 65536])
    
    # 应用阈值，构建平台直方图
    total_pixels = hist_input.size
    clipped_hist = np.copy(hist)
    clipped_hist[clipped_hist > threshold] = threshold # 裁剪平台
    
    # 计算累积分布函数(CDF)并创建查找表(LUT)
    cdf = np.cumsum(clipped_hist)
    
    # 归一化CDF并映射到[0, 255]
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    
    lut = cdf_normalized.astype('uint8')
    
    # 应用查找表，完成基础层处理
    base_layer_processed_8bit = cv2.LUT(cv2.normalize(hist_input, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U), lut)

    # 2.2 处理细节层
    print("阶段 2.2: 裁剪并缩放细节层...")
    sigma_r = np.std(detail_layer) * 2 # 使用标准差作为自适应裁剪范围
    print(f"细节层裁剪范围 (sigma_r): {-sigma_r:.2f} to {sigma_r:.2f}")

    # 将细节裁剪到 [-sigma_r, sigma_r] 范围内
    detail_layer_clipped = np.clip(detail_layer, -sigma_r, sigma_r)
    
    # --- 阶段 3: 融合 ---
    print("阶段 3: 融合处理后的基础层和细节层...")
    # 将基础层和细节层都归一化到相似的尺度再相加
    base_normalized = cv2.normalize(base_layer_processed_8bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    detail_normalized = cv2.normalize(detail_layer_clipped, None, 0, 100, cv2.NORM_MINMAX, dtype=cv2.CV_32F) # 细节的贡献权重可以调整

    final_image_f32 = base_normalized + detail_normalized

    # --- 最后: 归一化并保存 ---
    print("最终处理: 归一化到8位并保存...")
    # 使用MINMAX归一化将最终图像拉伸到[0, 255]范围
    final_image_8bit = cv2.normalize(final_image_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv2.imwrite(output_path, final_image_8bit)
    print(f"--- 算法完成 ---")
    print(f"✅ 结果已保存至: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="基于图像分解的红外图像增强算法。")
    parser.add_argument('-i', '--input', type=str, required=True, help="输入的16位TIF图像文件路径。")
    parser.add_argument('-o', '--output', type=str, required=True, help="处理后保存的8位图像文件路径。")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在。请检查路径。")
    else:
        apply_decomposition_enhancement(args.input, args.output)
