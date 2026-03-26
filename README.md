# 基于图像分解的高动态范围红外图像增强算法

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

一个基于 Python 和 OpenCV 实现的红外图像增强项目，旨在解决高动态范围（12/14/16位）红外图像在8位显示设备上对比度低、细节不清晰的问题。

## 📖 问题背景

红外热成像捕捉的是物体表面的温度分布，其原始数据（通常为12-16位）包含了极宽的温度动态范围。然而，人眼和标准显示设备只能感知和展示有限的8位（256级）灰度。这导致原始红外图像直接显示时，往往呈现为一片灰蒙蒙的景象，目标与背景的温差细节被严重压缩，难以分辨。

本项目采用了一种基于 **“基础层-细节层”** 分解的增强策略，它能在有效压缩图像动态范围的同时，显著强化我们感兴趣的局部纹理和边缘细节。

## ⚙️ 算法核心原理

该增强算法遵循“分解 → 处理 → 融合”的三步策略：

```math
I_{original} \xrightarrow{分解} (I_{base}, I_{detail}) \xrightarrow{处理} (I'_{base}, I'_{detail}) \xrightarrow{融合} I_{enhanced}
```

1.  **图像分解 (Decomposition)**: 使用**双边滤波器 (Bilateral Filter)** 从原图 $I_{original}$ 中分离出代表大尺度、平缓变化的基础层 $I_{base}$，因为它能在平滑的同时保持边缘。随后，通过原图与平滑后的基础层相减，得到代表边缘和纹理的细节层 $I_{detail}$。
    $$I_{base} = \text{BilateralFilter}(I_{original})$$
    $$I_{detail} = I_{original} - \text{GaussianBlur}(I_{base})$$

2.  **分层处理 (Processing)**:
    * **基础层处理**: 对 $I_{base}$ 应用一种改进的**平台直方图均衡化**，有效压缩其宽动态范围，同时避免噪声的过度放大，得到 $I'_{base}$。
    * **细节层处理**: 对 $I_{detail}$ 进行带限幅的**非线性增益**，在强化有用细节的同时抑制潜在噪声，得到 $I'_{detail}$。

3.  **图像融合 (Fusion)**: 将处理后的两层相加，并通过归一化操作将像素值线性拉伸到 `[0, 255]` 区间，得到最终的8位增强图像 $I_{enhanced}$。
```math
I_{enhanced} = \text{Normalize}(\,I'_{base} + I'_{detail}\,)
```

## 📊 效果演示

为了直观展示算法效果，我们将直接线性拉伸的8位图与本算法处理后的8位图进行对比。（原始的16位TIF图像无法直接在此处显示，但已包含在 `images/` 目录下）。

| 直接线性拉伸 (Stretched 8-bit) | 本算法增强后 (Enhanced 8-bit) |
| :---------------------------------: | :----------------------------------: |
|  ![直接拉伸图](images/stretched_8bit.png)   | ![算法增强图](images/enhanced_8bit.png)  |

从对比中可以明显看出，增强后的图像无论是在场景的整体对比度，还是在云层、地物等局部细节的表现力上，都得到了显著提升。

## 🚀 如何运行

### 1. 环境准备

克隆本项目并安装所需的 Python 依赖包。

```bash
git clone https://github.com/SeanWong17/Infrared-Image-Enhancement.git
cd Infrared-Image-Enhancement
pip install -r requirements.txt
```

### 2. 准备图像

将您的 **16位单通道 TIF 格式** 红外原图放入 `images/` 目录下。项目已包含一个名为 `original_16bit.tif` 的示例文件位置。

### 3. 执行增强脚本

当前仓库已经提供一个 `Open DDE v3-like` 的基础实现：

- `enhance.py`: 单张图像增强
- `enhance_v2.py`: 文件夹批处理

默认预设是 `balanced`，另外还提供了 `detail_plus`、`noise_safe`、`hot_scene`、`radiometric_safe` 等预设。

```bash
# 单张图像增强
python enhance.py -i images/original_16bit.tif -o output/enhanced_result.png

# 使用更激进的细节增强预设
python enhance.py -i images/original_16bit.tif -o output/enhanced_detail.png --preset detail_plus

# 批处理整个目录
python enhance_v2.py -i examples/raw -o output_batch --out_ext .png --preset balanced
```

单图处理完成后，输出会保存到指定路径；批处理会保持相对目录结构写入输出目录。

### 4. 基础验证

项目包含最小 smoke test，可用于确认核心管线和单图输出正常工作：

```bash
python -m unittest discover -s tests
```

### 5. 评估与可视化

项目已经提供了无参考评估和中间结果可视化脚本：

```bash
# 评估增强结果，相对于鲁棒线性拉伸基线输出 CSV
python scripts/evaluate.py --raw_dir examples/raw --enhanced_dir output_batch --csv reports/eval_metrics.csv

# 生成算法中间图层和最终结果的拼图面板
python scripts/visualize_pipeline.py -i images/original_16bit.tif -o comparisons/pipeline_panel.png
```

### 6. 目录结构

当前建议的目录约定如下：

- `src/ir_dde/`: 核心算法实现
- `scripts/`: 评估与可视化工具
- `tests/`: 最小 smoke tests
- `docs/`: DDE 公式拆解和 v3 设计文档
- `examples/raw/`: 原始红外样例
- `examples/linear/`: 线性拉伸样例
- `examples/enhanced/`: 增强结果样例

## 展望

本算法的效果依赖于若干关键参数，如双边滤波的 `sigmaColor`、`sigmaSpace`，以及细节裁剪的 `sigma_r` 等。在实际应用中，您可以：
* **参数调优**: 根据具体的成像设备、场景内容和应用需求进行仔细的调优。
* **自适应参数**: 进一步研究自适应的参数选择策略，使算法对不同场景更具鲁棒性。
* **算法融合**: 结合其他图像处理技术，如多尺度分解（小波变换）等，探索更优的增强效果。

## 设计资料

如果您希望把当前仓库继续演进为更完整的开源 `DDE-like` 红外增强项目，可先阅读以下设计文档：

- `docs/dde_formula_breakdown.md`: DDE 与分解类红外增强的公式级拆解。
- `docs/dde_v3_implementation_plan.md`: 将当前实现升级为“类 FLIR DDE v3”的工程方案。

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。
