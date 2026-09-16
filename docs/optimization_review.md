# 修复与视觉优化记录

本轮在 `9f6f774` 基础上修复评测错配、legacy 可视化、输入尺度约束和小图直方图退化。新增可选 `natural`，默认 balanced 和 legacy 的正常尺寸 uint16 输出风格保留。legacy 的小图平台阈值下限改为 1，避免整数截断把直方图清零。

## 视觉决定

执行 `python -m experiments.optimize_visual` 生成 12 张去重样例的原尺寸 PNG、全图拼图和固定路面裁剪。实际打开了全部 12 个场景的全图或原尺寸局部对照，并再次查看正式 natural 的道路局部与原始测试图。

- 道路候车亭、跟车 315/375 帧：balanced 发白，natural 恢复路面层次；legacy 的细纹更锐利，natural 更柔和。没有沿用上一轮两级粗基础层直接平台均衡的方案。
- 745/763/772 斑马线场景：natural 避免 balanced 的亮色颗粒，强映射 bounded8 的颗粒和灰度起伏更突出；选择 bounded4。
- 816/921/981/1041 街道场景：natural 保留道路裂纹、车辆和行人轮廓，天空不再整体抬亮。没有将更高对比的 bounded8 作为正式预设。
- Zenmuse 护栏：natural 的护栏亮暗结构和暗部枝条仍可辨，较 balanced 克制。
- original_16bit 测试图：balanced 的方框和条纹周围有明显描边；natural 去掉可见描边，但弱条纹也更温和。这是刻意保留的取舍，而非声称所有细节都更强。

最终保留温和候选为 `natural`，不改变默认，不宣称优于所有场景的 legacy。未做盲评、外部留出集或连续视频验证；本轮样例用于视觉选择，不能视作独立泛化证据。

## 算法

natural 使用输入鲁棒归一化，从局部方差最低四分位区域估计高通残差 MAD；以该噪声尺度控制小半径引导滤波。1024-bin 直方图的概率峰值受限，剩余概率均匀分配，CDF 插值后与线性基础层按 0.8/0.2 混合。细节采用保留幅度的软阈值，默认增益 1.5，不除以噪声估计值，不做输出百分位二次拉伸。

概率再分配后的 CDF 斜率小于 5，混合映射斜率小于 4.2。增益界限是基础层映射性质，不是对全管线或任意 d2br 的无光晕保证。噪声估计是显示处理启发式，不是传感器标定噪声。

## 验证

- `python -m unittest discover -s tests`：12 项通过，含 8 项新增回归。
- 单图 natural、11 张 natural 批处理及评测、legacy/natural 可视化入口成功。
- `python -m experiments.validate_optimization`：正式版本对已审阅原型的最大逐图 MAE 为 0.001932/255（边界/方差实现差异）。
- OpenCV 单线程、12 张 640×512 图、每方案每图预热后计时 3 次取中位数再平均：legacy 12.14 ms，balanced 42.99 ms，natural 22.10 ms。时间不含读写。
- 合成弱目标回归用 5 个随机种子验证 natural 平均 CNR 优于 balanced；另用阶跃检查可见过冲，测试归一化浮点/uint16 输出一致性。

生成物位于被 gitignore 排除的 `reports/optimization/`，包括 `*_full.png`、`*_crop.png`、各方案原图、`validation.csv`、`batch_metrics.csv` 和图层面板。上述两个实验命令可重新生成。较强 bounded8 只在实验脚本中，未进入生产预设。

natural 模式在库内使用输入百分位与 d2br，其余 v3 参数不参与此分支；CLI 明确拒绝不支持的覆盖项。legacy 只接受 uint16 原始 DN，避免静默解释归一化浮点输入；统计基于输入而非增强输出，noise_sigma 为归一化残差估计。评测严格按相对路径去扩展名配对，并拒绝歧义、缺失和图像尺寸不一致。
