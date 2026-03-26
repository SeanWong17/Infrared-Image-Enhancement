# Open DDE v3：工程实现方案

## 目标

把当前仓库收敛成一个先进、实用、可持续演进的开源红外 `DDE-like` 项目，满足以下要求：

- 单帧增强效果稳定；
- 对 `12/14/16-bit` 热像输入都有可预测表现；
- Python 参考实现清晰、可复现；
- 支持单图、批处理、评估、可视化；
- 后续方便继续优化到更快的 CPU 或 GPU 版本。

## 1. 项目定位

当前仓库应同时面向三类用途：

1. `显示增强`
   用于把热像数据转换成更易观察的 8 位图像。
2. `分析辅助`
   在尽量不破坏整体结构的前提下，让细节更容易被人观察或复核。
3. `工程基线`
   为研究者和产品团队提供一个可解释、可修改、可验证的开源 DDE 家族实现。

这意味着项目不应再是“几份脚本 + 几张图”的松散形态，而应当是：

- 一个清晰的 Python 包；
- 一套 CLI 入口；
- 一组样例与评估工具；
- 一份可持续扩展的设计文档。

## 2. 设计原则

### 2.1 算法原则

- 细节残差必须保持有符号；
- 优先使用边缘保持分解，不做简单锐化；
- 增益必须由局部结构和场景统计共同控制；
- 平坦区域优先抑噪；
- 高亮热点应受到保护；
- 最终映射保持单调，避免假边和过强“塑料感”；
- 所有关键参数都应可解释。

### 2.2 工程原则

- 根目录保持精简；
- 逻辑集中在包内，CLI 作为薄封装；
- 单图、批处理、评估、可视化都应走同一套核心实现；
- 用最小测试保证主路径可运行；
- 样例数据和文档素材分开管理。

## 3. 当前推荐的 v3 技术路线

推荐主线如下：

- 分解器：`guided filter` 或 `fast guided filter`
- 细节结构：`双尺度残差 + 可选 DoG`
- 基础层：`自适应对数压缩 + 轻量局部对比度增强`
- 细节层：`边缘感知局部增益 + 场景自适应全局增益`
- 噪声控制：`幅值门控 + 空间门控 + 鲁棒噪声估计`
- 融合：`热点保护 + 百分位输出映射`

这条路线比继续堆 bilateral 更适合开源、维护和后续优化。

## 4. v3 管线

设输入热像为 `I`。

### 阶段 0：鲁棒归一化

```math
I_0(x) = \mathrm{clip}\left(\frac{I(x) - p_{lo}}{p_{hi} - p_{lo} + \varepsilon}, 0, 1\right)
```

建议默认值：

- `p_lo = 0.1%`
- `p_hi = 99.9%`

作用：

- 稳定不同 bit-depth 下的参数行为；
- 抑制极端值对后续统计和滤波的影响。

### 阶段 1：场景统计

构建场景统计向量：

```math
s = \{r, \sigma, H, n_{bins}, \mu_{log}, h_r\}
```

其中：

- `r`：鲁棒动态范围；
- `sigma`：全局标准差；
- `H`：熵；
- `n_bins`：占用直方图 bin 数；
- `mu_log`：对数均值亮度；
- `h_r`：高亮区域比例。

这些量用于控制：

- 基础层压缩强度；
- 细节增益强度；
- 热点保护强度；
- 最终输出拉伸范围。

### 阶段 2：多尺度边缘保持分解

推荐使用引导滤波：

```math
B_1 = GF(I_0; r_1, \epsilon_1)
```

```math
B_2 = GF(B_1; r_2, \epsilon_2)
```

构造两个残差：

```math
D_1 = I_0 - B_1
```

```math
D_2 = B_1 - B_2
```

可选地加入 DoG：

```math
D_{dog} = G_{\sigma_1}(I_0) - G_{\sigma_2}(I_0)
```

```math
D_f = \alpha_1 D_1 + \alpha_2 D_2 + \alpha_3 D_{dog}
```

默认建议：

- `D_f = 0.7 D_1 + 0.3 D_2`
- `DoG` 默认关闭，先追求稳。

### 阶段 3：边缘置信度与噪声估计

先计算局部方差：

```math
v(x) = \mathrm{Var}_{\omega}(I_0)
```

再定义边缘置信度：

```math
m(x) = \frac{v(x)}{v(x) + \epsilon_m}
```

然后从细尺度残差估计噪声：

```math
\hat{\sigma}_n = 1.4826 \cdot \mathrm{median}(|D_1 - \mathrm{median}(D_1)|)
```

如果 `MAD` 退化为零，可用局部标准差做回退估计。

这一步输出两个最重要的控制信号：

- `m(x)`：哪里更像真实细节；
- `sigma_n`：多小的残差应该被视为噪声。

### 阶段 4：基础层压缩

推荐基础层分支采用“全局压缩 + 轻量局部补偿”：

```math
B_g = T_{global}(B_2; s)
```

```math
B_l = CLAHE(B_g; c, t)
```

```math
B_c = (1 - \alpha_b) B_g + \alpha_b B_l
```

推荐解释：

- `T_global` 负责整体亮度和动态范围压缩；
- `CLAHE` 只做轻量局部修正；
- `alpha_b` 不宜过大，通常 `0.15 ~ 0.30`。

当前仓库选择的是自适应对数压缩，这是一条适合继续迭代的主线。

### 阶段 5：细节层控制

#### 5.1 对称限幅

```math
\tau(x) = k_n \hat{\sigma}_n \cdot (1 + \beta_\tau m(x))
```

```math
D_{clip}(x) = \mathrm{clip}(D_f(x), -\tau(x), \tau(x))
```

含义：

- 边缘附近允许更大的细节波动；
- 平坦区更严格抑制噪声。

#### 5.2 局部增益

```math
g_{loc}(x) = g_{min} + (g_{max} - g_{min}) m(x)^\gamma
```

推荐初值：

- `g_min = 0.15`
- `g_max = 1.20`
- `gamma = 0.8`

#### 5.3 场景自适应全局增益

```math
g_{scn} = LUT(n_{bins}, \sigma, H, h_r)
```

推荐行为：

- 低对比、低杂波场景：细节增益可以适当升高；
- 高动态、高杂波场景：细节增益应保守；
- 强热点主导场景：优先压基础层，不要再过度加细节。

#### 5.4 幅值门控与空间门控

```math
w_a(x) = \sigma(k_a(|D_f(x)| - \tau_a))
```

```math
w_s(x) = \sigma(k_s(m(x) - \tau_s))
```

最终细节项：

```math
D_c(x) = w_a(x) w_s(x) g_{scn} g_{loc}(x) D_{clip}(x)
```

这就是对 `DDE threshold` 与 `spatial threshold` 最自然的公开工程解释。

### 阶段 6：热点保护

为匹配“高幅值信号需要被抑制”的思路，建议显式加入热点保护：

```math
h(x) = 1 - \eta \cdot \sigma\left(\frac{B_2(x) - \mu_h}{\sigma_h + \varepsilon}\right)
```

```math
F(x) = B_c(x) + \lambda h(x) D_c(x)
```

含义：

- 如果某区域本身已经主导基础层亮度，就减少额外细节注入；
- 把显示预算更多地给到弱目标和弱纹理。

### 阶段 7：最终显示映射

```math
q_l, q_h = Q(F; p_l, p_h)
```

```math
O(x) = 255 \cdot \mathrm{clip}\left(\frac{F(x) - q_l}{q_h - q_l + \varepsilon}, 0, 1\right)
```

建议默认值：

- `p_l = 0.5%`
- `p_h = 99.5%`

必要时可以加入 soft-knee，但不建议默认启用。

## 5. 参数层面的产品化映射

为了让用户更容易理解，建议对外暴露的参数尽量贴近 DDE 语义：

- `d2br`
  细节相对背景的融合比例
- `detail_gain_min`
  局部细节最小增益
- `detail_gain_max`
  局部细节最大增益
- `detail_threshold`
  细节幅值门限
- `spatial_threshold`
  细节空间门限
- `base_compression`
  基础层压缩强度
- `local_contrast_mix`
  基础层局部对比度混合比例
- `hotspot_protect`
  热点保护强度
- `output_percentile_low`
  最终输出低百分位
- `output_percentile_high`
  最终输出高百分位

## 6. 当前项目结构

当前仓库已经收敛到如下结构：

```text
docs/
  assets/
  dde_formula_breakdown.md
  dde_v3_implementation_plan.md
examples/
  single/
  batch/
ir_dde/
  __init__.py
  config.py
  filters.py
  stats.py
  metrics.py
  pipeline.py
  presets.py
  tone_map.py
  cli/
    enhance.py
    batch.py
    linear.py
    evaluate.py
    visualize.py
tests/
  test_pipeline.py
  test_metrics.py
pyproject.toml
```

这个结构满足几个目标：

- 根目录保持干净；
- 核心逻辑集中在包内；
- CLI 和库代码解耦；
- 示例数据与文档素材分离。

## 7. 实施阶段

### 阶段 A：基线清理

目标：

- 将脚本式实现收口到包；
- 统一单图、批处理、评估、可视化入口；
- 形成可测试、可安装的项目形态。

当前状态：

- 已完成。

### 阶段 B：v3 主线算法

目标：

- 从单尺度残差升级到双尺度残差；
- 加入边缘感知增益；
- 加入噪声门控和热点保护；
- 使用自适应基础层压缩与最终百分位映射。

当前状态：

- 已完成首版可用实现。

### 阶段 C：评估与预设

目标：

- 增加无参考评估指标；
- 输出 CSV 报告；
- 提供中间图层可视化；
- 增加多种场景预设。

当前状态：

- 已完成基础版本；
- 还可继续增强 benchmark 和报告自动化。

### 阶段 D：性能优化

目标：

- 分析瓶颈；
- 引入 fast guided filter 或更轻量近似；
- 为大分辨率或近实时场景做优化。

当前状态：

- 尚未系统展开，是下一阶段重点。

## 8. 评估协议

当前项目建议同时看：

### 8.1 无参考数值指标

- `entropy`
- `average gradient`
- `RMS contrast`
- `EME`
- `Laplacian variance`

这些指标已经接入项目 CLI，但不能孤立解释。

### 8.2 视觉检查项

每次调参至少观察：

- 弱目标是否更清晰；
- 边缘是否出现 halo；
- 平坦区是否变脏；
- 高亮区域是否过白；
- 整体亮度是否自然；
- 细节提升是否带来可用信息，而不是噪声。

### 8.3 场景覆盖

建议至少覆盖：

- 低对比室内；
- 复杂户外；
- 热点主导；
- 植被杂波；
- 路面和车辆场景。

## 9. 推荐默认参数

下面是一组适合作为首选平衡预设的初值：

```text
input_percentile_low = 0.1
input_percentile_high = 99.9
guided_radius_fine = 6
guided_eps_fine = 1e-4
guided_radius_coarse = 18
guided_eps_coarse = 4e-4
detail_mix_fine = 0.7
detail_mix_mid = 0.3
detail_gain_min = 0.15
detail_gain_max = 1.20
detail_gain_gamma = 0.8
detail_threshold_scale = 1.4
spatial_threshold = 0.15
base_local_contrast_mix = 0.20
hotspot_protect = 0.35
output_percentile_low = 0.5
output_percentile_high = 99.5
```

这组参数追求的是：

- 不明显过亮；
- 不明显起噪；
- 细节增强适中；
- 作为其他预设的中性起点。

## 10. 预设策略

当前适合保留的预设：

- `balanced`
  通用默认预设
- `detail_plus`
  更强调细节
- `noise_safe`
  更保守，更适合低信噪比
- `hot_scene`
  更强调热点保护
- `radiometric_safe`
  尽量少做过强本地化处理

预设是非常重要的产品化层，不应让用户一上来就面对全部底层参数。

## 11. 风险与已知问题

### 风险 1：过拟合样例图

应对：

- 维持多场景样例；
- 不要只盯着一张 demo 图调参。

### 风险 2：细节增强变成噪声增强

应对：

- 幅值门控和空间门控必须同时存在；
- 噪声估计必须稳定。

### 风险 3：局部对比度过强导致“塑料感”

应对：

- 基础层中的 CLAHE 只能做轻量混合；
- 不要叠加太多局部映射。

### 风险 4：性能退化

应对：

- 以 guided filter 为主；
- bilateral 只留作历史参考，不作为主线。

## 12. 对当前仓库的直接建议

如果继续往“成熟项目”推进，最值得做的是：

1. 增加更多带来源说明的 single 示例；
2. 为 README 提供更丰富的前后对比图；
3. 输出批量 benchmark 报告页；
4. 增加更贴近任务的指标，例如目标可见性相关度量；
5. 对 CLI 增加导出中间图层和完整 report 的能力；
6. 根据场景统计自动选择更合适的预设。

## 13. 成功标准

当前仓库可以合理宣称为“成熟的开源 DDE-like 红外增强项目”，至少需要满足：

- 包结构清晰；
- CLI 入口完整；
- 文档和示例齐全；
- 评估和可视化工具可用；
- 在多类场景上相对线性拉伸、HE、CLAHE 有稳定改进；
- 噪声与细节的平衡优于简单锐化方案；
- 便于继续优化和部署。

## 参考资料

- FLIR OEM support, `What is the Digital Detail Enhancement (DDE) filter?`  
  https://oem.flir.com/en-gb/support/support-center/knowledge-base/what-is-the-digital-detail-enhancement-dde-filter2/
- FLIR OEM support, `Boson & Tau SDK - Manipulating ACE and DDE`  
  https://flir.custhelp.com/app/answers/detail/a_id/3296/~/flir-oem---boson-%26-tau-sdk---manipulating-ace-%28active-contrast-enhancement%29%2C
- FLIR technical note, `Digital Detail Enhancement (DDE)`  
  https://www.flirmedia.com/MMC/CVS/Tech_Notes/TN_0003_EN.pdf
- FLIR Photon User's Manual  
  https://support.flir.com/answers/A6003/Photon_User_manual_v111.pdf
- FLIR patent, `Image processing method for detail enhancement and noise reduction`  
  https://patents.google.com/patent/US10255662B2/en
- Photonics 2024, `Infrared Image Enhancement Based on Adaptive Guided Filter and Global-Local Mapping`  
  https://www.mdpi.com/2304-6732/11/8/717
- PMC review/application note, `A Low-Delay Dynamic Range Compression and Contrast Enhancement Algorithm Based on an Uncooled Infrared Sensor with Local Optimal Contrast`  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10649624/
- PubMed abstract, `Dynamic range compression and detail enhancement algorithm for infrared image`  
  https://pubmed.ncbi.nlm.nih.gov/25321683/
