# DDE 与分解类红外增强：公式级拆解

## 适用范围

本文档有两个目的：

1. 把公开资料里能够确认的 `FLIR DDE` 信息和必须靠工程推断的部分明确分开。
2. 给当前仓库构建一个可实现、可解释、可开源复现的 `DDE-like` 数学框架。

边界先说清楚：

- FLIR 量产固件的精确公式没有公开。
- 本文中的公式是根据 FLIR 官方说明、专利、以及 `BF-DDE`、`GF-DDE`、`DRCDDE` 与近年分解类论文抽象出来的统一表达。
- 文中凡是出现 `DDE-like`，都表示“属于同一家族的公开工程实现”，不是声称和 FLIR 固件逐项一致。

## 1. FLIR DDE 公开可确认的信息

根据 FLIR 官方文档和 SDK 页面，可以确认这些事实：

- `DDE` 即 `Digital Detail Enhancement`。
- FLIR 将 DDE 描述为一种用于高动态热像场景、便于发现低对比目标的 `非线性` 图像处理算法。
- DDE 会保留高动态场景中的细节，并把这些细节增强到适合整体显示动态范围的程度。
- FLIR OEM 支持页明确提到：DDE 会增强 `高空间频率`，同时衰减 `高幅值信号`，而衰减量由 `场景统计量` 自动决定。
- 在 OEM SDK 的参数映射里，DDE 对应 `d2br`，即 `detail-to-background ratio`。
- Photon 手册提到：DDE 的边缘增强会随着场景动态变化，并与图像直方图中被占用的 bin 数量有关。
- Photon 手册还暴露了 `DDE gain`、`DDE threshold`、`spatial threshold` 等控制量，这说明它不是单一固定的锐化算子。

这些公开事实共同指向同一件事：DDE 更像是一套“基础层 / 细节层 / 场景自适应控制”的显示管线，而不是简单的 unsharp mask。

## 2. DDE-like 的统一数学框架

设输入热像为：

```math
I: \Omega \rightarrow [0, 2^B - 1]
```

其中 `B` 通常是 `12`、`14` 或 `16`。

### 2.1 鲁棒归一化

分解前通常先做鲁棒归一化，用于稳定后续滤波和统计：

```math
I_0(x) = \mathrm{clip}\left(\frac{I(x) - p_{lo}}{p_{hi} - p_{lo} + \varepsilon}, 0, 1\right)
```

说明：

- `p_lo`、`p_hi` 可取 `0.1%` 和 `99.9%`；
- 这一步不是最终显示映射；
- 目的只是抑制极端尾部像素对后续模块的影响。

### 2.2 边缘保持分解

基础层由边缘保持低通算子给出：

```math
B(x) = \mathcal{L}(I_0; \theta_{lp})
```

常见的 `L` 包括：

- 双边滤波；
- 引导滤波；
- 快速引导滤波；
- 局部边缘保持滤波；
- 加权最小二乘滤波。

细节层是残差：

```math
D(x) = I_0(x) - B(x)
```

很多文献还会对基础层再做一次轻微平滑，再求差分：

```math
D(x) = I_0(x) - \mathcal{S}(B)(x)
```

这样做的目的是减少梯度反转和局部伪边。你仓库早期版本里“`bilateral + Gaussian` 再相减”的逻辑，本质上就属于这一类。

### 2.3 多尺度细节

更稳定的做法是把单个残差拆成两个尺度：

```math
B_1 = \mathcal{L}(I_0; \theta_1), \quad B_2 = \mathcal{L}(B_1; \theta_2)
```

```math
D_1 = I_0 - B_1, \quad D_2 = B_1 - B_2
```

也可以加入 `DoG` 形式的带通补偿：

```math
D_{dog} = G_{\sigma_1}(I_0) - G_{\sigma_2}(I_0)
```

```math
D_f = \alpha_1 D_1 + \alpha_2 D_2 + \alpha_3 D_{dog}
```

近年的 `adaptive guided filter + DoG` 路线大体就是这个范式。

## 3. `d2br` 的公式含义

从公开接口名看，`d2br` 是理解 DDE 的关键。

最自然的数学解释是：

```math
F(x) = B_c(x) + \lambda(x) D_c(x)
```

其中：

- `B_c` 是被压缩和重映射后的基础层；
- `D_c` 是被限制和加权后的细节层；
- `lambda(x)` 或全局 `lambda` 就是“细节相对背景”的融合权重。

这说明 DDE 的目标不是“让边更硬”，而是“在有限显示动态范围内，把背景和细节的资源分配做得更合理”。

## 4. 基础层分支

基础层负责三件事：

- 控制整体亮度；
- 压缩过宽动态范围；
- 为细节层让出可显示空间。

可统一写成：

```math
B_c = T_b(B; s)
```

这里的 `s` 是场景统计量，例如：

- 占用直方图 bin 数；
- 熵；
- 全局标准差；
- 鲁棒动态范围；
- 对数均值亮度。

常见 `T_b`：

- 对数压缩；
- 分段 gamma；
- 平台均衡；
- CLAHE；
- 全局与局部的混合单调映射。

FLIR 文档里“衰减高幅值信号，为弱小目标腾出显示空间”的说法，正对应这一分支：

```math
\psi'(u) < 1 \quad \text{在高幅值区域}
```

也就是说，高亮热点、热路面、热天空、热背景不能继续线性占满显示资源。

## 5. 细节层分支

细节层分支是分解类算法差异最大的地方。

### 5.1 保持有符号细节

细节应该保持正负号：

```math
D_{clip}(x) = \mathrm{clip}(D(x), -\tau(x), \tau(x))
```

这是非常关键的工程点：

- 如果把细节层归一化成纯正值，整幅图会被系统性抬亮；
- 你仓库的早期版本就踩过这个坑；
- `v2` 和现在的 `v3-like` 实现已经修正了这一点。

### 5.2 局部细节增益

增益不应该是常数，而应根据局部结构强度自适应：

```math
g_{loc}(x) = g_{min} + (g_{max} - g_{min}) m(x)^\gamma
```

其中 `m(x)` 可以是边缘置信度。常见定义之一：

```math
m(x) = \frac{\sigma_\omega^2(x)}{\sigma_\omega^2(x) + \epsilon(x)}
```

解释：

- 平坦区方差低，`m(x)` 接近 `0`；
- 结构区方差高，`m(x)` 接近 `1`。

于是：

```math
D_g(x) = g_{loc}(x) \cdot D_{clip}(x)
```

### 5.3 场景自适应全局增益

FLIR 文档明确提到增强或衰减量由场景统计决定，因此还需要一个全局调节项：

```math
q(s) = \mathrm{LUT}(s)
```

```math
D_g(x) = q(s) \cdot g_{loc}(x) \cdot D_{clip}(x)
```

`s` 可以选：

- `occupied_histogram_bins(I_0)`；
- `entropy(I_0)`；
- `std(I_0)`；
- `robust_range(I_0)`。

这同时对应：

- Photon 手册里关于直方图占用 bin 的描述；
- FLIR 专利里使用标准差、熵、边缘信息构造权重的思路。

### 5.4 幅值门控和空间门控

既然官方接口里有 `DDE threshold` 和 `spatial threshold`，那么最合理的公开实现是：

```math
w_a(x) = \sigma(k_a(|D(x)| - \tau_a))
```

```math
w_s(x) = \sigma(k_s(m(x) - \tau_s))
```

于是：

```math
D_c(x) = w_a(x) \cdot w_s(x) \cdot D_g(x)
```

解释：

- `幅值门控`：抑制过小残差，避免噪声被放大；
- `空间门控`：抑制平坦区或低结构区的细节注入。

## 6. 融合与热点保护

最基本的融合为：

```math
F(x) = B_c(x) + \lambda D_c(x)
```

如果要匹配“热点要被抑制”的 DDE 风格，可以加入热点保护：

```math
h(x) = 1 - \eta \cdot \sigma\left(\frac{B(x) - \mu_h}{\sigma_h + \varepsilon}\right)
```

```math
F(x) = B_c(x) + h(x)\lambda D_c(x)
```

这不是 FLIR 逐式公开的原话，但它非常符合官方描述：当某些区域已经占据了很强的基础层能量时，不应该再给它们过多细节预算。

## 7. 最终显示映射

融合后的结果通常还不是最终 8 位显示图，需要做单调映射：

```math
O(x) = T_o(F(x))
```

比较稳妥的形式是百分位拉伸：

```math
O(x) = 255 \cdot \mathrm{clip}\left(\frac{F(x) - q_{1\%}}{q_{99\%} - q_{1\%} + \varepsilon}, 0, 1\right)
```

也可以附加轻量 soft-knee 或局部微调，但不要把最终显示映射做成过强的局部非单调操作，否则容易产生假边和“塑料感”。

## 8. DDE-like 统一总公式

把上面各模块合并，可得到一个适合开源实现的主公式：

```math
\begin{aligned}
I_0 &= \mathrm{robust\_normalize}(I) \\
B &= \mathcal{L}(I_0; \theta_{lp}) \\
D &= I_0 - \mathcal{S}(B) \\
s &= \phi(I_0) \\
B_c &= T_b(B; s) \\
m(x) &= \frac{\sigma_\omega^2(x)}{\sigma_\omega^2(x) + \epsilon(x)} \\
D_c(x) &= w_a(x) w_s(x) q(s) g_{loc}(x) \mathrm{clip}(D(x), -\tau(x), \tau(x)) \\
F(x) &= B_c(x) + h(x)\lambda D_c(x) \\
O(x) &= T_o(F(x))
\end{aligned}
```

这就是当前仓库应当坚持的公开表达：

- 边缘保持分解；
- 基础层压缩；
- 保持有符号的细节增强；
- 场景自适应控制；
- 最终显示映射。

## 9. 主要文献分支的差异

### 9.1 BF-DRP

典型形式：

```math
B = BF(I), \quad D = I - B, \quad O = N(\Gamma_b(B) + k\Gamma_d(D))
```

特点：

- 细节提升明显；
- 容易出现梯度反转和噪声抬升；
- 实时性一般。

### 9.2 BF-DDE

典型形式：

```math
B = BF(I), \quad D = I - G(B), \quad O = N(T_b(B) + kD)
```

特点：

- 在基础层和细节层之间做更细致的耦合；
- 比朴素 BF-DRP 更接近工程可用；
- 依然偏重，且对参数敏感。

### 9.3 GF-DDE

典型形式：

```math
B = GF(I), \quad D = I - B, \quad O = N(T_b(B) + g(x)D)
```

特点：

- 速度更快；
- 梯度伪影更少；
- 是当前仓库更值得坚持的路线。

### 9.4 DRCDDE

典型形式：

```math
B = BF(I), \quad D = I - B, \quad B_c = T_{adp}(B), \quad D_c = G_{adp}(D), \quad O = N(B_c + D_c)
```

特点：

- 把“基础层压缩”和“细节层增强”明确拆成两个优化问题；
- 更接近现代工程实现思路。

### 9.5 自适应引导滤波 + DoG + 全局局部映射

典型形式：

```math
B = AGF(I), \quad D_f = (I - B) + \beta DoG(I), \quad O = T_{glm}(B + g(x)D_f)
```

特点：

- 自适应性更强；
- 更适合做当前仓库的公开主线版本。

## 10. 常见失败模式

### 10.1 Halo

原因：

- 低通分解跨过强边缘；
- 边缘附近增益过大。

应对：

- 优先引导滤波或其它边缘保持滤波；
- 使用多尺度残差；
- 在强梯度处限制增益。

### 10.2 梯度反转

原因：

- 残差形状和真实边缘不匹配；
- 基础层过度平滑。

应对：

- 改进分解器；
- 使用更温和的残差重建；
- 控制细节幅度。

### 10.3 平坦区域噪声抬升

原因：

```math
g(x) \approx 常数
```

应对：

```math
g(x) \uparrow \text{在结构区}, \quad g(x) \downarrow \text{在平坦区}
```

### 10.4 整体偏亮

原因：

```math
D \rightarrow [0, D_{max}]
```

而不是：

```math
D \rightarrow [-D_{max}, D_{max}]
```

### 10.5 热点过白

原因：

- 基础层分支没有有效压缩；
- 热点区域又叠加了过强细节。

应对：

- 热点保护；
- 基础层对数压缩；
- 限制高亮区域细节注入。

## 11. 与当前仓库实现的对应关系

当前仓库的 `Open DDE v3-like` 实现已经具备以下核心模块：

- 基于引导滤波的多尺度分解；
- 有符号细节残差；
- 局部方差驱动的边缘置信度；
- 噪声估计和细节门控；
- 基础层自适应对数压缩；
- 高亮热点保护；
- 最终百分位显示映射。

与“更成熟的 DDE-like 开源实现”相比，还可以继续强化：

- 更强的场景分类和自适应预设；
- 更系统的 benchmark 报告；
- 更多与任务相关的评估指标；
- 更丰富的中间图层导出和报告生成能力。

## 12. 开源实现建议的表述

如果要把当前仓库对外描述得既准确又稳妥，最推荐的表述是：

`DDE-like thermal enhancement = 边缘保持分解 + 自适应基础层压缩 + 场景感知的有符号细节注入 + 单调显示映射`

这种描述：

- 对 FLIR 官方公开信息是忠实的；
- 对开源工程实现也是可验证、可维护的。

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
