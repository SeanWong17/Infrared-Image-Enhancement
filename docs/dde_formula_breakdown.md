# DDE and Decomposition-Based Infrared Enhancement: Formula-Level Breakdown

## Scope

This note has two goals:

1. Separate what is publicly confirmed about FLIR `DDE` from what must be inferred.
2. Build a practical mathematical model for an open-source, decomposition-based `DDE-like` algorithm.

Important boundary:

- The exact FLIR production formula is not public.
- The equations below are a synthesis of FLIR public documentation, FLIR patents, and open literature on `BF-DDE`, `GF-DDE`, `DRCDDE`, and newer decomposition methods.
- Whenever this document says `DDE-like`, it means an engineering reconstruction of the method family, not a claim of exact equivalence to FLIR firmware.

## 1. Publicly Confirmed Facts About FLIR DDE

The following points are directly supported by FLIR public materials:

- `DDE` stands for `Digital Detail Enhancement`.
- FLIR describes DDE as a `non-linear` image-processing algorithm for finding low-contrast targets in `high dynamic range` thermal scenes.
- FLIR states that DDE preserves detail in high-dynamic-range imagery and enhances the detail so that it matches the total dynamic range of the original image.
- FLIR OEM support describes DDE as a filter that enhances `high spatial frequencies` and attenuates `high-amplitude signals`, with attenuation chosen automatically from `scene statistics`.
- In the FLIR OEM SDK mapping, `DDE` corresponds to `d2br`, meaning `detail-to-background ratio`.
- The Photon manual states that DDE edge enhancement changes dynamically with scene content and refers to the number of occupied bins in the image histogram.
- The Photon manual exposes `DDE gain`, `DDE threshold`, and `spatial threshold`, which strongly suggests that the algorithm is not a single fixed sharpening operator.

These public facts imply a decomposition-style display pipeline:

- one branch manages `background` or `base` content;
- one branch manages `detail`;
- a scene-dependent controller adjusts how much detail is injected back.

## 2. Canonical DDE-Like Mathematical Framework

Let the input thermal frame be:

```math
I: \Omega \rightarrow [0, 2^B - 1]
```

where `B` is typically `12`, `14`, or `16` bits.

### 2.1 Robust Input Normalization

Before decomposition, practical systems usually remove unusable tails of the dynamic range:

```math
I_0(x) = \mathrm{clip}\left(\frac{I(x) - p_{lo}}{p_{hi} - p_{lo} + \varepsilon}, 0, 1\right)
```

- `p_lo`, `p_hi` are robust percentiles such as `0.1%` and `99.9%`.
- This is not the display mapping yet.
- It only stabilizes downstream filtering and statistics.

### 2.2 Edge-Preserving Decomposition

The base layer is obtained with an edge-preserving low-pass filter:

```math
B(x) = \mathcal{L}(I_0; \theta_{lp})
```

Typical choices:

- bilateral filter;
- guided filter;
- fast guided filter;
- local edge-preserving filter;
- weighted least squares filter.

The detail layer is the residual:

```math
D(x) = I_0(x) - B(x)
```

Many papers add a second smoothing step on the base before differencing:

```math
D(x) = I_0(x) - G_{\sigma}(B)(x)
```

This is exactly the logic already present in your current repository: bilateral decomposition, then subtract a lightly Gaussian-smoothed base to suppress gradient artifacts.

### 2.3 Optional Multi-Band Detail

A stronger modern design separates medium-scale and fine-scale detail:

```math
B_2 = \mathcal{L}(B; \theta_{coarse}), \quad D_1 = I_0 - B, \quad D_2 = B - B_2
```

or augments the residual with a band-pass operator such as `DoG`:

```math
D_{dog} = G_{\sigma_1}(I_0) - G_{\sigma_2}(I_0), \quad D_f = \alpha D + \beta D_{dog}
```

This is the direction used by recent `adaptive guided filter + DoG` work.

## 3. What "detail-to-background ratio" Means Mathematically

The SDK name `d2br` is the clearest public clue about DDE internals.

At a formula level, the most natural interpretation is:

```math
F(x) = B_c(x) + \lambda(x) \, D_c(x)
```

where:

- `B_c` is the compressed background or base;
- `D_c` is the controlled detail signal;
- `lambda(x)` or a global `lambda` is the effective `detail-to-background ratio`.

In other words, DDE is not just "make edges stronger"; it is "allocate display range between background and detail in a scene-adaptive way".

## 4. Base-Branch Processing

The base branch is responsible for dynamic-range compression and global brightness placement.

### 4.1 Generic Base Compression

```math
B_c = T_b(B; s)
```

where `s` is a scene statistic, for example:

- occupied histogram bins;
- entropy;
- global standard deviation;
- robust range;
- log-average intensity.

Common forms for `T_b`:

- piecewise gamma mapping;
- histogram equalization or plateau equalization;
- CLAHE;
- logarithmic or sigmoid mapping;
- global-local mixed mapping.

### 4.2 Why Base Compression Matters in DDE

FLIR public wording says DDE attenuates high-amplitude signals so that more display range becomes available for faint targets and details. In formula terms, this means the base branch cannot remain linear in high-dynamic scenes.

An open implementation should therefore treat base compression as mandatory rather than optional:

```math
B_c = \psi(B), \quad \psi'(u) < 1 \text{ in high-amplitude regions}
```

This reduces domination by hot backgrounds, warm roofs, sky gradients, or sun-heated clutter.

## 5. Detail-Branch Processing

The detail branch is where decomposition methods differ most.

### 5.1 Symmetric Detail Clipping

Practical DDE-like systems should keep detail signed:

```math
D_{clip}(x) = \mathrm{clip}(D(x), -\tau(x), \tau(x))
```

This is critical.

- If the detail branch is mapped to only positive values, the whole image gets a brightness bias.
- Your `enhance.py` had this issue.
- Your `enhance_v2.py` already fixed this by preserving positive and negative detail contributions.

### 5.2 Local Gain Control

The gain should be lower in flat regions and higher near real texture or edges:

```math
g(x) = g_{min} + (g_{max} - g_{min}) \cdot m(x)^\gamma
```

where `m(x)` is a local edge-confidence or masking term.

A strong choice from guided-filter literature is:

```math
m(x) = \frac{\sigma_\omega^2(x)}{\sigma_\omega^2(x) + \epsilon(x)}
```

Interpretation:

- in flat regions, local variance is low, so `m(x)` is close to `0`;
- near structure and edges, local variance is high, so `m(x)` approaches `1`.

This gives:

```math
D_g(x) = g(x) \cdot D_{clip}(x)
```

### 5.3 Scene-Adaptive Global Gain

FLIR public documentation says the amount of attenuation or enhancement is selected using scene statistics.

A DDE-like open version can model this with:

```math
q(s) = \mathrm{LUT}(s)
```

and:

```math
D_g(x) = q(s) \cdot g(x) \cdot D_{clip}(x)
```

Example scene statistics:

- `s = occupied_histogram_bins(I_0)`;
- `s = entropy(I_0)`;
- `s = std(I_0)`;
- `s = robust_range(I_0)`.

This matches both:

- FLIR's histogram-bin description in the Photon manual;
- the FLIR patent family describing a weighting derived from an information measure such as standard deviation, entropy, or edge measure.

### 5.4 Threshold and Spatial Threshold

The Photon manual exposes `DDE threshold` and `spatial threshold`.
At a formula level, these are naturally modeled as:

```math
w_a(x) = \mathbb{1}(|D(x)| > \tau_a)
```

```math
w_s(x) = \mathbb{1}(m(x) > \tau_s)
```

or in soft form:

```math
w_a(x) = \sigma(k_a(|D(x)| - \tau_a))
```

```math
w_s(x) = \sigma(k_s(m(x) - \tau_s))
```

Then:

```math
D_c(x) = w_a(x) \cdot w_s(x) \cdot D_g(x)
```

Interpretation:

- `amplitude threshold` rejects tiny fluctuations likely to be noise;
- `spatial threshold` rejects detail injection in texture-poor or visually sensitive flat areas.

## 6. Fusion

The fused image is:

```math
F(x) = B_c(x) + \lambda \, D_c(x)
```

or multi-band:

```math
F(x) = B_c(x) + \lambda_1 D_{1,c}(x) + \lambda_2 D_{2,c}(x)
```

To protect bright hotspots and avoid washed-out output, it is often beneficial to add a hot-region suppressor:

```math
h(x) = 1 - \eta \, \sigma\left(\frac{B(x) - \mu_h}{\sigma_h + \varepsilon}\right)
```

and:

```math
F(x) = B_c(x) + h(x)\lambda D_c(x)
```

This is an engineering extension, not something directly disclosed by FLIR, but it is consistent with the public statement that large-amplitude signals should be attenuated.

## 7. Final Display Mapping

The fused image is still not necessarily suitable for `8-bit` display. A final monotone remap is required:

```math
O(x) = T_o(F(x))
```

Practical choices:

- linear clip to `[0, 255]`;
- percentile stretch;
- monotone sigmoid;
- global-local hybrid mapping;
- light CLAHE after fusion.

The safest open-source default is usually:

```math
O(x) = 255 \cdot \mathrm{clip}\left(\frac{F(x) - q_{1\%}}{q_{99\%} - q_{1\%} + \varepsilon}, 0, 1\right)
```

with an optional local-contrast correction.

## 8. Unified DDE-Like Equation

A practical open-source `DDE-like` master equation can be written as:

```math
\begin{aligned}
I_0 &= \mathrm{robust\_normalize}(I) \\
B &= \mathcal{L}(I_0; \theta_{lp}) \\
D &= I_0 - \mathcal{S}(B) \\
s &= \phi(I_0) \\
B_c &= T_b(B; s) \\
m(x) &= \frac{\sigma_\omega^2(x)}{\sigma_\omega^2(x) + \epsilon(x)} \\
D_c(x) &= w_a(x) \, w_s(x) \, q(s) \, g(x) \, \mathrm{clip}(D(x), -\tau(x), \tau(x)) \\
F(x) &= B_c(x) + h(x)\lambda D_c(x) \\
O(x) &= T_o(F(x))
\end{aligned}
```

This equation captures the main structure of the DDE family:

- edge-preserving decomposition;
- base compression;
- signed detail enhancement;
- scene-adaptive detail weighting;
- final display remapping.

## 9. Where the Main Literature Variants Differ

### 9.1 BF-DRP

Canonical form:

```math
B = BF(I), \quad D = I - B, \quad O = N(\Gamma_b(B) + k \Gamma_d(D))
```

Main issue:

- sharp local detail;
- prone to gradient reversal and noise lift.

### 9.2 BF-DDE

Canonical form:

```math
B = BF(I), \quad D = I - G(B), \quad O = N(T_b(B) + kD)
```

Main improvement over plain BF-DRP:

- detail is taken against a corrected or smoothed base to reduce artifacts;
- stronger practical detail enhancement;
- still expensive and not ideal for real time.

### 9.3 GF-DDE

Canonical form:

```math
B = GF(I), \quad D = I - B, \quad O = N(T_b(B) + g(x)D)
```

Main trade-off:

- lower complexity;
- fewer bilateral artifacts;
- usually weaker micro-detail than BF-DDE.

### 9.4 DRCDDE

Canonical form:

```math
B = BF(I), \quad D = I - B, \quad B_c = T_{adp}(B), \quad D_c = G_{adp}(D), \quad O = N(B_c + D_c)
```

Main idea:

- treat dynamic-range compression and detail enhancement as separate but coupled control problems.

### 9.5 Newer AGF + DoG + Global-Local Mapping

Canonical form:

```math
B = AGF(I), \quad D_f = (I - B) + \beta DoG(I), \quad O = T_{glm}(B + g(x)D_f)
```

Main idea:

- scene-adaptive decomposition;
- edge-aware detail gain;
- better global-local contrast balance.

## 10. Failure Modes and Their Formula-Level Causes

### 10.1 Halo

Cause:

- low-pass filter crosses strong edges;
- large gain is applied to a residual that already contains edge bias.

Mitigation:

- guided filter or edge-aware filter instead of naive smoothing;
- subtract `G(B)` rather than `B` when needed;
- reduce gain near very large gradients.

### 10.2 Gradient Reversal

Cause:

- residual shape no longer matches the true edge shape after filtering.

Mitigation:

- tighter edge-preserving decomposition;
- multi-band detail instead of one large residual;
- bounded gain and soft clipping.

### 10.3 Noise Lift in Flat Regions

Cause:

```math
g(x) \approx const
```

instead of spatially selective gain.

Mitigation:

```math
g(x) \uparrow \text{ on edges}, \quad g(x) \downarrow \text{ in flat regions}
```

### 10.4 Global Brightness Bias

Cause:

```math
D \rightarrow [0, D_{max}]
```

instead of keeping detail signed.

Mitigation:

```math
D \rightarrow [-D_{max}, D_{max}]
```

### 10.5 Over-Whitening of Hot Regions

Cause:

- the base branch uses too much of the display range for already dominant signals.

Mitigation:

- compress the base;
- protect highlights;
- inject detail preferentially where the base is not already saturated.

## 11. Mapping This Back to the Current Repository

Your current code already sits inside the DDE family:

- `enhance.py`
  - `B = bilateral(I)`
  - `D = I - Gaussian(B)`
  - histogram-based base compression
  - detail fusion
- `enhance_v2.py`
  - keeps signed detail contribution
  - introduces explicit detail amplitude control
  - moves closer to a proper DDE-like implementation

The biggest remaining gaps relative to a strong `DDE v3` are:

- the gain is still mostly global rather than local edge-aware;
- the decomposition is still single-scale;
- the base mapping is not yet a robust global-local hybrid;
- scene-statistics control is still shallow;
- hotspot attenuation is not explicit.

## 12. Recommended Open-Source Interpretation of DDE

If the goal is an advanced and practical open-source DDE repository, the most defensible public framing is:

`DDE-like thermal enhancement = edge-preserving decomposition + adaptive base compression + scene-aware signed detail injection + monotone output remapping`

That framing is both:

- faithful to public FLIR descriptions;
- broad enough to incorporate the strongest open literature.

## Sources

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
