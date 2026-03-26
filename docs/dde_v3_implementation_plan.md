# Open DDE v3: Practical Implementation Plan

## Goal

Build this repository into a practical, advanced, open-source thermal `DDE-like` enhancement project with:

- strong single-frame enhancement quality;
- predictable behavior on `12/14/16-bit` thermal data;
- clean Python reference implementation;
- fast enough CPU path for batch use and moderate real-time use;
- room for later FPGA, C++, or CUDA acceleration.

## 1. Product Positioning

The repository should target three use cases:

1. `Display enhancement`
   For converting raw thermal frames into visually interpretable `8-bit` output.
2. `Analysis-preserving enhancement`
   For users who want better display but need to keep radiometric structure as intact as possible.
3. `Engineering baseline`
   For researchers and product teams who need an open DDE family reference instead of a black-box camera firmware feature.

This means the project should not be a single script anymore. It should become a small algorithm library with reproducible presets and diagnostics.

## 2. v3 Design Principles

### 2.1 Design Principles

- Preserve signed detail.
- Prefer edge-aware decomposition over brute-force sharpening.
- Use scene statistics to adapt gain and compression.
- Protect flat regions from noise lift.
- Protect bright or dominant thermal regions from over-whitening.
- Keep the whole pipeline monotone and debuggable.
- Expose parameters with names that match the DDE mental model.

### 2.2 v3 Technical Direction

Recommended `v3` baseline:

- decomposition filter: `guided filter` or `fast guided filter`;
- detail structure: `two-band residual` plus optional `DoG` edge boost;
- base mapping: `global-local hybrid monotone mapping`;
- gain control: `edge-aware local gain x scene-adaptive global gain`;
- noise control: `detail threshold + spatial threshold + robust noise estimate`;
- final mapping: `soft percentile remap`.

This is more realistic for a public `DDE-like` system than trying to exactly clone legacy bilateral-filter papers.

## 3. Proposed v3 Pipeline

Let `I` be the input raw thermal frame.

### Stage 0. Robust Normalization

```math
I_0(x) = \mathrm{clip}\left(\frac{I(x) - p_{lo}}{p_{hi} - p_{lo} + \varepsilon}, 0, 1\right)
```

Recommended defaults:

- `p_lo = 0.1%`
- `p_hi = 99.9%`

Purpose:

- remove useless tails;
- keep parameter behavior stable across scenes and bit depths.

### Stage 1. Scene Statistics

Compute a compact statistics vector:

```math
s = \{r, \sigma, H, n_{bins}, \mu_{log}\}
```

where:

- `r`: robust range;
- `sigma`: global standard deviation;
- `H`: entropy;
- `n_bins`: occupied histogram bins;
- `mu_log`: log-average intensity.

Use these to drive:

- detail gain;
- base compression strength;
- hotspot protection strength;
- final remap aggressiveness.

### Stage 2. Multi-Scale Edge-Preserving Decomposition

Use guided filtering:

```math
B_1 = GF(I_0; r_1, \epsilon_1)
```

```math
B_2 = GF(B_1; r_2, \epsilon_2)
```

Define two residual bands:

```math
D_1 = I_0 - B_1
```

```math
D_2 = B_1 - B_2
```

Optional edge boost:

```math
D_{dog} = G_{\sigma_1}(I_0) - G_{\sigma_2}(I_0)
```

```math
D_f = \alpha_1 D_1 + \alpha_2 D_2 + \alpha_3 D_{dog}
```

Recommended default:

- start with `D_f = 0.7 D_1 + 0.3 D_2`
- leave `DoG` disabled by default until artifacts are controlled.

Why this design:

- `D_1` handles fine texture;
- `D_2` handles medium detail and soft edges;
- this is more stable than one single large residual.

### Stage 3. Edge Awareness and Noise Visibility

Compute local variance:

```math
v(x) = \mathrm{Var}_{\omega}(I_0)
```

Define edge confidence:

```math
m(x) = \frac{v(x)}{v(x) + \epsilon_m}
```

Estimate robust noise from the fine residual:

```math
\hat{\sigma}_n = 1.4826 \cdot \mathrm{median}(|D_1 - \mathrm{median}(D_1)|)
```

This gives two necessary signals:

- `m(x)` says where detail is probably real;
- `sigma_n` says how much tiny residual energy should be ignored.

### Stage 4. Base-Branch Compression

This should replace the current simple LUT-only branch.

Use a hybrid mapping:

```math
B_g = T_{global}(B_2; s)
```

```math
B_l = CLAHE(B_g; c, t)
```

```math
B_c = (1 - \alpha_b) B_g + \alpha_b B_l
```

Recommended interpretation:

- `T_global` sets the global brightness and dynamic-range compression;
- `CLAHE` only adds restrained local contrast;
- `alpha_b` should stay modest, for example `0.15` to `0.30`.

Good `T_global` candidates:

- log or sigmoid mapping;
- plateau-limited histogram mapping;
- monotone spline driven by scene brightness zones.

Recommended first implementation:

```math
B_g = \frac{\log(1 + k_b B_2)}{\log(1 + k_b)}
```

with `k_b` adapted from `mu_log` and `n_bins`.

### Stage 5. Detail-Branch Control

#### 5.1 Symmetric clipping

```math
\tau(x) = k_n \hat{\sigma}_n \cdot (1 + \beta_\tau m(x))
```

```math
D_{clip}(x) = \mathrm{clip}(D_f(x), -\tau(x), \tau(x))
```

This gives larger allowed detail amplitude near real structure and tighter clipping in flat regions.

#### 5.2 Local detail gain

```math
g_{loc}(x) = g_{min} + (g_{max} - g_{min}) m(x)^\gamma
```

Recommended initial range:

- `g_min = 0.15`
- `g_max = 1.20`
- `gamma = 0.8`

#### 5.3 Scene-adaptive global gain

```math
g_{scn} = LUT(n_{bins}, \sigma, H)
```

Example behavior:

- high dynamic and high clutter scenes: reduce aggressive detail injection;
- low-contrast and low-clutter scenes: raise detail injection moderately;
- scenes dominated by a few hot structures: bias toward base compression rather than more detail gain.

#### 5.4 Amplitude and spatial gating

```math
w_a(x) = \sigma(k_a(|D_f(x)| - \tau_a))
```

```math
w_s(x) = \sigma(k_s(m(x) - \tau_s))
```

and:

```math
D_c(x) = w_a(x) w_s(x) g_{scn} g_{loc}(x) D_{clip}(x)
```

This is the cleanest open-source interpretation of `DDE threshold` and `spatial threshold`.

### Stage 6. Hot-Region Protection

To match FLIR's public statement that high-amplitude signals are attenuated, add an explicit protection term:

```math
h(x) = 1 - \eta \cdot \sigma\left(\frac{B_2(x) - \mu_h}{\sigma_h + \varepsilon}\right)
```

Then:

```math
F(x) = B_c(x) + \lambda \, h(x) \, D_c(x)
```

Interpretation:

- if a region is already dominant in the base layer, reduce extra detail injection there;
- spend more display budget on faint structure elsewhere.

This is one of the most important upgrades relative to a naive decomposition pipeline.

### Stage 7. Final Display Remap

Use a controlled monotone remap:

```math
q_l, q_h = Q(F; p_l, p_h)
```

```math
O(x) = 255 \cdot \mathrm{clip}\left(\frac{F(x) - q_l}{q_h - q_l + \varepsilon}, 0, 1\right)
```

Recommended defaults:

- `p_l = 0.5%`
- `p_h = 99.5%`

Optional soft-knee:

```math
O_s(x) = \frac{1}{1 + e^{-a(O(x)-b)}}
```

Use the soft-knee only as an optional preset, not as the default.

## 4. Mapping v3 Parameters to FLIR-Like Controls

Expose user-facing controls that match the DDE mental model:

- `d2br`
  Effective detail-to-background ratio. Internally maps to `lambda x g_scn`.
- `detail_gain_min`
  Minimum local detail gain.
- `detail_gain_max`
  Maximum local detail gain.
- `detail_threshold`
  Minimum residual amplitude before enhancement starts.
- `spatial_threshold`
  Minimum edge confidence before enhancement starts.
- `base_compression`
  Strength of base dynamic-range compression.
- `local_contrast_mix`
  Weight of local contrast correction in the base branch.
- `hotspot_protect`
  Strength of dominant-region attenuation.
- `output_percentile_low`
  Lower display remap percentile.
- `output_percentile_high`
  Upper display remap percentile.

This parameter surface will make the project more understandable to both users and future contributors.

## 5. Recommended Repository Refactor

Move from single scripts to a small package layout:

```text
docs/
  dde_formula_breakdown.md
  dde_v3_implementation_plan.md
src/
  ir_dde/
    __init__.py
    pipeline.py
    filters.py
    stats.py
    detail.py
    tone_map.py
    presets.py
    metrics.py
    io.py
cli/
  enhance_image.py
  enhance_folder.py
tests/
  test_pipeline.py
  test_filters.py
  test_metrics.py
scripts/
  benchmark_folder.py
  compare_presets.py
```

Recommended module responsibilities:

- `filters.py`: bilateral, guided, fast guided, DoG, utility kernels
- `stats.py`: histogram occupancy, entropy, robust range, log-average gray, local variance
- `detail.py`: clipping, gain maps, gating, hotspot protection
- `tone_map.py`: base compression and final remap
- `pipeline.py`: end-to-end algorithm graph
- `presets.py`: security, airborne, outdoor daylight, low-contrast indoor, radiometric-safe
- `metrics.py`: AG, EME, entropy, SCRG, BSF, target contrast

## 6. Implementation Phases

### Phase A. Baseline Cleanup

Replace script-only code with library code while preserving current behavior.

Tasks:

- move current `enhance_v2.py` logic into `src/ir_dde/pipeline.py`;
- add CLI wrappers for single image and folder processing;
- normalize path handling;
- add config dataclass or YAML preset loading.

Deliverable:

- reproducible `v2-baseline` package.

### Phase B. v3 Core Algorithm

Tasks:

- replace bilateral base split with guided or fast guided filter;
- add two-band residuals;
- add local variance mask;
- add robust noise estimate;
- add symmetric adaptive clipping;
- add scene-adaptive global gain;
- add hotspot protection;
- add robust percentile output remap.

Deliverable:

- first practical `v3` algorithm.

### Phase C. Evaluation and Presets

Tasks:

- add batch benchmark script;
- dump intermediate maps for debugging;
- implement presets for typical scene types;
- compare `v2`, `v3`, and classical baselines.

Deliverable:

- reproducible comparison report and example images.

### Phase D. Performance Path

Tasks:

- profile bottlenecks;
- switch to fast guided filter or integral-box approximations;
- vectorize local statistics;
- optionally add OpenCV or Numba acceleration.

Deliverable:

- real-time capable CPU path for moderate resolutions.

## 7. Evaluation Protocol

Use both visual and task-oriented metrics.

### 7.1 Objective Metrics

- `AG` average gradient
- `EME` or local contrast metrics
- entropy
- `SCRG` signal-to-clutter ratio gain
- `BSF` background suppression factor
- robust noise estimate on manually selected flat regions

Do not optimize only for entropy or AG; they can reward noisy output.

### 7.2 Visual Checklist

For each test scene inspect:

- faint target visibility;
- edge crispness;
- flat-region graininess;
- halo near strong edges;
- hot-region whitening;
- global brightness naturalness.

### 7.3 Scene Types

At minimum, collect examples for:

- low-contrast indoor
- complex outdoor daylight
- hot-target dominant
- foggy or low-SNR scene
- cluttered vegetation or rooftop scene

This matters because decomposition methods often look great on one scene and fail badly on another.

## 8. Recommended v3 Defaults

These are starting points, not final tuned values:

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
detail_threshold = 1.5 * sigma_n
spatial_threshold = 0.15
base_local_contrast_mix = 0.20
hotspot_protect = 0.35
output_percentile_low = 0.5
output_percentile_high = 99.5
```

These defaults should produce a balanced image before any scene-specific tuning.

## 9. Preset Strategy

Provide a few named presets instead of exposing every parameter first:

- `balanced`
  General-purpose default.
- `detail_plus`
  More aggressive structure enhancement for surveillance review.
- `noise_safe`
  Conservative preset for low-SNR scenes.
- `hot_scene`
  Stronger hotspot suppression.
- `radiometric_safe`
  Minimal local remap for users who mainly care about display support rather than aggressive enhancement.

This is a better public API than forcing users to understand all internal controls immediately.

## 10. Risks and Expected Failure Modes

### Risk 1. Overfitting to Sample Images

Mitigation:

- maintain a scene-diverse benchmark folder;
- never tune only on the demo image.

### Risk 2. Detail Gain Becomes Noise Gain

Mitigation:

- keep amplitude and spatial gating separate;
- use robust noise estimation from the residual.

### Risk 3. Too Much Local Contrast Makes the Image Look Synthetic

Mitigation:

- keep local contrast in the base branch low-weight;
- avoid stacking strong CLAHE with aggressive detail gain.

### Risk 4. Real-Time Performance Regresses

Mitigation:

- guided filter first;
- fast guided filter next;
- keep bilateral only as a reference baseline.

## 11. Direct Next Steps for This Repository

The fastest path from the current state is:

1. Keep `enhance_v2.py` as the reference baseline.
2. Refactor it into a package.
3. Implement a new `OpenDDEV3Config`.
4. Replace the single bilateral split with guided multi-scale decomposition.
5. Add local edge-aware gain and robust noise gating.
6. Add hotspot protection and scene-adaptive base compression.
7. Add benchmark scripts and intermediate debug output.

## 12. Definition of Success

This repository can reasonably claim to be an advanced practical open-source `DDE-like` project when it satisfies all of the following:

- clearly documented decomposition pipeline;
- reproducible presets and CLI;
- visible improvement over linear, HE, CLAHE, and current `enhance_v2.py`;
- better noise-detail balance than a naive bilateral residual method;
- code structure suitable for further optimization or deployment.

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
