# Open DDE: Decomposition-Based Infrared Image Enhancement

[中文](README.md) | [English](README.en.md)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An infrared image enhancement project built with Python and OpenCV. The goal is to provide an advanced, practical, open-source `DDE-like` toolkit for improving high-dynamic-range thermal images on 8-bit displays.

## Background

Thermal imagery records surface temperature distributions. Raw thermal frames are often stored in `12/14/16-bit` formats, while common displays and human visual perception are limited to `8-bit` grayscale. Direct display therefore tends to look flat and washed out, with weak local contrast and poor target visibility.

The current implementation follows an `Open DDE v3-like` strategy: multi-scale edge-preserving decomposition, local detail gain, noise gating, hotspot protection, and display-oriented remapping.

## Core Idea

The enhancement pipeline follows `decompose -> process -> fuse`:

```math
I_{original} \xrightarrow{decompose} (I_{base}, I_{detail}) \xrightarrow{process} (I'_{base}, I'_{detail}) \xrightarrow{fuse} I_{enhanced}
```

1. **Decomposition**: use guided filtering to build two base layers and one multi-scale detail signal.
2. **Base processing**: compress dynamic range with adaptive log mapping and mild local contrast enhancement.
3. **Detail processing**: apply signed detail gain with clipping, edge-aware masking, and noise gating.
4. **Fusion**: combine processed base and detail, then remap to `[0, 255]`.

## Visual Comparison

| Linear Stretch | Enhanced Output |
| :------------: | :-------------: |
| ![Linear](docs/assets/stretched_8bit.png) | ![Enhanced](docs/assets/enhanced_8bit.png) |

## Installation

```bash
git clone https://github.com/SeanWong17/Infrared-Image-Enhancement.git
cd Infrared-Image-Enhancement
pip install -e .
```

## Usage

The repository includes a single-image example at `examples/single/original_16bit.tif` and a batch dataset under `examples/batch/raw/`.

```bash
# Enhance one image
ir-dde-enhance -i examples/single/original_16bit.tif -o output/enhanced_result.png

# Use a stronger preset
ir-dde-enhance -i examples/single/original_16bit.tif -o output/enhanced_detail.png --preset detail_plus

# Batch processing
ir-dde-batch -i examples/batch/raw -o output_batch --out_ext .png --preset balanced

# Generate linear baseline images
ir-dde-linear -i examples/batch/raw -o output_linear --out_ext .png
```

## Validation

```bash
python -m unittest discover -s tests
```

## Evaluation and Visualization

```bash
# Evaluate enhanced results against a robust linear baseline
ir-dde-eval --raw_dir examples/batch/raw --enhanced_dir output_batch --csv reports/eval_metrics.csv

# Render a panel with intermediate maps
ir-dde-viz -i examples/single/original_16bit.tif -o comparisons/pipeline_panel.png
```

## Project Layout

- `ir_dde/`: core library code
- `ir_dde/cli/`: command-line entry points
- `tests/`: automated tests
- `docs/`: design notes and README assets
- `examples/single/`: single-image example
- `examples/batch/raw/`: raw thermal examples
- `examples/batch/linear/`: linear baseline examples
- `examples/batch/enhanced/`: enhanced example outputs
- `pyproject.toml`: project metadata and console script configuration

## Design Notes

- `docs/dde_formula_breakdown.md`: formula-level breakdown of DDE-like enhancement
- `docs/dde_v3_implementation_plan.md`: implementation plan for the Open DDE v3-like pipeline

## License

Released under the [MIT License](LICENSE).
