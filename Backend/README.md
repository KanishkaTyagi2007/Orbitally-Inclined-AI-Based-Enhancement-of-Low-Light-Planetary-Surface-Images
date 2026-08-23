#  AI-Assisted Planetary Image Enhancement Pipeline

## Problem statement

Planetary exploration missions can produce images with low illumination,
noise, and limited dynamic range. AuraNet is an AI-assisted image-enhancement
pipeline for scientific visualization that:

1. **Improves visibility** in dark, low-contrast, low-SNR planetary imagery.
2. **Minimizes hallucinated structure** — every enhancement is architecturally
   constrained or explicitly flagged, never presented as unqualified fact.
3. **Preserves scientific fidelity** — radiometric calibration and
   geospatial metadata survive the pipeline unmodified.
4. **Provides quantitative image-quality evaluation** for every processed
   scene (PSNR, SSIM, NIQE, BRISQUE, entropy gain), with an explicit
   structure-preservation guardrail.

## Why this architecture minimizes hallucination (design rationale)

Every stage below was chosen not just for restoration quality but for a
specific, inspectable reason it _can't_ invent scene content it wasn't given:

| Stage                           | Role                                                            | Anti-hallucination property                                                                                                                                                       |
| ------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `physics_frontend`              | Radiometric calibration, cosmic-ray scrubbing, Anscombe VST     | Purely physics-based, zero learned weights — every correction traces to a calibration constant                                                                                    |
| `wavelet_dequant` (SWT)         | Splits image into illumination (LL) vs. detail (LH/HL/HH) bands | Shift-invariant decomposition avoids block-edge artifacts a decimated transform would introduce                                                                                   |
| `wavelet_dequant` (dequantizer) | Removes 16-bit banding artifacts                                | Correction is hard-clamped to ±½ quantization step — architecturally incapable of inventing structure larger than one bin                                                         |
| `zero_dce`                      | Illumination enhancement (LL band only)                         | Applies only a monotonic per-pixel tone curve — cannot synthesize new spatial structure, and needs no paired "ground truth bright" data that doesn't exist for planetary scenes   |
| `nafnet_denoiser`               | Detail restoration (LH/HL/HH bands)                             | Activation-free design (SimpleGate + Simplified Channel Attention replace nonlinear MLP gates) — fewer nonlinear degrees of freedom to fabricate plausible-but-fictitious texture |
| `uncertainty_head`              | Per-pixel (μ, log-var) → trust map                              | Every output pixel ships a confidence value; low-confidence regions are surfaced to the scientist rather than silently presented as fact                                          |
| `evaluation.metrics`            | SSIM structure guardrail                                        | Flags any run where enhanced-vs-calibrated-raw structural similarity falls below a configurable threshold                                                                         |

## Project structure

```
aura_net/
├── config/
│   └── default_config.yaml          # Hyperparameters, sensor calibration gains, thresholds
├── data/
│   ├── raw/                         # Raw 16-bit PDS4 / GeoTIFF planetary images
│   └── output/                      # Multi-band enhanced GeoTIFFs + Trust maps
├── models/
│   ├── __init__.py
│   ├── physics_frontend.py          # Anscombe VST, Cosmic Ray Scrubber, Radiometric Calib
│   ├── wavelet_dequant.py           # SWT, ISWT, Implicit De-quantization
│   ├── zero_dce.py                  # Multi-order curve estimator (LL band)
│   ├── nafnet_denoiser.py           # Linear activation-free detail restorer (LH, HL, HH)
│   └── uncertainty_head.py          # Bayesian heteroscedastic (μ, log-var) head
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # PSNR, SSIM, NIQE, BRISQUE, Entropy Gain
│   └── exporter.py                  # Geospatial metadata-preserving GeoTIFF writer
├── aura_pipeline.py                 # Unified End-to-End Model & Execution Pipeline
├── requirements.txt                 # torch, rasterio, numpy, scikit-image, tifffile, ...
└── README.md
```

## Pipeline flow

```
raw 16-bit DN
  -> physics_frontend    radiometric calibration, cosmic-ray scrub, Anscombe VST
  -> wavelet_dequant      SWT decomposition -> LL + {LH,HL,HH} per level, de-quantize
  -> zero_dce             reference-free illumination curve, applied to LL only
  -> nafnet_denoiser      activation-free detail restoration, applied to LH/HL/HH
  -> wavelet_dequant      ISWT reconstruction back to a single image
  -> uncertainty_head     per-pixel (mu, log-var) -> trust map
  -> physics_frontend     inverse Anscombe VST
  -> evaluation.metrics   PSNR / SSIM / NIQE / BRISQUE / entropy gain (+ guardrail)
  -> evaluation.exporter  enhanced GeoTIFF (+ trust band) + trust map GeoTIFF sidecar
```

## Installation

```bash
cd backend
pip install -r requirements.txt
```

`rasterio` requires GDAL; on some systems `pip install rasterio` alone is
sufficient (it ships GDAL wheels for common platforms), but if it fails,
install GDAL via your system package manager first (e.g.
`apt install gdal-bin libgdal-dev` on Debian/Ubuntu).

## Usage

### Smoke test (no real data required)

Generates a synthetic low-light, noisy, low-dynamic-range scene (with
injected cosmic-ray hits) and runs it through the full pipeline:

```bash
python backend_pipeline.py --demo
```

### Process a real scene

```bash
python backend_pipeline.py --input data/raw/your_scene.tif --output-dir data/output
```

### With trained weights

```bash
python backend_pipeline.py --input data/raw/your_scene.tif --checkpoint checkpoints/auranet_v1.pt
```

Each run writes, into `--output-dir`:

- `<name>_enhanced.tif` — enhanced image (band 1) + trust map (band 2), CRS/transform preserved from the input
- `<name>_trust.tif` — standalone trust map GeoTIFF
- `<name>_metrics.json` — PSNR, SSIM, NIQE, BRISQUE, entropy gain, guardrail pass/fail, cosmic-ray/low-trust/saturation pixel fractions

## Training status

`zero_dce`, `nafnet_denoiser`, and `uncertainty_head` are the three learned
submodules; they ship here with random initialization plus their complete
loss functions (`zero_dce.py`: `SpatialConsistencyLoss`,
`ExposureControlLoss`, `IlluminationSmoothnessLoss`;
`uncertainty_head.py`: `HeteroscedasticNLLLoss`). Training scripts are
mission-dataset-specific and are the natural next step once real imagery
(e.g. LRO NAC, HiRISE, or a specific mission's PDS4 archive) is available —
`AuraNetPipeline.load_checkpoint()` is already wired up to load trained
weights once you have them.

## Configuration

All tunables — sensor gain/bias/dark/flat calibration, cosmic-ray
sigma-clip threshold, wavelet family/depth, Zero-DCE iteration count,
NAFNet width/blocks, uncertainty trust thresholds, and evaluation metric
selection/guardrail — live in `config/default_config.yaml`. Point the
pipeline at a different config with `--config path/to/your.yaml` to adapt
to a new sensor or mission without touching code.
