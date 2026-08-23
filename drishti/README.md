# AURA-NET — Physics-Constrained Planetary Enhancement Pipeline

Enhancement of low-illumination, noisy, low-dynamic-range planetary surface
imagery into a scientifically usable product: calibrated radiance plus a
per-pixel Bayesian trust map, in a lossless georeferenced container.

The design goal is not "make the image look better". It is to make previously
invisible surface detail visible **while keeping every step auditable**, so a
scientist can tell recovered signal from invented structure.

## Pipeline

```
[RAW INPUT]  16-bit PDS4 / FITS / GeoTIFF (uncalibrated DN + spatial metadata)
   |
STAGE 1  SCIENTIFIC INGESTION & PHYSICS FRONT-END
   1.1  Ingestion            16-bit linear arrays + affine / CRS preserved
   1.2  Radiometric calib.   DN -> spectral radiance (W m^-2 sr^-1 um^-1)
   1.3  Cosmic ray scrubber  Laplacian edge rejection of SEU spikes
   1.4  Anscombe VST         mixed Poisson-Gaussian -> AWGN (sigma ~ 1)
   |
STAGE 2  FREQUENCY DECOUPLING & DE-QUANTIZATION
   2.1  Implicit de-quant.   sub-bin continuous offset mapping
   2.2  Stationary WT        translation-invariant sub-bands
   |
   +--- LL (low-frequency field) -------+--- LH / HL / HH (high-frequency) ---+
   |                                    |                                     |
STAGE 3A  ILLUMINATION CURVE          STAGE 3B  DETAIL & PSF DECONVOLUTION
   Zero-DCE++ depthwise backbone         differentiable PSF deconvolution
   multi-order recurrent curves          NAFNet (SimpleGate + SCA)
   zero-synthesis monotonic curves       linear activation-free subtraction
   |                                    |
   +----------------+-------------------+
                    |
STAGE 3C  FREQUENCY RECOMBINATION   inverse SWT -> full-spectrum radiance
   |
STAGE 4  DYNAMIC RANGE COMPRESSION & PHOTOMETRIC NORMALIZATION
   4.1  Bilateral-guided log tone mapping   shadows up, highlights unclipped
   4.2  Lommel-Seeliger correction          normalize incidence / emission
   |
STAGE 5  PHYSICS-BASED VERIFICATION & UNCERTAINTY ESTIMATION
   5.1  Photometric flux conservation across scales
   5.2  Sobel gradient consistency guardrail
   5.3  Heteroscedastic head -> mean radiance (mu) + log-variance (s)
   |
STAGE 6  SCIENTIFIC METRIC HARNESS & LOSSLESS EXPORT
   6.1  NIQE / BRISQUE / entropy gain (+ PSNR / SSIM)
   6.2  Frozen topological crater detector (mIoU / true detection)
   6.3  Lossless 32-bit float multi-band GeoTIFF
   |
[FINAL DELIVERABLE]
   Band 1: Calibrated enhanced radiance (mu)
   Band 2: Per-pixel Bayesian trust map (T in [0, 1])
   Tags:   Affine transform, planetary CRS, radiometric metadata
```

## Why this minimizes hallucinated structure

Each constraint below is architectural or measured — not a hoped-for training
outcome.

| Stage | Mechanism | Why it cannot fabricate |
| --- | --- | --- |
| 1.2–1.4 | Calibration, scrubbing, VST | Zero learned weights; every correction traces to a calibration constant recorded in the output tags |
| 1.3 | Laplacian edge rejection | Separates cosmic rays from real relief by *edge sharpness*, so a bright crater rim is not scrubbed away |
| 2.1 | Implicit de-quantization | Correction passes through `tanh` scaled by half a quantizer bin — architecturally incapable of exceeding one bin |
| 2.2 | Stationary (undecimated) WT | Shift-invariant, so independently editing LL and detail bands cannot produce block-boundary ringing at recombination |
| 3A | Monotonic curve estimation | The network outputs curve parameters, never pixels. `\|A\| <= 0.999` forces `dLE/dx > 0`, so equal inputs always map to equal outputs — order-preserving, hence no new spatial pattern |
| 3B | Wiener/RL with a gain ceiling | Per-frequency amplification is hard-capped, so information destroyed beyond the optical cutoff cannot be "restored" |
| 3B | NAFNet activation-free path | SimpleGate + SCA replace nonlinear activations, shrinking the representable function family |
| 5.1 | Flux conservation | Area-averaged energy balance across scales, gated on the learned path |
| 5.2 | Gradient consistency | Rank correlation of pre-smoothed Sobel magnitudes — reported, **not gated**; see the calibration note below |
| 5.3 | Heteroscedastic head | Every pixel ships a confidence value; saturated and cosmic-ray pixels are forced to zero trust regardless of what the network predicts |

Each run records `zero_synthesis_guarantee_held`, `flux_conservation_passed`,
`gradient_correlation`, `trust_map_informative` and `all_guardrails_passed` in
its metrics JSON, so the claims are evidenced per scene rather than asserted
once in a README. Which of these actually gate — and which are reported for a
human to weigh — is spelled out under *Guardrail calibration* below.

## Installation

```bash
pip install -r requirements.txt
```

`rasterio` needs GDAL. The wheels bundle it on common platforms; if the install
fails, install GDAL from your system package manager first (for example
`apt install gdal-bin libgdal-dev`).

## Usage

Smoke test on a synthetic low-light scene (no mission data needed). The demo
scene is written as a georeferenced GeoTIFF on a lunar sphere, so it exercises
the CRS/transform preservation path rather than skipping it:

```bash
python aura_pipeline.py --demo
```

Process a real scene:

```bash
python aura_pipeline.py --input data/raw/your_scene.tif --output-dir data/output
```

With trained weights:

```bash
python aura_pipeline.py --input data/raw/your_scene.tif --checkpoint checkpoints/auranet_v1.pt
```

Each run writes into the output directory:

- `<name>_enhanced.tif` — band 1 enhanced radiance (mu), band 2 trust map,
  32-bit float, CRS/transform/radiometric tags inherited from the input
- `<name>_trust.tif` — the trust map alone, for independent GIS overlay
- `<name>_metrics.json` — every stage-6 metric plus all guardrail verdicts
- `previews/` — human-viewable PNGs (see below)

## Viewable images

The GeoTIFF above is the scientific product: 32-bit float radiance in physical
units. No ordinary image viewer can display it meaningfully — the values sit in
a narrow band around 0.3 W·m⁻²·sr⁻¹·µm⁻¹, so Windows Photos or Preview shows a
flat grey rectangle. Every run therefore also writes viewable PNGs into a
`previews/` folder beside the products:

```
data/output/previews/
├── <name>_1_raw.png           the "before": calibrated input, display-stretched
├── <name>_2_enhanced.png      the "after": the enhanced image  <-- open this one
├── <name>_3_trust.png         trust map in viridis (dark = untrusted)
└── <name>_4_comparison.png    all three side by side, labelled
```

The CLI prints the folder and points at the enhanced image when the run
finishes. Control it with:

```bash
python aura_pipeline.py --demo --preview-dir some/other/folder
```

```bash
python aura_pipeline.py --demo --no-preview
```

Everything else is under `export.preview` in the config: output format, which
of the four images to write, 8- or 16-bit depth, `max_dimension` downscaling for
very large scenes, whether to colourise the trust map, and the contrast stretch
(`percentile`, the robust default; `minmax`; or `asinh`, the standard
astronomical stretch that keeps deep shadow and sunlit terrain legible in one
frame).

**Previews are display renderings, not measurements.** Each is independently
contrast-stretched so it looks its best on screen — the honest way to compare
"the best you can see in the raw" against "the best you can see in the enhanced
product" — but it means preview brightness is *not* proportional to radiance,
and preview pixels must never be used for photometry. The limits actually
applied are recorded per run in the metrics JSON under `preview_stretch_limits`,
so any preview traces back to the radiance it came from.

Runs without trained weights label the enhanced panel `physics-only — no trained
weights`. This matters: an uncheckpointed run deconvolves without denoising, so
it sharpens noise along with signal and the output legitimately looks grainier
than the input. Unlabelled, that reads as the pipeline underperforming rather
than as a run with no weights loaded.

## Configuration

Everything tunable lives in `config/default_config.yaml`, ordered to mirror the
pipeline stages: sensor calibration, cosmic-ray thresholds, wavelet family and
depth, Zero-DCE++ iterations, PSF model and deconvolution regularization, tone
mapping, photometric geometry, guardrail thresholds, and export settings. Point
at a different file with `--config` to adapt to a new sensor without touching
code.

The calibration block ships with placeholder constants. **Replace them with the
mission calibration kernel values** before treating band 1 as photometry —
`radiance_scale` in particular is an instrument responsivity, not a universal
constant. If you run without a calibration kernel, set `radiance_units` to
something honest so the GeoTIFF header does not carry a false unit claim.

`config/ch2_tmc.yaml` is a worked example for Chandrayaan-2 TMC-2 Nadir raw
products, and doubles as a template for how to document a sensor swap: it
separates constants taken from the PDS4 label, constants measured from the
product itself, and constants left deliberately neutral because they are
unknown. Run it with:

```bash
python aura_pipeline.py --input data/raw/your_tile.tif --config config/ch2_tmc.yaml
```

Note what it does *not* do: the default `bias_offset_dn` + `black_level_dn` of
192 DN is above that product's global minimum of 130 DN and would clip the
darkest terrain to zero, so both are set to 0 rather than guessed. Subtracting
an unknown offset is worse than subtracting none.

## Training status

`zero_dce`, the `detail_restorer` (PSF deconvolution + NAFNet), the implicit
dequantizer, and `uncertainty_head` are the learned submodules. They ship
**zero-initialized at their output layers**, so an uncheckpointed run is a
physics-only pass — the learned stages are identity pass-throughs rather than
random weights scribbling over real data.

This matters most for the trust map. An untrained variance head would emit a
spatially varying but entirely random confidence field: a trust map that looks
meaningful and is not. Zero-initialized, it emits a uniform map in which only
physically known-bad pixels (saturated wells, scrubbed cosmic-ray hits, nodata)
carry zero trust, and every run reports `trust_map_informative: false` until
trained weights are loaded.

Their complete loss functions are included, beside the architectures they
constrain: `ZeroDCELoss` (spatial consistency, exposure control, illumination
smoothness), `HeteroscedasticNLLLoss`, `PhotometricFluxConservationLoss` and
`SobelGradientConsistencyLoss`. Training scripts are mission-dataset specific
and are the natural next step once real imagery is available;
`AuraNetPipeline.load_checkpoint()` is already wired to accept the weights.

## Guardrail calibration and known limits

These are measured properties of the current defaults, stated so they are not
mistaken for guarantees:

- **Flux conservation (5.1) is the gate that works.** It is scale-stable
  (drift 0.013–0.029 measured from 192 to 512 px), rejects invented large-scale
  energy by an order of magnitude (0.49 against a 0.05 tolerance), and correctly
  accepts a pure global gain. Together with the architectural zero-synthesis
  guarantee, this is what `all_guardrails_passed` rests on.
- **Gradient consistency (5.2) is reported, not gated.** No fixed threshold for
  it survived calibration. Across 4 scene sizes (192–512 px) × 3 seeds × 3
  fabrication controls, every candidate formulation — raw and smoothed Pearson,
  Spearman, percentile-masked, edge-weighted, blockwise minimum/quantile, and
  scene-normalized — had some genuine run scoring below some fabricated one.
  The decisive control: running with PSF deconvolution disabled leaves only a
  monotone tone map, which *provably* cannot move an edge, and it still scored
  0.783 at 512 px while fabricated craters at 192 px scored 0.796. The cause is
  the statistic, not the pipeline — in a noisy scene the gradient field of flat
  terrain is dominated by photon noise, so the achievable correlation is set by
  the scene's structure-to-noise composition, which varies enormously between
  real products. Each run reports `gradient_correlation` and
  `gradient_fidelity_vs_baseline` (the same statistic divided by a
  structure-preserving denoise of that scene, which is the more interpretable
  number). Set `verification.gradient_consistency.gate: true` only after
  calibrating `min_correlation` on real imagery from your instrument, at your
  typical scene size and terrain. The statistic is still worth reading — warped
  edges score far below everything else in every configuration measured — it is
  just not safe to automate on yet.
- **SSIM vs raw** is a gross structural gate, not a hallucination detector. It
  is measured against the *noisy* calibrated scene, so it is bounded above by
  the noise the enhancement is meant to remove — on the synthetic scene even a
  plain Gaussian denoise of the reference against itself only reaches 0.78–0.82.
  The gate sits at 0.60 to catch washout and geometric collapse.
- **Crater detection (6.2)** is a downstream *utility* measure, not an
  integrity test: the revealed-crater count alone does not separate genuine
  from fabricated scenes. Use the reported trust value at each revealed crater
  to weigh individual detections.
- **NIQE** falls back to a self-referential `niqe_lite` proxy unless `pyiqa` is
  installed; the backend actually used is reported as `niqe_backend`. **BRISQUE**
  reports `null` rather than a wrong number when neither `pyiqa` nor `piq` is
  present.
- **Memory** is full-frame: the SWT keeps every sub-band at native resolution,
  so peak usage scales with image area times sub-band count. Very large mosaics
  need tiling, which this pipeline does not yet do.

## Project structure

```
drishti/
├── config/
│   └── default_config.yaml          # Hyperparameters, sensor calibration gains, thresholds
├── data/
│   ├── raw/                         # Raw 16-bit PDS4 / FITS / GeoTIFF planetary images
│   └── output/                      # Multi-band enhanced GeoTIFFs + Trust maps
├── models/
│   ├── __init__.py
│   ├── physics_frontend.py          # Stage 1 (ingest, calibration, CR scrub, VST)
│   │                                # Stage 4 (tone mapping, Lommel-Seeliger)
│   ├── wavelet_dequant.py           # Stage 2 (implicit de-quant, SWT) + Stage 3C (ISWT)
│   ├── zero_dce.py                  # Stage 3A multi-order curve estimator (LL band)
│   ├── nafnet_denoiser.py           # Stage 3B PSF deconvolution + detail restorer
│   └── uncertainty_head.py          # Stage 5 flux/gradient guardrails + (mu, log-var) head
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Stage 6.1 metrics + 6.2 frozen crater detector
│   └── exporter.py                  # Stage 6.3 lossless GeoTIFF writer
│                                    # + viewable PNG preview renderer
├── aura_pipeline.py                 # Unified End-to-End Model & Execution Pipeline
└── requirements.txt
```
