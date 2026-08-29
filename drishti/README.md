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

Stage 3B's denoiser is the one component with none of the above — an
unconstrained convolutional network on the detail bands. What holds it is the
training objective, which has no term that adding structure would minimize:

| Objective | Why it cannot fabricate |
| --- | --- |
| Noisier2Noise reconstruction | The target *is* the measured tile; the input is that tile plus one extra realization of the sensor's own noise. Lowering the loss means removing noise the pipeline itself added — nothing else |
| Structure-addition penalty | One-sided hinge: charges the network for producing more gradient energy than the measurement had, and stays silent when it produces less. Deconvolution sharpening gets a bounded allowance; invention does not |
| Identity anchor | Fed an un-degraded tile, the restorer is asked for that tile back, fixing its behaviour on real input at the identity |

Each run records `zero_synthesis_guarantee_held`, `flux_conservation_passed`,
`gradient_correlation`, `trust_map_informative` and `all_guardrails_passed` in
its metrics JSON, so the claims are evidenced per scene rather than asserted
once in a README. A trained run additionally carries `checkpoint_*` fields —
the weights' own audit on products no training phase saw, including
`checkpoint_added_structure_fraction`. Which of these actually gate — and which
are reported for a human to weigh — is spelled out under *Guardrail
calibration* below.

## Installation

Use a virtual environment. On Windows in particular, a bare `python` on PATH
often resolves to an MSYS2 or Store shim with no packages, and the failure looks
like `ModuleNotFoundError: No module named 'numpy'` when the app starts:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r drishti/requirements.txt
```

On Linux/macOS the interpreter is `.venv/bin/python`. Run everything with that
interpreter — `.venv/Scripts/python drishti/app.py`, not `python drishti/app.py`.

`rasterio` needs GDAL. The wheels bundle it on common platforms; if the install
fails, install GDAL from your system package manager first (for example
`apt install gdal-bin libgdal-dev`).

**Windows path length.** PyTorch loads native DLLs through a path-length-limited
API. A virtualenv nested deep under a long path (a temp directory, a synced
folder several levels down) fails at `import torch` with
`[WinError 206] The filename or extension is too long`, even though `pip install`
succeeded. Keep the venv near the drive root or at the project root.

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
python aura_pipeline.py --input data/raw/your_scene.tif --checkpoint checkpoints/aura_net_tmc.pt
```

Process a Chandrayaan-2 PDS4 product straight out of an extracted ISSDC
download — point at the `.xml` label, never the `.img`:

```bash
python aura_pipeline.py --input <bundle>/data/calibrated/20260810/ch2_tmc_ncn_20260810T0544521099_d_img_d18.xml
```

Run every product in a bundle folder and collect the results into one table:

```bash
python batch_run.py --data <bundle folder> --checkpoint checkpoints/aura_net_tmc.pt
```

Each run writes into the output directory:

- `<name>_enhanced.tif` — band 1 enhanced radiance (mu), band 2 trust map,
  32-bit float, CRS/transform/radiometric tags inherited from the input
- `<name>_trust.tif` — the trust map alone, for independent GIS overlay
- `<name>_metrics.json` — every stage-6 metric plus all guardrail verdicts
- `previews/` — human-viewable PNGs (see below)

## Dashboard

A minimal local web front end, for when you would rather drop a file on a page
than read a wall of JSON:

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. Drop in a scene, pick a sensor config, and it
runs the pipeline with a live stage-by-stage progress bar, then shows the
enhanced image, the raw input, the trust map and the side-by-side sheet, the
full statistics grouped into cards, and download links for the GeoTIFF products
and the metrics JSON.

### Getting a scene in

An ISSDC download is a *directory tree*, not a file. The science raster is a
~2 GB headerless `.img` beside the `.xml` label that gives it a shape, and the
`browse/` folder holds a contrast-stretched quicklook PNG that opens perfectly
well and is not the data. There are three ways in, and all of them run the same
bundle discovery, so the quicklook is never processed by mistake:

1. **Local folder** — paste the path of the extracted product folder, e.g.
   `C:\Users\you\Downloads\ch2_tmc_ncn_20260810T0544521099_d_img_d18`.
   Nothing is uploaded or copied; the pipeline opens the label in place. This is
   the practical route for a full-size product, and **Inspect** resolves the
   path and reports what it found — instrument, sensor, processing level, pixel
   dimensions, solar incidence — before you commit minutes to a run.
2. **Bundle ZIP** — upload the archive exactly as downloaded.
3. **Loose files or a whole folder** — a GeoTIFF, or a hand-picked `.xml` +
   `.img` pair. Uploading the `.img` alone is the natural mistake to make, and
   the dashboard says so explicitly instead of failing with a misleading "not a
   TIFF file".

When a bundle holds several products the calibrated nadir scene is preferred,
and the choice is reported in the run notes; point at a single `.xml` to
override it. Also accepts GeoTIFF/TIFF, FITS and PNG/JPEG.

**The label's illumination geometry travels with the scene.** `solar_incidence`
is per product — it ranges from 39° to 83° across the products in this repo's
test bundle — and stage 4.2 divides by `cos i / (cos i + cos e)`. The ingestor
reads the angle out of the PDS4 label and hands it to the photometric stage,
which reports `incidence_source: product_metadata` when that worked and
`config_default` when it fell back to the config constant. Only the first is
right for anything but the one scene the config was written from.

Five views along the top: **Overview** (stage progress, headline numbers, tonal
distribution chart, crater breakdown, run conditions, upload), **Pipeline** (a
stage-by-stage flowchart of the implementation, annotated with the current
run's own values), **Imagery**, **Metrics**, and **Guardrails**. It is
responsive down to a 375 px phone — grids collapse three columns → two → one,
the nav scrolls horizontally, and touch targets grow on coarse pointers.

The **Weights** selector picks a checkpoint from `checkpoints/`, or *Physics
only* to run the learned stages as identity pass-throughs. It defaults to the
newest checkpoint present, because a physics-only run is the honest fallback
rather than the preferred mode.

Requests that start a run or read the filesystem must carry an
`X-Aura-Client` header. A page on another origin can forge a plain form POST to
localhost, but setting a custom header forces a CORS preflight this server never
answers — which is what stops a stray browser tab from driving the dashboard
into reading local paths and rendering them back.

Two things it does deliberately:

- **It surfaces the caveats next to the numbers.** Physics-only runs, an
  uninformative trust map, an ungated gradient guardrail and cropped inputs are
  all called out as banners. A dashboard that showed only the pretty numbers
  would undo the work the pipeline does to stay auditable.
- **It crops big scenes loudly.** The pipeline is full-frame, so a 592-megapixel
  strip would need tens of GB. Anything above the *Max edge* setting is
  centre-cropped with a windowed read (never a full decode, and georeferencing
  is carried across to the crop), and every report says so. Set *If larger* to
  **Reject** if you would rather be stopped than cropped.

It is a local tool, not a service: it binds to localhost, runs one job at a
time, keeps job state in memory, and has no authentication. Don't expose it.

### Front-end (React + Vite + TypeScript)

The UI lives in `frontend/` as a typed React app. The **built bundle is
committed** to `static/`, so `python drishti/app.py` works straight after a
clone with no Node installed — deliberate for a local tool with no CI.

To change the UI you need Node:

```bash
cd drishti/frontend && npm install
```

```bash
npm run dev
```

`npm run dev` serves on :5173 and proxies `/api` to Flask on :5000 (so run
`app.py` alongside it) — one origin, no CORS. When you're done:

```bash
npm run build
```

which writes `index.html` + `assets/` into `static/`, replacing what Flask
serves. `npm run build` runs `tsc -b` first, so type errors fail the build.

```
frontend/src/
├── types.ts                  # API response types (Metrics, Job, JobResult…)
├── api.ts                    # typed fetch client
├── styles.css                # design tokens + responsive rules
├── App.tsx                   # shell, view routing, job polling
└── components/
    ├── primitives.tsx        # formatting, StatCard, Card
    ├── Hero.tsx              # stage bars, KPI tiles
    ├── HistogramChart.tsx    # inline-SVG raw vs enhanced distribution
    ├── OverviewCards.tsx     # summary, provenance, notes, crater bubbles
    ├── UploadCard.tsx        # drop zone + run options
    └── Views.tsx             # Imagery / Metrics / Guardrails
```

Charts are hand-drawn inline SVG rather than a charting library: two polylines
and a crosshair don't justify the dependency, and it keeps the bundle free of
anything that would need a CDN at runtime. Total build is ~209 kB JS (66 kB
gzipped) with React itself the bulk of it.

## Performance

A 1536 px Chandrayaan-2 tile went from **78.4 s to 21.2 s** (3.7x). Nothing was
traded away for it — the enhanced product is bit-identical and all 75 checks
still pass. Measured with `cProfile`, not guessed:

| Change | Saved | Why it is safe |
| --- | --- | --- |
| Skip provably-identity learned stages | 39 s | With no checkpoint, the denoiser, de-quantizer and curve head all sit at their zero initialization, so each is *algebraically* the identity (`x - 0`, `x + 0`, `x + 0·x(1-x)`). Verified bit-exact; loading weights re-enables full computation automatically. |
| Scale-space pyramid for the crater detector | 12 s | The LoG operator is scale-invariant, so coarse scales are evaluated on a decimated grid. Measured on real TMC data: **identical** detections (25/25 and 83/83 matched, median IoU 1.0000) and identical SSIM. |
| `torch.var_mean` in LayerNorm2d | 6 s | One pass over the channel axis instead of two. Helps trained runs, where the denoiser actually runs. |
| Reuse the cosmic-ray median filter | 1.4 s | The same window median was computed twice — once for detection, once for inpainting. |
| Rank-correlation sample cap 2M → 400k | 2 s | Standard error ~0.0016 on the correlation, far below the precision it is read at. |

Zero-initializing the Zero-DCE curve head was a **correctness** fix before a
speed one: previously an uncheckpointed run applied a random (if bounded and
monotone) tone curve to real science data, which contradicted this README's
claim that such a run is physics-only. It now genuinely is.

The identity shortcut is gated on `eval` mode. The zero-initialized state is
also where *training* starts, and a module that returns its input unchanged has
no gradient path to its own weights — left ungated, the optimizer would fail on
`element 0 of tensors does not require grad` and the module could never leave
the identity it was initialized to. Inference runs in `eval`, so the saving
above is unaffected.

Set `evaluation.crater_detector.pyramid: false` to force full-resolution
scale-space evaluation.

What remains is real work: the L.A.Cosmic median filters (~5 s), tone mapping,
verification and export. GPU is used automatically when `project.device: cuda`
and CUDA is available.

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

## Training

Four submodules carry learned weights — the implicit de-quantizer (2.1),
Zero-DCE++ (3A), the detail restorer (3B) and the uncertainty head (5.3). All
four ship **zero-initialized at their output layers**, so an uncheckpointed run
is a physics-only pass: the learned stages are identity pass-throughs rather
than random weights scribbling over real data. `trained_weights_loaded` says
which mode any given run was in.

`train.py` trains all four on real PDS4 products, with **no ground truth** —
nobody can re-photograph a crater at a different sun angle to supervise
against.

```bash
python train.py all --data <bundle folder>          # build tile cache, then train
python train.py prepare --data <bundle folder>      # just the cache
python train.py fit                                 # just the training
```

`prepare` writes a tile cache to `data/training/`, and `fit` adds the
materialized per-phase domains beside it — together roughly **1 GB** at the
default 32 tiles per product. It is git-ignored, but a cloud-synced working
directory will still try to upload it, so pass `--cache <somewhere local>` if
that matters. Both commands take the same `--cache`.

`--instrument` defaults to `tmc`: stage 1 converts DN to electrons using the
config's gain and exposure, so a product from a different camera would enter the
VST with the wrong noise model. Products from other instruments are listed and
skipped rather than quietly mixed in.

Training writes `checkpoints/aura_net_tmc.pt` (~7 MB) and a
`.report.json` beside it holding the per-epoch history and the held-out audit.
`checkpoints/` is git-ignored; force-add the `.pt` if you want a clone to come
up with trained weights instead of falling back to physics-only.

### The domains are the inference domains

Each module is trained on exactly what it sees at inference, produced by the
same objects the pipeline uses. `training/tiles.py` samples stratified windows
from every product, runs the real `PhysicsFrontend` over each one, and caches
the result; `DomainCache` then derives each phase's input with the pipeline's
own `WaveletDequantizer` and normalization helpers. The training distribution is
not an approximation of the inference distribution — it is produced by the same
code.

The phases run in pipeline order, because each module's input is the previous
module's output:

| Phase | Module | Domain | Objective |
|---|---|---|---|
| A | de-quantizer | VST-stabilized radiance | edge-weighted TV + fidelity |
| B1 | Zero-DCE++ | normalized LL band | spatial consistency, exposure, curve smoothness |
| B2 | detail restorer | normalized LH/HL/HH | Noisier2Noise + structure penalty + identity anchor |
| C | uncertainty head | tone-mapped radiance | heteroscedastic NLL + flux conservation |

By phase C the head is looking at what the *trained* pipeline hands it, which is
the only way its variance means anything. Nothing back-propagates across the SWT
(`pywt` is numpy), which is why the phases are separate rather than end-to-end;
freezing each phase's upstream also makes its domain fixed data, so it is
materialized once instead of once per epoch.

### Why the result does not hallucinate

Two of the four modules cannot invent structure whatever their weights, and the
guarantee is architectural rather than trained:

- The de-quantizer's offset passes through `tanh` scaled by `step/2`, so it
  cannot move a pixel outside its own quantizer bin. It can erase banding; it
  cannot invent a crater.
- Zero-DCE emits curve *parameters*, never pixels, and the only operation
  applied is `LE(x) = x + A·x·(1−x)` with `|A| < 1`. Every application is
  strictly increasing, so two pixels of equal input radiance always receive
  equal output radiance — a pointwise monotone map can reorder brightness but
  cannot create a spatial pattern. Verified per run as
  `zero_synthesis_guarantee_held`.

The detail restorer is the one unconstrained network, so the objectives are
what police it:

- **The target is the measurement.** There is no clean reference, so each tile
  is cached twice: as measured, and with one extra realization of the sensor's
  own Poisson-Gaussian noise drawn in the electron domain from the config's gain
  and read-noise constants. Training noisy → measured (Noisier2Noise) means the
  only way to lower the loss is to remove noise the pipeline itself added.
- **A one-sided gradient penalty.** `StructureAdditionPenalty` charges the
  network for producing more edge energy than the measurement had, and says
  nothing when it produces less. The asymmetry is the point: removing gradient
  energy is denoising, adding it is either PSF deconvolution — legitimate, and
  what the 50 % margin allows for — or invention.
- **An identity anchor.** Fed an un-degraded tile, the restorer is asked for
  that tile back. This defines its fixed point: on real input it should be close
  to the identity, and any departure is an unrequested change to science data.

The uncertainty head is trained against a target it genuinely cannot fully
recover, so it cannot drive its variance to the floor and produce a
confident-looking uniform trust map — the artefact the zero-initialization was
chosen to avoid in the first place.

How strict to be is a knob: `--w-structure` (default 2.0) weights the gradient
penalty, `--w-identity` (0.5) the anchor, and `--structure-margin` (0.5) sets
how much sharpening is allowed before the penalty bites. Raising the first or
lowering the third buys a more conservative restorer at the cost of some
deconvolution sharpening; the held-out `added_structure_fraction` below is what
to watch while changing them.

### The audit

Training ends by re-measuring on **products no phase ever saw** — the split is
by product, not by tile, because tiles from one strip overlap in terrain and
illumination and a random tile split would report a validation loss that
flatters the model. The audit lands in `<checkpoint>.report.json`:

| Field | Meaning |
|---|---|
| `added_structure_fraction` | share of pixels whose gradient magnitude exceeds the sharpening allowance — the direct measurement of invented edges |
| `detail_reconstruction` | held-out Charbonnier error of the restorer |
| `zero_synthesis_guarantee_held` | `max\|A\| < 1` over the held-out set |
| `log_var_spread` | whether the variance head learned to distinguish anything |
| `trust_map_informative` | whether the resulting trust map carries more than the physical masks |

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

### What training measured, including what it cost

Measured over the whole bundle — 28 TMC windows of 1024 px, two per product,
the same windows both ways. The shipped checkpoint against a physics-only run:

| | physics-only | trained | |
|---|---|---|---|
| `all_guardrails_passed` | 28/28 | 28/28 | — |
| `trust_map_informative` | 0/28 | **28/28** | ✅ |
| SSIM vs raw | 0.852 | **0.886** | ✅ |
| NIQE (lower better) | 1.268 | **1.117** | ✅ |
| `craters_matched` | 43.8 | 43.0 | ✗ −1.8 % |
| `craters_lost` | 3.1 | 3.9 | ✗ |
| `crater_miou` | 0.969 | 0.938 | ✗ |
| `gradient_correlation` | 0.912 | 0.875 | ✗ |
| `entropy_gain` | −0.340 | −0.485 | ✗ |

Read that honestly: the headline gain is the **trust map**, which goes from
carrying nothing but the physical masks to being informative on every window —
that is the whole point of stage 5.3 and it only exists once the variance head
is trained. Structure fidelity and no-reference quality also improve. Against
that, crater retention is marginally *worse* (−0.8 of 43.8 matched) and
localisation slips 0.03 of mIoU. The pipeline is not uniformly better after
training; it is better at saying how much to trust itself, slightly better
structurally, and slightly worse at crater retention — and the section below
identifies which stage each of those belongs to.

Two further findings, both of them things the guardrails caught rather than
things that were designed in.

**Stage 3A does not earn its place on TMC data.** Loading the checkpoint one
section at a time on a held-out 1024 px window, against the frozen crater
detector:

| Sections loaded | SSIM | grad_corr | craters | matched/18 | lost | mIoU | guardrails |
|---|---|---|---|---|---|---|---|
| physics-only | 0.813 | 0.922 | 87 | 18 | 1 | 0.985 | pass |
| de-quantizer (2.1) | 0.832 | 0.927 | 83 | 18 | 1 | 0.967 | pass |
| detail restorer (3B) | **0.891** | **0.931** | 127 | **18** | **1** | 0.972 | pass |
| Zero-DCE (3A) | 0.797 | 0.871 | 69 | 14 | 5 | 0.954 | pass |
| all four | 0.881 | 0.884 | 78 | 16 | 3 | 0.936 | pass |

The detail restorer — the one unconstrained network, and the one the training
objectives were written to police — is a clean win: it raises SSIM *above* the
physics-only baseline while retaining every matched crater. Stage 3A costs two
of them, at every setting tried. That is not a tuning failure so much as the
curve doing what a curve does: `dLE/dx = 1 + A(1-2x)` is below 1 for `x > 0.5`,
so lifting shadows compresses highlights, and stage 4.1 has already performed
the dynamic-range compression that 3A duplicates. If crater retention matters
more than shadow lift, set `zero_dce.num_iterations: 0` — with the curve applied
zero times the stage is an exact bit-for-bit bypass, verified against the
trained weights, and the other three stages are unaffected.

The first attempt was worse and is worth recording. At the Zero-DCE paper's
`target_exposure: 0.6` / weight 10.0 — tuned for consumer photography, where the
curve *is* the enhancement — the exposure term ran ~14x larger than the spatial
consistency term meant to restrain it, and the trained pipeline produced **313
"revealed" craters per window against 58 physics-only, while matched craters
fell (43.8 → 39.8) and lost craters rose (3.1 → 7.1)**. More detections, fewer
real ones: the hallucination signature. `structure_guardrail_passed` dropped to
14/28 windows and the run was correctly rejected. The measured LL-band mean is
0.286, so 0.6 was asking for a +0.31 lift on every patch; the config now ships
0.30 with the measurement and the three-way comparison recorded beside it.

**The config is per instrument, and the guardrails enforce it.** The bundle's
OHRC product (12000 samples, 0.25 m/px, 162.1 ms line exposure) fails the
structure gate under `ch2_tmc.yaml` (SSIM 0.243 and 0.537) while all 14 TMC
products pass. Stage 1.2 converts DN to electrons with the config's gain and
divides by its exposure time, so the wrong config is a wrong radiometric
conversion — and the gate caught it without being told to. `train.py
--instrument` excludes such products from the tile cache for the same reason;
`batch_run.py` flags them as `instrument_mismatch` rather than quietly
reporting the number.

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
├── training/
│   ├── __init__.py
│   ├── tiles.py                     # PDS4 tile sampling, stage-1 cache, noise pairs
│   └── objectives.py                # Structure-addition penalty, identity anchor,
│                                    # banding suppression
├── frontend/                        # React + Vite + TypeScript dashboard source
├── static/                          # Built dashboard bundle (tracked, no Node needed)
├── pds4_bundle.py                   # ISSDC bundle discovery + label geometry
├── train.py                         # Self-supervised trainer (phases A -> B -> C)
├── batch_run.py                     # Whole-bundle sweep -> summary.csv
├── app.py                           # Local Flask server for the dashboard
├── aura_pipeline.py                 # Unified End-to-End Model & Execution Pipeline
└── requirements.txt
```

`app.py` and `static/` are the front end and are entirely optional — the
pipeline runs headless from `aura_pipeline.py` with no web dependency.
