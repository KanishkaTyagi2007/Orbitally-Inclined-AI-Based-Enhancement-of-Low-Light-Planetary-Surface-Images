# Orbitally-Inclined: AI-Based Enhancement of Low-Light Planetary Surface Images

An end-to-end deep learning framework that enhances low-illumination, noisy
planetary imagery into high-fidelity, interpretable products — with quantitative
validation attached to every pixel.

The hard part of this problem is not making a dark image brighter. It is making
it brighter **without inventing terrain that was never observed**, and being
able to prove it afterwards. A crater that appears only because a network
hallucinated it is worse than a crater left in shadow, because it looks exactly
as convincing as a real one.

So the pipeline is built around three commitments:

- **Most stages have no learned weights at all.** Calibration, cosmic-ray
  rejection, noise stabilization, tone mapping and photometric normalization are
  fixed physics — traceable to a sensor constant or a published law.
- **The stages that do learn are constrained by construction.** The illumination
  stage emits monotonic curve *parameters*, never pixels, so it cannot create a
  spatial pattern. The de-quantizer cannot move a pixel outside its own
  quantizer bin. Both hold for any weights whatsoever, trained or not.
- **Every run reports whether it stayed honest.** Flux conservation across
  scales, gradient consistency against the raw scene, and a per-pixel trust map
  ship with the product, and a run that fails a guardrail says so.

Training is self-supervised on real Chandrayaan-2 PDS4 products — there is no
ground truth, because nobody can re-photograph a crater at a different sun
angle. The objectives are anchored to the measurement itself, and the one
unconstrained network is charged for any edge energy it produces that the data
did not contain.

## Quick start

```bash
pip install -r drishti/requirements.txt
cd drishti
```

```bash
python aura_pipeline.py --demo                       # synthetic smoke test
python app.py                                        # dashboard on :5000
```

Enhance one real scene — point at the PDS4 `.xml` label, never the `.img`:

```bash
python aura_pipeline.py --input <bundle>/data/calibrated/<date>/<product>.xml
```

Train on a folder of products, then sweep the whole folder:

```bash
python train.py all --data <bundle folder>
python batch_run.py --data <bundle folder> --checkpoint checkpoints/aura_net_tmc.pt
```

In the dashboard, paste the path of an extracted ISSDC download and press
**Inspect**: it finds the science raster inside the bundle tree, reports what it
found, and reads the product in place — no upload, no unpacking by hand.

## Documentation

**[drishti/README.md](drishti/README.md)** — the full write-up: stage-by-stage
pipeline, the training scheme and its audit, guardrail calibration and measured
limits, configuration reference, and the dashboard.
