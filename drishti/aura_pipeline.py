"""
aura_pipeline.py
=================
AURA-NET -- Physics-Constrained Planetary Enhancement Pipeline.
Unified end-to-end model and execution pipeline.

    [RAW INPUT]  16-bit PDS4 / FITS / GeoTIFF (uncalibrated DN + spatial metadata)
        |
    STAGE 1  SCIENTIFIC INGESTION & PHYSICS FRONT-END
        1.1  ingestion: 16-bit linear arrays + affine/CRS metadata preserved
        1.2  radiometric calibration: DN -> spectral radiance
        1.3  cosmic ray / SEU scrubber: Laplacian edge rejection
        1.4  Anscombe VST: Poisson-Gaussian -> AWGN (sigma ~ 1)
        |
    STAGE 2  FREQUENCY DECOUPLING & DE-QUANTIZATION
        2.1  implicit de-quantization: sub-bin continuous offset mapping
        2.2  stationary wavelet transform -> translation-invariant sub-bands
        |
        +---- LL (low-frequency field) -----+---- LH/HL/HH (high-frequency) ----+
        |                                   |                                  |
    STAGE 3A  ILLUMINATION CURVE          STAGE 3B  DETAIL RESTORATION & PSF
        Zero-DCE++ depthwise backbone         differentiable PSF deconvolution
        multi-order recurrent curves          NAFNet (SimpleGate + SCA)
        zero-synthesis monotonic curves       linear activation-free subtraction
        |                                   |
        +----------------+------------------+
                         |
    STAGE 3C  FREQUENCY RECOMBINATION -- inverse SWT -> full-spectrum radiance
        |
    STAGE 4  DYNAMIC RANGE COMPRESSION & PHOTOMETRIC NORMALIZATION
        4.1  bilateral-guided log tone mapping
        4.2  Lommel-Seeliger incidence/emission correction
        |
    STAGE 5  PHYSICS-BASED VERIFICATION & UNCERTAINTY ESTIMATION
        5.1  photometric flux conservation across scales
        5.2  Sobel gradient consistency guardrail
        5.3  heteroscedastic head -> mean radiance (mu) + log-variance (s)
        |
    STAGE 6  SCIENTIFIC METRIC HARNESS & LOSSLESS EXPORT
        6.1  NIQE / BRISQUE / entropy gain (+ PSNR / SSIM guardrail)
        6.2  frozen topological crater detector (mIoU / true detection)
        6.3  lossless 32-bit float multi-band GeoTIFF
        |
    [FINAL DELIVERABLE]
        Band 1: calibrated enhanced radiance (mu)
        Band 2: per-pixel Bayesian trust map (T in [0, 1])
        Tags:   affine transform, planetary CRS, radiometric metadata

The three learned submodules (Zero-DCE++, NAFNet, uncertainty head) ship
zero-initialized at their output layers, so an uncheckpointed run is a
physics-only pass rather than random noise applied to real data. Load trained
weights with `--checkpoint`; see `AuraNetPipeline.load_checkpoint`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import yaml

from models.physics_frontend import PhotometricBackend, PhysicsFrontend
from models.wavelet_dequant import WaveletDequantizer
from models.zero_dce import ZeroDCE
from models.nafnet_denoiser import DetailRestorer
from models.uncertainty_head import PhysicsVerifier, UncertaintyHead
from evaluation.metrics import evaluate_all
from evaluation.exporter import (
    export_geotiff,
    export_metrics_report,
    export_preview_images,
    export_trust_sidecar,
)

PACKAGE_DIR = Path(__file__).resolve().parent


# =============================================================================
# Normalization helpers
# =============================================================================
def _minmax_normalize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """[0, 1] scaling -- used for bands that must be non-negative (LL, images)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    return ((x - lo) / (hi - lo)).astype(np.float32), lo, hi


def _minmax_denormalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo


def _symmetric_normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Scales by peak magnitude to [-1, 1], preserving the zero point and sign.
    Wavelet detail coefficients are signed and zero-centred; a [0, 1] min-max
    would shift zero and make the denoiser's residual meaningless.
    """
    scale = float(np.max(np.abs(x)))
    if scale < 1e-12:
        return np.zeros_like(x, dtype=np.float32), 1.0
    return (x / scale).astype(np.float32), scale


# =============================================================================
# Pipeline
# =============================================================================
class AuraNetPipeline:
    """End-to-end AURA-NET inference pipeline (stages 1 through 6)."""

    def __init__(self, config_path: str | Path = "config/default_config.yaml",
                 device: Optional[str] = None):
        self.config_path = self._resolve(config_path)
        self.config = self._load_config(self.config_path)
        self._set_seed(int(self.config["project"]["seed"]))

        requested = device or self.config["project"]["device"]
        self.device = torch.device(
            "cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        # Set by load_checkpoint. Recorded in the metrics and stamped on the
        # preview so a physics-only run is never mistaken for a trained one.
        self.checkpoint_loaded = False
        self.checkpoint_name: Optional[str] = None
        self.checkpoint_audit: Optional[dict] = None

        # -- STAGE 1 + 4: physics operators (no learned weights) -----------
        self.frontend = PhysicsFrontend(self.config)
        self.photometry = PhotometricBackend(self.config)

        # -- STAGE 2 + 3C: wavelet decoupling / recombination ---------------
        self.wavelet = WaveletDequantizer(self.config)
        self.wavelet.dequantizer.to(self.device)

        # -- STAGE 3A: illumination curve estimator (LL band) ---------------
        z = self.config["zero_dce"]
        self.zero_dce = ZeroDCE(
            in_channels=1,
            num_filters=int(z["num_filters"]),
            num_conv_layers=int(z["num_conv_layers"]),
            num_iterations=int(z["num_iterations"]),
            depthwise=bool(z.get("depthwise", True)),
            enforce_monotonic=bool(z.get("enforce_monotonic", True)),
        ).to(self.device)

        # -- STAGE 3B: PSF deconvolution + detail denoiser (LH/HL/HH) -------
        self.detail_restorer = DetailRestorer(self.config, in_channels=3).to(self.device)

        # -- STAGE 5: verification + uncertainty ----------------------------
        u = self.config["uncertainty"]
        self.uncertainty_head = UncertaintyHead(
            in_channels=1,
            hidden_channels=int(u["hidden_channels"]),
            min_log_var=float(u["min_log_var"]),
            max_log_var=float(u["max_log_var"]),
        ).to(self.device)
        self.verifier = PhysicsVerifier(self.config)

        for module in (self.wavelet.dequantizer, self.zero_dce,
                       self.detail_restorer, self.uncertainty_head):
            module.eval()

    # -- setup helpers -----------------------------------------------------
    @staticmethod
    def _resolve(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (PACKAGE_DIR / p)

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_checkpoint(self, checkpoint_path: str) -> list[str]:
        """
        Loads trained weights. The checkpoint is a dict which may contain any
        of: 'zero_dce', 'detail_restorer' (or legacy 'nafnet'),
        'uncertainty_head', 'dequantizer'. Returns the keys actually applied.
        """
        state = torch.load(checkpoint_path, map_location=self.device)
        targets = {
            "zero_dce": self.zero_dce,
            "detail_restorer": self.detail_restorer,
            "nafnet": self.detail_restorer.denoiser,
            "uncertainty_head": self.uncertainty_head,
            "dequantizer": self.wavelet.dequantizer,
        }
        loaded = []
        for key, module in targets.items():
            if key in state:
                module.load_state_dict(state[key])
                loaded.append(key)
        self.checkpoint_loaded = bool(loaded)
        self.checkpoint_name = Path(checkpoint_path).name

        # `train.py` measures the trained weights on products no training phase
        # ever saw, and stores the result in the checkpoint. Carrying it into
        # every run's metrics means a reader can see what this model was shown
        # to do on held-out data without going and finding the training report --
        # it qualifies the numbers beside it, which is the whole point.
        self.checkpoint_audit = state.get("audit")
        return loaded

    # -- main entry point ---------------------------------------------------
    @torch.no_grad()
    def process(self, input_path: str, output_dir: Optional[str] = None,
                preview_dir: Optional[str] = None,
                progress: Optional[Callable[[int, str], None]] = None) -> dict:
        """
        Runs stages 1-6 on one scene.

        Args:
            input_path: raw PDS4 / FITS / GeoTIFF product.
            output_dir: where the scientific products go; defaults to
                config paths.output_dir.
            preview_dir: where the viewable PNGs go; defaults to the
                `export.preview.directory` subfolder of `output_dir`.
            progress: optional callback invoked as progress(percent, stage_label)
                at each stage boundary. A full-resolution scene takes minutes,
                so a long-running caller (the dashboard) needs somewhere to read
                state from. Purely observational -- it cannot alter the result.
        """
        started = time.perf_counter()
        report = progress or (lambda pct, label: None)
        cfg = self.config
        out_dir = Path(output_dir) if output_dir else self._resolve(cfg["paths"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        # =====================================================================
        # STAGE 1 -- SCIENTIFIC INGESTION & PHYSICS FRONT-END
        # =====================================================================
        report(5, "Stage 1: ingestion & physics front-end")
        scene = self.frontend.ingest(input_path)                  # 1.1
        fe = self.frontend.process(scene)                         # 1.2 -> 1.4
        stabilized = fe.stabilized                                # VST domain

        # =====================================================================
        # STAGE 2 -- FREQUENCY DECOUPLING & DE-QUANTIZATION
        # =====================================================================
        report(20, "Stage 2: de-quantization & wavelet decomposition")
        dequantized = self.wavelet.dequantize(stabilized, self.device)   # 2.1
        bands, ll, original_shape = self.wavelet.decompose(dequantized)  # 2.2

        # =====================================================================
        # STAGE 3A -- ILLUMINATION CURVE ESTIMATION (LL band)
        # =====================================================================
        report(35, "Stage 3A: illumination curve estimation")
        ll_norm, ll_lo, ll_hi = _minmax_normalize(ll)
        ll_tensor = torch.from_numpy(ll_norm)[None, None].to(self.device)
        ll_enhanced_t, curve_map = self.zero_dce(ll_tensor)
        ll_enhanced = _minmax_denormalize(
            ll_enhanced_t[0, 0].cpu().numpy(), ll_lo, ll_hi
        )
        zero_synthesis_ok = ZeroDCE.is_zero_synthesis(curve_map)

        # =====================================================================
        # STAGE 3B -- DETAIL RESTORATION & PSF DECONVOLUTION (LH/HL/HH)
        # =====================================================================
        # All SWT levels are restored. Every band sits on the native pixel grid
        # (the transform is undecimated), so one shared PSF/denoiser applies to
        # each level identically -- stacked along the batch axis.
        report(45, "Stage 3B: PSF deconvolution & detail restoration")
        detail_stack = np.stack(
            [np.stack([b.LH, b.HL, b.HH], axis=0) for b in bands], axis=0
        )
        detail_norm, detail_scale = _symmetric_normalize(detail_stack)
        detail_tensor = torch.from_numpy(detail_norm).float().to(self.device)
        restored = self.detail_restorer(detail_tensor).cpu().numpy() * detail_scale

        for level, band in enumerate(bands):
            band.LH, band.HL, band.HH = (restored[level, 0],
                                         restored[level, 1],
                                         restored[level, 2])

        # =====================================================================
        # STAGE 3C -- FREQUENCY RECOMBINATION
        # =====================================================================
        report(60, "Stage 3C: frequency recombination")
        reconstructed_vst = self.wavelet.reconstruct(bands, ll_enhanced, original_shape)
        # The inverse VST returns the reconstruction to linear spectral
        # radiance, which is the domain stage 4 photometry is defined in.
        reconstructed_radiance = self.frontend.vst.inverse(reconstructed_vst)

        # =====================================================================
        # STAGE 4 -- DYNAMIC RANGE COMPRESSION & PHOTOMETRIC NORMALIZATION
        # =====================================================================
        report(68, "Stage 4: tone mapping & photometric normalization")
        toned_radiance, photometric_record = self.photometry.process(
            reconstructed_radiance, scene.metadata
        )                                                          # 4.1 -> 4.2

        # =====================================================================
        # STAGE 5 -- PHYSICS-BASED VERIFICATION & UNCERTAINTY ESTIMATION
        # =====================================================================
        # 5.3 heteroscedastic head, run on a normalized copy so the learned
        # variance is scene-scale independent; mu is mapped back to radiance.
        report(78, "Stage 5: physics verification & uncertainty")
        toned_norm, t_lo, t_hi = _minmax_normalize(toned_radiance)
        toned_tensor = torch.from_numpy(toned_norm)[None, None].to(self.device)
        mu_t, log_var_t = self.uncertainty_head(toned_tensor)
        mu_norm = mu_t[0, 0].cpu().numpy()
        log_var = log_var_t[0, 0].cpu().numpy()
        enhanced_radiance = _minmax_denormalize(mu_norm, t_lo, t_hi).astype(np.float32)

        # 5.1 + 5.2 physics guardrails, against the calibrated raw radiance.
        # Flux conservation is gated on the stage-3C reconstruction so it
        # polices the learned path rather than the deliberate stage-4 tone
        # mapping; gradient consistency is gated on the final product, since
        # edge locations must survive every stage.
        verification = self.verifier.verify(
            reference=fe.radiance_linear,
            enhanced=enhanced_radiance,
            log_var=log_var,
            reconstruction=reconstructed_radiance,
            saturation_mask=fe.saturation_mask,
            cosmic_ray_mask=fe.cosmic_ray_mask,
            nodata_mask=fe.nodata_mask,
        )
        trust_map = verification.trust_map

        # =====================================================================
        # STAGE 6 -- SCIENTIFIC METRIC HARNESS & LOSSLESS EXPORT
        # =====================================================================
        # 6.1 + 6.2 -- both scenes stretched to [0, 1] so full-reference
        # metrics measure structure rather than the intended brightness change.
        report(85, "Stage 6.1/6.2: metrics & crater detection")
        raw_ref_norm, _, _ = _minmax_normalize(fe.radiance_linear)
        enhanced_norm, _, _ = _minmax_normalize(enhanced_radiance)
        metrics = evaluate_all(raw_ref_norm, enhanced_norm, cfg, trust_map=trust_map)

        metrics.update(verification.record)
        metrics.update(photometric_record)
        metrics.update({
            "zero_synthesis_guarantee_held": zero_synthesis_ok,
            "curve_map_abs_max": float(curve_map.abs().max().item()),
            "cosmic_ray_hit_fraction": float(fe.cosmic_ray_mask.mean()),
            "saturated_pixel_fraction": float(fe.saturation_mask.mean()),
            "radiance_units": fe.units,
            "radiance_mean": float(np.mean(enhanced_radiance)),
            "radiance_min": float(np.min(enhanced_radiance)),
            "radiance_max": float(np.max(enhanced_radiance)),
            "source_format": scene.source_format,
            "georeferenced": scene.crs is not None,
            "input_shape": list(scene.dn.shape),
            "device": str(self.device),
            "trained_weights_loaded": self.checkpoint_loaded,
            "checkpoint_name": self.checkpoint_name,
        })
        if self.checkpoint_audit:
            # Namespaced, because these describe the *model*, not this scene.
            metrics.update({f"checkpoint_{k}": v
                            for k, v in self.checkpoint_audit.items()})
        metrics["all_guardrails_passed"] = bool(
            metrics.get("structure_guardrail_passed", True)
            and metrics.get("physics_verification_passed", True)
            and zero_synthesis_ok
        )
        metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)

        # 6.3 -- lossless export -------------------------------------------
        report(94, "Stage 6.3: lossless export & previews")
        exp_cfg = cfg["export"]
        enhanced_path = out_dir / f"{stem}_enhanced.tif"
        export_geotiff(
            enhanced_radiance, str(enhanced_path),
            trust_map=trust_map if exp_cfg["write_trust_band"] else None,
            source_profile=scene.profile,
            config=cfg,
            tags={
                "SSIM_VS_RAW": metrics.get("ssim"),
                "ALL_GUARDRAILS_PASSED": metrics["all_guardrails_passed"],
                "MEAN_TRUST": metrics.get("mean_trust"),
            },
        )

        trust_path = None
        if exp_cfg["write_trust_sidecar"]:
            trust_path = out_dir / f"{stem}_trust.tif"
            export_trust_sidecar(trust_map, str(trust_path),
                                 source_profile=scene.profile, config=cfg)

        # 6.3b -- human-viewable renderings of the same products ------------
        preview = export_preview_images(
            output_dir=str(preview_dir or (out_dir / exp_cfg.get("preview", {})
                                           .get("directory", "previews"))),
            stem=stem,
            enhanced=enhanced_radiance,
            raw=fe.radiance_linear,
            trust_map=trust_map,
            config=cfg,
            trained=self.checkpoint_loaded,
        )
        metrics["preview_stretch"] = preview.get("preview_stretch")
        metrics["preview_stretch_limits"] = preview.get("preview_stretch_limits")

        report_path = export_metrics_report(
            metrics, str(out_dir / f"{stem}_metrics.json"),
            report_format=cfg["evaluation"].get("report_format", "json"),
        )

        report(100, "Complete")
        return {
            "enhanced_path": str(enhanced_path),
            "trust_path": str(trust_path) if trust_path else None,
            "metrics_path": report_path,
            "preview_dir": preview.get("preview_dir"),
            "preview_image": preview.get("preview_enhanced"),
            "preview_comparison": preview.get("preview_comparison"),
            "previews": {k: v for k, v in preview.items()
                         if k.startswith("preview_") and isinstance(v, str)},
            "metrics": metrics,
        }


# =============================================================================
# Synthetic demo scene
# =============================================================================
def make_synthetic_scene(size: int = 256, seed: int = 0) -> np.ndarray:
    """
    Builds a synthetic 16-bit low-illumination planetary scene for smoke
    testing: compressed dynamic range, photon-limited Poisson noise, Gaussian
    read noise, cosmic-ray hits, and bowl-shaped craters with sunlit rims.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    scene = 0.15 + 0.05 * (xx / size)          # smooth albedo gradient
    for _ in range(8):
        cy, cx = rng.integers(0, size, 2)
        r = rng.integers(8, max(9, size // 8))
        d = np.hypot(xx - cx, yy - cy)
        scene += 0.10 * np.exp(-((d - r) ** 2) / (2 * (r / 4) ** 2))   # rim
        scene -= 0.05 * np.exp(-(d ** 2) / (2 * (r / 2) ** 2))         # bowl
    scene = np.clip(scene, 0.01, 0.35)          # compressed low dynamic range

    photons = scene * 4000.0                    # low photon count
    noisy = rng.poisson(photons).astype(np.float64)
    noisy += rng.normal(0.0, 4.5, size=noisy.shape)     # read noise

    hits = 15                                    # cosmic ray / SEU spikes
    noisy[rng.integers(0, size, hits), rng.integers(0, size, hits)] += \
        rng.uniform(3000, 8000, hits)

    return np.clip(noisy + 128 + 64, 0, 65000).astype(np.uint16)


def write_demo_scene(path: Path, size: int = 256, seed: int = 0) -> Path:
    """
    Writes the synthetic scene as a georeferenced GeoTIFF on a lunar sphere,
    so the demo actually exercises CRS/transform preservation instead of
    silently skipping it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    array = make_synthetic_scene(size=size, seed=seed)

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        crs = CRS.from_proj4("+proj=eqc +R=1737400 +units=m +no_defs")
        transform = from_origin(-5000.0, 5000.0, 10.0, 10.0)   # 10 m/pixel
        profile = {
            "driver": "GTiff", "height": size, "width": size, "count": 1,
            "dtype": "uint16", "crs": crs, "transform": transform,
            "compress": "deflate",
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
            dst.update_tags(
                INCIDENCE_ANGLE="72.5", EMISSION_ANGLE="8.0",
                INSTRUMENT="AURA-NET synthetic demo",
            )
    except Exception:
        import tifffile
        tifffile.imwrite(str(path), array)

    return path


# =============================================================================
# CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AURA-NET -- physics-constrained planetary image enhancement.",
    )
    p.add_argument("--input", type=str,
                   help="Raw 16-bit PDS4 / FITS / GeoTIFF scene.")
    p.add_argument("--config", type=str, default="config/default_config.yaml",
                   help="Config YAML (relative paths resolve against the package).")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--preview-dir", type=str, default=None,
                   help="Folder for the viewable PNGs "
                        "(default: a 'previews' subfolder of the output dir).")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip the viewable PNG renderings entirely.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Trained weights for the learned submodules.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                   help="Override the configured device.")
    p.add_argument("--demo", action="store_true",
                   help="Generate and process a synthetic low-light scene.")
    p.add_argument("--demo-size", type=int, default=256,
                   help="Edge length of the synthetic demo scene.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    pipeline = AuraNetPipeline(config_path=args.config, device=args.device)

    if args.no_preview:
        pipeline.config["export"].setdefault("preview", {})["enabled"] = False

    if args.checkpoint:
        loaded = pipeline.load_checkpoint(args.checkpoint)
        print(f"[AURA-NET] Loaded checkpoint sections: {', '.join(loaded) or 'none'}")
    else:
        print("[AURA-NET] No checkpoint supplied -- running physics-only "
              "(learned stages are zero-initialized pass-throughs).")

    if args.demo:
        demo_path = write_demo_scene(
            pipeline._resolve(pipeline.config["paths"]["raw_dir"]) / "synthetic_demo_scene.tif",
            size=args.demo_size, seed=int(pipeline.config["project"]["seed"]),
        )
        input_path = str(demo_path)
        print(f"[AURA-NET] Synthetic demo scene: {input_path}")
    elif args.input:
        input_path = args.input
    else:
        raise SystemExit("Provide --input <path>, or --demo to smoke-test the pipeline.")

    result = pipeline.process(input_path, output_dir=args.output_dir,
                              preview_dir=args.preview_dir)

    print(json.dumps(result["metrics"], indent=2, default=str))

    print(f"\n[AURA-NET] Enhanced product : {result['enhanced_path']}")
    print(f"[AURA-NET] Trust map        : {result['trust_path']}")
    print(f"[AURA-NET] Metric report    : {result['metrics_path']}")

    if result.get("preview_dir"):
        print(f"\n[AURA-NET] Viewable images  : {result['preview_dir']}")
        for key, path in sorted(result.get("previews", {}).items()):
            if key in ("preview_dir", "preview_stretch"):
                continue
            print(f"             {key.removeprefix('preview_'):<12s}-> {Path(path).name}")
        if result.get("preview_image"):
            print(f"\n[AURA-NET] Open this one    : {result['preview_image']}")

    if not result["metrics"].get("all_guardrails_passed", True):
        print("\n[AURA-NET] WARNING: one or more physics guardrails FAILED. "
              "Treat this product as unvalidated pending review.")


if __name__ == "__main__":
    main()
