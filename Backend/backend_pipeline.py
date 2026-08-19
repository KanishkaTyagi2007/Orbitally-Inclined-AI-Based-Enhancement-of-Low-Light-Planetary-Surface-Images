"""
aura_pipeline.py
==================
AuraNet — Unified End-to-End Model & Execution Pipeline.

Problem statement
------------------
Planetary exploration missions can produce images with low illumination,
noise, and limited dynamic range. This pipeline improves visibility while
minimizing hallucinated structure, preserves scientific (radiometric)
fidelity, and reports quantitative image-quality evaluation for every run.

Stage order
-----------
    raw 16-bit DN
      -> [physics_frontend]   radiometric calibration, cosmic-ray scrub, Anscombe VST
      -> [wavelet_dequant]    SWT decomposition -> LL + {LH,HL,HH} per level
                               + implicit de-quantization per band
      -> [zero_dce]           reference-free illumination curve, applied to LL only
      -> [nafnet_denoiser]    activation-free detail restoration, applied to LH/HL/HH
      -> [wavelet_dequant]    ISWT reconstruction back to a single image
      -> [uncertainty_head]   per-pixel (mu, log-var) -> trust map
      -> [physics_frontend]   inverse Anscombe VST
      -> [evaluation.metrics] PSNR / SSIM / NIQE / BRISQUE / entropy gain
      -> [evaluation.exporter] enhanced GeoTIFF + trust map GeoTIFF

Every learned stage (Zero-DCE, NAFNet, uncertainty head) is untrained by
default (randomly initialized weights) — this scaffold defines the full
inference *and* loss-function machinery needed to train each stage, but
actual weights depend on the mission dataset the pipeline is pointed at.
Load trained checkpoints via `--checkpoint` (see AuraNetPipeline.load_checkpoint).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from models.physics_frontend import PhysicsFrontend
from models.wavelet_dequant import WaveletDequantizer
from models.zero_dce import ZeroDCE
from models.nafnet_denoiser import NAFNetDenoiser
from models.uncertainty_head import UncertaintyHead, compute_trust_map
from evaluation.metrics import evaluate_all
from evaluation.exporter import read_source_profile, export_geotiff, export_trust_sidecar


# =============================================================================
# Utility: read raw raster (rasterio if available/georeferenced, else tifffile)
# =============================================================================
def _read_raw(input_path: str):
    profile = None
    try:
        profile = read_source_profile(input_path)
        import rasterio
        with rasterio.open(input_path) as src:
            array = src.read(1)
    except Exception:
        import tifffile
        array = tifffile.imread(input_path)
        if array.ndim == 3:
            array = array[..., 0]  # single-band planetary product assumption
    return array, profile


def _minmax_normalize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    return ((x - lo) / (hi - lo)).astype(np.float32), lo, hi


def _minmax_denormalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)[None, None]  # (1,1,H,W)


# =============================================================================
# Pipeline
# =============================================================================
class AuraNetPipeline:
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config = self._load_config(config_path)
        self._set_seed(self.config["project"]["seed"])

        requested_device = self.config["project"]["device"]
        self.device = torch.device(
            requested_device if (requested_device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )

        # Physics-based, non-learned stage
        self.frontend = PhysicsFrontend(self.config)

        # Wavelet decomposition + implicit de-quantization
        self.wavelet = WaveletDequantizer(self.config)
        self.wavelet.dequantizer.to(self.device)

        # Zero-DCE: illumination curve estimator (LL band, 1 channel)
        z_cfg = self.config["zero_dce"]
        self.zero_dce = ZeroDCE(
            in_channels=1,
            num_filters=int(z_cfg["num_filters"]),
            num_conv_layers=int(z_cfg["num_conv_layers"]),
            num_iterations=int(z_cfg["num_iterations"]),
        ).to(self.device)

        # NAFNet: detail restorer (LH, HL, HH stacked as 3 channels)
        n_cfg = self.config["nafnet"]
        self.nafnet = NAFNetDenoiser(
            in_channels=3,
            width=int(n_cfg["width"]),
            enc_blocks=tuple(n_cfg["enc_blocks"]),
            middle_blocks=int(n_cfg["middle_blocks"]),
            dec_blocks=tuple(n_cfg["dec_blocks"]),
            dw_expand=int(n_cfg["dw_expand"]),
            ffn_expand=int(n_cfg["ffn_expand"]),
        ).to(self.device)

        # Uncertainty head: operates on the fully reconstructed image
        u_cfg = self.config["uncertainty"]
        self.uncertainty_head = UncertaintyHead(
            in_channels=1,
            hidden_channels=int(u_cfg["hidden_channels"]),
            min_log_var=float(u_cfg["min_log_var"]),
            max_log_var=float(u_cfg["max_log_var"]),
        ).to(self.device)

        self.zero_dce.eval()
        self.nafnet.eval()
        self.uncertainty_head.eval()

    # -------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_checkpoint(self, checkpoint_path: str):
        """Loads trained weights for the three learned submodules. Expects a
        dict with keys 'zero_dce', 'nafnet', 'uncertainty_head', and
        optionally 'dequantizer'."""
        state = torch.load(checkpoint_path, map_location=self.device)
        if "zero_dce" in state:
            self.zero_dce.load_state_dict(state["zero_dce"])
        if "nafnet" in state:
            self.nafnet.load_state_dict(state["nafnet"])
        if "uncertainty_head" in state:
            self.uncertainty_head.load_state_dict(state["uncertainty_head"])
        if "dequantizer" in state:
            self.wavelet.dequantizer.load_state_dict(state["dequantizer"])

    # -------------------------------------------------------------------
    @torch.no_grad()
    def process(self, input_path: str, output_dir: Optional[str] = None) -> dict:
        cfg = self.config
        output_dir = Path(output_dir or cfg["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        # 1) Load raw scene -----------------------------------------------
        raw_dn, source_profile = _read_raw(input_path)

        # 2) Physics frontend: calibration, cosmic-ray scrub, Anscombe VST -
        fe = self.frontend.process(raw_dn)
        stabilized = fe.stabilized  # variance-stabilized domain, arbitrary range

        norm_stab, lo, hi = _minmax_normalize(stabilized)

        # 3) SWT decomposition + implicit de-quantization ------------------
        bands, LL = self.wavelet.decompose(norm_stab)
        LL_dq = self.wavelet.dequantize_band(LL)
        # Only the finest level's detail bands are restored by NAFNet; coarser
        # levels (if any) are passed through unchanged to limit the denoiser's
        # receptive footprint to genuinely noisy high-frequency content.
        finest = bands[-1]
        LH_dq = self.wavelet.dequantize_band(finest.LH)
        HL_dq = self.wavelet.dequantize_band(finest.HL)
        HH_dq = self.wavelet.dequantize_band(finest.HH)

        # 4) Zero-DCE on LL (illumination) ---------------------------------
        LL_norm, LL_lo, LL_hi = _minmax_normalize(LL_dq)
        LL_tensor = _to_tensor(LL_norm, self.device)
        LL_enhanced_t, curve_params = self.zero_dce(LL_tensor)
        LL_enhanced = _minmax_denormalize(
            LL_enhanced_t[0, 0].cpu().numpy(), LL_lo, LL_hi
        )

        # 5) NAFNet on detail bands (LH, HL, HH) ----------------------------
        detail_stack = np.stack([LH_dq, HL_dq, HH_dq], axis=0)
        detail_norm, d_lo, d_hi = _minmax_normalize(detail_stack)
        detail_tensor = torch.from_numpy(detail_norm).float().to(self.device)[None]
        detail_denoised_t = self.nafnet(detail_tensor)
        detail_denoised = _minmax_denormalize(
            detail_denoised_t[0].cpu().numpy(), d_lo, d_hi
        )
        finest.LH[:] = detail_denoised[0]
        finest.HL[:] = detail_denoised[1]
        finest.HH[:] = detail_denoised[2]

        # 6) ISWT reconstruction --------------------------------------------
        reconstructed_norm = self.wavelet.reconstruct(bands, LL_enhanced)

        # 7) Uncertainty head -> trust map -----------------------------------
        recon_tensor = _to_tensor(reconstructed_norm.astype(np.float32), self.device)
        mu_t, log_var_t = self.uncertainty_head(recon_tensor)
        mu = mu_t[0, 0].cpu().numpy()
        log_var = log_var_t[0, 0].cpu().numpy()
        trust_map, low_trust_mask = compute_trust_map(
            log_var, low_trust_threshold=cfg["uncertainty"]["trust_map"]["low_trust_threshold"]
        )

        # 8) Undo min-max normalization, then inverse Anscombe VST -----------
        mu_stab_domain = _minmax_denormalize(mu, lo, hi)
        enhanced_linear = self.frontend.vst.inverse(mu_stab_domain)

        # Reference for full-reference metrics: physics-calibrated raw,
        # contrast-stretched to [0, 1] the same way the enhanced output is.
        raw_ref_norm, _, _ = _minmax_normalize(fe.calibrated_linear)
        enhanced_norm, _, _ = _minmax_normalize(enhanced_linear)

        # 9) Quantitative evaluation ------------------------------------------
        metrics = evaluate_all(raw_ref_norm, enhanced_norm, cfg)
        metrics["low_trust_pixel_fraction"] = float(low_trust_mask.mean())
        metrics["cosmic_ray_hit_fraction"] = float(fe.cosmic_ray_mask.mean())
        metrics["saturated_pixel_fraction"] = float(fe.saturation_mask.mean())

        # 10) Export ------------------------------------------------------------
        exp_cfg = cfg["export"]
        enhanced_path = output_dir / f"{stem}_enhanced.tif"
        export_geotiff(
            enhanced_norm, str(enhanced_path), source_profile=source_profile,
            trust_map=trust_map if exp_cfg["write_trust_band"] else None,
            dtype=exp_cfg["output_dtype"], compress=exp_cfg["compress"],
        )

        trust_path = None
        if exp_cfg["write_trust_sidecar"]:
            trust_path = output_dir / f"{stem}_trust.tif"
            export_trust_sidecar(trust_map, str(trust_path), source_profile=source_profile)

        report_path = output_dir / f"{stem}_metrics.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return {
            "enhanced_path": str(enhanced_path),
            "trust_path": str(trust_path) if trust_path else None,
            "metrics_path": str(report_path),
            "metrics": metrics,
        }


# =============================================================================
# CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AuraNet — AI-assisted planetary image enhancement pipeline."
    )
    p.add_argument("--input", type=str, help="Path to a raw 16-bit PDS4/GeoTIFF image.")
    p.add_argument("--config", type=str, default="config/default_config.yaml")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Optional path to trained weights (zero_dce/nafnet/uncertainty_head).")
    p.add_argument("--demo", action="store_true",
                    help="Run on a synthetic low-light/noisy scene instead of --input, "
                         "to smoke-test the pipeline without real mission data.")
    return p


def _make_synthetic_scene(size: int = 256, seed: int = 0) -> np.ndarray:
    """Synthetic 16-bit low-illumination, noisy, low-dynamic-range scene for
    smoke-testing the pipeline end to end without real mission data."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    # a few "craters" (dark ring + bright rim) plus a smooth albedo gradient
    scene = 0.15 + 0.05 * (xx / size)
    for _ in range(6):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(10, 30)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        scene += 0.10 * np.exp(-((d - r) ** 2) / (2 * (r / 4) ** 2))
        scene -= 0.05 * np.exp(-(d ** 2) / (2 * (r / 2) ** 2))
    scene = np.clip(scene, 0.01, 0.35)  # compressed low dynamic range

    dn_scale = 4000.0  # low photon count -> strong Poisson noise
    photon_signal = scene * dn_scale
    noisy = rng.poisson(photon_signal).astype(np.float64)
    noisy += rng.normal(0, 4.5, size=noisy.shape)  # read noise
    # sprinkle a few cosmic ray hits
    n_hits = 15
    hit_y = rng.integers(0, size, n_hits)
    hit_x = rng.integers(0, size, n_hits)
    noisy[hit_y, hit_x] += rng.uniform(3000, 8000, n_hits)

    dn = np.clip(noisy + 128 + 64, 0, 65000)  # add bias + black level
    return dn.astype(np.uint16)


def main():
    args = _build_arg_parser().parse_args()
    pipeline = AuraNetPipeline(config_path=args.config)

    if args.checkpoint:
        pipeline.load_checkpoint(args.checkpoint)

    if args.demo:
        import tifffile
        demo_dir = Path(pipeline.config["paths"]["raw_dir"])
        demo_dir.mkdir(parents=True, exist_ok=True)
        demo_path = demo_dir / "synthetic_demo_scene.tif"
        tifffile.imwrite(str(demo_path), _make_synthetic_scene())
        input_path = str(demo_path)
        print(f"[AuraNet] Generated synthetic demo scene at {input_path}")
    elif args.input:
        input_path = args.input
    else:
        raise SystemExit("Provide --input <path> or use --demo to smoke-test the pipeline.")

    result = pipeline.process(input_path, output_dir=args.output_dir)

    print(f"[AuraNet] Enhanced image: {result['enhanced_path']}")
    print(f"[AuraNet] Trust map:      {result['trust_path']}")
    print(f"[AuraNet] Metrics report: {result['metrics_path']}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
