"""
train.py
=========
AURA-NET self-supervised trainer, for real planetary products with no ground
truth.

    python train.py prepare --data <bundle folder>      # build the tile cache
    python train.py fit                                 # train all four modules
    python train.py all --data <bundle folder>          # both, in one go

Why the phases are ordered
--------------------------
The four learned modules sit at different depths of the pipeline, and each one
sees a domain that the modules *before* it produced. Training them
simultaneously would train each on an input distribution that its upstream
neighbour is still changing, so they are trained in pipeline order and every
phase generates its data by running the already-trained stages:

    A  de-quantizer      stage 2.1   VST-stabilized radiance
    B  Zero-DCE++        stage 3A    normalized LL band  <- after A
       detail restorer   stage 3B    normalized LH/HL/HH <- after A
    C  uncertainty head  stage 5.3   normalized tone-mapped radiance <- after A, B

By phase C the head is looking at exactly what the trained pipeline will hand
it at inference, which is the only way its variance means anything.

Why the result does not hallucinate
-----------------------------------
No objective in this file is minimized by inventing structure.

  * Phase A optimizes flatness in already-flat regions plus fidelity, under an
    architectural bound of half a quantizer bin.
  * Phase B's reconstruction target is the *measured* tile. The input is that
    same tile with one extra realization of the sensor's own noise, so the only
    way to lower the loss is to remove noise the pipeline itself added. Two
    further terms charge the network for gradient energy it did not receive
    (`StructureAdditionPenalty`) and for altering an un-degraded tile at all
    (`IdentityAnchor`).
  * Zero-DCE is zero-reference by construction and monotonic by architecture.
  * Phase C's target is the measured scene and its input is a noisier copy, so
    the head cannot drive its variance to the floor -- the residual it is
    modelling is real and irreducible.

`--audit` re-measures the structure-addition fraction on held-out products
after training and writes it into the report, so the claim is evidenced rather
than asserted.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from aura_pipeline import AuraNetPipeline, _minmax_normalize, _symmetric_normalize
from models.uncertainty_head import (
    HeteroscedasticNLLLoss,
    PhotometricFluxConservationLoss,
    SobelGradientConsistencyLoss,
)
from models.zero_dce import ZeroDCELoss
from training.objectives import (
    BandingSuppressionLoss,
    CharbonnierLoss,
    IdentityAnchor,
    StructureAdditionPenalty,
    structure_addition_fraction,
)
from training.tiles import TileCache, build_cache

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PACKAGE_DIR / "data" / "training" / "tmc_tiles"
DEFAULT_CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"


# =============================================================================
# Hyper-parameters
# =============================================================================
@dataclass
class TrainSettings:
    batch_size: int = 4
    epochs_dequant: int = 6
    epochs_detail: int = 8
    epochs_zero_dce: int = 8
    epochs_uncertainty: int = 6
    lr_dequant: float = 1e-4
    lr_detail: float = 2e-4
    lr_zero_dce: float = 1e-4
    lr_uncertainty: float = 2e-4
    grad_clip: float = 1.0

    # Anti-hallucination term weights (phase B).
    w_reconstruction: float = 1.0
    w_identity: float = 0.5
    w_structure: float = 2.0
    w_gradient: float = 0.2
    structure_margin: float = 0.5
    identity_samples: int = 1        # tiles per step for the anchor

    # Phase C.
    w_flux: float = 0.5

    val_fraction: float = 0.12
    seed: int = 42


# =============================================================================
# Batch preparation -- stages 2.1 / 2.2 run exactly as the pipeline runs them
# =============================================================================
class DomainCache:
    """
    Materializes each phase's input domain once, using the pipeline's own
    `WaveletDequantizer`, Zero-DCE and normalization helpers.

    The de-quantizer, and later Zero-DCE and the restorer, are *frozen* by the
    time the phase that consumes their output starts -- that is what the phase
    ordering buys. So the SWT decompositions and the tone-mapped scenes are
    fixed data, not something to recompute every epoch: derived once, written to
    a memory-mapped `.npy` beside the tile cache, and read back per batch. On
    CPU this is the difference between a run dominated by `pywt` and one
    dominated by the optimizer.

    Nothing here back-propagates across the SWT -- `pywt` is numpy -- which is
    the reason the modules are trained in phases rather than end-to-end.
    """

    def __init__(self, pipeline: AuraNetPipeline, cache: TileCache,
                 work_dir: Optional[Path] = None):
        self.pipeline = pipeline
        self.cache = cache
        self.device = pipeline.device
        self.work_dir = Path(work_dir or (cache.dir / "domains"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.levels = int(pipeline.config["wavelet"]["levels"])
        self._ll: Optional[np.ndarray] = None
        self._details: Optional[np.ndarray] = None
        self._toned: Optional[np.ndarray] = None

    # -- stage 2.1 + 2.2 ---------------------------------------------------
    def _decompose(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One tile -> (LL, detail stack (levels, 3, H, W)), post de-quantization."""
        dequantized = self.pipeline.wavelet.dequantize(image, self.device)
        bands, ll, _ = self.pipeline.wavelet.decompose(dequantized)
        details = np.stack([np.stack([b.LH, b.HL, b.HH], axis=0) for b in bands])
        return ll, details

    def _memmap(self, name: str, shape: tuple) -> np.ndarray:
        from numpy.lib.format import open_memmap
        return open_memmap(self.work_dir / name, mode="w+", dtype=np.float32,
                           shape=shape)

    # -- phase B domain ----------------------------------------------------
    def build_band_domain(self, verbose: bool = True) -> None:
        """
        Writes the Zero-DCE and detail-restorer inputs for every tile.

        `details` holds, per tile, three stacks: the noisy input, the measured
        target scaled to match it, and the measured tile normalized on its own
        scale (the identity anchor's input). Inference normalizes the detail
        stack by its own peak magnitude, so the input is scaled that way to keep
        the domain identical; scaling the target by that same peak is what makes
        the difference between them a reconstruction error rather than a
        comparison of two differently-stretched images.
        """
        n, size = len(self.cache), self.cache.tile_size
        planes = self.levels * 3
        self._details = self._memmap("details.npy", (n, 3, planes, size, size))
        self._ll = self._memmap("ll.npy", (n, size, size))

        started = time.perf_counter()
        for index in range(n):
            noisy = self.cache.plane(index, "stab_noisy")
            clean = self.cache.plane(index, "stab_clean")

            ll_clean, det_clean = self._decompose(clean)
            _, det_noisy = self._decompose(noisy)

            noisy_norm, noisy_scale = _symmetric_normalize(det_noisy)
            measured_norm, _ = _symmetric_normalize(det_clean)

            shape = (planes, size, size)
            self._details[index, 0] = noisy_norm.reshape(shape)
            self._details[index, 1] = (det_clean / noisy_scale).reshape(shape)
            self._details[index, 2] = measured_norm.reshape(shape)
            self._ll[index] = _minmax_normalize(ll_clean)[0]

            if verbose and (index + 1) % 100 == 0:
                print(f"    bands {index + 1}/{n} "
                      f"({time.perf_counter() - started:.0f}s)")

        self._details.flush()
        self._ll.flush()
        if verbose:
            print(f"    band domain built in {time.perf_counter() - started:.0f}s")

    def ll_batch(self, indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self._ll[indices], dtype=np.float32)[:, None]).to(self.device)

    def detail_batch(self, indices: np.ndarray, which: int) -> torch.Tensor:
        """
        (B*levels, 3, H, W) -- the exact shape stage 3B sees, where the SWT
        levels ride the batch axis because the transform is undecimated and
        every level sits on the native pixel grid.
        """
        block = np.asarray(self._details[indices, which], dtype=np.float32)
        b, _, h, w = block.shape
        return torch.from_numpy(block.reshape(b * self.levels, 3, h, w)).to(self.device)

    # -- phase C domain ----------------------------------------------------
    @torch.no_grad()
    def build_toned_domain(self, verbose: bool = True) -> None:
        """
        Runs stages 2 through 4 with the phase A/B weights in place, so the
        uncertainty head's input is what the *trained* pipeline produces rather
        than what the physics-only one would have.
        """
        n, size = len(self.cache), self.cache.tile_size
        self._toned = self._memmap("toned.npy", (n, 2, size, size))
        started = time.perf_counter()

        for index in range(n):
            toned_noisy = self._through_stage_four(self.cache.plane(index, "stab_noisy"))
            toned_clean = self._through_stage_four(self.cache.plane(index, "stab_clean"))
            normalized, lo, hi = _minmax_normalize(toned_noisy)
            self._toned[index, 0] = normalized
            # Same limits for the target: inference has only the observed scene
            # to normalize from, so the head must learn in that frame.
            self._toned[index, 1] = np.clip(
                (toned_clean - lo) / max(hi - lo, 1e-12), -1.0, 2.0)

            if verbose and (index + 1) % 100 == 0:
                print(f"    toned {index + 1}/{n} "
                      f"({time.perf_counter() - started:.0f}s)")

        self._toned.flush()
        if verbose:
            print(f"    toned domain built in {time.perf_counter() - started:.0f}s")

    def toned_batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        block = np.asarray(self._toned[indices], dtype=np.float32)
        pair = torch.from_numpy(block).to(self.device)
        return pair[:, 0:1], pair[:, 1:2]

    def _through_stage_four(self, stabilized: np.ndarray) -> np.ndarray:
        """Stages 2.1 -> 3A/3B -> 3C -> inverse VST -> 4, as `process` does."""
        pipeline = self.pipeline
        dequantized = pipeline.wavelet.dequantize(stabilized, self.device)
        bands, ll, shape = pipeline.wavelet.decompose(dequantized)

        ll_norm, lo, hi = _minmax_normalize(ll)
        ll_tensor = torch.from_numpy(ll_norm)[None, None].to(self.device)
        enhanced_ll = pipeline.zero_dce(ll_tensor)[0][0, 0].cpu().numpy()
        enhanced_ll = enhanced_ll * (hi - lo) + lo

        details = np.stack([np.stack([b.LH, b.HL, b.HH], axis=0) for b in bands])
        normalized, scale = _symmetric_normalize(details)
        restored = pipeline.detail_restorer(
            torch.from_numpy(normalized).float().to(self.device)
        ).cpu().numpy() * scale
        for level, band in enumerate(bands):
            band.LH, band.HL, band.HH = restored[level]

        reconstructed = pipeline.frontend.vst.inverse(
            pipeline.wavelet.reconstruct(bands, enhanced_ll, shape))
        return pipeline.photometry.process(reconstructed, {})[0]


# =============================================================================
# Phases
# =============================================================================
def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(indices)
    for start in range(0, len(order) - batch_size + 1, batch_size):
        yield order[start:start + batch_size]


def _log_epoch(phase: str, epoch: int, epochs: int, stats: dict,
               seconds: float) -> None:
    body = "  ".join(f"{k}={v:.5g}" for k, v in stats.items())
    print(f"  [{phase}] epoch {epoch}/{epochs}  {body}  ({seconds:.1f}s)")


def train_dequantizer(pipeline: AuraNetPipeline, domains: DomainCache,
                      train_idx: np.ndarray, val_idx: np.ndarray,
                      settings: TrainSettings, rng: np.random.Generator) -> list[dict]:
    """Phase A -- stage 2.1, zero-reference banding suppression."""
    module = pipeline.wavelet.dequantizer
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=settings.lr_dequant)
    criterion = BandingSuppressionLoss()
    history = []

    for epoch in range(1, settings.epochs_dequant + 1):
        started, totals, count = time.perf_counter(), {}, 0
        for batch in _batches(train_idx, settings.batch_size, rng):
            measured = torch.from_numpy(
                domains.cache.batch(batch, "stab_clean")).to(domains.device)
            loss, parts = criterion(module(measured), measured)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), settings.grad_clip)
            optimizer.step()

            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1

        stats = {k: v / max(count, 1) for k, v in totals.items()}
        with torch.no_grad():
            measured = torch.from_numpy(
                domains.cache.batch(val_idx[:settings.batch_size], "stab_clean")
            ).to(domains.device)
            stats["val_total"] = float(criterion(module(measured), measured)[0])
        history.append({"epoch": epoch, **stats})
        _log_epoch("A dequant", epoch, settings.epochs_dequant, stats,
                   time.perf_counter() - started)

    module.eval()
    return history


def train_zero_dce(pipeline: AuraNetPipeline, domains: DomainCache,
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   settings: TrainSettings, rng: np.random.Generator) -> list[dict]:
    """Phase B1 -- stage 3A, the paper's three zero-reference objectives."""
    module = pipeline.zero_dce
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=settings.lr_zero_dce)
    criterion = ZeroDCELoss(pipeline.config).to(domains.device)
    history = []

    for epoch in range(1, settings.epochs_zero_dce + 1):
        started, totals, count = time.perf_counter(), {}, 0
        for batch in _batches(train_idx, settings.batch_size, rng):
            ll = domains.ll_batch(batch)
            enhanced, curve_map = module(ll)
            loss, parts = criterion(enhanced, ll, curve_map)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), settings.grad_clip)
            optimizer.step()

            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            totals["curve_abs_max"] = max(totals.get("curve_abs_max", 0.0),
                                          float(curve_map.abs().max()))
            count += 1

        stats = {k: (v if k == "curve_abs_max" else v / max(count, 1))
                 for k, v in totals.items()}
        history.append({"epoch": epoch, **stats})
        _log_epoch("B1 zero-dce", epoch, settings.epochs_zero_dce, stats,
                   time.perf_counter() - started)

    module.eval()
    return history


def train_detail_restorer(pipeline: AuraNetPipeline, domains: DomainCache,
                          train_idx: np.ndarray, val_idx: np.ndarray,
                          settings: TrainSettings,
                          rng: np.random.Generator) -> list[dict]:
    """
    Phase B2 -- stage 3B. Noisier2Noise reconstruction, held to the measurement
    by the identity anchor and the structure-addition penalty.
    """
    module = pipeline.detail_restorer
    module.train()
    # The PSF deconvolution in front of the denoiser is a fixed physical
    # operator, not something to fit -- only the denoiser's weights are trained.
    optimizer = torch.optim.Adam(module.denoiser.parameters(), lr=settings.lr_detail)

    reconstruction = CharbonnierLoss()
    identity = IdentityAnchor()
    structure = StructureAdditionPenalty(settings.structure_margin).to(domains.device)
    gradient = SobelGradientConsistencyLoss(pipeline.config).to(domains.device)
    history = []

    for epoch in range(1, settings.epochs_detail + 1):
        started, totals, count = time.perf_counter(), {}, 0
        for batch in _batches(train_idx, settings.batch_size, rng):
            noisy = domains.detail_batch(batch, 0)
            target = domains.detail_batch(batch, 1)
            # The anchor is a regularizer, not a fit target, so it is evaluated
            # on a slice rather than the whole batch -- a second full pass
            # through NAFNet is the single most expensive thing in this loop and
            # the term's gradient is just as informative from a few rows.
            anchor_rows = max(1, settings.identity_samples) * domains.levels
            measured = domains.detail_batch(batch, 2)[:anchor_rows]

            restored = module(noisy)
            restored_measured = module(measured)

            l_recon = reconstruction(restored, target)
            l_identity = identity(restored_measured, measured)
            l_structure = structure(restored, target)
            l_gradient = gradient(restored, target)

            loss = (settings.w_reconstruction * l_recon
                    + settings.w_identity * l_identity
                    + settings.w_structure * l_structure
                    + settings.w_gradient * l_gradient)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.denoiser.parameters(), settings.grad_clip)
            optimizer.step()

            for key, value in (("reconstruction", l_recon), ("identity", l_identity),
                               ("structure_add", l_structure), ("gradient", l_gradient),
                               ("total", loss)):
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            count += 1

        stats = {k: v / max(count, 1) for k, v in totals.items()}
        with torch.no_grad():
            val_batch = val_idx[:settings.batch_size]
            restored = module(domains.detail_batch(val_batch, 0))
            target = domains.detail_batch(val_batch, 1)
            stats["val_reconstruction"] = float(reconstruction(restored, target))
            stats["val_added_structure_fraction"] = structure_addition_fraction(
                restored, target, settings.structure_margin)
        history.append({"epoch": epoch, **stats})
        _log_epoch("B2 detail", epoch, settings.epochs_detail, stats,
                   time.perf_counter() - started)

    module.eval()
    return history


def train_uncertainty(pipeline: AuraNetPipeline, domains: DomainCache,
                      train_idx: np.ndarray, val_idx: np.ndarray,
                      settings: TrainSettings, rng: np.random.Generator) -> list[dict]:
    """
    Phase C -- stage 5.3. Heteroscedastic NLL against the measured scene, from a
    noisier copy of it, so the learned variance tracks real predictive error.
    """
    module = pipeline.uncertainty_head
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=settings.lr_uncertainty)
    nll = HeteroscedasticNLLLoss()
    flux = PhotometricFluxConservationLoss(pipeline.config).to(domains.device)
    history = []

    for epoch in range(1, settings.epochs_uncertainty + 1):
        started, totals, count = time.perf_counter(), {}, 0
        for batch in _batches(train_idx, settings.batch_size, rng):
            noisy, target = domains.toned_batch(batch)
            mu, log_var = module(noisy)

            l_nll = nll(mu, log_var, target)
            # Radiometry is stage 4's business, not the head's: the flux term
            # holds mu's multi-scale mean structure to the input it is refining.
            l_flux = flux(mu.clamp_min(1e-6), noisy.clamp_min(1e-6))
            loss = l_nll + settings.w_flux * l_flux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), settings.grad_clip)
            optimizer.step()

            for key, value in (("nll", l_nll), ("flux", l_flux), ("total", loss)):
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            totals["log_var_std"] = totals.get("log_var_std", 0.0) + float(log_var.std())
            count += 1

        stats = {k: v / max(count, 1) for k, v in totals.items()}
        with torch.no_grad():
            noisy, target = domains.toned_batch(val_idx[:settings.batch_size])
            mu, log_var = module(noisy)
            stats["val_nll"] = float(nll(mu, log_var, target))
            # A variance field with no spread is a uniform trust map, which is
            # the uninformative case the dashboard already warns about.
            stats["val_log_var_spread"] = float(log_var.max() - log_var.min())
        history.append({"epoch": epoch, **stats})
        _log_epoch("C uncertainty", epoch, settings.epochs_uncertainty, stats,
                   time.perf_counter() - started)

    module.eval()
    return history


# =============================================================================
# Audit
# =============================================================================
@torch.no_grad()
@torch.no_grad()
def audit(pipeline: AuraNetPipeline, domains: DomainCache, val_idx: np.ndarray,
          settings: TrainSettings, max_batches: int = 8) -> dict:
    """
    Post-training measurement on held-out products.

    These are the numbers that support the claim, so they are measured on
    products no phase ever saw, and they are reported whatever they say.

    Every module is put in eval mode first, so what is measured is the network
    the pipeline will actually run -- including the zero-init identity shortcuts,
    which are live in eval and suppressed during training.
    """
    for module in (pipeline.detail_restorer, pipeline.zero_dce,
                   pipeline.uncertainty_head, pipeline.wavelet.dequantizer):
        module.eval()

    added, recon, curve_max, spreads = [], [], 0.0, []
    reconstruction = CharbonnierLoss()

    for count, start in enumerate(range(0, len(val_idx), settings.batch_size)):
        if count >= max_batches:
            break
        batch = val_idx[start:start + settings.batch_size]
        if len(batch) < 1:
            break

        target = domains.detail_batch(batch, 1)
        restored = pipeline.detail_restorer(domains.detail_batch(batch, 0))
        added.append(structure_addition_fraction(restored, target,
                                                 settings.structure_margin))
        recon.append(float(reconstruction(restored, target)))

        ll = domains.ll_batch(batch)
        curve_max = max(curve_max, float(pipeline.zero_dce(ll)[1].abs().max()))

        toned_noisy, _ = domains.toned_batch(batch)
        log_var = pipeline.uncertainty_head(toned_noisy)[1]
        spreads.append(float(log_var.max() - log_var.min()))

    return {
        "held_out_tiles": int(min(len(val_idx), max_batches * settings.batch_size)),
        "added_structure_fraction": round(float(np.mean(added)), 6) if added else None,
        "detail_reconstruction": round(float(np.mean(recon)), 6) if recon else None,
        "zero_synthesis_guarantee_held": bool(curve_max < 1.0),
        "curve_map_abs_max": round(curve_max, 6),
        "log_var_spread": round(float(np.mean(spreads)), 4) if spreads else None,
        "trust_map_informative": bool(spreads and float(np.mean(spreads)) > 1e-3),
    }


# =============================================================================
# Orchestration
# =============================================================================
def do_prepare(args) -> dict:
    from pds4_bundle import discover_products

    config = yaml.safe_load(Path(_resolve(args.config)).read_text(encoding="utf-8"))
    from models.physics_frontend import PhysicsFrontend

    products = [p for p in discover_products(args.data) if p.readable]
    if not products:
        raise SystemExit(f"No readable PDS4 products under {args.data}")

    # The calibration constants in the config belong to one instrument. Stage 1
    # converts DN to electrons with that gain and divides by that exposure time,
    # so a product from a different camera enters the VST with the wrong noise
    # model -- and the models would learn to denoise a noise level nothing in
    # the target domain has. Filtering here beats discovering it in the metrics.
    if args.instrument:
        wanted = args.instrument.lower()
        kept = [p for p in products if p.instrument.lower() == wanted]
        dropped = [p for p in products if p.instrument.lower() != wanted]
        if dropped:
            print(f"[train] excluding {len(dropped)} non-{wanted.upper()} product(s): "
                  + ", ".join(sorted({p.instrument.upper() for p in dropped})))
            for p in dropped:
                print(f"          {p.name}")
        products = kept
        if not products:
            raise SystemExit(
                f"No {wanted.upper()} products under {args.data}. "
                "Pass --instrument '' to train on everything found.")

    print(f"[train] {len(products)} products under {args.data}")
    manifest = build_cache(
        products, args.cache, config, PhysicsFrontend(config),
        tiles_per_product=args.tiles_per_product, tile_size=args.tile_size,
        noise_boost=args.noise_boost, seed=args.seed,
    )
    print(f"[train] cached {manifest['tile_count']} tiles from "
          f"{len(manifest['products'])} products in {manifest['build_seconds']}s "
          f"-> {args.cache}")
    print(f"[train] rejected windows: {manifest['rejected']}")
    return manifest


def do_fit(args) -> dict:
    settings = TrainSettings(
        batch_size=args.batch, epochs_dequant=args.epochs_dequant,
        epochs_detail=args.epochs_detail, epochs_zero_dce=args.epochs_zero_dce,
        epochs_uncertainty=args.epochs_uncertainty, seed=args.seed,
        w_structure=args.w_structure, w_identity=args.w_identity,
        structure_margin=args.structure_margin,
    )
    cache = TileCache(args.cache)
    pipeline = AuraNetPipeline(config_path=args.config, device=args.device)
    domains = DomainCache(pipeline, cache)
    rng = np.random.default_rng(settings.seed)

    # Retraining one phase is the common case once something is diagnosed:
    # `--resume <ckpt> --phases B1` re-fits the illumination curve and leaves
    # the other three modules exactly as they were. Without --resume, a
    # --phases subset would silently ship zero-initialized weights for the
    # phases it skipped, so the two flags belong together.
    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()]         if args.phases else ["A", "B1", "B2", "C"]
    unknown = set(phases) - {"A", "B1", "B2", "C"}
    if unknown:
        raise SystemExit(f"Unknown phase(s): {', '.join(sorted(unknown))}. "
                         "Choose from A, B1, B2, C.")

    resumed: list[str] = []
    if args.resume:
        resumed = pipeline.load_checkpoint(str(_resolve(args.resume)))
        print(f"[train] resumed from '{Path(args.resume).name}': "
              f"{', '.join(resumed) or 'nothing matched'}")
    elif set(phases) != {"A", "B1", "B2", "C"}:
        print("[train] WARNING: training a subset of phases without --resume. "
              "The phases you skipped will be saved zero-initialized, i.e. as "
              "identity pass-throughs.")

    train_idx, val_idx = cache.split(settings.val_fraction, settings.seed)
    if len(train_idx) < settings.batch_size or len(val_idx) < 1:
        raise SystemExit(
            f"Cache too small to split: {len(train_idx)} train / {len(val_idx)} "
            "validation tiles. Rebuild with more --tiles-per-product."
        )

    held_out = sorted({cache.manifest["tiles"][int(i)]["product"] for i in val_idx})
    print(f"[train] {len(train_idx)} training tiles, {len(val_idx)} held out "
          f"from {len(held_out)} product(s):")
    for name in held_out:
        print(f"          {name}")
    print(f"[train] device: {pipeline.device}   config: {Path(args.config).name}")

    started = time.perf_counter()
    history: dict[str, list[dict]] = {}
    print(f"[train] phases: {', '.join(phases)}")

    # Phase A -- stage 2.1. Reads the cached stage-1 planes directly.
    if "A" in phases:
        history["A_dequantizer"] = train_dequantizer(
            pipeline, domains, train_idx, val_idx, settings, rng)

    # The de-quantizer is frozen from here, so every tile's SWT decomposition is
    # fixed: derive them all once instead of once per epoch. Needed by B1, B2
    # and (through `_through_stage_four`) the audit.
    if {"B1", "B2"} & set(phases) or val_idx.size:
        print("[train] materializing the stage-2 band domain...")
        domains.build_band_domain()

    if "B1" in phases:
        history["B1_zero_dce"] = train_zero_dce(
            pipeline, domains, train_idx, val_idx, settings, rng)
    if "B2" in phases:
        history["B2_detail_restorer"] = train_detail_restorer(
            pipeline, domains, train_idx, val_idx, settings, rng)

    # Stages 2 and 3 are frozen too, so the tone-mapped scene the uncertainty
    # head will see at inference is likewise fixed data.
    print("[train] materializing the stage-4 tone-mapped domain...")
    domains.build_toned_domain()

    if "C" in phases:
        history["C_uncertainty_head"] = train_uncertainty(
            pipeline, domains, train_idx, val_idx, settings, rng)
    elapsed = time.perf_counter() - started

    print("[train] auditing on held-out products...")
    audit_record = audit(pipeline, domains, val_idx, settings)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.name

    torch.save({
        "dequantizer": pipeline.wavelet.dequantizer.state_dict(),
        "zero_dce": pipeline.zero_dce.state_dict(),
        "detail_restorer": pipeline.detail_restorer.state_dict(),
        "uncertainty_head": pipeline.uncertainty_head.state_dict(),
        "config_name": Path(args.config).name,
        "tile_manifest": {k: v for k, v in cache.manifest.items() if k != "tiles"},
        "settings": settings.__dict__,
        "audit": audit_record,
    }, checkpoint_path)

    report = {
        "checkpoint": str(checkpoint_path),
        "phases_trained": phases,
        "resumed_from": str(args.resume) if args.resume else None,
        "resumed_sections": resumed,
        "config": str(_resolve(args.config)),
        "device": str(pipeline.device),
        "train_tiles": int(len(train_idx)),
        "validation_tiles": int(len(val_idx)),
        "held_out_products": held_out,
        "train_seconds": round(elapsed, 1),
        "settings": settings.__dict__,
        "audit": audit_record,
        "history": history,
    }
    report_path = checkpoint_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[train] checkpoint -> {checkpoint_path}")
    print(f"[train] report     -> {report_path}")
    print(f"[train] audit      -> {json.dumps(audit_record, indent=2)}")
    return report


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PACKAGE_DIR / p)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AURA-NET self-supervised trainer for real PDS4 products.")
    parser.add_argument("command", choices=["prepare", "fit", "all"])
    parser.add_argument("--data", type=str, default=None,
                        help="Bundle folder to sample training tiles from.")
    parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--config", type=str, default="config/ch2_tmc.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--name", type=str, default="aura_net_tmc.pt")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])

    parser.add_argument("--instrument", type=str, default="tmc",
                        help="Keep only products from this instrument, whose "
                             "calibration constants the config describes. "
                             "Pass '' to keep everything.")
    parser.add_argument("--tiles-per-product", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=160,
                        help="Must be a multiple of 2**wavelet.levels.")
    parser.add_argument("--noise-boost", type=float, default=2.0,
                        help="Poisson variance multiplier for the noisier input.")

    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs-dequant", type=int, default=6)
    parser.add_argument("--epochs-zero-dce", type=int, default=10)
    parser.add_argument("--epochs-detail", type=int, default=6)
    parser.add_argument("--epochs-uncertainty", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Load an existing checkpoint before training, so "
                             "phases you skip keep their trained weights.")
    parser.add_argument("--phases", type=str, default=None,
                        help="Comma-separated subset of A,B1,B2,C. "
                             "Default: all four. Use with --resume.")

    # The anti-hallucination knobs. Raising --w-structure or lowering
    # --structure-margin makes the restorer more conservative about producing
    # edge energy the measurement did not contain, at the cost of some
    # deconvolution sharpening; the held-out `added_structure_fraction` in the
    # training report is what to watch when changing them.
    parser.add_argument("--w-structure", type=float, default=2.0,
                        help="Weight on the structure-addition penalty.")
    parser.add_argument("--w-identity", type=float, default=0.5,
                        help="Weight on the identity anchor.")
    parser.add_argument("--structure-margin", type=float, default=0.5,
                        help="Sharpening allowance before the penalty applies; "
                             "0.5 permits half again the measured gradient.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.command in ("prepare", "all"):
        if not args.data:
            raise SystemExit("--data <bundle folder> is required to build the cache.")
        do_prepare(args)
    if args.command in ("fit", "all"):
        do_fit(args)


if __name__ == "__main__":
    main()
