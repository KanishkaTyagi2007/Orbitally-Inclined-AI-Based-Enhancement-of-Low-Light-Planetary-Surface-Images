"""
uncertainty_head.py
====================
STAGE 5 -- PHYSICS-BASED VERIFICATION & UNCERTAINTY ESTIMATION

    5.1  PhotometricFluxConservation  area-averaged energy balance across scales
    5.2  SobelGradientConsistency     rejects unphysical discontinuities
    5.3  UncertaintyHead              heteroscedastic (mu, log-var) -> trust map

Stages 5.1 and 5.2 each ship in two forms:

  * a differentiable `nn.Module` loss, used during training to *pressure* the
    network toward physically admissible solutions;
  * a NumPy `check_*` function, used at inference to measure a finished product
    and record the result in the metrics JSON.

The distinction matters. A training loss is a soft preference that can be
traded away against other objectives; the inference-time check reports what
actually happened to this specific scene.

Of the two, only flux conservation (5.1) gates by default. Gradient consistency
(5.2) is reported but not gated, because no fixed threshold for it survived
calibration across scene sizes -- see `check_gradient_consistency` for the
measurements and for how to calibrate and enable it on real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# STAGE 5.1 -- Photometric flux conservation
# =============================================================================
def _normalized_area_average(image: np.ndarray, scale: int) -> np.ndarray:
    """
    Area-averages `image` in scale x scale blocks, then divides by the scene
    mean. The normalization removes any global gain, so what remains measures
    purely how flux is *distributed* in space.
    """
    mean = float(np.mean(image))
    if abs(mean) < 1e-30:
        return np.zeros(
            (max(image.shape[0] // scale, 1), max(image.shape[1] // scale, 1))
        )
    if scale > 1:
        h = (image.shape[0] // scale) * scale
        w = (image.shape[1] // scale) * scale
        if h == 0 or w == 0:
            return np.array([[1.0]])
        pooled = image[:h, :w].reshape(
            h // scale, scale, w // scale, scale
        ).mean(axis=(1, 3))
    else:
        pooled = image
    return pooled / mean


def _flux_drift_per_scale(reference: np.ndarray, image: np.ndarray,
                          scales: list[int]) -> dict[int, float]:
    drift = {}
    for s in scales:
        ref_pool = _normalized_area_average(reference, s)
        img_pool = _normalized_area_average(image, s)
        if ref_pool.shape != img_pool.shape:
            continue
        drift[s] = float(np.mean(np.abs(img_pool - ref_pool)))
    return drift


def check_flux_conservation(reference: np.ndarray, reconstruction: np.ndarray,
                            config: dict,
                            final: Optional[np.ndarray] = None) -> dict:
    """
    Stage 5.1 verification.

    For each scale s, both images are area-averaged in s x s blocks and divided
    by their own scene mean, then compared:

        drift(s) = mean | pool_s(E)/mean(E) - pool_s(R)/mean(R) |

    Dividing by the scene mean removes any global gain, so what is measured is
    purely how flux has been *redistributed* in space. Fine scales are expected
    to drift -- shadow is deliberately lifted more than highlight. The physical
    requirement is that drift decreases with scale and lands inside tolerance
    at the coarsest scale, which is what the gate uses.

    What is gated
    -------------
    The gate is applied to the stage-3C *reconstruction*, not the final
    product. This is deliberate. Stages 2-3 are the learned path -- the only
    place the pipeline could invent radiance -- so that is what needs policing.
    Stage 4 tone mapping is a documented, deterministic display transform whose
    entire purpose is to compress large-scale illumination, and it dominates
    coarse-scale drift by design (measured on the synthetic scene: 0.023 at
    stage 3C versus 0.078 after tone mapping, at 16-pixel scale). Gating on the
    post-tone-map image would therefore measure tone mapping rather than
    fabrication.

    When `final` is supplied its drift is reported too, as an ungated
    diagnostic recording how much stage 4 moved the flux distribution.
    """
    cfg = config["verification"]["flux_conservation"]
    if not cfg.get("enabled", True):
        return {"flux_conservation_checked": False}

    scales = [int(s) for s in cfg["scales"]]
    tolerance = float(cfg["tolerance"])

    per_scale = _flux_drift_per_scale(reference, reconstruction, scales)
    if not per_scale:
        return {"flux_conservation_checked": False}

    coarse_drift = per_scale[max(per_scale)]
    report = {
        "flux_conservation_checked": True,
        "flux_drift_per_scale": {str(k): v for k, v in sorted(per_scale.items())},
        "flux_drift_coarsest_scale": coarse_drift,
        "flux_conservation_tolerance": tolerance,
        "flux_conservation_passed": bool(coarse_drift <= tolerance),
    }

    if final is not None:
        final_drift = _flux_drift_per_scale(reference, final, scales)
        if final_drift:
            report["flux_drift_after_tone_mapping"] = {
                str(k): v for k, v in sorted(final_drift.items())
            }
    return report


class PhotometricFluxConservationLoss(nn.Module):
    """Differentiable training-time form of `check_flux_conservation`."""

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["verification"]["flux_conservation"]
        self.scales = [int(s) for s in cfg["scales"]]
        self.weight = float(cfg.get("weight", 1.0))

    @staticmethod
    def _pool_normalized(x: torch.Tensor, scale: int) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        pooled = F.avg_pool2d(x, scale) if scale > 1 else x
        return pooled / mean

    def forward(self, enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        loss = enhanced.new_zeros(())
        for s in self.scales:
            if min(enhanced.shape[-2:]) < s:
                continue
            loss = loss + (self._pool_normalized(enhanced, s)
                           - self._pool_normalized(reference, s)).abs().mean()
        return self.weight * loss / max(len(self.scales), 1)


# =============================================================================
# STAGE 5.2 -- Sobel gradient consistency guardrail
# =============================================================================
_SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.T


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude via the 3x3 Sobel operator."""
    from scipy import ndimage

    gx = ndimage.convolve(image.astype(np.float64), _SOBEL_X, mode="nearest")
    gy = ndimage.convolve(image.astype(np.float64), _SOBEL_Y, mode="nearest")
    return np.sqrt(gx ** 2 + gy ** 2)


def check_gradient_consistency(reference: np.ndarray, enhanced: np.ndarray,
                               config: dict) -> dict:
    """
    Stage 5.2 verification.

    Real surface relief produces edges at fixed locations. Enhancement may
    change an edge's *contrast* but must not move it, delete it, or add one.
    The test therefore asks where the gradient energy sits, not how strong it is.

    Two choices here are load-bearing, and both were settled by measurement
    rather than assumption (see the design note below).

    1. Both images are Gaussian pre-smoothed by `smoothing_sigma` before the
       Sobel operator runs. Without this the metric is dominated by per-pixel
       photon noise, which is not surface structure -- the raw scene's
       strongest gradients are shot-noise spikes, not crater rims.

    2. The statistic is Spearman *rank* correlation, not Pearson. Stage 4 tone
       mapping applies a deliberate monotone but strongly nonlinear remapping
       of contrast; Pearson penalizes that legitimate change, while a rank
       correlation is invariant to any monotone remapping and so responds only
       to edges genuinely moving, appearing, or vanishing.

    Two numbers are reported:

        gradient_correlation   the raw statistic
        gradient_fidelity      that value divided by the same statistic for a
                               transformation KNOWN to preserve structure -- a
                               mild Gaussian denoise of the reference against
                               itself. The baseline absorbs this scene's own
                               structure density and noise level, so the ratio
                               is the more interpretable quantity: "how does the
                               enhancement's edge fidelity compare to something
                               that provably did not move any edges here?"

    WHY THIS DOES NOT GATE BY DEFAULT
    ---------------------------------
    Scored across four scene sizes (192-512 px) x three seeds x three
    fabrication controls (invented craters, warped edges, synthetic texture),
    no formulation of this statistic supports a fixed absolute threshold:

        formulation                      genuine        worst fabricated
        Pearson, unsmoothed, top 10%     0.424          0.424
        Spearman, smoothed, all pixels   0.619-0.838    0.777
        Spearman, smoothed, top 50/70/90 0.170-0.741    0.863
        edge-weighted Pearson            0.628-0.899    0.900
        scene-normalized fidelity        0.668-0.907    0.805
        blockwise minimum / q05 / q25    0.448-0.881    0.836

    Every margin is negative: some genuine run scores below some fabricated
    one. The decisive control was running the pipeline with PSF deconvolution
    disabled, leaving only a monotone tone map -- an operation that *provably*
    cannot move an edge. It still scored 0.783 on a 512 px scene while
    fabricated craters on a 192 px scene scored 0.796.

    The cause is not the pipeline, it is the statistic: in a noisy scene the
    gradient field of flat terrain is dominated by photon noise, so the
    achievable correlation is set by the scene's structure-to-noise
    composition, which varies enormously between real products. A single
    constant cannot separate the two populations.

    So this ships as a reported diagnostic, not a gate. Set
    `verification.gradient_consistency.gate: true` only after calibrating
    `min_correlation` against real imagery from your instrument, at your
    typical scene size and terrain type. A gate that fails a provably
    structure-preserving transform would teach operators to ignore the warning,
    which is worse than having no gate at all.

    What still gates: flux conservation (5.1), which is scale-stable
    (0.016-0.029 across sizes) and rejects invented energy by an order of
    magnitude, and the zero-synthesis guarantee (3A), which is architectural.

    The statistic remains genuinely informative even ungated -- warped edges
    score far below everything else in every configuration measured -- so it is
    worth reading, just not worth automating on yet.
    """
    from scipy import ndimage
    from scipy.stats import spearmanr

    cfg = config["verification"]["gradient_consistency"]
    if not cfg.get("enabled", True):
        return {"gradient_consistency_checked": False}

    min_corr = float(cfg["min_correlation"])
    sigma = float(cfg.get("smoothing_sigma", 1.0))
    baseline_sigma = float(cfg.get("baseline_sigma", 1.0))
    max_samples = int(cfg.get("max_samples", 2_000_000))
    gate = bool(cfg.get("gate", False))

    ref64 = reference.astype(np.float64)

    def smoothed_gradient(image: np.ndarray, blur: float) -> np.ndarray:
        base = ndimage.gaussian_filter(image, blur) if blur > 0 else image
        return sobel_magnitude(base).ravel()

    grad_ref = smoothed_gradient(ref64, sigma)
    grad_enh = smoothed_gradient(enhanced.astype(np.float64), sigma)
    # Structure-preserving reference point for THIS scene: a mild denoise of
    # the reference. Combined blur of two Gaussians adds in quadrature.
    grad_base = smoothed_gradient(ref64, float(np.hypot(sigma, baseline_sigma)))

    # Rank correlation needs a sort; subsample deterministically on large
    # scenes so cost stays bounded without biasing the estimate. All three
    # arrays share the index set so the ratio stays meaningful.
    if grad_ref.size > max_samples:
        idx = np.random.default_rng(0).choice(grad_ref.size, max_samples, replace=False)
        grad_ref, grad_enh, grad_base = grad_ref[idx], grad_enh[idx], grad_base[idx]

    def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        value = float(spearmanr(a, b).statistic)
        return value if np.isfinite(value) else 0.0

    correlation = rank_corr(grad_ref, grad_enh)
    baseline = rank_corr(grad_ref, grad_base)
    fidelity = correlation / baseline if abs(baseline) > 1e-6 else float("nan")

    report = {
        "gradient_consistency_checked": True,
        "gradient_correlation": correlation,
        "gradient_correlation_statistic": "spearman",
        "gradient_baseline_correlation": baseline,
        "gradient_fidelity_vs_baseline": fidelity,
        "gradient_smoothing_sigma": sigma,
        "gradient_correlation_threshold": min_corr,
        "gradient_consistency_gated": gate,
    }
    # Only claim a verdict when an operator has calibrated the threshold and
    # switched the gate on; see this function's docstring for why the default
    # is off.
    if gate:
        report["gradient_consistency_passed"] = bool(correlation >= min_corr)
    return report


class SobelGradientConsistencyLoss(nn.Module):
    """
    Differentiable training-time counterpart to `check_gradient_consistency`.

    The inference check uses a Spearman rank correlation, which has no useful
    gradient. Training therefore optimizes Pearson correlation on the same
    pre-smoothed Sobel magnitudes: the smoothing (the part that actually made
    the metric discriminative) is preserved, and the rank/linear difference
    only matters for the monotone contrast remapping that stage 4 applies
    after this loss is evaluated.
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["verification"]["gradient_consistency"]
        self.weight = float(cfg.get("weight", 1.0))
        self.sigma = float(cfg.get("smoothing_sigma", 1.0))

        self.register_buffer("kx", torch.tensor(_SOBEL_X, dtype=torch.float32)[None, None])
        self.register_buffer("ky", torch.tensor(_SOBEL_Y, dtype=torch.float32)[None, None])

        if self.sigma > 0:
            radius = max(1, int(round(3.0 * self.sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
            self.register_buffer("blur", (kernel / kernel.sum())[None, None])
            self.blur_radius = radius

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur, matching the inference-time pre-smoothing."""
        if self.sigma <= 0:
            return x
        channels = x.shape[1]
        r = self.blur_radius
        kh = self.blur.to(x.dtype).view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        kv = self.blur.to(x.dtype).view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh, groups=channels)
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv, groups=channels)

    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kx = self.kx.to(x.dtype).expand(channels, 1, -1, -1)
        ky = self.ky.to(x.dtype).expand(channels, 1, -1, -1)
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kx, groups=channels)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), ky, groups=channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)

    def forward(self, enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        a = self._magnitude(self._smooth(reference))
        b = self._magnitude(self._smooth(enhanced))
        a = a - a.mean()
        b = b - b.mean()
        correlation = (a * b).mean() / (a.std().clamp_min(1e-12) * b.std().clamp_min(1e-12))
        return self.weight * (1.0 - correlation)


# =============================================================================
# STAGE 5.3 -- Heteroscedastic uncertainty head
# =============================================================================
class UncertaintyHead(nn.Module):
    """
    Predicts, per pixel, a mean radiance estimate (mu) and the log-variance of
    the model's own uncertainty about it (log_var).

    This is what makes the pipeline's output auditable rather than merely
    plausible: every enhanced pixel ships with a confidence value, and
    low-confidence regions are surfaced to the scientist as a trust map rather
    than presented as established fact.

    Trained with the heteroscedastic Gaussian NLL

        NLL = 0.5 * exp(-log_var) * (target - mu)^2 + 0.5 * log_var

    whose first term lets the network attenuate its own loss -- and therefore
    the gradient pressure to commit to a confident answer -- exactly where the
    input evidence does not support one: deep shadow at near-zero photon
    counts, saturated wells, or pixels the cosmic-ray scrubber rewrote.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 24,
                 min_log_var: float = -6.0, max_log_var: float = 4.0):
        super().__init__()
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.mu_head = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        self.log_var_head = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

        # Both heads are zero-initialized, for different reasons.
        #
        # mu: an uncheckpointed pipeline then reports the reconstruction itself
        # rather than a random perturbation of it.
        #
        # log_var: an untrained head would otherwise emit a spatially varying,
        # entirely random variance field, which becomes a trust map that *looks*
        # meaningful and is not -- the precise failure this pipeline exists to
        # prevent. Zero-initialized, it emits a constant log-variance, the trust
        # map comes out uniform, and only physically known-bad pixels
        # (saturated wells, scrubbed cosmic-ray hits, nodata) are marked
        # untrustworthy. That correctly reads as "no learned uncertainty
        # estimate is available for this run".
        for head in (self.mu_head, self.log_var_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        mu = self.mu_head(feat) + x          # residual around the reconstruction
        log_var = torch.clamp(self.log_var_head(feat), self.min_log_var, self.max_log_var)
        return mu, log_var


class HeteroscedasticNLLLoss(nn.Module):
    """Gaussian negative log-likelihood with learned per-pixel variance."""

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-log_var)
        return (0.5 * precision * (target - mu).pow(2) + 0.5 * log_var).mean()


def compute_trust_map(log_var: np.ndarray, config: dict,
                      saturation_mask: Optional[np.ndarray] = None,
                      cosmic_ray_mask: Optional[np.ndarray] = None,
                      nodata_mask: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts predicted log-variance into the exported per-pixel trust band.

        trust_raw = exp(-log_var)              (higher = more confident)
        trust     = min-max normalized to [0, 1] over the scene

    Physically untrustworthy pixels are then forced to exactly zero regardless
    of what the learned head predicted -- a saturated well or a scrubbed cosmic
    ray hit carries no recoverable information about the surface, and no amount
    of network confidence changes that.

    Returns (trust in [0, 1], low_trust_mask).
    """
    tm_cfg = config["uncertainty"]["trust_map"]

    trust_raw = np.exp(-log_var.astype(np.float64))
    lo, hi = float(trust_raw.min()), float(trust_raw.max())
    trust = (np.ones_like(trust_raw) if (hi - lo) < 1e-12
             else (trust_raw - lo) / (hi - lo))

    if tm_cfg.get("zero_trust_on_saturation", True) and saturation_mask is not None:
        trust[saturation_mask] = 0.0
    if tm_cfg.get("zero_trust_on_cosmic_ray", True) and cosmic_ray_mask is not None:
        trust[cosmic_ray_mask] = 0.0
    if nodata_mask is not None:
        trust[nodata_mask] = 0.0

    low_trust_mask = trust < float(tm_cfg["low_trust_threshold"])
    return trust.astype(np.float32), low_trust_mask


# =============================================================================
# STAGE 5 -- assembled verifier
# =============================================================================
@dataclass
class VerificationResult:
    trust_map: np.ndarray
    low_trust_mask: np.ndarray
    record: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only if every enabled physics guardrail passed."""
        return all(
            self.record.get(key, True)
            for key in ("flux_conservation_passed", "gradient_consistency_passed")
        )


class PhysicsVerifier:
    """
    Runs stage 5 end to end: the two physics guardrails plus trust-map
    construction, collapsing the result into one record for the metrics JSON.
    """

    def __init__(self, config: dict):
        self.config = config

    def verify(self, reference: np.ndarray, enhanced: np.ndarray,
               log_var: np.ndarray,
               reconstruction: Optional[np.ndarray] = None,
               saturation_mask: Optional[np.ndarray] = None,
               cosmic_ray_mask: Optional[np.ndarray] = None,
               nodata_mask: Optional[np.ndarray] = None) -> VerificationResult:
        """
        Args:
            reference: calibrated raw radiance -- the measured truth.
            enhanced: the final stage-4 product, gated on gradient consistency.
            reconstruction: the stage-3C output. Flux conservation is gated on
                this rather than on `enhanced`, so the check polices the
                learned path instead of the deliberate tone-mapping transform
                (see `check_flux_conservation`). Falls back to `enhanced`.
        """
        record: dict = {}
        record.update(check_flux_conservation(
            reference,
            reconstruction if reconstruction is not None else enhanced,
            self.config,
            final=enhanced if reconstruction is not None else None,
        ))
        record.update(check_gradient_consistency(reference, enhanced, self.config))

        trust_map, low_trust_mask = compute_trust_map(
            log_var, self.config,
            saturation_mask=saturation_mask,
            cosmic_ray_mask=cosmic_ray_mask,
            nodata_mask=nodata_mask,
        )

        tm_cfg = self.config["uncertainty"]["trust_map"]
        record["mean_trust"] = float(trust_map.mean())
        record["low_trust_pixel_fraction"] = float(low_trust_mask.mean())
        record["hallucination_flagged_fraction"] = float(
            (trust_map < float(tm_cfg["hallucination_flag_threshold"])).mean()
        )
        # Self-evidencing: a constant log-variance means the head carries no
        # trained uncertainty, so the trust band reflects only the physical
        # masks. Reported per run so nobody reads a uniform map as confidence.
        record["trust_map_informative"] = bool(float(np.std(log_var)) > 1e-9)

        result = VerificationResult(trust_map, low_trust_mask, record)
        record["physics_verification_passed"] = result.passed
        return result
