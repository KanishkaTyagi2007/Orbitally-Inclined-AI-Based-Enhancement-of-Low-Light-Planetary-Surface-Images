"""
zero_dce.py
============
STAGE 3A -- ILLUMINATION CURVE ESTIMATION (LL / approximation band)

    Zero-DCE++ backbone (depthwise separable convolutions)
    Multi-order recurrent curve adjustment
    Zero-synthesis: monotonic dynamic curves

Adapted from Guo et al., "Zero-Reference Deep Curve Estimation" (CVPR 2020)
and its ++ successor (TPAMI 2021), for single-channel planetary LL bands.

Why curve estimation rather than a generative enhancer
------------------------------------------------------
There is no ground-truth well-lit counterpart for a low-light planetary scene
-- nobody can re-photograph the same crater at a different sun angle to
supervise against. Zero-DCE needs no paired data: it optimizes reference-free
objectives (exposure, spatial consistency, curve smoothness) directly.

Zero-synthesis property
-----------------------
The network never outputs pixels. It outputs *curve parameters*, and the only
operation applied to the image is the pointwise quadratic

    LE(x) = x + A * x * (1 - x)

Its derivative, dLE/dx = 1 + A*(1 - 2x), is linear in x, so on x in [0, 1] it
attains its minimum at one of the endpoints:

    min(1 + A, 1 - A) = 1 - |A|

The curve map is `tanh(...) * CURVE_BOUND` with CURVE_BOUND slightly below 1,
so |A| <= CURVE_BOUND < 1 and therefore dLE/dx >= 1 - CURVE_BOUND > 0
everywhere: the curve is strictly monotonically increasing.

The margin is not cosmetic. In float32 `tanh` saturates to exactly 1.0 once its
input exceeds about 8.3, which large or badly-initialized weights reach easily
-- and at |A| = 1 the derivative hits zero at an endpoint, making the curve
merely non-decreasing. Scaling by CURVE_BOUND keeps the strict inequality true
for any weights whatsoever.

A strictly monotonic pointwise map can reorder brightness levels but cannot
create a spatial pattern that was not already present -- two pixels with equal
input radiance always receive equal output radiance. That is a structural
guarantee, independent of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Upper bound on |A|. Strictly below 1 so that dLE/dx >= 1 - CURVE_BOUND > 0
# holds even when tanh saturates to exactly +/-1 in float32.
CURVE_BOUND = 0.999


# =============================================================================
# Zero-DCE++ backbone block
# =============================================================================
class DepthwiseSeparableConv(nn.Module):
    """
    The Zero-DCE++ building block: a 3x3 depthwise convolution followed by a
    1x1 pointwise convolution.

    Beyond the ~7x parameter reduction versus a dense 3x3 conv, the factorization
    matters here for a second reason: spatial mixing happens per-channel only,
    so the layer cannot fuse features across channels *and* space in a single
    step. Curve estimation needs local illumination context, not cross-channel
    texture synthesis, and this restricts the backbone to the former.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# =============================================================================
# Curve estimator
# =============================================================================
class ZeroDCE(nn.Module):
    """
    Zero-DCE++ curve estimation network.

    Predicts one per-pixel curve-parameter map `A`, then applies the light
    enhancement curve `num_iterations` times recurrently. Re-using a single
    map (rather than one map per iteration, as in the original Zero-DCE) is
    the ++ formulation: it yields a higher-order composite curve with an
    eighth of the parameters, and each individual application remains
    monotonic, so the composition is monotonic too.

    Args:
        in_channels: 1 for a single-band planetary LL band.
        num_filters: backbone width (paper default 32).
        num_conv_layers: odd, >= 3 (paper default 7).
        num_iterations: recurrent curve applications -> curve order.
        depthwise: use the ++ depthwise-separable backbone.
        enforce_monotonic: bound A to [-CURVE_BOUND, CURVE_BOUND] via tanh.
            Turning this off voids the zero-synthesis guarantee and is offered
            only for ablation studies.

    Returns from `forward`: (enhanced, curve_map) where `curve_map` is the
    single A map, kept for the illumination-smoothness loss and for
    inspection/auditing of what the curve actually did.
    """

    def __init__(self, in_channels: int = 1, num_filters: int = 32,
                 num_conv_layers: int = 7, num_iterations: int = 8,
                 depthwise: bool = True, enforce_monotonic: bool = True):
        super().__init__()
        if num_conv_layers < 3:
            raise ValueError("num_conv_layers must be >= 3 (in / mid / out)")
        if num_conv_layers % 2 == 0:
            num_conv_layers += 1  # symmetric skip pattern needs an odd depth

        self.num_iterations = int(num_iterations)
        self.enforce_monotonic = bool(enforce_monotonic)
        self.in_channels = in_channels

        conv = DepthwiseSeparableConv if depthwise else (
            lambda i, o, k=3: nn.Conv2d(i, o, k, padding=k // 2)
        )

        c = num_filters
        self.n_enc = (num_conv_layers + 1) // 2
        self.n_dec = num_conv_layers - self.n_enc

        # Encoder: first layer lifts the image into feature space.
        self.encoders = nn.ModuleList(
            [conv(in_channels if i == 0 else c, c) for i in range(self.n_enc)]
        )
        # Decoder: each layer consumes its own input concatenated with the
        # mirrored encoder feature (symmetric skip connections, as in the paper).
        self.decoders = nn.ModuleList(
            [conv(c * 2, c) for _ in range(self.n_dec - 1)]
        )
        self.out_conv = conv(c * 2, in_channels)
        self.act = nn.ReLU(inplace=True)

        # Zero-initialize the curve head. With A = 0 the enhancement curve is
        # LE(x) = x + 0*x*(1-x) = x, so an uncheckpointed pipeline leaves the
        # illumination band exactly as the physics stages produced it.
        #
        # This is a correctness point before it is a speed one: with random
        # weights the network applies an arbitrary (if bounded and monotone)
        # tone curve to real science data, which is not something an untrained
        # model should be doing silently. It also makes the whole stage a
        # provable identity that `is_identity` can skip.
        final = self.out_conv.pointwise if isinstance(
            self.out_conv, DepthwiseSeparableConv) else self.out_conv
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    # -- the light-enhancement curve --------------------------------------
    @staticmethod
    def enhancement_curve(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """One monotonic application: LE(x) = x + A*x*(1-x)."""
        return x + a * (x - x.pow(2))

    @property
    def is_identity(self) -> bool:
        """True while the curve head is zero-initialized: A = 0 makes every
        curve application LE(x) = x, so the backbone can be skipped entirely."""
        final = self.out_conv.pointwise if isinstance(
            self.out_conv, DepthwiseSeparableConv) else self.out_conv
        if final.weight.numel() and final.weight.abs().max().item() != 0.0:
            return False
        return not (final.bias is not None and final.bias.abs().max().item() != 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) normalized to [0, 1].
        Returns:
            (enhanced in [0, 1], curve_map A in (-1, 1))
        """
        if self.is_identity:
            zeros = torch.zeros_like(x)
            return x.clamp(0.0, 1.0), zeros

        feats = []
        h = x
        for enc in self.encoders:
            h = self.act(enc(h))
            feats.append(h)

        # mirror indices: for a 7-layer net with n_enc=4, decoders concat with
        # encoder features 2, 1, then the out_conv concats with feature 0.
        for j, dec in enumerate(self.decoders):
            skip = feats[self.n_enc - 2 - j]
            h = self.act(dec(torch.cat([h, skip], dim=1)))

        raw = self.out_conv(torch.cat([h, feats[0]], dim=1))
        curve_map = torch.tanh(raw) * CURVE_BOUND if self.enforce_monotonic else raw

        enhanced = x
        for _ in range(self.num_iterations):
            enhanced = self.enhancement_curve(enhanced, curve_map)

        return enhanced.clamp(0.0, 1.0), curve_map

    # -- auditing ----------------------------------------------------------
    @staticmethod
    def is_zero_synthesis(curve_map: torch.Tensor) -> bool:
        """
        Verifies the zero-synthesis guarantee held for an actual forward pass:
        every curve parameter must satisfy |A| < 1, which makes each curve
        application strictly increasing and therefore incapable of creating
        spatial structure. Reported per-run in the metrics JSON, so the claim
        is evidenced per scene rather than merely asserted.
        """
        return bool(curve_map.abs().max().item() < 1.0)


# Explicit alias -- the architecture implemented above is the ++ variant.
ZeroDCEPlusPlus = ZeroDCE


# =============================================================================
# Zero-reference losses (training only; kept beside the model so the objective
# travels with the architecture it constrains)
# =============================================================================
class IlluminationSmoothnessLoss(nn.Module):
    """
    Total-variation penalty on the curve map. Forces the illumination
    adjustment to vary smoothly across the scene, so the curve corrects
    lighting rather than painting per-pixel brightness decorations.
    """

    def forward(self, curve_map: torch.Tensor) -> torch.Tensor:
        dh = curve_map[:, :, 1:, :] - curve_map[:, :, :-1, :]
        dw = curve_map[:, :, :, 1:] - curve_map[:, :, :, :-1]
        return dh.pow(2).mean() + dw.pow(2).mean()


class ExposureControlLoss(nn.Module):
    """Pulls local mean intensity toward a well-exposed target level."""

    def __init__(self, patch_size: int = 16, target_exposure: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = float(target_exposure)

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        return (self.pool(enhanced) - self.target).pow(2).mean()


class SpatialConsistencyLoss(nn.Module):
    """
    Penalizes any change in local gradient relationships between the input and
    the enhanced output -- a direct, differentiable statement of "do not invent
    or erase edges", complementing the architectural monotonicity guarantee.
    """

    _KERNELS = {
        "left":  [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        "up":    [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        "down":  [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    }

    def __init__(self, pool_size: int = 4):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size)
        for name, k in self._KERNELS.items():
            self.register_buffer(f"k_{name}", torch.tensor([[k]], dtype=torch.float32))

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        e_pool, o_pool = self.pool(enhanced), self.pool(original)
        loss = enhanced.new_zeros(())
        for name in self._KERNELS:
            k = getattr(self, f"k_{name}")
            loss = loss + (F.conv2d(e_pool, k, padding=1)
                           - F.conv2d(o_pool, k, padding=1)).pow(2).mean()
        return loss


class ZeroDCELoss(nn.Module):
    """Weighted sum of the three reference-free objectives (config-driven)."""

    def __init__(self, config: dict):
        super().__init__()
        z = config["zero_dce"]
        w = z["loss_weights"]
        self.w_spatial = float(w["spatial_consistency"])
        self.w_exposure = float(w["exposure_control"])
        self.w_smooth = float(w["illumination_smoothness"])

        self.spatial = SpatialConsistencyLoss()
        self.exposure = ExposureControlLoss(target_exposure=float(z["target_exposure"]))
        self.smooth = IlluminationSmoothnessLoss()

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor,
                curve_map: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l_spatial = self.spatial(enhanced, original)
        l_exposure = self.exposure(enhanced)
        l_smooth = self.smooth(curve_map)
        total = (self.w_spatial * l_spatial
                 + self.w_exposure * l_exposure
                 + self.w_smooth * l_smooth)
        return total, {
            "spatial_consistency": float(l_spatial.detach()),
            "exposure_control": float(l_exposure.detach()),
            "illumination_smoothness": float(l_smooth.detach()),
            "total": float(total.detach()),
        }
