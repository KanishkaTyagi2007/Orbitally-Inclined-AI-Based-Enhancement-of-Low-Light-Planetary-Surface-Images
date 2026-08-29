"""
training/objectives.py
=======================
The training-time half of the pipeline's anti-hallucination contract.

The architecture already forbids some kinds of invention outright, for any
weights whatsoever:

  * Zero-DCE emits curve parameters, never pixels, and |A| < 1 makes every
    curve application strictly monotonic -- equal inputs get equal outputs, so
    no spatial pattern can be created (`zero_dce.CURVE_BOUND`).
  * The de-quantizer's offset passes through tanh scaled by step/2, so it
    cannot move a pixel outside its own quantizer bin
    (`wavelet_dequant.ImplicitDequantizer`).

Two things those guarantees do not cover, and this module supplies:

  * The detail restorer is an unconstrained convolutional network on the SWT
    detail bands. Nothing structural stops it from writing a crater rim into
    LH/HL/HH. `StructureAdditionPenalty` and `IdentityAnchor` are what stop it.
  * The uncertainty head is only useful if its variance is *earned*. Trained on
    a target it can see, it would drive log-var to the floor everywhere and
    produce a confident-looking uniform trust map -- exactly the artefact the
    zero-initialization was chosen to avoid. It is trained against a target it
    genuinely cannot fully recover (`train.py`, phase 2), so the variance it
    reports is real predictive uncertainty.

Every loss here is either zero-reference or anchored to measured data. None of
them can reward a structure that is absent from the input, because no term in
any of them is minimized by adding one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


# =============================================================================
# Robust reconstruction distance
# =============================================================================
class CharbonnierLoss(nn.Module):
    """
    Smooth L1 (Charbonnier), sqrt(x^2 + eps^2).

    Preferred over MSE here because the residual between a measured tile and a
    noisier copy of it is heavy-tailed -- cosmic-ray skirts and the occasional
    bright rim dominate an L2 loss and drag the restorer toward blurring
    everything to hedge against them.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = float(eps) ** 2

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).pow(2) + self.eps2).mean()


# =============================================================================
# "Do not add structure that was not measured"
# =============================================================================
class StructureAdditionPenalty(nn.Module):
    """
    One-sided hinge on gradient magnitude: penalizes the restorer for producing
    *more* edge energy than its input had, and says nothing when it produces
    less.

    The asymmetry is the point. Removing gradient energy is denoising, which is
    the job. Adding it is either deconvolution -- legitimate, bounded, and the
    reason for `margin` -- or invention, which is not. A symmetric penalty
    (plain gradient matching) would forbid both equally and defeat stage 3B's
    PSF deconvolution.

        loss = mean( relu( |grad(out)| - (1 + margin) * |grad(in)| ) )

    `margin` is the sharpening allowance. At the default 0.5 the restorer may
    raise local edge magnitude by half again -- comfortably more than the
    Wiener deconvolution of a sigma = 1.1 px Gaussian PSF asks for -- and is
    charged for anything beyond.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = float(margin)
        self.register_buffer("kx", torch.tensor(_SOBEL_X)[None, None])
        self.register_buffer("ky", torch.tensor(_SOBEL_Y)[None, None])

    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kx = self.kx.to(x.dtype).expand(channels, 1, -1, -1)
        ky = self.ky.to(x.dtype).expand(channels, 1, -1, -1)
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, kx, groups=channels)
        gy = F.conv2d(padded, ky, groups=channels)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)

    def forward(self, output: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        excess = self._magnitude(output) - (1.0 + self.margin) * self._magnitude(measured)
        return F.relu(excess).mean()


class IdentityAnchor(nn.Module):
    """
    Charges the restorer for changing a tile that was never degraded.

    Trained only on noisy -> measured pairs, a network is free to learn an
    aggressive transform and apply it to everything, including inputs that are
    already as clean as the data gets. Feeding it the *measured* tile and asking
    for the measured tile back defines its fixed point: on real, un-augmented
    input the restorer should be close to the identity, and any departure is a
    change to science data that nothing asked for.

    This is also what keeps a trained checkpoint honest against the
    zero-initialized one it replaces -- both leave un-degraded input alone.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.distance = CharbonnierLoss(eps)

    def forward(self, output_on_measured: torch.Tensor,
                measured: torch.Tensor) -> torch.Tensor:
        return self.distance(output_on_measured, measured)


# =============================================================================
# De-quantization (stage 2.1)
# =============================================================================
class BandingSuppressionLoss(nn.Module):
    """
    Zero-reference objective for the implicit de-quantizer.

    The target is contouring: where a smooth radiance ramp crossed a quantizer
    bin edge, the recorded image has a staircase, and a later contrast stretch
    turns each step into a visible false contour that reads as terrain. The
    de-quantizer's job is to put the ramp back.

        loss = w_tv * edge_weighted_TV(dequantized)
             + w_fidelity * ||dequantized - input||^2

    Both terms are needed. Total variation alone is minimized by a constant
    image, so it is gated by an edge weight computed from the *input*: flat
    regions (where banding lives) are penalized for roughness, and real edges
    are exempt. The fidelity term then holds the result near the measurement,
    on top of the architectural half-bin bound that already caps how far it
    could possibly move.

    `edge_sigma` sets what counts as flat, in units of the input's own gradient
    scale, so the loss needs no per-scene tuning.
    """

    def __init__(self, tv_weight: float = 1.0, fidelity_weight: float = 0.1,
                 edge_sigma: float = 1.0):
        super().__init__()
        self.tv_weight = float(tv_weight)
        self.fidelity_weight = float(fidelity_weight)
        self.edge_sigma = float(edge_sigma)

    @staticmethod
    def _differences(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (x[:, :, 1:, :] - x[:, :, :-1, :],
                x[:, :, :, 1:] - x[:, :, :, :-1])

    def forward(self, dequantized: torch.Tensor, measured: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        dy_in, dx_in = self._differences(measured)
        dy_out, dx_out = self._differences(dequantized)

        # Edge weight in (0, 1]: ~1 where the input is flat, ~0 at real edges.
        # Scaled by the batch's own gradient RMS so it is scene-independent.
        scale = self.edge_sigma * (
            torch.sqrt(dy_in.pow(2).mean() + dx_in.pow(2).mean()).detach() + 1e-8
        )
        wy = torch.exp(-(dy_in / scale).pow(2)).detach()
        wx = torch.exp(-(dx_in / scale).pow(2)).detach()

        tv = (wy * dy_out.pow(2)).mean() + (wx * dx_out.pow(2)).mean()
        fidelity = (dequantized - measured).pow(2).mean()
        total = self.tv_weight * tv + self.fidelity_weight * fidelity
        return total, {
            "banding_tv": float(tv.detach()),
            "dequant_fidelity": float(fidelity.detach()),
            "total": float(total.detach()),
        }


# =============================================================================
# Reporting helper
# =============================================================================
@torch.no_grad()
def structure_addition_fraction(output: torch.Tensor, measured: torch.Tensor,
                                margin: float = 0.5) -> float:
    """
    Fraction of pixels where the output's gradient magnitude exceeds the
    allowance -- the audit number behind `StructureAdditionPenalty`.

    Reported per epoch during training and again per scene at inference. It is
    the closest thing to a direct measurement of "how often did this model add
    an edge that was not in the data", and driving it to zero is the training
    objective's whole purpose.
    """
    penalty = StructureAdditionPenalty(margin).to(output.device)
    excess = (penalty._magnitude(output)
              - (1.0 + margin) * penalty._magnitude(measured))
    return float((excess > 0).float().mean())
