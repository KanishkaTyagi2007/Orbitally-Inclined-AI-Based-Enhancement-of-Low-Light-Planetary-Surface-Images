"""
wavelet_dequant.py
===================
STAGE 2 -- FREQUENCY DECOUPLING & DE-QUANTIZATION

    2.1  ImplicitDequantizer   sub-bin continuous offset mapping
                               (resolves false contouring / banding)
    2.2  swt_decompose/iswt    stationary wavelet transform into
                               translation-invariant sub-bands

Why an *undecimated* (stationary) transform: the LL and detail bands are
enhanced by two different networks and then re-summed. A decimated DWT is not
shift-invariant, so independent per-band edits reappear after reconstruction
as block-boundary ringing -- fine-scale structure that was never in the scene.
SWT keeps every band at full resolution and is shift-invariant, so that entire
artifact class cannot occur.

Why de-quantization comes *first*: 16-bit products of a very dark scene occupy
only a handful of adjacent code values, so a subsequent contrast stretch turns
each quantizer bin edge into a visible false contour. Removing the staircase
before decomposition means the wavelet detail bands carry real texture rather
than quantizer edges, which would otherwise be amplified by stage 3B as if
they were genuine surface detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn


# =============================================================================
# STAGE 2.1 -- Implicit de-quantization (sub-bin continuous offset mapping)
# =============================================================================
class ImplicitDequantizer(nn.Module):
    """
    An implicit continuous field that predicts, for every pixel, a *sub-bin*
    offset which un-does the original quantizer's rounding.

    The quantizer mapped a continuous radiance onto a discrete code, losing
    everything inside one bin; the true value lies somewhere in
    [code - step/2, code + step/2]. This module learns where inside that
    interval it most plausibly sat, conditioned on the local neighbourhood and
    on Fourier-encoded pixel coordinates (the "implicit" part -- the field is
    queried continuously in space rather than stored per pixel).

    The anti-hallucination property is architectural, not learned: the output
    offset passes through `tanh` scaled by `quantization_step / 2`, so the
    correction is *incapable* of exceeding half a quantizer bin. It can erase
    banding; it cannot invent a crater. That bound holds for random weights,
    trained weights, or adversarial weights alike.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 16,
                 num_layers: int = 3, quantization_step: float = 1.0,
                 fourier_features: int = 8):
        super().__init__()
        self.quant_step = float(quantization_step)
        self.fourier_features = int(fourier_features)

        # Coordinate encoding bands: pi, 2pi, 4pi, ... (standard NeRF-style)
        freqs = (2.0 ** torch.arange(self.fourier_features).float()) * np.pi
        self.register_buffer("freqs", freqs, persistent=False)

        coord_dim = 4 * self.fourier_features  # {sin, cos} x {x, y}
        in_dim = in_channels + coord_dim

        layers: list[nn.Module] = [nn.Conv2d(in_dim, hidden_channels, 3, padding=1),
                                   nn.GELU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                       nn.GELU()]
        layers += [nn.Conv2d(hidden_channels, in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

        # Start as a no-op: zero-initialized output layer means an untrained
        # pipeline passes the signal through untouched rather than injecting
        # random sub-bin noise.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _coordinate_features(self, height: int, width: int,
                             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        feats = []
        freqs = self.freqs.to(device=device, dtype=dtype)
        for coord in (grid_y, grid_x):
            scaled = coord[None, :, :] * freqs[:, None, None]
            feats.append(torch.sin(scaled))
            feats.append(torch.cos(scaled))
        return torch.cat(feats, dim=0)[None]  # (1, 4F, H, W)

    @property
    def is_identity(self) -> bool:
        """
        True while the output layer is zero-initialized: the field then predicts
        a zero offset (tanh(0) = 0) and the module returns its input unchanged.
        Skipping it is exact, and avoids building the Fourier coordinate stack
        for a result that is bit-identical to the input.
        """
        last = self.net[-1]
        if last.weight.numel() and last.weight.abs().max().item() != 0.0:
            return False
        return not (last.bias is not None and last.bias.abs().max().item() != 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `self.training` gates the shortcut, not just `is_identity`. The
        # zero-initialized state is the *starting point* of training, and a
        # module that returns its input unchanged has no gradient path to its
        # own weights -- the optimizer would see `element 0 of tensors does not
        # require grad` and the module could never leave the identity it was
        # initialized to. In eval mode, which is what the pipeline runs, the
        # shortcut applies exactly as before.
        if self.is_identity and not self.training:
            return x
        b, _, h, w = x.shape
        coords = self._coordinate_features(h, w, x.device, x.dtype).expand(b, -1, -1, -1)

        # The network sees a standardized copy, not the raw signal. Its input
        # arrives in whatever units stage 1.4 produced -- ~200 for this TMC
        # product in VST units, ~1e6 if a radiometric kernel put it in SI
        # radiance -- while the Fourier coordinate channels beside it are
        # bounded by 1. Feeding both raw makes the image channel swamp the
        # coordinates by six orders of magnitude and leaves the layer badly
        # conditioned. Standardizing per sample makes the predicted offset
        # depend on relative local structure, which is scale-free, so one set
        # of weights transfers across sensors and processing levels.
        #
        # The offset is still added to the raw signal and is still bounded by
        # tanh * step/2, so the half-bin guarantee is untouched.
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized = (x - mean) / std

        offset = torch.tanh(self.net(torch.cat([normalized, coords], dim=1)))
        return x + offset * (self.quant_step / 2.0)


# =============================================================================
# STAGE 2.2 -- Stationary wavelet transform
# =============================================================================
@dataclass
class WaveletBands:
    """One SWT level's detail coefficients, three orientations."""

    LH: np.ndarray   # horizontal detail (vertical edges)
    HL: np.ndarray   # vertical detail (horizontal edges)
    HH: np.ndarray   # diagonal detail


def pad_to_multiple(image: np.ndarray, multiple: int
                    ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    `pywt.swt2` requires each axis to be divisible by 2**level. Planetary
    products are rarely such a size, so pad symmetrically (reflect, which does
    not introduce a false edge at the border) and record the original shape
    for cropping after reconstruction.
    """
    h, w = image.shape
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (h, w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (h, w)


def swt_decompose(image: np.ndarray, wavelet: str = "sym4", level: int = 2
                  ) -> Tuple[List[WaveletBands], np.ndarray]:
    """
    Multi-level 2D stationary wavelet decomposition.

    Returns:
        bands: per-level WaveletBands, coarsest-first (pywt.swt2 ordering with
               trim_approx=True).
        LL:    the single coarsest approximation band. SWT keeps only one LL
               for the whole decomposition, unlike a Laplacian pyramid.
    """
    coeffs = pywt.swt2(image.astype(np.float64), wavelet=wavelet,
                       level=level, trim_approx=True)
    ll = coeffs[0]
    bands = [WaveletBands(LH=lh, HL=hl, HH=hh) for (lh, hl, hh) in coeffs[1:]]
    return bands, ll


def iswt_reconstruct(bands: List[WaveletBands], ll: np.ndarray,
                     wavelet: str = "sym4") -> np.ndarray:
    """STAGE 3C -- inverts `swt_decompose` back to a single full-band image."""
    coeffs: list = [ll]
    coeffs.extend((b.LH, b.HL, b.HH) for b in bands)
    return pywt.iswt2(coeffs, wavelet=wavelet)


# =============================================================================
# Config-driven wrapper
# =============================================================================
class WaveletDequantizer:
    """
    Bundles stage 2.1 (de-quantization) and stage 2.2 (SWT), plus the stage 3C
    inverse transform, behind one config-driven object.

    Usage mirrors the pipeline diagram exactly:

        dequantized      = wd.dequantize(stabilized)      # 2.1
        bands, ll, shape = wd.decompose(dequantized)      # 2.2
        ...                                               # 3A / 3B
        recon            = wd.reconstruct(bands, ll, shape)  # 3C
    """

    def __init__(self, config: dict):
        wv = config["wavelet"]
        self.wavelet = str(wv["family"])
        self.level = int(wv["levels"])
        self.multiple = 2 ** self.level

        dq = wv["dequant"]
        self.dequant_enabled = bool(dq["enabled"])
        self.dequantizer = ImplicitDequantizer(
            in_channels=1,
            hidden_channels=int(dq["hidden_channels"]),
            num_layers=int(dq["num_layers"]),
            quantization_step=float(dq["quantization_step"]),
            fourier_features=int(dq.get("fourier_features", 8)),
        )

    # -- 2.1 ---------------------------------------------------------------
    @torch.no_grad()
    def dequantize(self, image: np.ndarray,
                   device: Optional[torch.device] = None) -> np.ndarray:
        """Applies the sub-bin offset field to a full 2D image."""
        if not self.dequant_enabled:
            return image
        device = device or next(self.dequantizer.parameters()).device
        self.dequantizer.eval()
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None].to(device)
        return self.dequantizer(tensor)[0, 0].cpu().numpy().astype(image.dtype)

    # -- 2.2 ---------------------------------------------------------------
    def decompose(self, image: np.ndarray
                  ) -> Tuple[List[WaveletBands], np.ndarray, Tuple[int, int]]:
        padded, original_shape = pad_to_multiple(image, self.multiple)
        bands, ll = swt_decompose(padded, wavelet=self.wavelet, level=self.level)
        return bands, ll, original_shape

    # -- 3C ---------------------------------------------------------------
    def reconstruct(self, bands: List[WaveletBands], ll: np.ndarray,
                    original_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        recon = iswt_reconstruct(bands, ll, wavelet=self.wavelet)
        if original_shape is not None:
            h, w = original_shape
            recon = recon[:h, :w]
        return recon
