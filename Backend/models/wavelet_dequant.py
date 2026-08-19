"""
wavelet_dequant.py
-------------------
Splits the physics-stabilized image into an approximation band (LL — coarse
illumination/albedo content) and detail bands (LH, HL, HH — edges/texture at
each orientation) using a **stationary** (undecimated) wavelet transform.

SWT (vs. a decimated DWT) is used deliberately: it is shift-invariant, which
avoids the block-boundary artifacts a decimated transform would introduce
after the LL and detail bands are enhanced independently and re-summed —
that boundary ringing is a common source of "hallucinated" fine structure at
tile edges in prior planetary-image enhancers.

Also implements an implicit de-quantization network: a tiny residual conv
net that predicts a smooth, continuous correction to remove staircase
(banding) artifacts introduced by the original sensor's finite bit depth,
without hallucinating texture beyond the quantization step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn


# =============================================================================
# SWT / ISWT decomposition
# =============================================================================
@dataclass
class WaveletBands:
    """One level of SWT2 detail coefficients (three orientations)."""
    LH: np.ndarray
    HL: np.ndarray
    HH: np.ndarray


def swt_decompose(image: np.ndarray, wavelet: str = "sym4", level: int = 2
                   ) -> Tuple[List[WaveletBands], np.ndarray]:
    """
    Multi-level 2D stationary wavelet decomposition.

    Returns:
        bands: list of per-level WaveletBands (LH, HL, HH), coarsest-first,
               matching pywt.swt2's level ordering with trim_approx=True.
        LL:    single coarsest-level approximation band, shared across the
               whole decomposition (SWT keeps only one LL, unlike a pyramid).
    """
    image = image.astype(np.float64)
    coeffs = pywt.swt2(image, wavelet=wavelet, level=level, trim_approx=True)
    # coeffs[0] = LL of coarsest level; coeffs[1:] = (LH,HL,HH) per level, coarse->fine
    LL = coeffs[0]
    bands = [WaveletBands(LH=LH, HL=HL, HH=HH) for (LH, HL, HH) in coeffs[1:]]
    return bands, LL


def iswt_reconstruct(bands: List[WaveletBands], LL: np.ndarray,
                      wavelet: str = "sym4") -> np.ndarray:
    """Inverts swt_decompose's output back to a single 2D image."""
    coeffs = [LL]
    for b in bands:
        coeffs.append((b.LH, b.HL, b.HH))
    return pywt.iswt2(coeffs, wavelet=wavelet)


# =============================================================================
# Implicit de-quantization network
# =============================================================================
class ImplicitDequantizer(nn.Module):
    """
    Small fully-convolutional residual net that maps a quantized band to a
    continuous-valued correction.

    Design intent ("implicit"): the network is *not* shown pixel coordinates
    and is *not* free to output arbitrary values — its correction is clamped
    to +/- half the original quantization step (config: quantization_step).
    This hard-clamps the network's capacity to remove banding/staircase
    artifacts while making it architecturally incapable of inventing
    structure larger than one quantization bin — a direct, inspectable
    anti-hallucination constraint rather than a hoped-for training outcome.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 16,
                 num_layers: int = 3, quantization_step: float = 1.0):
        super().__init__()
        self.quant_step = quantization_step

        layers = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hidden_channels, in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = torch.tanh(self.net(x)) * (self.quant_step / 2.0)
        return x + correction


class WaveletDequantizer:
    """
    Config-driven convenience wrapper combining SWT decomposition with an
    (optional) implicit dequantizer applied per-band.
    """

    def __init__(self, config: dict):
        wv = config["wavelet"]
        self.wavelet = wv["family"]
        self.level = int(wv["levels"])

        dq = wv["dequant"]
        self.dequant_enabled = bool(dq["enabled"])
        self.dequantizer = ImplicitDequantizer(
            in_channels=1,
            hidden_channels=int(dq["hidden_channels"]),
            num_layers=int(dq["num_layers"]),
            quantization_step=float(dq["quantization_step"]),
        )

    def decompose(self, image: np.ndarray) -> Tuple[List[WaveletBands], np.ndarray]:
        return swt_decompose(image, wavelet=self.wavelet, level=self.level)

    def reconstruct(self, bands: List[WaveletBands], LL: np.ndarray) -> np.ndarray:
        return iswt_reconstruct(bands, LL, wavelet=self.wavelet)

    @torch.no_grad()
    def dequantize_band(self, band: np.ndarray) -> np.ndarray:
        if not self.dequant_enabled:
            return band
        self.dequantizer.eval()
        t = torch.from_numpy(band).float()[None, None]  # (1,1,H,W)
        out = self.dequantizer(t)
        return out[0, 0].numpy()
