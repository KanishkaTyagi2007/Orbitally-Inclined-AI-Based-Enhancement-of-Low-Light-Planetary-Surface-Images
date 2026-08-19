"""
nafnet_denoiser.py
--------------------
Nonlinear Activation Free Network (NAFNet, Chen et al. ECCV 2022), applied
to the LH/HL/HH wavelet detail bands.

Why NAFNet for detail bands specifically: the detail bands carry the
high-frequency edge/texture information most vulnerable to noise, and also
the information most damaging to "scientific fidelity" if a denoiser
over-smooths or invents texture. NAFNet reaches state-of-the-art restoration
quality while removing nonlinear activations (ReLU/GELU/Sigmoid) from the
main feature path in favor of SimpleGate (a channel-split-and-multiply gate)
and Simplified Channel Attention (SCA, a single global-pooled 1x1 conv gate
instead of a learned nonlinear MLP as in classic SE blocks). Fewer nonlinear
degrees of freedom means less capacity for the network to hallucinate
plausible-looking-but-fictitious high-frequency structure, at comparable
restoration quality to attention-heavy transformer denoisers.

The 3 detail orientations (LH, HL, HH) are stacked as input channels and
processed jointly (shared receptive field across orientations helps the
network distinguish oriented noise from oriented real edges).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# Building blocks
# =============================================================================
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (normalizes over C only)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies the halves — a linear-in-each-
    argument gating mechanism that replaces ReLU/GELU on the main path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    One activation-free residual block:
      LN -> 1x1 conv (expand) -> 3x3 depthwise conv -> SimpleGate
         -> Simplified Channel Attention -> 1x1 conv (project) -> +residual
      LN -> 1x1 conv (expand) -> SimpleGate -> 1x1 conv (project) -> +residual
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        # --- spatial mixing sub-block ---
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1)
        self.dwconv = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.gate1 = SimpleGate()
        # Simplified Channel Attention: global-avg-pool -> 1x1 conv -> scale.
        # No nonlinearity between pool and scale (unlike SE's ReLU-Sigmoid MLP).
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv2 = nn.Conv2d(dw_ch // 2, channels, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- channel-mixing ("FFN") sub-block ---
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_ch, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_ch // 2, channels, 1)
        self.alpha2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + self.alpha1 * y

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        x = x + self.alpha2 * y
        return x


# =============================================================================
# Full U-Net-style NAFNet
# =============================================================================
class NAFNetDenoiser(nn.Module):
    """
    Encoder-decoder NAFNet sized for wavelet detail-band restoration
    (small inputs relative to full-resolution RGB photography, hence a
    shallower default than the original paper's SIDD/GoPro configs).
    """

    def __init__(self, in_channels: int = 3, width: int = 32,
                 enc_blocks=(1, 1, 2), middle_blocks: int = 2,
                 dec_blocks=(1, 1, 1), dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n_blocks in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(n_blocks)])
            )
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(middle_blocks)]
        )

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n_blocks in dec_blocks:
            self.ups.append(
                nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.PixelShuffle(2))
            )
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(n_blocks)])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (-H) % (2 ** len(self.downs))
        pad_w = (-W) % (2 ** len(self.downs))
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        out = self.ending(feat) + x
        return out[:, :, :H, :W]
