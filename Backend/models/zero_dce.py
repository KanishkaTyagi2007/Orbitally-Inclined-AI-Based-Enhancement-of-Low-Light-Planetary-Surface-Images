"""
zero_dce.py
------------
Zero-Reference Deep Curve Estimation, adapted from Guo et al. (CVPR 2020)
for single-channel planetary LL (illumination/approximation) bands.

Why Zero-DCE for this problem specifically: planetary datasets have no
"ground truth well-lit" counterpart to train against (there is no second,
brighter exposure of the same lunar crater taken under different sun angle
to supervise on). Zero-DCE sidesteps this by learning pixel-wise higher-order
curve parameters and optimizing *reference-free* losses (exposure, spatial
consistency, smoothness) — it never needs paired low/normal-light data,
which makes it the right fit when none exists.

The network only ever applies a monotonic tone curve per pixel; it cannot
synthesize new spatial structure, which is the main reason it is used here
instead of a generative low-light enhancer (e.g. a GAN or diffusion model)
that could hallucinate craters/rocks that were never in the scene.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroDCE(nn.Module):
    """
    7-layer conv net (paper default) predicting `num_iterations` maps of
    per-pixel curve parameters `A`, applied iteratively:

        LE_0(x) = x
        LE_{n+1}(x) = LE_n(x) + A_n * LE_n(x) * (1 - LE_n(x))

    Each application is a quadratic curve bounded in [0,1] when A in [-1,1]
    and x in [0,1], guaranteeing the output stays in the valid range without
    clipping artifacts.
    """

    def __init__(self, in_channels: int = 1, num_filters: int = 32,
                 num_conv_layers: int = 7, num_iterations: int = 8):
        super().__init__()
        assert num_conv_layers >= 3, "need at least an in/mid/out layer"
        self.num_iterations = num_iterations

        c = num_filters
        self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.mid_convs = nn.ModuleList(
            [nn.Conv2d(c, c, 3, padding=1) for _ in range(num_conv_layers - 2)]
        )
        # Symmetric skip-concat like the original Zero-DCE (halves channel
        # count back down after concatenation with the input-stage features).
        self.skip_convs = nn.ModuleList(
            [nn.Conv2d(c * 2, c, 3, padding=1) for _ in range(len(self.mid_convs) // 2)]
        )
        self.out_conv = nn.Conv2d(c, num_iterations * in_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 1, H, W) normalized to [0, 1].
        Returns: (enhanced, curve_params) where curve_params has shape
                 (B, num_iterations, H, W) for inspection / regularization.
        """
        feats = [self.act(self.in_conv(x))]
        h = feats[0]
        n_mid = len(self.mid_convs)
        for i, conv in enumerate(self.mid_convs):
            h = self.act(conv(h))
            feats.append(h)
            # symmetric skip connections at the halfway point onward
            mirror_idx = n_mid - 1 - i
            if i >= n_mid // 2 and (n_mid - 1 - i) < len(self.skip_convs):
                skip_feat = feats[mirror_idx] if mirror_idx < len(feats) else feats[0]
                h = self.act(self.skip_convs[n_mid - 1 - i](torch.cat([h, skip_feat], dim=1)))

        curve_params = torch.tanh(self.out_conv(h))  # (B, T, H, W), range [-1, 1]

        enhanced = x
        params_per_step = torch.chunk(curve_params, self.num_iterations, dim=1)
        for A in params_per_step:
            enhanced = enhanced + A * (enhanced - enhanced.pow(2))
        return enhanced.clamp(0.0, 1.0), curve_params


# =============================================================================
# Zero-reference losses (used only during training; kept here so the model
# and its objective travel together)
# =============================================================================
class IlluminationSmoothnessLoss(nn.Module):
    """Total-variation-style penalty on the curve parameter maps."""

    def forward(self, curve_params: torch.Tensor) -> torch.Tensor:
        dh = curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :]
        dw = curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1]
        return dh.pow(2).mean() + dw.pow(2).mean()


class ExposureControlLoss(nn.Module):
    """Pulls local mean intensity toward a well-exposed target level."""

    def __init__(self, patch_size: int = 16, target_exposure: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = target_exposure

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        mean_patch = self.pool(enhanced)
        return (mean_patch - self.target).pow(2).mean()


class SpatialConsistencyLoss(nn.Module):
    """Encourages the enhanced image's local gradients to track the input's,
    directly discouraging invented edges/structure not present pre-enhancement."""

    def __init__(self):
        super().__init__()
        kernels = {
            "left": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up": [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, k in kernels.items():
            self.register_buffer(f"k_{name}", torch.tensor([[k]], dtype=torch.float32))

    def _grad(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=1)

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        pool = nn.AvgPool2d(4)
        e_pool, o_pool = pool(enhanced), pool(original)
        loss = 0.0
        for name in ["left", "right", "up", "down"]:
            k = getattr(self, f"k_{name}")
            d_e = self._grad(e_pool, k)
            d_o = self._grad(o_pool, k)
            loss = loss + (d_e - d_o).pow(2).mean()
        return loss
