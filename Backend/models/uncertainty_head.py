"""
uncertainty_head.py
---------------------
Bayesian heteroscedastic head: predicts, per pixel, both a mean estimate
(mu) and a log-variance (log_var) of the model's own uncertainty about that
estimate. This is the piece of AuraNet most directly responsible for
"minimizing hallucinated structures" at the *output* level (as opposed to
architectural constraints elsewhere in the pipeline): every enhanced pixel
ships with a confidence value, and low-confidence regions are surfaced to
the scientist as a trust map rather than silently presented as fact.

Trained with the standard heteroscedastic Gaussian negative log-likelihood:

    NLL = 0.5 * exp(-log_var) * (target - mu)^2 + 0.5 * log_var

which lets the network attenuate the loss (and therefore its gradient
pressure to "invent" a confident answer) in regions where the input
evidence genuinely doesn't support one — e.g. deep shadow with near-zero
photon counts, or areas the cosmic-ray scrubber heavily modified.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    """
    Lightweight conv head taking the reconstructed feature map and producing
    (mu, log_var), each the same spatial size as the input.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 24,
                 min_log_var: float = -6.0, max_log_var: float = 4.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.mu_head = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        self.log_var_head = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        mu = self.mu_head(feat) + x  # residual around the pre-enhanced input
        log_var = self.log_var_head(feat)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        return mu, log_var


class HeteroscedasticNLLLoss(nn.Module):
    """Gaussian negative log-likelihood with learned per-pixel variance."""

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-log_var)
        return (0.5 * precision * (target - mu).pow(2) + 0.5 * log_var).mean()


def compute_trust_map(log_var: np.ndarray, low_trust_threshold: float = 0.35
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a log-variance map into a normalized [0, 1] trust map and a
    boolean low-confidence mask for downstream flagging / export.

        trust_raw = exp(-log_var)                 (higher = more confident)
        trust     = min-max normalize(trust_raw) over the scene

    Returns:
        trust: float32 array in [0, 1]
        low_trust_mask: bool array, True where trust < low_trust_threshold
    """
    trust_raw = np.exp(-log_var)
    lo, hi = trust_raw.min(), trust_raw.max()
    if hi - lo < 1e-8:
        trust = np.ones_like(trust_raw)
    else:
        trust = (trust_raw - lo) / (hi - lo)
    low_trust_mask = trust < low_trust_threshold
    return trust.astype(np.float32), low_trust_mask
