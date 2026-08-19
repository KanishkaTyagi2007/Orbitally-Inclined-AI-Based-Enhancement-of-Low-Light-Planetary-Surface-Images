"""
physics_frontend.py
--------------------
Physics-grounded preprocessing that runs *before* any learned component
touches the pixels. Keeping this stage purely physics-based (no learned
weights) is what lets AuraNet later claim "scientific fidelity": every
correction here is traceable to a sensor calibration constant, not a
network's prior.

Pipeline order (see PhysicsFrontend.process):
    raw DN (uint16)
      -> RadiometricCalibrator   (bias/black-level/gain/dark/flat correction)
      -> CosmicRayScrubber       (impulsive high-energy particle hit removal)
      -> AnscombeVST.forward     (Poisson-Gaussian -> ~unit-variance Gaussian)

The inverse Anscombe transform is applied at the very end of the full
AuraNet pipeline (see aura_pipeline.py), after reconstruction, so that all
learned stages operate in the variance-stabilized domain where Gaussian
noise assumptions (used by NAFNet's training loss) actually hold.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Radiometric calibration
# =============================================================================
class RadiometricCalibrator:
    """
    Converts raw sensor DN into calibrated radiance-proportional units:

        signal_e  = (DN - bias - black_level) * gain_e_per_dn
        corrected = (signal_e - dark_frame_e) / flat_field   [if provided]

    All gains/offsets are pulled from config['calibration'] so a mission or
    sensor swap is a config edit, not a code edit.
    """

    def __init__(self, config: dict):
        cal = config["calibration"]
        self.gain = float(cal["gain_e_per_dn"])
        self.read_noise_e = float(cal["read_noise_e"])
        self.bias_offset = float(cal["bias_offset_dn"])
        self.black_level = float(cal["black_level_dn"])
        self.saturation_dn = float(cal["saturation_dn"])

        self.dark_frame = self._load_optional(cal.get("dark_frame_path"))
        self.flat_field = self._load_optional(cal.get("flat_field_path"))

    @staticmethod
    def _load_optional(path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        import tifffile
        return tifffile.imread(path).astype(np.float32)

    def saturation_mask(self, raw_dn: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels at/above sensor full-well (unrecoverable)."""
        return raw_dn >= self.saturation_dn

    def calibrate(self, raw_dn: np.ndarray) -> np.ndarray:
        raw_dn = raw_dn.astype(np.float32)
        signal_e = (raw_dn - self.bias_offset - self.black_level) * self.gain
        signal_e = np.clip(signal_e, a_min=0.0, a_max=None)

        if self.dark_frame is not None:
            signal_e = signal_e - self.dark_frame

        if self.flat_field is not None:
            flat = np.clip(self.flat_field, 1e-3, None)
            flat = flat / np.median(flat)
            signal_e = signal_e / flat

        return np.clip(signal_e, 0.0, None)


# =============================================================================
# Cosmic ray scrubber
# =============================================================================
class CosmicRayScrubber:
    """
    Detects and inpaints impulsive high-energy-particle hits (cosmic rays),
    which appear as single/few-pixel outliers uncorrelated with real scene
    structure — the classic failure mode that a naive contrast-stretch or a
    generative model would otherwise amplify into a "hallucinated" bright
    speck.

    Method: local sigma-clipping against a median-filtered baseline.
    A pixel is flagged if it deviates from its neighborhood median by more
    than `sigma_threshold` robust standard deviations (MAD-based).
    """

    def __init__(self, config: dict):
        cr = config["cosmic_ray"]
        self.enabled = bool(cr["enabled"])
        self.window = int(cr["window_size"])
        self.sigma_threshold = float(cr["sigma_threshold"])
        self.max_hit_fraction = float(cr["max_hit_fraction"])

    def scrub(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cleaned_image, hit_mask)."""
        if not self.enabled:
            return image, np.zeros_like(image, dtype=bool)

        median = ndimage.median_filter(image, size=self.window)
        residual = image - median
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-6
        robust_std = 1.4826 * mad

        hit_mask = np.abs(residual) > (self.sigma_threshold * robust_std)

        hit_fraction = hit_mask.mean()
        if hit_fraction > self.max_hit_fraction:
            # Likely a calibration/threshold problem rather than real cosmic
            # ray flux — fail safe by not touching the image, since aggressive
            # scrubbing of real structure is exactly the "hallucination in
            # reverse" failure this module exists to prevent.
            return image, np.zeros_like(image, dtype=bool)

        cleaned = np.where(hit_mask, median, image)
        return cleaned, hit_mask


# =============================================================================
# Anscombe variance-stabilizing transform
# =============================================================================
class AnscombeVST:
    """
    Forward:  z = 2 * sqrt(x + 3/8)
    Inverse (closed-form unbiased, Makitalo & Foi 2011 approximation):
        x_hat = (z/2)^2 - 1/8
                + (1/4)*sqrt(3/2)*z^-1
                - (11/8)*z^-2
                + (5/8)*sqrt(3/2)*z^-3
                - 1/8

    Photon-limited planetary sensors exhibit signal-dependent (Poisson)
    noise. Stabilizing variance before the learned denoiser (NAFNet) lets us
    use a plain Gaussian heteroscedastic loss downstream instead of a
    signal-dependent one — simpler and better conditioned.
    """

    def __init__(self, config: dict):
        vst = config["anscombe"]
        self.enabled = bool(vst["apply"])
        self.eps = float(vst["epsilon"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        return 2.0 * np.sqrt(np.clip(x, 0.0, None) + self.eps)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return z
        z = np.clip(z, 1e-3, None)
        return (
            (z / 2.0) ** 2
            - 1.0 / 8.0
            + (1.0 / 4.0) * np.sqrt(1.5) * (z ** -1)
            - (11.0 / 8.0) * (z ** -2)
            + (5.0 / 8.0) * np.sqrt(1.5) * (z ** -3)
            - 1.0 / 8.0
        )


# =============================================================================
# Unified frontend
# =============================================================================
@dataclass
class FrontendResult:
    stabilized: np.ndarray          # ready for wavelet decomposition
    calibrated_linear: np.ndarray   # calibrated, pre-VST (for metric reference)
    cosmic_ray_mask: np.ndarray
    saturation_mask: np.ndarray
    vst: AnscombeVST = field(repr=False)


class PhysicsFrontend:
    """Chains RadiometricCalibrator -> CosmicRayScrubber -> AnscombeVST."""

    def __init__(self, config: dict):
        self.calibrator = RadiometricCalibrator(config)
        self.scrubber = CosmicRayScrubber(config)
        self.vst = AnscombeVST(config)

    def process(self, raw_dn: np.ndarray) -> FrontendResult:
        sat_mask = self.calibrator.saturation_mask(raw_dn)
        calibrated = self.calibrator.calibrate(raw_dn)
        cleaned, cr_mask = self.scrubber.scrub(calibrated)
        stabilized = self.vst.forward(cleaned)

        return FrontendResult(
            stabilized=stabilized.astype(np.float32),
            calibrated_linear=cleaned.astype(np.float32),
            cosmic_ray_mask=cr_mask,
            saturation_mask=sat_mask,
            vst=self.vst,
        )
