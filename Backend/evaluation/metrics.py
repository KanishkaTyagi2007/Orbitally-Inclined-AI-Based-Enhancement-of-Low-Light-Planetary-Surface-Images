"""
metrics.py
-----------
Quantitative image-quality evaluation, as required by the project brief:
"provide quantitative image-quality evaluation."

Two families of metric are reported, because a single-number score cannot
distinguish "enhanced" from "hallucinated":

Full-reference (compare enhanced output against the physics-calibrated,
contrast-stretched *raw* image — same scene content, so these measure
whether structure was preserved, not invented):
    - PSNR
    - SSIM   <- gated against config['evaluation']['min_acceptable_ssim_vs_raw_structure']
               as an explicit anti-hallucination guardrail: if enhancement
               diverges too far structurally from the calibrated raw signal,
               the run is flagged rather than silently shipped.

No-reference / naturalness (assess the enhanced image on its own perceptual
merits, since a good enhancement *should* look different from the noisy raw
input by design, so full-reference alone would penalize genuine improvement):
    - NIQE
    - BRISQUE

Plus:
    - Entropy gain (Shannon entropy of enhanced minus raw) as a simple,
      interpretable proxy for "how much previously-invisible information
      became visible."

NIQE/BRISQUE use `pyiqa` (pretrained, corpus-calibrated) when installed —
this is the recommended path for publication-grade numbers. If `pyiqa` is
unavailable, BRISQUE falls back to `piq`, and NIQE falls back to a
self-referential "niqe_lite" approximation (see docstring below) so the
pipeline still runs end-to-end without extra downloads.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy


# =============================================================================
# Full-reference metrics
# =============================================================================
def compute_psnr(reference: np.ndarray, test: np.ndarray,
                  data_range: float = 1.0) -> float:
    return float(peak_signal_noise_ratio(reference, test, data_range=data_range))


def compute_ssim(reference: np.ndarray, test: np.ndarray,
                  data_range: float = 1.0) -> float:
    return float(structural_similarity(reference, test, data_range=data_range))


# =============================================================================
# Entropy gain
# =============================================================================
def compute_entropy_gain(raw: np.ndarray, enhanced: np.ndarray) -> float:
    """Shannon entropy (bits) of enhanced minus raw, over a 256-bin histogram."""
    return float(shannon_entropy(enhanced) - shannon_entropy(raw))


# =============================================================================
# No-reference: BRISQUE
# =============================================================================
def compute_brisque(image: np.ndarray) -> Optional[float]:
    """Lower is better (more natural-looking). Returns None if no backend
    is installed, rather than a silently wrong number."""
    img01 = np.clip(image, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(img01)[None, None].repeat(1, 3, 1, 1)  # (1,3,H,W)

    try:
        import pyiqa
        metric = pyiqa.create_metric("brisque", device="cpu")
        return float(metric(tensor).item())
    except ImportError:
        pass

    try:
        import piq
        return float(piq.brisque(tensor, data_range=1.0).item())
    except ImportError:
        warnings.warn(
            "Neither 'pyiqa' nor 'piq' is installed — BRISQUE skipped. "
            "Install one of them for no-reference naturalness scoring."
        )
        return None


# =============================================================================
# No-reference: NIQE
# =============================================================================
def _mscn_coefficients(image: np.ndarray, c: float = 1.0 / 255.0) -> np.ndarray:
    """Mean-Subtracted Contrast-Normalized coefficients, the core NSS
    feature underlying both the original NIQE and BRISQUE."""
    from scipy.ndimage import gaussian_filter
    mu = gaussian_filter(image, sigma=7 / 6, truncate=3.5)
    mu_sq = gaussian_filter(image ** 2, sigma=7 / 6, truncate=3.5)
    sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0)) + c
    return (image - mu) / sigma


def niqe_lite(image: np.ndarray) -> float:
    """
    Self-referential NIQE approximation.

    The true NIQE (Mittal et al. 2013) scores an image by fitting a
    multivariate Gaussian to 36-dim NSS features over 96x96 patches and
    measuring its distance from a *pretrained* MVG fitted on a large corpus
    of pristine natural images. That pretrained corpus model isn't bundled
    here (see module docstring — use `pyiqa` for the real metric).

    As a dependency-free fallback, this computes the variance of MSCN
    coefficients' deviation from the ideal unit-Gaussian shape (real NSS
    theory: pristine-image MSCN coefficients are closely Gaussian; noise,
    blur, and compression artifacts systematically distort that shape).
    Lower = closer to natural-image statistics. This is a *proxy*, not a
    calibrated NIQE score — treat it as directionally informative only.
    """
    mscn = _mscn_coefficients(image.astype(np.float64))
    # Ideal pristine MSCN ~ N(0,1); score = deviation of empirical moments
    # (mean, variance, excess kurtosis) from the Gaussian ideal (0, 1, 0).
    mean = mscn.mean()
    var = mscn.var()
    kurtosis = ((mscn - mean) ** 4).mean() / (var ** 2 + 1e-8) - 3.0
    score = abs(mean) + abs(var - 1.0) + abs(kurtosis)
    return float(score)


def compute_niqe(image: np.ndarray) -> float:
    img01 = np.clip(image, 0.0, 1.0).astype(np.float32)
    try:
        import pyiqa
        tensor = torch.from_numpy(img01)[None, None]
        metric = pyiqa.create_metric("niqe", device="cpu")
        return float(metric(tensor).item())
    except ImportError:
        return niqe_lite(img01)


# =============================================================================
# Unified evaluation entry point
# =============================================================================
def evaluate_all(raw_reference: np.ndarray, enhanced: np.ndarray, config: dict) -> dict:
    """
    Runs every metric named in config['evaluation']['metrics'] and applies
    the structure-preservation guardrail.

    `raw_reference` should be the physics-calibrated (not learned-enhanced)
    image, contrast-stretched to the same [0,1] range as `enhanced`, so PSNR/
    SSIM measure structural fidelity rather than penalizing intended
    brightness/contrast changes.
    """
    ev_cfg = config["evaluation"]
    requested = set(ev_cfg["metrics"])
    results = {}

    if "psnr" in requested:
        results["psnr"] = compute_psnr(raw_reference, enhanced)
    if "ssim" in requested:
        results["ssim"] = compute_ssim(raw_reference, enhanced)
    if "niqe" in requested:
        results["niqe"] = compute_niqe(enhanced)
    if "brisque" in requested:
        results["brisque"] = compute_brisque(enhanced)
    if "entropy_gain" in requested:
        results["entropy_gain"] = compute_entropy_gain(raw_reference, enhanced)

    if "ssim" in results:
        threshold = float(ev_cfg["min_acceptable_ssim_vs_raw_structure"])
        results["structure_guardrail_passed"] = bool(results["ssim"] >= threshold)
        results["structure_guardrail_threshold"] = threshold

    return results
