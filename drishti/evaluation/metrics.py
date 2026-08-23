"""
metrics.py
===========
STAGE 6.1 / 6.2 -- SCIENTIFIC METRIC HARNESS

    6.1  Reference-free quality metrics   NIQE, BRISQUE, entropy gain (dH)
         plus full-reference PSNR / SSIM against the calibrated raw scene
    6.2  Downstream utility proof         frozen topological crater detector

Two metric families are reported, because no single number distinguishes
"enhanced" from "hallucinated":

Full-reference, against the physics-calibrated raw scene stretched to the same
range as the output. Same scene content, so these measure whether structure was
*preserved*, not whether it was invented:
    PSNR, SSIM  -- SSIM is gated against
                   evaluation.min_acceptable_ssim_vs_raw_structure as an
                   explicit anti-hallucination guardrail.

Reference-free, assessing the output on its own merits, since a successful
enhancement *should* differ from the noisy input and full-reference metrics
alone would penalize genuine improvement:
    NIQE, BRISQUE, entropy gain.

NIQE/BRISQUE use `pyiqa` (pretrained, corpus-calibrated) when installed, which
is the path for publication-grade numbers; BRISQUE falls back to `piq`, and
NIQE to the self-referential `niqe_lite` proxy documented below. Where no
backend exists the metric reports `None` rather than a silently wrong number.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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
# Information entropy gain
# =============================================================================
def histogram_entropy(image: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy in bits over a *fixed* `bins`-bin histogram spanning [0, 1].

    The fixed shared binning is essential and is why `skimage.shannon_entropy`
    is not used here. That function counts unique values, which for a
    continuous float image is very nearly one bin per pixel -- so it returns
    approximately log2(N_pixels) for *any* float image regardless of content.
    Measured on the synthetic demo scene it reported 15.16 bits for the
    enhanced image against a log2(36864) = 15.17 bit ceiling, making the
    apparent 6-bit "entropy gain" a pure artifact of the raw image being
    quantized and the output not being. On a shared 256-bin histogram the same
    pair differs by 0.09 bits, which is the real figure.
    """
    hist, _ = np.histogram(np.clip(image, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def compute_entropy_gain(raw: np.ndarray, enhanced: np.ndarray,
                         bins: int = 256) -> float:
    """
    Information entropy gain (dH): entropy of the enhanced image minus the raw
    image, both on the same fixed histogram -- an interpretable proxy for how
    much previously indistinguishable information became distinguishable.
    """
    return histogram_entropy(enhanced, bins) - histogram_entropy(raw, bins)


# =============================================================================
# No-reference: BRISQUE
# =============================================================================
def compute_brisque(image: np.ndarray) -> Optional[float]:
    """Lower is better. Returns None when no backend is installed."""
    img01 = np.clip(image, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(img01)[None, None]

    try:
        import pyiqa
        metric = pyiqa.create_metric("brisque", device="cpu")
        return float(metric(tensor.repeat(1, 3, 1, 1)).item())
    except Exception:
        pass

    try:
        import piq
        return float(piq.brisque(tensor, data_range=1.0).item())
    except Exception:
        warnings.warn(
            "Neither 'pyiqa' nor 'piq' is installed -- BRISQUE skipped. "
            "Install one of them for no-reference naturalness scoring."
        )
        return None


# =============================================================================
# No-reference: NIQE
# =============================================================================
def _mscn_coefficients(image: np.ndarray, c: float = 1.0 / 255.0) -> np.ndarray:
    """
    Mean-Subtracted Contrast-Normalized coefficients -- the natural scene
    statistics feature underlying both NIQE and BRISQUE.
    """
    mu = ndimage.gaussian_filter(image, sigma=7 / 6, truncate=3.5)
    mu_sq = ndimage.gaussian_filter(image ** 2, sigma=7 / 6, truncate=3.5)
    sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0)) + c
    return (image - mu) / sigma


def niqe_lite(image: np.ndarray) -> float:
    """
    Self-referential NIQE approximation (a proxy, not a calibrated NIQE).

    True NIQE (Mittal et al., 2013) scores an image by fitting a multivariate
    Gaussian to 36-dimensional NSS features over 96x96 patches and measuring
    its distance from a *pretrained* MVG fitted on a corpus of pristine natural
    images. That corpus model is not bundled here -- install `pyiqa` for the
    real metric.

    As a dependency-free fallback this measures how far the MSCN coefficients
    depart from the unit-Gaussian shape that pristine images exhibit; noise,
    blur and quantization each distort that shape in characteristic ways.
    Lower is closer to natural-image statistics. Directionally informative
    only, and labelled `niqe_lite` in the report so it is never mistaken for
    a published NIQE value.
    """
    mscn = _mscn_coefficients(image.astype(np.float64))
    mean = mscn.mean()
    var = mscn.var()
    kurtosis = ((mscn - mean) ** 4).mean() / (var ** 2 + 1e-8) - 3.0
    return float(abs(mean) + abs(var - 1.0) + abs(kurtosis))


def compute_niqe(image: np.ndarray) -> tuple[float, str]:
    """Returns (score, backend_name)."""
    img01 = np.clip(image, 0.0, 1.0).astype(np.float32)
    try:
        import pyiqa
        metric = pyiqa.create_metric("niqe", device="cpu")
        return float(metric(torch.from_numpy(img01)[None, None]).item()), "pyiqa"
    except Exception:
        return niqe_lite(img01), "niqe_lite"


# =============================================================================
# STAGE 6.2 -- Frozen topological crater detector
# =============================================================================
class TopologicalCraterDetector:
    """
    A deterministic, non-learned crater detector used as a downstream utility
    proof.

    "Frozen" is the load-bearing word: the detector has no trainable
    parameters and is applied identically to the raw and the enhanced scene.
    Any difference in what it finds is therefore attributable to the
    enhancement alone -- there is no possibility of the detector having been
    tuned, deliberately or accidentally, to flatter the enhanced product.

    "Topological" refers to the feature it keys on. A crater is a bowl-shaped
    depression, i.e. a local minimum of the surface with a closed rim, which in
    an image appears as a dark blob at some characteristic scale. Scale-
    normalized Laplacian-of-Gaussian response

        R(x, y; sigma) = sigma^2 * (grad^2 G_sigma * I)(x, y)

    peaks at the centre of a dark blob whose radius is sigma*sqrt(2). Sweeping
    sigma builds a scale space; local maxima in that 3-D volume are candidate
    craters, and overlapping candidates are suppressed largest-response-first.
    """

    def __init__(self, config: dict):
        cd = config["evaluation"]["crater_detector"]
        self.enabled = bool(cd.get("enabled", True))
        self.min_radius = float(cd["min_radius_px"])
        self.max_radius = float(cd["max_radius_px"])
        self.num_scales = int(cd["num_scales"])
        self.threshold = float(cd["detection_threshold"])
        self.overlap_threshold = float(cd["overlap_threshold"])
        self.match_iou = float(cd["match_iou"])
        self.pyramid = bool(cd.get("pyramid", True))
        self.pyramid_min_sigma = float(cd.get("pyramid_min_sigma", 4.0))

    # -- scale space -------------------------------------------------------
    @staticmethod
    def _block_mean(image: np.ndarray, k: int) -> np.ndarray:
        h = int(np.ceil(image.shape[0] / k)) * k
        w = int(np.ceil(image.shape[1] / k)) * k
        padded = np.pad(image, ((0, h - image.shape[0]), (0, w - image.shape[1])),
                        mode="edge")
        return padded.reshape(h // k, k, w // k, k).mean(axis=(1, 3))

    def _log_response(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Scale-normalized Laplacian of Gaussian, sigma^2 * grad^2(G_sigma * I).

        Cost of a separable Gaussian grows linearly with sigma, and the coarse
        scales dominate: on a 1536 px tile the eight-scale sweep measured 13.9 s
        of a 39 s run. The operator is scale-invariant, so a coarse scale can be
        evaluated on a decimated grid -- blur sigma/k on a k-fold block-mean of
        the image, then resample the response back. The blob it responds to is
        tens of pixels across, so the k-pixel positional quantisation is far
        below the feature size.

        Set `pyramid: false` in the config to force full-resolution evaluation
        at every scale.
        """
        if not self.pyramid or sigma <= self.pyramid_min_sigma:
            return (sigma ** 2 * ndimage.gaussian_laplace(
                image, sigma, mode="nearest")).astype(np.float32)

        k = int(2 ** np.floor(np.log2(sigma / self.pyramid_min_sigma)))
        k = int(np.clip(k, 1, 8))
        if k == 1:
            return (sigma ** 2 * ndimage.gaussian_laplace(
                image, sigma, mode="nearest")).astype(np.float32)

        small = self._block_mean(image, k)
        response = (sigma / k) ** 2 * ndimage.gaussian_laplace(
            small, sigma / k, mode="nearest")
        zoomed = ndimage.zoom(response, k, order=1, mode="nearest")
        return zoomed[:image.shape[0], :image.shape[1]].astype(np.float32)

    # -- geometry ----------------------------------------------------------
    @staticmethod
    def circle_iou(a: np.ndarray, b: np.ndarray) -> float:
        """Exact intersection-over-union of two circles (y, x, r)."""
        (y1, x1, r1), (y2, x2, r2) = a[:3], b[:3]
        d = float(np.hypot(y1 - y2, x1 - x2))
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            small, large = min(r1, r2), max(r1, r2)
            return float((small ** 2) / (large ** 2))

        r1sq, r2sq, dsq = r1 ** 2, r2 ** 2, d ** 2
        alpha = np.arccos(np.clip((dsq + r1sq - r2sq) / (2 * d * r1), -1.0, 1.0))
        beta = np.arccos(np.clip((dsq + r2sq - r1sq) / (2 * d * r2), -1.0, 1.0))
        intersection = (r1sq * (alpha - np.sin(2 * alpha) / 2)
                        + r2sq * (beta - np.sin(2 * beta) / 2))
        union = np.pi * (r1sq + r2sq) - intersection
        return float(intersection / union) if union > 0 else 0.0

    # -- detection ---------------------------------------------------------
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Returns an (N, 4) array of detections: (y, x, radius_px, response).
        Input is normalized internally, so `detection_threshold` is
        scene-independent.
        """
        if not self.enabled:
            return np.zeros((0, 4))

        img = image.astype(np.float64)
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-12:
            return np.zeros((0, 4))
        img = (img - lo) / (hi - lo)

        sigmas = np.linspace(self.min_radius, self.max_radius,
                             self.num_scales) / np.sqrt(2.0)
        # float32: the scale space is num_scales x H x W and dominates memory on
        # large products; single precision is far beyond what a detection
        # threshold needs.
        scale_space = np.stack([self._log_response(img, s) for s in sigmas])

        # Local maxima in the (scale, y, x) volume.
        peaks = (scale_space == ndimage.maximum_filter(scale_space, size=(3, 3, 3),
                                                       mode="nearest"))
        peaks &= scale_space > self.threshold

        idx = np.argwhere(peaks)
        if idx.size == 0:
            return np.zeros((0, 4))

        responses = scale_space[idx[:, 0], idx[:, 1], idx[:, 2]]
        radii = sigmas[idx[:, 0]] * np.sqrt(2.0)
        candidates = np.column_stack(
            [idx[:, 1].astype(float), idx[:, 2].astype(float), radii, responses]
        )
        return self._suppress_overlaps(candidates)

    def _suppress_overlaps(self, candidates: np.ndarray) -> np.ndarray:
        """
        Greedy non-maximum suppression, strongest response first.

        Spatially indexed: two circles can only overlap if their centres are
        closer than the sum of their radii, so each candidate is compared only
        against neighbours inside that reach instead of against every survivor.
        On a real 2048x2048 lunar tile a naive all-pairs sweep is quadratic in
        the candidate count, which runs into thousands once the detection
        threshold is lowered.
        """
        from scipy.spatial import cKDTree

        order = np.argsort(-candidates[:, 3])
        cand = candidates[order]
        tree = cKDTree(cand[:, :2])
        max_radius = float(cand[:, 2].max())

        suppressed = np.zeros(len(cand), dtype=bool)
        kept: list[int] = []
        for i in range(len(cand)):
            if suppressed[i]:
                continue
            kept.append(i)
            reach = cand[i, 2] + max_radius
            for j in tree.query_ball_point(cand[i, :2], reach):
                if j > i and not suppressed[j]:
                    if self.circle_iou(cand[i], cand[j]) > self.overlap_threshold:
                        suppressed[j] = True
        return cand[kept] if kept else np.zeros((0, 4))

    # -- comparison --------------------------------------------------------
    def match(self, detections_a: np.ndarray, detections_b: np.ndarray
              ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """
        Greedy highest-IoU-first matching between two detection sets.
        Returns (matches as (i, j, iou), unmatched_a, unmatched_b).

        Spatially indexed for the same reason as `_suppress_overlaps`: the full
        cross product is O(len(a) * len(b)), and a pair whose centres are
        further apart than the sum of their radii has zero IoU by construction.
        """
        from scipy.spatial import cKDTree

        if len(detections_a) == 0 or len(detections_b) == 0:
            return [], list(range(len(detections_a))), list(range(len(detections_b)))

        tree_b = cKDTree(detections_b[:, :2])
        reach = float(detections_a[:, 2].max() + detections_b[:, 2].max())

        pairs = []
        for i in range(len(detections_a)):
            for j in tree_b.query_ball_point(detections_a[i, :2], reach):
                iou = self.circle_iou(detections_a[i], detections_b[j])
                if iou >= self.match_iou:
                    pairs.append((iou, i, j))
        pairs.sort(reverse=True)

        used_a: set[int] = set()
        used_b: set[int] = set()
        matches = []
        for iou, i, j in pairs:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j, iou))

        unmatched_a = [i for i in range(len(detections_a)) if i not in used_a]
        unmatched_b = [j for j in range(len(detections_b)) if j not in used_b]
        return matches, unmatched_a, unmatched_b


def crater_detection_utility(raw: np.ndarray, enhanced: np.ndarray, config: dict,
                             trust_map: Optional[np.ndarray] = None) -> dict:
    """
    Stage 6.2 -- runs the frozen detector on both scenes and reports whether
    the enhancement made craters more findable.

    Reported quantities:
        craters_detected_raw / _enhanced  detector yield on each scene
        craters_matched                   detections agreeing across both
        crater_miou                       mean IoU over matched pairs -- how
                                          consistently the two scenes localize
                                          the same craters
        craters_revealed                  present only after enhancement
        craters_lost                      present before but not after
        detection_gain                    enhanced / raw yield ratio

    When a trust map is supplied, the mean and minimum trust at newly revealed
    crater sites is also reported, so a reviewer can weigh each recovery: a
    revealed crater in a high-trust region is a plausible genuine recovery,
    while one in a low-trust region should be treated as a candidate for
    follow-up imaging rather than as a detection.

    Read this as a *utility* measure, not an integrity test. Measured across
    eight synthetic seeds, the revealed-crater count does not by itself
    separate a genuine enhancement from one with deliberately fabricated
    craters (16.4 mean revealed versus 20.6, with per-seed overlap). The
    integrity gates are stage 5.1 and 5.2; this stage answers the different
    question of whether the product is more useful downstream.
    """
    cd_cfg = config["evaluation"]["crater_detector"]
    if not cd_cfg.get("enabled", True):
        return {"crater_detection_checked": False}

    detector = TopologicalCraterDetector(config)
    dets_raw = detector.detect(raw)
    dets_enh = detector.detect(enhanced)
    matches, _, unmatched_enh = detector.match(dets_raw, dets_enh)

    n_raw, n_enh = len(dets_raw), len(dets_enh)
    miou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    report = {
        "crater_detection_checked": True,
        "craters_detected_raw": n_raw,
        "craters_detected_enhanced": n_enh,
        "craters_matched": len(matches),
        "crater_miou": miou,
        "craters_revealed": len(unmatched_enh),
        "craters_lost": n_raw - len(matches),
        "detection_gain": float(n_enh / n_raw) if n_raw else float(n_enh),
    }

    if trust_map is not None and unmatched_enh:
        trusts = []
        for j in unmatched_enh:
            y, x = int(round(dets_enh[j, 0])), int(round(dets_enh[j, 1]))
            if 0 <= y < trust_map.shape[0] and 0 <= x < trust_map.shape[1]:
                trusts.append(float(trust_map[y, x]))
        if trusts:
            report["revealed_crater_mean_trust"] = float(np.mean(trusts))
            report["revealed_crater_min_trust"] = float(np.min(trusts))

    return report


# =============================================================================
# STAGE 6.1 -- unified evaluation entry point
# =============================================================================
def evaluate_all(raw_reference: np.ndarray, enhanced: np.ndarray, config: dict,
                 trust_map: Optional[np.ndarray] = None) -> dict:
    """
    Runs every metric named in config['evaluation']['metrics'], applies the
    SSIM structure-preservation guardrail, and appends the stage 6.2 crater
    detection utility proof.

    `raw_reference` must be the physics-calibrated (not learned-enhanced)
    scene, stretched to the same [0, 1] range as `enhanced`, so PSNR/SSIM
    measure structural fidelity instead of penalizing intended brightness and
    contrast changes.
    """
    ev_cfg = config["evaluation"]
    requested = set(ev_cfg["metrics"])
    results: dict = {}

    if "psnr" in requested:
        results["psnr"] = compute_psnr(raw_reference, enhanced)
    if "ssim" in requested:
        results["ssim"] = compute_ssim(raw_reference, enhanced)
    if "niqe" in requested:
        score, backend = compute_niqe(enhanced)
        results["niqe"] = score
        results["niqe_backend"] = backend
    if "brisque" in requested:
        results["brisque"] = compute_brisque(enhanced)
    if "entropy_gain" in requested:
        bins = int(ev_cfg.get("entropy_bins", 256))
        results["entropy_raw"] = histogram_entropy(raw_reference, bins)
        results["entropy_enhanced"] = histogram_entropy(enhanced, bins)
        results["entropy_gain"] = results["entropy_enhanced"] - results["entropy_raw"]
        results["entropy_bins"] = bins

    if ev_cfg.get("record_histograms", True):
        # The tonal distributions the entropy figures are derived from, kept so
        # a reader (or a dashboard) can see *how* the histogram was redistributed
        # rather than only the single-number entropy delta. Downsampled to 64
        # bins: enough to show the shape, small enough to sit in a JSON report.
        chart_bins = int(ev_cfg.get("histogram_bins", 64))
        for label, image in (("raw", raw_reference), ("enhanced", enhanced)):
            counts, _ = np.histogram(np.clip(image, 0.0, 1.0),
                                     bins=chart_bins, range=(0.0, 1.0))
            results[f"histogram_{label}"] = counts.astype(int).tolist()
        results["histogram_bins"] = chart_bins

    if "ssim" in results:
        threshold = float(ev_cfg["min_acceptable_ssim_vs_raw_structure"])
        results["structure_guardrail_threshold"] = threshold
        results["structure_guardrail_passed"] = bool(results["ssim"] >= threshold)

    results.update(
        crater_detection_utility(raw_reference, enhanced, config, trust_map)
    )
    return results
