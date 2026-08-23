"""
physics_frontend.py
====================
AURA-NET physics operators. This module holds every stage that is pure
physics -- zero learned weights, every correction traceable to a sensor
calibration constant or a published photometric law.

    STAGE 1  SCIENTIFIC INGESTION & PHYSICS FRONT-END
      1.1  PlanetaryIngestor       PDS4 / FITS / GeoTIFF -> 16-bit linear array
                                    + affine transform + CRS metadata
      1.2  RadiometricCalibrator   DN -> spectral radiance [W m^-2 sr^-1 um^-1]
      1.3  CosmicRayScrubber       Laplacian edge rejection of SEU spikes
      1.4  AnscombeVST             mixed Poisson-Gaussian -> AWGN (sigma ~ 1)

    STAGE 4  DYNAMIC RANGE COMPRESSION & PHOTOMETRIC NORMALIZATION
      4.1  BilateralLogToneMapper  shadow recovery without highlight clipping
      4.2  LommelSeeligerCorrection  normalize incidence/emission geometry

Stage 1 is exposed as `PhysicsFrontend`, stage 4 as `PhotometricBackend`.
Keeping both in one module is deliberate: they are inverse halves of the same
radiometric contract, and a sensor swap should require editing exactly one
file plus the config.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage


# =============================================================================
# STAGE 1.1 -- Scientific ingestion
# =============================================================================
@dataclass
class IngestedScene:
    """A raw planetary product plus everything needed to write it back out."""

    dn: np.ndarray                      # 16-bit linear digital numbers (float64 view)
    profile: Optional[dict] = None      # rasterio profile (driver/dtype/crs/transform)
    crs: Any = None                     # coordinate reference system (planetary)
    transform: Any = None               # affine geotransform
    metadata: dict = field(default_factory=dict)   # product-level tags
    source_format: str = "unknown"
    nodata_mask: Optional[np.ndarray] = None


class PlanetaryIngestor:
    """
    Reads a raw planetary raster and preserves its spatial metadata.

    Format resolution order:
      1. rasterio/GDAL  -- GeoTIFF, ISIS3, PDS3/PDS4 (via GDAL drivers), and
         anything else GDAL understands. This is the only path that recovers
         the affine transform + planetary CRS, so it is always tried first.
      2. astropy.io.fits -- FITS products (common for orbital instruments).
      3. pds4_tools      -- PDS4 XML-labelled products GDAL declines.
      4. tifffile        -- last-resort plain TIFF read (no georeferencing).

    Whichever path succeeds, the returned `IngestedScene` carries the raw
    linear DN array untouched; no scaling, stretching or dtype narrowing
    happens here, because every later stage depends on DN being linear.
    """

    def __init__(self, config: dict):
        io_cfg = config["io"]
        self.band_index = int(io_cfg.get("band_index", 1))
        self.fits_hdu = int(io_cfg.get("fits_hdu", 0))
        self.nodata_value = io_cfg.get("nodata_value", None)

    # -- individual readers ------------------------------------------------
    def _read_rasterio(self, path: str) -> Optional[IngestedScene]:
        try:
            import rasterio
        except ImportError:
            return None
        try:
            with rasterio.open(path) as src:
                band = min(self.band_index, src.count)
                array = src.read(band)
                nodata = src.nodata if src.nodata is not None else self.nodata_value
                mask = (array == nodata) if nodata is not None else None
                return IngestedScene(
                    dn=array.astype(np.float64),
                    profile=src.profile.copy(),
                    crs=src.crs,
                    transform=src.transform,
                    metadata=dict(src.tags()),
                    source_format=src.driver,
                    nodata_mask=mask,
                )
        except Exception:
            return None

    def _read_fits(self, path: str) -> Optional[IngestedScene]:
        try:
            from astropy.io import fits
        except ImportError:
            return None
        try:
            with fits.open(path) as hdul:
                hdu = hdul[self.fits_hdu]
                if hdu.data is None:  # primary HDU may be header-only
                    hdu = next(h for h in hdul if h.data is not None)
                array = np.asarray(hdu.data, dtype=np.float64)
                while array.ndim > 2:
                    array = array[0]
                header = {k: str(v) for k, v in hdu.header.items() if k}
                return IngestedScene(
                    dn=array, metadata=header, source_format="FITS",
                )
        except Exception:
            return None

    def _read_pds4(self, path: str) -> Optional[IngestedScene]:
        try:
            import pds4_tools
        except ImportError:
            return None
        try:
            structures = pds4_tools.read(path, quiet=True)
            for struct in structures:
                if getattr(struct, "is_array", lambda: False)():
                    array = np.asarray(struct.data, dtype=np.float64)
                    while array.ndim > 2:
                        array = array[0]
                    meta = {}
                    label = getattr(structures, "label", None)
                    if label is not None:
                        meta["pds4_label"] = str(label.to_string()[:4096])
                    return IngestedScene(
                        dn=array, metadata=meta, source_format="PDS4",
                    )
        except Exception:
            return None
        return None

    def _read_tifffile(self, path: str) -> IngestedScene:
        import tifffile
        array = np.asarray(tifffile.imread(path), dtype=np.float64)
        while array.ndim > 2:
            array = array[..., 0] if array.shape[-1] <= 4 else array[0]
        warnings.warn(
            f"Read {Path(path).name} with tifffile -- no CRS/affine transform "
            "recovered. Install rasterio (with GDAL) to preserve georeferencing."
        )
        return IngestedScene(dn=array, source_format="TIFF (no georeferencing)")

    # -- public entry point -------------------------------------------------
    def read(self, path: str) -> IngestedScene:
        suffix = Path(path).suffix.lower()

        # Extension picks the first reader to try; the rest follow as fallbacks
        # (GDAL declines some vendor products that astropy/pds4_tools accept,
        # and vice versa).
        readers = [self._read_rasterio, self._read_fits, self._read_pds4]
        if suffix in (".fits", ".fit", ".fts"):
            readers.insert(0, readers.pop(readers.index(self._read_fits)))
        elif suffix in (".xml", ".lbl"):
            readers.insert(0, readers.pop(readers.index(self._read_pds4)))

        for reader in readers:
            scene = reader(path)
            if scene is not None:
                if scene.nodata_mask is None and self.nodata_value is not None:
                    scene.nodata_mask = scene.dn == self.nodata_value
                return scene

        return self._read_tifffile(path)


# =============================================================================
# STAGE 1.2 -- Radiometric calibration (DN -> spectral radiance)
# =============================================================================
class RadiometricCalibrator:
    """
    Converts raw sensor DN into SI spectral radiance:

        electrons  = (DN - bias - black_level) * gain_e_per_dn
        electrons -= dark_frame                        [if provided]
        electrons /= normalized_flat_field              [if provided]
        radiance   = electrons / exposure_time_s * radiance_scale
                     [W m^-2 sr^-1 um^-1]

    `radiance_scale` folds pixel solid angle, aperture area, quantum
    efficiency and filter bandwidth into a single traceable responsivity
    constant, so a mission swap is a config edit.

    The electrons-per-radiance-unit factor is retained (`electrons_per_radiance`)
    because the Anscombe VST in stage 1.4 must reason about photon counts,
    not radiance, to stabilize Poisson noise correctly.
    """

    def __init__(self, config: dict):
        cal = config["calibration"]
        self.gain = float(cal["gain_e_per_dn"])
        self.read_noise_e = float(cal["read_noise_e"])
        self.bias_offset = float(cal["bias_offset_dn"])
        self.black_level = float(cal["black_level_dn"])
        self.saturation_dn = float(cal["saturation_dn"])
        self.exposure_time = float(cal.get("exposure_time_s", 1.0))
        self.radiance_scale = float(cal.get("radiance_scale", 1.0))
        self.bandwidth_um = float(cal.get("bandwidth_um", 1.0))
        # Declared units for the calibrated output. Config-driven because a
        # product processed without its instrument's radiometric calibration
        # kernel is NOT in SI radiance, and stamping SI units on the GeoTIFF
        # header would be a false provenance claim.
        self._units = str(cal.get("radiance_units", "W m^-2 sr^-1 um^-1"))

        self.dark_frame = self._load_optional(cal.get("dark_frame_path"))
        self.flat_field = self._load_optional(cal.get("flat_field_path"))

        # radiance = electrons * k  ->  electrons = radiance / k
        self.radiance_per_electron = self.radiance_scale / max(self.exposure_time, 1e-12)
        self.electrons_per_radiance = 1.0 / max(self.radiance_per_electron, 1e-30)

    @staticmethod
    def _load_optional(path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        import tifffile
        return np.asarray(tifffile.imread(path), dtype=np.float64)

    def saturation_mask(self, raw_dn: np.ndarray) -> np.ndarray:
        """Pixels at/above sensor full-well: unrecoverable, never trustworthy."""
        return raw_dn >= self.saturation_dn

    def to_electrons(self, raw_dn: np.ndarray) -> np.ndarray:
        signal_e = (raw_dn.astype(np.float64) - self.bias_offset - self.black_level) * self.gain
        signal_e = np.clip(signal_e, 0.0, None)

        if self.dark_frame is not None:
            signal_e = signal_e - self.dark_frame

        if self.flat_field is not None:
            flat = np.clip(self.flat_field, 1e-3, None)
            flat = flat / np.median(flat)
            signal_e = signal_e / flat

        return np.clip(signal_e, 0.0, None)

    def calibrate(self, raw_dn: np.ndarray) -> np.ndarray:
        """Returns spectral radiance in W m^-2 sr^-1 um^-1."""
        return self.to_electrons(raw_dn) * self.radiance_per_electron

    @property
    def units(self) -> str:
        return self._units


# =============================================================================
# STAGE 1.3 -- Cosmic ray / SEU scrubber (Laplacian edge rejection)
# =============================================================================
class CosmicRayScrubber:
    """
    Detects and inpaints single-event ionization spikes -- cosmic ray hits and
    SEU-induced hot pixels -- which appear as sharp-edged outliers uncorrelated
    with real surface structure. Left in place, a downstream contrast stretch
    amplifies them into convincing but entirely fictitious bright specks.

    Default method ("laplacian") implements van Dokkum (2001) L.A.Cosmic
    edge-rejection, which separates cosmic rays from real sharp features by
    their *edge sharpness* rather than their amplitude alone:

      1. 2x subsample, convolve with the discrete Laplacian, keep the positive
         part, rebin to native scale. Cosmic rays -- being undersampled
         delta-like events -- produce a far stronger Laplacian response than
         PSF-limited real structure of the same brightness.
      2. Build the per-pixel noise model from the median-filtered signal:
             sigma = sqrt(median_e + read_noise_e^2)
      3. Form S = L / (2*sigma), and subtract its own median filter to remove
         the smooth sampling-flux component: S_prime.
      4. Build the fine-structure image F (median3 - median7), which is large
         for genuine PSF-scale detail and small for delta-like hits.
      5. Flag pixels where S_prime > sigma_threshold AND L/F > laplacian_ratio.

    Step 4/5 is what makes this safe on real terrain: a crater rim has a large
    F and is therefore *not* flagged even when it is bright and sharp.

    "sigma_clip_median" is retained as a cheaper fallback for products whose
    PSF is already heavily oversampled.
    """

    _LAPLACIAN = np.array([[0.0, -1.0, 0.0],
                           [-1.0, 4.0, -1.0],
                           [0.0, -1.0, 0.0]])

    def __init__(self, config: dict):
        cr = config["cosmic_ray"]
        self.enabled = bool(cr["enabled"])
        self.method = str(cr.get("method", "laplacian"))
        self.window = int(cr["window_size"])
        self.sigma_threshold = float(cr["sigma_threshold"])
        self.laplacian_ratio = float(cr.get("laplacian_ratio", 2.0))
        self.max_hit_fraction = float(cr["max_hit_fraction"])
        self.dilate_iterations = int(cr.get("dilate_iterations", 1))
        self.read_noise_e = float(config["calibration"]["read_noise_e"])

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _subsample2(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

    @staticmethod
    def _rebin2(a: np.ndarray) -> np.ndarray:
        """Inverse of `_subsample2`: block-averages 2x2 back to native scale."""
        h, w = a.shape[0] // 2, a.shape[1] // 2
        return a.reshape(h, 2, w, 2).mean(axis=(1, 3))

    def _laplacian_detect(self, electrons: np.ndarray,
                          baseline: np.ndarray) -> np.ndarray:
        # 1) sharpness response at 2x sampling. _subsample2 always yields even
        #    dimensions, so _rebin2 returns exactly the native shape.
        sub = self._subsample2(electrons)
        lap = ndimage.convolve(sub, self._LAPLACIAN, mode="nearest")
        L = self._rebin2(np.clip(lap, 0.0, None))

        # 2) per-pixel Poisson+read noise model, in electrons. `baseline` is the
        #    window-sized median, computed once by the caller and reused for
        #    inpainting -- a median filter is the most expensive operation in
        #    stage 1, so it is not worth running twice.
        sigma = np.sqrt(np.clip(baseline, 0.0, None) + self.read_noise_e ** 2) + 1e-8

        # 3) normalized sharpness, sampling-flux removed
        S = L / (2.0 * sigma)
        S_prime = S - ndimage.median_filter(S, size=5, mode="nearest")

        # 4) fine-structure image: large for real PSF-scale detail
        med3 = ndimage.median_filter(electrons, size=3, mode="nearest")
        F = med3 - ndimage.median_filter(med3, size=7, mode="nearest")
        F = np.clip(F, 0.01, None)

        # 5) joint criterion
        return (S_prime > self.sigma_threshold) & ((L / F) > self.laplacian_ratio)

    def _sigma_clip_detect(self, electrons: np.ndarray,
                           baseline: np.ndarray) -> np.ndarray:
        residual = electrons - baseline
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-6
        robust_std = 1.4826 * mad
        return np.abs(residual) > (self.sigma_threshold * robust_std)

    # -- public ------------------------------------------------------------
    def scrub(self, electrons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Operates in the electron domain (where the Poisson noise model is
        valid). Returns (cleaned_electrons, hit_mask).
        """
        if not self.enabled:
            return electrons, np.zeros(electrons.shape, dtype=bool)

        # One median filter serves both detection and inpainting.
        baseline = ndimage.median_filter(electrons, size=self.window, mode="nearest")

        if self.method == "laplacian":
            hit_mask = self._laplacian_detect(electrons, baseline)
        else:
            hit_mask = self._sigma_clip_detect(electrons, baseline)

        if self.dilate_iterations > 0 and hit_mask.any():
            # Grow by one pixel to catch the charge-bleed skirt around a hit.
            hit_mask = ndimage.binary_dilation(
                hit_mask, iterations=self.dilate_iterations
            )

        hit_fraction = float(hit_mask.mean())
        if hit_fraction > self.max_hit_fraction:
            # More plausibly a calibration/threshold error than a genuine
            # particle flux. Scrubbing real terrain away is hallucination in
            # reverse, so fail safe and touch nothing.
            warnings.warn(
                f"Cosmic-ray scrub flagged {hit_fraction:.2%} of pixels "
                f"(cap {self.max_hit_fraction:.2%}) -- skipping scrub. "
                "Check cosmic_ray.sigma_threshold and the calibration constants."
            )
            return electrons, np.zeros(electrons.shape, dtype=bool)

        cleaned = np.where(hit_mask, baseline, electrons)
        return cleaned, hit_mask


# =============================================================================
# STAGE 1.4 -- Anscombe variance-stabilizing transform
# =============================================================================
class AnscombeVST:
    """
    Maps mixed Poisson-Gaussian sensor noise onto approximately additive white
    Gaussian noise of unit variance, so every learned stage downstream can
    assume a single, signal-independent noise level.

    Generalized (Poisson + read noise), operating in electrons:
        z = 2 * sqrt(e + 3/8 + sigma_read^2)

    Inverse: the closed-form unbiased inverse of Makitalo & Foi (2011),
        e_hat = (z/2)^2 - 1/8 + (1/4)*sqrt(3/2)*z^-1
                - (11/8)*z^-2 + (5/8)*sqrt(3/2)*z^-3
        (then the read-noise offset is removed)

    The transform is applied to *radiance*, per the pipeline diagram, but
    internally converts to electrons first via `electrons_per_radiance`
    because photon counts -- not SI radiance -- are what is Poisson
    distributed. Forward and inverse are exact mutual inverses in radiance.
    """

    def __init__(self, config: dict, electrons_per_radiance: float = 1.0,
                 read_noise_e: Optional[float] = None):
        vst = config["anscombe"]
        self.enabled = bool(vst["apply"])
        self.eps = float(vst["epsilon"])
        self.use_generalized = bool(vst.get("use_generalized", True))
        self.electrons_per_radiance = float(electrons_per_radiance)
        self.radiance_per_electron = 1.0 / max(self.electrons_per_radiance, 1e-30)

        rn = float(config["calibration"]["read_noise_e"] if read_noise_e is None
                   else read_noise_e)
        self.read_noise_var = rn ** 2 if self.use_generalized else 0.0

    def forward(self, radiance: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return radiance
        electrons = np.clip(radiance, 0.0, None) * self.electrons_per_radiance
        return 2.0 * np.sqrt(electrons + self.eps + self.read_noise_var)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return z
        z = np.clip(z, 1e-3, None)
        electrons = (
            (z / 2.0) ** 2
            - 1.0 / 8.0
            + 0.25 * np.sqrt(1.5) * z ** -1
            - (11.0 / 8.0) * z ** -2
            + (5.0 / 8.0) * np.sqrt(1.5) * z ** -3
        )
        electrons = electrons - self.read_noise_var
        return np.clip(electrons, 0.0, None) * self.radiance_per_electron


# =============================================================================
# STAGE 1 -- assembled front-end
# =============================================================================
@dataclass
class FrontendResult:
    stabilized: np.ndarray           # VST domain -- input to stage 2
    radiance_linear: np.ndarray      # calibrated radiance, pre-VST (metric reference)
    cosmic_ray_mask: np.ndarray
    saturation_mask: np.ndarray
    nodata_mask: Optional[np.ndarray]
    scene: IngestedScene = field(repr=False)
    vst: AnscombeVST = field(repr=False)
    units: str = "W m^-2 sr^-1 um^-1"


class PhysicsFrontend:
    """Stage 1: ingestion -> calibration -> cosmic-ray scrub -> Anscombe VST."""

    def __init__(self, config: dict):
        self.config = config
        self.ingestor = PlanetaryIngestor(config)
        self.calibrator = RadiometricCalibrator(config)
        self.scrubber = CosmicRayScrubber(config)
        self.vst = AnscombeVST(
            config,
            electrons_per_radiance=self.calibrator.electrons_per_radiance,
        )

    def ingest(self, path: str) -> IngestedScene:
        """Stage 1.1 in isolation (used by the pipeline to grab metadata early)."""
        return self.ingestor.read(path)

    def process(self, source: str | IngestedScene) -> FrontendResult:
        scene = self.ingest(source) if isinstance(source, str) else source
        raw_dn = scene.dn

        # 1.2 radiometric calibration --------------------------------------
        sat_mask = self.calibrator.saturation_mask(raw_dn)
        electrons = self.calibrator.to_electrons(raw_dn)

        # 1.3 cosmic ray / SEU scrub (electron domain: Poisson model valid) --
        cleaned_e, cr_mask = self.scrubber.scrub(electrons)
        radiance = cleaned_e * self.calibrator.radiance_per_electron

        # 1.4 Anscombe VST ---------------------------------------------------
        stabilized = self.vst.forward(radiance)

        return FrontendResult(
            stabilized=stabilized.astype(np.float32),
            radiance_linear=radiance.astype(np.float32),
            cosmic_ray_mask=cr_mask,
            saturation_mask=sat_mask,
            nodata_mask=scene.nodata_mask,
            scene=scene,
            vst=self.vst,
            units=self.calibrator.units,
        )


# =============================================================================
# STAGE 4.1 -- Bilateral-guided log tone mapping
# =============================================================================
def _bilateral_filter_piecewise(image: np.ndarray, sigma_spatial: float,
                                 sigma_range: float, max_segments: int = 24
                                 ) -> np.ndarray:
    """
    Durand & Dorsey (2002) piecewise-linear bilateral filter approximation.

    The exact bilateral filter is O(N * r^2); this approximation quantizes the
    intensity range into segments, runs a plain Gaussian blur per segment, and
    linearly interpolates between them -- O(S * N log N) with S <= 24, using
    nothing but `scipy.ndimage.gaussian_filter`. Accuracy is well within the
    noise floor for tone-mapping purposes and it avoids an OpenCV dependency.
    """
    lo, hi = float(image.min()), float(image.max())
    span = hi - lo
    if span < 1e-12:
        return image.copy()

    n_seg = int(np.clip(np.ceil(span / max(sigma_range, 1e-6)) + 1, 2, max_segments))
    segments = np.linspace(lo, hi, n_seg)

    stack = np.empty((n_seg,) + image.shape, dtype=np.float64)
    for j, s in enumerate(segments):
        g = np.exp(-((image - s) ** 2) / (2.0 * sigma_range ** 2))
        k = ndimage.gaussian_filter(g, sigma_spatial, mode="nearest")
        h = ndimage.gaussian_filter(g * image, sigma_spatial, mode="nearest")
        stack[j] = h / np.maximum(k, 1e-12)

    # linear interpolation across the segment axis at each pixel's own value
    pos = np.clip((image - lo) / (span / (n_seg - 1)), 0.0, n_seg - 1 - 1e-6)
    j0 = np.floor(pos).astype(np.intp)
    frac = pos - j0
    j1 = np.minimum(j0 + 1, n_seg - 1)
    rows, cols = np.indices(image.shape)
    return stack[j0, rows, cols] * (1.0 - frac) + stack[j1, rows, cols] * frac


class BilateralLogToneMapper:
    """
    Stage 4.1 -- compresses scene dynamic range so deep-shadow detail becomes
    visible without clipping directly-lit highlights.

        L_log  = log10(L + eps)
        base   = bilateral(L_log)          <- large-scale illumination
        detail = L_log - base              <- surface texture, preserved
        out    = base * base_compression + detail * detail_gain

    Only the *base* layer is compressed. Because the base layer is extracted
    with an edge-preserving bilateral filter rather than a Gaussian, strong
    illumination boundaries (shadow edges, crater rims) do not bleed, which is
    what would otherwise produce the halo artifacts that read as invented
    structure.

    A soft knee above `highlight_knee` rolls off the top of the range
    asymptotically, so solar highlights compress rather than clip.
    """

    def __init__(self, config: dict):
        tm = config["tone_mapping"]
        self.enabled = bool(tm["enabled"])
        self.sigma_spatial = float(tm["sigma_spatial"])
        self.sigma_range = float(tm["sigma_range"])
        self.base_compression = float(tm["base_compression"])
        self.detail_gain = float(tm["detail_gain"])
        self.highlight_knee = float(tm["highlight_knee"])
        self.eps = float(tm["epsilon"])

    @staticmethod
    def _soft_knee(x: np.ndarray, knee: float) -> np.ndarray:
        """Linear below `knee`, asymptotic to 1.0 above it -- never clips."""
        if knee >= 1.0:
            return np.clip(x, 0.0, 1.0)
        out = x.copy()
        over = x > knee
        if over.any():
            head = 1.0 - knee
            out[over] = knee + head * np.tanh((x[over] - knee) / max(head, 1e-6))
        return np.clip(out, 0.0, 1.0)

    def apply(self, radiance: np.ndarray) -> np.ndarray:
        """Input: linear radiance. Output: tone-mapped radiance, same units."""
        if not self.enabled:
            return radiance

        positive = np.clip(radiance, 0.0, None)
        log_l = np.log10(positive + self.eps)

        base = _bilateral_filter_piecewise(
            log_l, self.sigma_spatial, self.sigma_range
        )
        detail = log_l - base

        # Compress the base layer about its own mean so the scene's mean
        # radiance level is preserved (flux conservation, stage 5.1).
        base_mean = float(base.mean())
        compressed = base_mean + (base - base_mean) * self.base_compression
        out_log = compressed + detail * self.detail_gain

        mapped = np.power(10.0, out_log) - self.eps
        mapped = np.clip(mapped, 0.0, None)

        # Soft-knee the top end in normalized space, then restore scale.
        peak = float(mapped.max())
        if peak > 0:
            mapped = self._soft_knee(mapped / peak, self.highlight_knee) * peak

        return mapped.astype(np.float32)


# =============================================================================
# STAGE 4.2 -- Lommel-Seeliger photometric correction
# =============================================================================
class LommelSeeligerCorrection:
    """
    Stage 4.2 -- removes the illumination-geometry signature so brightness
    reflects surface albedo rather than where the sun happened to be.

    The Lommel-Seeliger (single-scattering) disk function for a dark,
    particulate regolith:

        D(i, e) = cos(i) / (cos(i) + cos(e))

    Observed radiance L_obs = A * D(i, e), so the geometry-normalized
    quantity is L_corr = L_obs / D(i, e). This is the standard photometric
    model for lunar and asteroid regolith at moderate-to-high phase angles,
    where a Lambertian (cos i) correction over-brightens the limb.

    Geometry comes from, in priority order:
      1. per-pixel backplanes (`incidence_map_path` / `emission_map_path`),
      2. product metadata keys (`incidence_key` / `emission_key`),
      3. the scene-constant config fallbacks.

    `min_cos` floors cos(i) so terminator pixels -- where D -> 0 and the
    correction would explode into pure amplified noise -- stay bounded.
    """

    def __init__(self, config: dict):
        ph = config["photometric"]
        self.enabled = bool(ph["lommel_seeliger"])
        self.incidence_deg = float(ph["incidence_deg"])
        self.emission_deg = float(ph["emission_deg"])
        self.incidence_key = str(ph.get("incidence_key", "INCIDENCE_ANGLE"))
        self.emission_key = str(ph.get("emission_key", "EMISSION_ANGLE"))
        self.incidence_map_path = ph.get("incidence_map_path")
        self.emission_map_path = ph.get("emission_map_path")
        self.min_cos = float(ph["min_cos"])
        self.normalize_to_unit_mean = bool(ph["normalize_to_unit_mean"])

    @staticmethod
    def _load_map(path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        import tifffile
        return np.asarray(tifffile.imread(path), dtype=np.float64)

    def _angle_field(self, which: str, metadata: dict, shape: tuple
                     ) -> tuple[np.ndarray, str]:
        """Returns (angle field in radians, provenance of that field)."""
        path = self.incidence_map_path if which == "incidence" else self.emission_map_path
        backplane = self._load_map(path)
        if backplane is not None and backplane.shape == shape:
            return np.deg2rad(backplane), "backplane"

        key = self.incidence_key if which == "incidence" else self.emission_key
        default = self.incidence_deg if which == "incidence" else self.emission_deg
        value, source = default, "config_default"
        for meta_key, meta_value in (metadata or {}).items():
            if meta_key.upper() == key.upper():
                try:
                    value = float(str(meta_value).split()[0])
                    source = "product_metadata"
                except (ValueError, IndexError):
                    value, source = default, "config_default"
                break
        return np.full(shape, np.deg2rad(value), dtype=np.float64), source

    def disk_function(self, metadata: dict, shape: tuple
                      ) -> tuple[np.ndarray, dict]:
        inc, inc_src = self._angle_field("incidence", metadata, shape)
        emi, emi_src = self._angle_field("emission", metadata, shape)
        cos_i = np.clip(np.cos(inc), self.min_cos, 1.0)
        cos_e = np.clip(np.cos(emi), self.min_cos, 1.0)
        geometry = {
            # The angles actually applied, not the config fallbacks -- this is
            # provenance, so it has to say what happened.
            "incidence_deg_applied": float(np.rad2deg(inc).mean()),
            "emission_deg_applied": float(np.rad2deg(emi).mean()),
            "incidence_source": inc_src,
            "emission_source": emi_src,
        }
        return cos_i / (cos_i + cos_e), geometry

    def apply(self, radiance: np.ndarray, metadata: Optional[dict] = None
              ) -> tuple[np.ndarray, dict]:
        """Returns (geometry-normalized radiance, applied-geometry record)."""
        if not self.enabled:
            return radiance, {"lommel_seeliger_applied": False}

        d, geometry = self.disk_function(metadata or {}, radiance.shape)
        corrected = radiance / np.maximum(d, 1e-6)

        if self.normalize_to_unit_mean:
            src_mean = float(np.mean(radiance))
            out_mean = float(np.mean(corrected))
            if out_mean > 1e-30:
                corrected = corrected * (src_mean / out_mean)

        record = {
            "lommel_seeliger_applied": True,
            "disk_function_mean": float(d.mean()),
            "disk_function_min": float(d.min()),
            **geometry,
        }
        return corrected.astype(np.float32), record


# =============================================================================
# STAGE 4 -- assembled back-end
# =============================================================================
class PhotometricBackend:
    """Stage 4: bilateral-guided log tone mapping -> Lommel-Seeliger."""

    def __init__(self, config: dict):
        self.tone_mapper = BilateralLogToneMapper(config)
        self.photometry = LommelSeeligerCorrection(config)

    def process(self, radiance: np.ndarray, metadata: Optional[dict] = None
                ) -> tuple[np.ndarray, dict]:
        tone_mapped = self.tone_mapper.apply(radiance)
        corrected, record = self.photometry.apply(tone_mapped, metadata)
        record["tone_mapping_applied"] = self.tone_mapper.enabled
        return corrected, record
