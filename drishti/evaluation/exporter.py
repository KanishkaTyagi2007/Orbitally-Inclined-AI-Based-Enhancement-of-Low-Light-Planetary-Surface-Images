"""
exporter.py
============
STAGE 6.3 -- LOSSLESS GEOTIFF EXPORT

Writes the final deliverable:

    Band 1 : Calibrated enhanced radiance (mu)  [W m^-2 sr^-1 um^-1]
    Band 2 : Per-pixel Bayesian trust map (T in [0, 1])
    Tags   : affine transform, planetary CRS, radiometric metadata

Two properties are non-negotiable here.

*Lossless.* The container is 32-bit IEEE float with DEFLATE compression and
the floating-point predictor (predictor=3), which is bit-exact. Band 1 holds
radiance in physical units, not a 0-255 or 0-65535 display stretch -- a
quantized product cannot be used for photometry, and re-deriving radiance from
a stretched image is impossible once the stretch parameters are gone.

*Georeferenced.* CRS and affine transform are copied from the source product.
A visually improved image that has lost its planetary georeferencing cannot be
mosaicked, cannot be compared against other instruments, and is not a
scientific product regardless of how it looks.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


BAND_DESCRIPTIONS = (
    "enhanced_radiance_mu",
    "bayesian_trust_map",
)


# =============================================================================
# Source metadata
# =============================================================================
def read_source_profile(input_path: str) -> dict:
    """Reads CRS/transform/nodata from the source raster so outputs inherit
    them exactly. Raises informatively when rasterio is unavailable."""
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to read/preserve geospatial metadata. "
            "Install it via `pip install rasterio` (see requirements.txt)."
        )
    with rasterio.open(input_path) as src:
        return src.profile.copy()


def build_radiometric_tags(config: dict, extra: Optional[dict] = None) -> dict:
    """
    Assembles the radiometric provenance written into the GeoTIFF header.

    Recording the calibration constants alongside the pixels is what makes the
    product reversible: a later user can recover raw DN from radiance, or
    re-derive the product under updated calibration, without needing this
    repository or its config file.
    """
    cal = config["calibration"]
    tags: dict[str, Any] = {
        "AURA_NET_PIPELINE": config["project"]["name"],
        "RADIANCE_UNITS": cal.get("radiance_units", "W m^-2 sr^-1 um^-1"),
        "BAND_1": "enhanced spectral radiance (posterior mean mu)",
        "BAND_2": "per-pixel Bayesian trust map, 0 (untrusted) to 1 (trusted)",
        "CAL_GAIN_E_PER_DN": cal["gain_e_per_dn"],
        "CAL_BIAS_OFFSET_DN": cal["bias_offset_dn"],
        "CAL_BLACK_LEVEL_DN": cal["black_level_dn"],
        "CAL_READ_NOISE_E": cal["read_noise_e"],
        "CAL_SATURATION_DN": cal["saturation_dn"],
        "CAL_EXPOSURE_TIME_S": cal.get("exposure_time_s", 1.0),
        "CAL_RADIANCE_SCALE": cal.get("radiance_scale", 1.0),
        "CAL_BANDWIDTH_UM": cal.get("bandwidth_um", 1.0),
        "VST_APPLIED": config["anscombe"]["apply"],
        "TONE_MAPPING": config["tone_mapping"]["method"]
        if config["tone_mapping"]["enabled"] else "none",
        "PHOTOMETRIC_MODEL": "lommel_seeliger"
        if config["photometric"]["lommel_seeliger"] else "none",
    }
    if extra:
        tags.update(extra)
    return {k: str(v) for k, v in tags.items()}


# =============================================================================
# Main export
# =============================================================================
def export_geotiff(radiance_mu: np.ndarray, output_path: str,
                   trust_map: Optional[np.ndarray] = None,
                   source_profile: Optional[dict] = None,
                   config: Optional[dict] = None,
                   tags: Optional[dict] = None,
                   nodata: Optional[float] = None) -> str:
    """
    Writes the multi-band scientific deliverable.

    Args:
        radiance_mu: (H, W) enhanced radiance in physical units -- written
            verbatim as band 1, with no stretch or rescaling.
        trust_map: (H, W) trust values in [0, 1] -- written as band 2.
        source_profile: profile from `read_source_profile`; CRS, transform and
            nodata are inherited from it.
        config: pipeline config, used for compression settings and the
            radiometric tag block.
        tags: additional key/value pairs merged into the GeoTIFF header.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    band1 = np.ascontiguousarray(radiance_mu, dtype=np.float32)
    bands = [band1]
    if trust_map is not None:
        bands.append(np.ascontiguousarray(
            np.clip(trust_map, 0.0, 1.0), dtype=np.float32))

    exp_cfg = (config or {}).get("export", {})
    compress = str(exp_cfg.get("compress", "deflate"))
    predictor = int(exp_cfg.get("predictor", 3))

    header = build_radiometric_tags(config, tags) if config else (tags or {})

    if not _HAS_RASTERIO:
        import tifffile
        warnings.warn(
            "rasterio not installed -- writing a plain 32-bit float TIFF with "
            "NO geospatial metadata. Install rasterio to preserve CRS/transform."
        )
        tifffile.imwrite(
            output_path, np.stack(bands, axis=0),
            metadata={k: str(v) for k, v in header.items()},
        )
        return output_path

    profile = dict(source_profile) if source_profile else {}
    profile.update({
        "driver": "GTiff",
        "count": len(bands),
        "dtype": "float32",
        "height": band1.shape[0],
        "width": band1.shape[1],
        "compress": compress,
        "tiled": False,
    })
    if compress.lower() in ("deflate", "lzw", "zstd"):
        profile["predictor"] = predictor
    if nodata is not None:
        profile["nodata"] = float(nodata)
    else:
        profile.pop("nodata", None)
    # A source profile may carry settings that make no sense for float output.
    for key in ("photometric", "interleave", "blockxsize", "blockysize"):
        profile.pop(key, None)

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)
            if i <= len(BAND_DESCRIPTIONS):
                dst.set_band_description(i, BAND_DESCRIPTIONS[i - 1])
        if header:
            dst.update_tags(**header)

    return output_path


def export_trust_sidecar(trust_map: np.ndarray, output_path: str,
                         source_profile: Optional[dict] = None,
                         config: Optional[dict] = None) -> str:
    """
    Writes the trust map as its own single-band float32 GeoTIFF, so it can be
    thresholded and overlaid in GIS software without touching the radiance
    band's math.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trust32 = np.ascontiguousarray(np.clip(trust_map, 0.0, 1.0), dtype=np.float32)

    if not _HAS_RASTERIO:
        import tifffile
        warnings.warn(
            "rasterio not installed -- writing the trust map as a plain TIFF "
            "with NO geospatial metadata."
        )
        tifffile.imwrite(output_path, trust32)
        return output_path

    exp_cfg = (config or {}).get("export", {})
    compress = str(exp_cfg.get("compress", "deflate"))

    profile = dict(source_profile) if source_profile else {}
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "height": trust32.shape[0],
        "width": trust32.shape[1],
        "compress": compress,
        "tiled": False,
    })
    if compress.lower() in ("deflate", "lzw", "zstd"):
        profile["predictor"] = int(exp_cfg.get("predictor", 3))
    profile.pop("nodata", None)
    for key in ("photometric", "interleave", "blockxsize", "blockysize"):
        profile.pop(key, None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(trust32, 1)
        dst.set_band_description(1, BAND_DESCRIPTIONS[1])
        dst.update_tags(TRUST_MAP_RANGE="0 (untrusted) to 1 (trusted)")

    return output_path


# =============================================================================
# Display rendering -- human-viewable preview images
# =============================================================================
# The GeoTIFF above is the scientific product: 32-bit float radiance in physical
# units. No ordinary image viewer can display it meaningfully -- the values sit
# in a narrow band around 0.3 W m^-2 sr^-1 um^-1, so a viewer shows a flat grey
# rectangle. The functions below render viewable 8-bit images beside it.
#
# IMPORTANT: previews are *display renderings*, not measurements. Each is
# independently contrast-stretched so it looks its best on screen, which is the
# honest way to compare "the best you can see in the raw" against "the best you
# can see in the enhanced product" -- but it means preview brightness is NOT
# proportional to radiance, and preview pixels must never be used for
# photometry. The stretch limits actually applied are recorded in the metrics
# JSON so any preview can be traced back to the radiance it came from.

# Viridis control points: perceptually uniform and colour-vision-deficiency
# safe, which matters for a map whose whole job is to be read correctly.
_VIRIDIS = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
    (31, 158, 137), (53, 183, 121), (109, 205, 89), (253, 231, 37),
], dtype=np.float64)


def _finite_mask(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array)


def stretch_for_display(array: np.ndarray, method: str = "percentile",
                        percentile_low: float = 2.0, percentile_high: float = 98.0,
                        asinh_softening: float = 0.1
                        ) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Maps physical values onto [0, 1] for display.

    Methods:
        "percentile" -- clip to the given percentiles, then linear. The default,
            because a single cosmic-ray residual or hot pixel destroys a
            min-max stretch: one outlier at 50x the scene median compresses
            everything real into the bottom 2% of the range.
        "minmax" -- full range, for when outliers are known to be absent.
        "asinh" -- inverse hyperbolic sine, the standard astronomical stretch.
            Near-linear for faint signal and logarithmic for bright, so deep
            shadow detail and sunlit terrain stay visible in one frame.

    Returns (values in [0, 1], the (low, high) limits applied).
    """
    data = np.asarray(array, dtype=np.float64)
    finite = _finite_mask(data)
    if not finite.any():
        return np.zeros(data.shape), (0.0, 0.0)

    valid = data[finite]
    if method == "minmax":
        lo, hi = float(valid.min()), float(valid.max())
    else:
        lo = float(np.percentile(valid, percentile_low))
        hi = float(np.percentile(valid, percentile_high))
        if hi - lo < 1e-30:
            lo, hi = float(valid.min()), float(valid.max())

    if hi - lo < 1e-30:
        return np.zeros(data.shape), (lo, hi)

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0

    if method == "asinh":
        soft = max(float(asinh_softening), 1e-6)
        scaled = np.arcsinh(scaled / soft) / np.arcsinh(1.0 / soft)

    return scaled, (lo, hi)


def apply_colormap(gray01: np.ndarray) -> np.ndarray:
    """Maps [0, 1] onto viridis RGB, returned as uint8 (H, W, 3)."""
    x = np.clip(np.asarray(gray01, dtype=np.float64), 0.0, 1.0)
    pos = x * (len(_VIRIDIS) - 1)
    i0 = np.floor(pos).astype(np.intp)
    i1 = np.minimum(i0 + 1, len(_VIRIDIS) - 1)
    frac = (pos - i0)[..., None]
    rgb = _VIRIDIS[i0] * (1.0 - frac) + _VIRIDIS[i1] * frac
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _to_uint(gray01: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    peak = 65535 if bit_depth == 16 else 255
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    return np.clip(np.rint(np.clip(gray01, 0.0, 1.0) * peak), 0, peak).astype(dtype)


def _load_font(size: int):
    """Best available font at `size`, degrading to Pillow's bitmap default."""
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _resize_if_needed(image, max_dimension: int):
    from PIL import Image

    if max_dimension <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_dimension:
        return image
    scale = max_dimension / float(max(w, h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.LANCZOS)


def compose_comparison(panels: list[np.ndarray], labels: list[str],
                       background: int = 18) -> "np.ndarray":
    """
    Tiles labelled RGB panels into one contact sheet.

    Orientation follows the source aspect: wide scenes stack vertically and
    tall ones side by side, so the sheet never ends up an unusable 10:1 strip.
    """
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("compose_comparison needs at least one panel")

    h, w = panels[0].shape[:2]
    bar = max(18, int(round(h * 0.06)))
    font_size = max(11, int(round(bar * 0.62)))
    font = _load_font(font_size)
    text_y = max(1, (bar - font_size) // 2)
    gap = max(4, int(round(min(h, w) * 0.01)))
    horizontal = h >= w

    tile_w, tile_h = w, h + bar
    n = len(panels)
    if horizontal:
        sheet_w = tile_w * n + gap * (n - 1)
        sheet_h = tile_h
    else:
        sheet_w = tile_w
        sheet_h = tile_h * n + gap * (n - 1)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (background,) * 3)
    draw = ImageDraw.Draw(sheet)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = i * (tile_w + gap) if horizontal else 0
        y = 0 if horizontal else i * (tile_h + gap)
        draw.text((x + gap, y + text_y), label, fill=(235, 235, 235), font=font)
        sheet.paste(Image.fromarray(panel), (x, y + bar))

    return np.asarray(sheet)


def export_preview_images(output_dir: str, stem: str,
                          enhanced: np.ndarray,
                          raw: Optional[np.ndarray] = None,
                          trust_map: Optional[np.ndarray] = None,
                          config: Optional[dict] = None,
                          trained: bool = True) -> dict:
    """
    Renders viewable images into a dedicated folder and returns their paths
    plus the stretch limits used.

    Written (subject to the config flags):
        <stem>_1_raw.png          calibrated input, display-stretched
        <stem>_2_enhanced.png     the enhanced product -- the headline image
        <stem>_3_trust.png        per-pixel trust, viridis (dark = untrusted)
        <stem>_4_comparison.png   all of the above side by side, labelled

    Args:
        output_dir: folder to write into; created if absent.
        stem: source scene name, used as the filename prefix.
        enhanced: enhanced radiance (H, W) in physical units.
        raw: calibrated raw radiance, for the before/after panels.
        trust_map: per-pixel trust in [0, 1].
        config: pipeline config; reads the `export.preview` block.
        trained: whether trained weights were loaded. When False the enhanced
            panel is labelled "physics-only", because an uncheckpointed run
            deconvolves without denoising -- it sharpens noise along with
            signal, and the output legitimately looks grainier than the input.
            An unlabelled panel would read as the pipeline underperforming
            rather than as a run with no weights loaded.
    """
    from PIL import Image

    prev_cfg = ((config or {}).get("export", {}) or {}).get("preview", {}) or {}
    if not prev_cfg.get("enabled", True):
        return {"preview_enabled": False}

    method = str(prev_cfg.get("stretch", "percentile"))
    p_low = float(prev_cfg.get("percentile_low", 2.0))
    p_high = float(prev_cfg.get("percentile_high", 98.0))
    softening = float(prev_cfg.get("asinh_softening", 0.1))
    bit_depth = int(prev_cfg.get("bit_depth", 8))
    max_dim = int(prev_cfg.get("max_dimension", 4096))
    fmt = str(prev_cfg.get("format", "png")).lower().lstrip(".")
    colorize_trust = bool(prev_cfg.get("trust_colormap", True))

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    def stretch(a):
        return stretch_for_display(a, method, p_low, p_high, softening)

    written: dict = {"preview_enabled": True, "preview_dir": str(folder)}
    limits: dict = {}
    panels: list[np.ndarray] = []
    labels: list[str] = []

    def save(gray01, name, rgb=None):
        array = rgb if rgb is not None else _to_uint(gray01, bit_depth)
        mode = None
        if rgb is None and bit_depth == 16:
            mode = "I;16"
        image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
        image = _resize_if_needed(image, max_dim)
        path = folder / f"{stem}_{name}.{fmt}"
        image.save(path)
        return str(path)

    # --- raw ("before") ---------------------------------------------------
    if raw is not None and prev_cfg.get("save_raw", True):
        raw01, raw_limits = stretch(raw)
        written["preview_raw"] = save(raw01, "1_raw")
        limits["raw"] = list(raw_limits)
        panels.append(np.repeat(_to_uint(raw01, 8)[..., None], 3, axis=2))
        labels.append("RAW (calibrated radiance)")

    # --- enhanced ("after") -- the image the user asked for ---------------
    enh01, enh_limits = stretch(enhanced)
    if prev_cfg.get("save_enhanced", True):
        written["preview_enhanced"] = save(enh01, "2_enhanced")
    limits["enhanced"] = list(enh_limits)
    panels.append(np.repeat(_to_uint(enh01, 8)[..., None], 3, axis=2))
    labels.append("ENHANCED (AURA-NET)" if trained
                  else "ENHANCED (physics-only -- no trained weights)")

    # --- trust map ---------------------------------------------------------
    if trust_map is not None and prev_cfg.get("save_trust", True):
        trust01 = np.clip(np.nan_to_num(trust_map, nan=0.0), 0.0, 1.0)
        trust_rgb = apply_colormap(trust01) if colorize_trust else None
        written["preview_trust"] = save(trust01, "3_trust", rgb=trust_rgb)
        panels.append(trust_rgb if trust_rgb is not None
                      else np.repeat(_to_uint(trust01, 8)[..., None], 3, axis=2))
        labels.append("TRUST MAP (dark = untrusted)")

    # --- labelled contact sheet -------------------------------------------
    if prev_cfg.get("save_comparison", True) and len(panels) > 1:
        sheet = compose_comparison(panels, labels)
        image = _resize_if_needed(Image.fromarray(sheet), max_dim)
        path = folder / f"{stem}_4_comparison.{fmt}"
        image.save(path)
        written["preview_comparison"] = str(path)

    written["preview_stretch"] = method
    written["preview_stretch_limits"] = limits
    written["preview_trained_weights"] = bool(trained)
    return written


# =============================================================================
# Metric report
# =============================================================================
def export_metrics_report(metrics: dict, output_path: str,
                          report_format: str = "json") -> str:
    """Writes the stage 6 metric harness output beside the raster products."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if report_format == "csv":
        import csv
        path = path.with_suffix(".csv")
        flat = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(flat.items())
        return str(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return str(path)
