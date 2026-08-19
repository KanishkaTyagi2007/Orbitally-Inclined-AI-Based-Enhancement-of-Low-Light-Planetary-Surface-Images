"""
exporter.py
------------
Writes AuraNet's outputs back to disk as GeoTIFFs that preserve the source
raster's geospatial metadata (CRS, affine transform, nodata value) — a hard
requirement for any product downstream scientists will load into GIS tools
(QGIS, ArcGIS) or mosaic against other planetary basemaps. A visually
enhanced image that has lost its georeferencing is not scientifically usable
regardless of how good it looks.

Two artifacts are written per input scene:
    1. <name>_enhanced.tif — the enhanced image, same CRS/transform as input,
       optionally with the trust map appended as an extra band.
    2. <name>_trust.tif    — the trust map alone, single band, float32,
       same CRS/transform, so it can be overlaid/thresholded independently
       in GIS software without touching the enhanced image band math.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.profiles import Profile
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


def read_source_profile(input_path: str) -> dict:
    """Reads CRS/transform/nodata/etc. from the source raster so outputs can
    inherit them exactly. Raises informatively if rasterio is unavailable."""
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to read/preserve geospatial metadata. "
            "Install it via `pip install rasterio` (see requirements.txt)."
        )
    with rasterio.open(input_path) as src:
        return src.profile.copy()


def _to_export_dtype(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint16":
        scaled = np.clip(array, 0.0, 1.0) * 65535.0
        return scaled.astype(np.uint16)
    if dtype == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported export dtype: {dtype}")


def export_geotiff(image: np.ndarray, output_path: str, source_profile: Optional[dict] = None,
                    trust_map: Optional[np.ndarray] = None, dtype: str = "uint16",
                    compress: str = "deflate") -> str:
    """
    Writes `image` (H, W), normalized to [0, 1], to `output_path` as a
    GeoTIFF. If `source_profile` is supplied (from read_source_profile),
    CRS/transform/nodata are copied over exactly. If `trust_map` is given,
    it is appended as band 2 (float32-in-a-uint16-scaled-band is avoided by
    forcing multi-band output to float32 whenever a trust band is attached).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_bands = 2 if trust_map is not None else 1
    export_dtype = "float32" if trust_map is not None else dtype

    band1 = _to_export_dtype(image, export_dtype)

    if not _HAS_RASTERIO:
        # Fallback: plain (non-georeferenced) TIFF via tifffile. Geospatial
        # metadata is lost in this path — surfaced to the caller via a
        # warning rather than failing silently.
        import warnings
        import tifffile
        warnings.warn(
            "rasterio not installed — writing a plain TIFF with NO "
            "geospatial metadata. Install rasterio to preserve CRS/transform."
        )
        stack = band1[None] if trust_map is None else np.stack(
            [band1, trust_map.astype(export_dtype)], axis=0)
        tifffile.imwrite(output_path, stack)
        return output_path

    profile = dict(source_profile) if source_profile else {}
    profile.update({
        "driver": "GTiff",
        "count": n_bands,
        "dtype": export_dtype,
        "compress": compress,
        "height": image.shape[0],
        "width": image.shape[1],
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(band1, 1)
        dst.set_band_description(1, "enhanced_radiance")
        if trust_map is not None:
            dst.write(trust_map.astype(export_dtype), 2)
            dst.set_band_description(2, "trust_map")

    return output_path


def export_trust_sidecar(trust_map: np.ndarray, output_path: str,
                          source_profile: Optional[dict] = None) -> str:
    """Writes the trust map as its own single-band float32 GeoTIFF."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trust32 = trust_map.astype(np.float32)

    if not _HAS_RASTERIO:
        import warnings
        import tifffile
        warnings.warn(
            "rasterio not installed — writing trust map as a plain TIFF "
            "with NO geospatial metadata."
        )
        tifffile.imwrite(output_path, trust32)
        return output_path

    profile = dict(source_profile) if source_profile else {}
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "compress": "deflate",
        "height": trust_map.shape[0],
        "width": trust_map.shape[1],
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(trust32, 1)
        dst.set_band_description(1, "trust_map")

    return output_path
