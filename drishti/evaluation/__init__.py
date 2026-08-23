"""
AURA-NET evaluation subpackage -- STAGE 6.

    metrics.py   6.1 PSNR, SSIM, NIQE, BRISQUE, information entropy gain
                 6.2 frozen topological crater detector (mIoU / true detection)
    exporter.py  6.3 lossless 32-bit float multi-band GeoTIFF export with
                     affine transform, planetary CRS and radiometric tags,
                     plus human-viewable preview renderings of the same products
"""

from .metrics import (
    TopologicalCraterDetector,
    compute_brisque,
    compute_entropy_gain,
    compute_niqe,
    compute_psnr,
    compute_ssim,
    crater_detection_utility,
    evaluate_all,
    histogram_entropy,
    niqe_lite,
)
from .exporter import (
    apply_colormap,
    build_radiometric_tags,
    export_geotiff,
    export_metrics_report,
    export_preview_images,
    export_trust_sidecar,
    read_source_profile,
    stretch_for_display,
)

__all__ = [
    "evaluate_all", "compute_psnr", "compute_ssim", "compute_niqe",
    "compute_brisque", "compute_entropy_gain", "histogram_entropy", "niqe_lite",
    "TopologicalCraterDetector", "crater_detection_utility",
    "export_geotiff", "export_trust_sidecar", "export_metrics_report",
    "export_preview_images", "stretch_for_display", "apply_colormap",
    "read_source_profile", "build_radiometric_tags",
]
