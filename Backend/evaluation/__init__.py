"""
AuraNet evaluation subpackage.

metrics.py  : PSNR, SSIM, NIQE, BRISQUE, Shannon entropy gain
exporter.py : Metadata-preserving GeoTIFF export (enhanced image + trust map)
"""

from .metrics import evaluate_all
from .exporter import export_geotiff, export_trust_sidecar

__all__ = ["evaluate_all", "export_geotiff", "export_trust_sidecar"]
