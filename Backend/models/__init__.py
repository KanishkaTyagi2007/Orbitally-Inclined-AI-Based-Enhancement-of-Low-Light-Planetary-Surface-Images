"""
AuraNet model subpackage.

Modules
-------
physics_frontend   : Radiometric calibration, cosmic-ray scrubbing, Anscombe VST
wavelet_dequant    : SWT / ISWT band decomposition + implicit de-quantization
zero_dce           : Zero-reference deep curve estimator (illumination, LL band)
nafnet_denoiser    : Activation-free detail restorer (LH/HL/HH bands)
uncertainty_head   : Bayesian heteroscedastic (mu, log-var) confidence head
"""

from .physics_frontend import PhysicsFrontend
from .wavelet_dequant import WaveletDequantizer
from .zero_dce import ZeroDCE
from .nafnet_denoiser import NAFNetDenoiser
from .uncertainty_head import UncertaintyHead

__all__ = [
    "PhysicsFrontend",
    "WaveletDequantizer",
    "ZeroDCE",
    "NAFNetDenoiser",
    "UncertaintyHead",
]
