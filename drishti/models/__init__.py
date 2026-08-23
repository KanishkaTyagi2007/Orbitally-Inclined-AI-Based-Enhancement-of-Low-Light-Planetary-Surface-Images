"""
AURA-NET model subpackage.

Modules map one-to-one onto the pipeline stages:

    physics_frontend   STAGE 1  ingestion, DN -> radiance calibration,
                                Laplacian cosmic-ray scrubbing, Anscombe VST
                       STAGE 4  bilateral-guided log tone mapping,
                                Lommel-Seeliger photometric correction
    wavelet_dequant    STAGE 2  implicit sub-bin de-quantization, SWT
                       STAGE 3C inverse SWT recombination
    zero_dce           STAGE 3A Zero-DCE++ illumination curve estimation (LL)
    nafnet_denoiser    STAGE 3B PSF deconvolution + activation-free denoising
                                (LH/HL/HH)
    uncertainty_head   STAGE 5  flux conservation, Sobel gradient guardrail,
                                heteroscedastic (mu, log-var) trust mapping
"""

from .physics_frontend import (
    AnscombeVST,
    BilateralLogToneMapper,
    CosmicRayScrubber,
    FrontendResult,
    IngestedScene,
    LommelSeeligerCorrection,
    PhotometricBackend,
    PhysicsFrontend,
    PlanetaryIngestor,
    RadiometricCalibrator,
)
from .wavelet_dequant import (
    ImplicitDequantizer,
    WaveletBands,
    WaveletDequantizer,
    iswt_reconstruct,
    swt_decompose,
)
from .zero_dce import (
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
    SpatialConsistencyLoss,
    ZeroDCE,
    ZeroDCELoss,
    ZeroDCEPlusPlus,
)
from .nafnet_denoiser import (
    DetailRestorer,
    DifferentiablePSFDeconvolution,
    NAFBlock,
    NAFNetDenoiser,
    SimpleGate,
    build_psf,
)
from .uncertainty_head import (
    HeteroscedasticNLLLoss,
    PhotometricFluxConservationLoss,
    PhysicsVerifier,
    SobelGradientConsistencyLoss,
    UncertaintyHead,
    VerificationResult,
    check_flux_conservation,
    check_gradient_consistency,
    compute_trust_map,
    sobel_magnitude,
)

__all__ = [
    # stage 1 + 4
    "PhysicsFrontend", "PlanetaryIngestor", "RadiometricCalibrator",
    "CosmicRayScrubber", "AnscombeVST", "FrontendResult", "IngestedScene",
    "PhotometricBackend", "BilateralLogToneMapper", "LommelSeeligerCorrection",
    # stage 2 + 3C
    "WaveletDequantizer", "ImplicitDequantizer", "WaveletBands",
    "swt_decompose", "iswt_reconstruct",
    # stage 3A
    "ZeroDCE", "ZeroDCEPlusPlus", "ZeroDCELoss", "SpatialConsistencyLoss",
    "ExposureControlLoss", "IlluminationSmoothnessLoss",
    # stage 3B
    "DetailRestorer", "NAFNetDenoiser", "DifferentiablePSFDeconvolution",
    "NAFBlock", "SimpleGate", "build_psf",
    # stage 5
    "UncertaintyHead", "HeteroscedasticNLLLoss", "PhysicsVerifier",
    "VerificationResult", "PhotometricFluxConservationLoss",
    "SobelGradientConsistencyLoss", "check_flux_conservation",
    "check_gradient_consistency", "compute_trust_map", "sobel_magnitude",
]
