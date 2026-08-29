"""
training
=========
Self-supervised training for AURA-NET's four learned submodules, on real
Chandrayaan-2 products and without any ground truth.

    tiles.py       stratified tile sampling from PDS4 strips, stage-1 physics
                   applied once, cached in the exact domain each module sees at
                   inference; measured/noisier pair generation
    objectives.py  the training-time anti-hallucination terms -- structure
                   addition penalty, identity anchor, banding suppression

The trainer that drives both is `drishti/train.py`.
"""

from .objectives import (
    BandingSuppressionLoss,
    CharbonnierLoss,
    IdentityAnchor,
    StructureAdditionPenalty,
    structure_addition_fraction,
)
from .tiles import TileCache, TileRef, add_sensor_noise, build_cache, sample_windows

__all__ = [
    "BandingSuppressionLoss",
    "CharbonnierLoss",
    "IdentityAnchor",
    "StructureAdditionPenalty",
    "structure_addition_fraction",
    "TileCache",
    "TileRef",
    "add_sensor_noise",
    "build_cache",
    "sample_windows",
]
