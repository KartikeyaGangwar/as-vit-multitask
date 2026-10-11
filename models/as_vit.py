"""
================================================================================
Adaptive Subspace Vision Transformer (AS-ViT) -> PRISM Transition Module.
Provides backward compatibility aliases for PRISM architecture.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeyagangwar@proton.me
================================================================================
"""
from models.prism import (
    PRISM,
    PRISMMultiTask,
    PRISMBlock,
    ASViT,
    AdaptiveSubspaceBlock,
)

__all__ = [
    "PRISM",
    "PRISMMultiTask",
    "PRISMBlock",
    "ASViT",
    "AdaptiveSubspaceBlock",
]
