"""
Two-Stage Discover-and-Deploy Training Orchestrator for PRISM (formerly AS-ViT).
Backward compatibility wrapper forwarding to train_prism.py.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeyagangwar@proton.me
"""
from train_prism import (
    TwoStagePRISMTrainer,
    TwoStageASViTTrainer,
    set_seed,
    main,
)

if __name__ == "__main__":
    main()
