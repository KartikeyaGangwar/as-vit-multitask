"""
================================================================================
Two-Stage Discover-and-Deploy Training Orchestrator for AS-ViT.
Stage 1: Autonomous Latent Feature AMR Discovery.
Stage 2: Clean Production Retraining with Fresh Optimizer Momentum.

Author: Kartikey Singh (Department of Mathematics, University of Delhi)
Contact: kartikeysingh525@protonmail.com
================================================================================
"""
import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.as_vit import ASViT
from engine.gram_clash_profiler import VectorizedGramClashProfiler
from engine.subspace_amr_manager import LatentSubspaceAMRManager
from engine.loss_functions import MultiTaskLossModule
from engine.optimizers.pcgrad import PCGrad
from engine.optimizers.cagrad import CAGrad
from data.nyuv2_dataset import NYUv2Dataset
from eval_multitask import evaluate_model, MultiTaskEvaluator


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TwoStageASViTTrainer:
    """
    Two-Stage Discover-and-Deploy Trainer for AS-ViT:
      - STAGE 1: Online feature-space AMR discovery with vectorized Gram clash profiling.
      - STAGE 2: Clean production training on discovered topology with fresh optimizer state.
    """
    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        img_size: int = 224,
        batch_size: int = 8,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        set_seed(seed)

        self.loss_module = MultiTaskLossModule().to(self.device)
        self.profiler = VectorizedGramClashProfiler(conflict_threshold=0.15, ema_decay=0.85)
        self.amr_manager = LatentSubspaceAMRManager(
            max_subspaces_per_block=8,
            min_centroid_distance_factor=0.35,
            conflict_threshold=0.15,
        )

    def get_dataloaders(self, synthetic: bool = True, num_train: int = 64, num_val: int = 32):
        train_dataset = NYUv2Dataset(
            split="train",
            synthetic=synthetic,
            synthetic_samples=num_train,
            img_size=(self.img_size, self.img_size),
        )
        val_dataset = NYUv2Dataset(
            split="val",
            synthetic=synthetic,
            synthetic_samples=num_val,
            img_size=(self.img_size, self.img_size),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def run_stage1_discovery(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        disc_epochs: int = 10,
        profile_freq: int = 2,
    ) -> Dict:
        print("\n" + "=" * 75)
        print("  STAGE 1: AUTONOMOUS FEATURE-SPACE AMR TOPOLOGY DISCOVERY")
        print(f"  Device: {self.device} | Depth: {self.depth} | EmbedDim: {self.embed_dim}")
        print("=" * 75)

        # Initialize Probe Model with N=1 subspace per block
        probe_model = ASViT(
            img_size=self.img_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            initial_subspaces_per_block=1,
        ).to(self.device)

        optimizer = torch.optim.AdamW(probe_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, disc_epochs + 1):
            probe_model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}

                optimizer.zero_grad()
                preds, inter_tokens, _ = probe_model(images, return_intermediates=True)
                total_loss, ind_losses = self.loss_module(preds, targets)
                task_loss_list = [ind_losses[k] for k in ["segmentation", "depth", "surface_normals"] if k in ind_losses]

                # Online AMR Profiling & Cleavage check
                if (epoch % profile_freq == 0) and (batch_idx == 0):
                    for b_idx in range(len(probe_model.blocks)):
                        block = probe_model.blocks[b_idx]
                        z_in = inter_tokens[b_idx]
                        cleaved = self.amr_manager.evaluate_and_cleave_block(
                            block_idx=b_idx,
                            block=block,
                            latent_tokens=z_in,
                            task_losses=task_loss_list,
                            profiler=self.profiler,
                            epoch=epoch,
                        )
                        if cleaved:
                            # Refresh optimizer parameter groups
                            optimizer = torch.optim.AdamW(
                                probe_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                            )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += total_loss.item()

            if epoch % max(1, disc_epochs // 5) == 0 or epoch == disc_epochs:
                subspace_counts = [b.num_subspaces for b in probe_model.blocks]
                print(f"  [Stage 1 Epoch {epoch:3d}/{disc_epochs:3d}] Loss: {epoch_loss/len(train_loader):.4f} | Subspaces: {subspace_counts}")

        topology = probe_model.get_topology()
        print(f"\n  [+] Stage 1 Discovery Complete! Final Subspace Counts: {[b.num_subspaces for b in probe_model.blocks]}")
        return {
            "topology": topology,
            "subspace_counts": [b.num_subspaces for b in probe_model.blocks],
            "probe_model": probe_model,
        }

    def run_stage2_production(
        self,
        discovered_info: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        prod_epochs: int = 15,
    ) -> Tuple[ASViT, Dict[str, float]]:
        print("\n" + "=" * 75)
        print("  STAGE 2: CLEAN PRODUCTION RETRAINING (ZERO MOMENTUM CORRUPTION)")
        print(f"  Subspace Allocation: {discovered_info['subspace_counts']}")
        print("=" * 75)

        # Initialize Clean Production Model with Discovered Subspaces
        prod_model = ASViT(
            img_size=self.img_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            initial_subspaces_per_block=discovered_info["subspace_counts"],
        ).to(self.device)

        # Load Discovered Centroids and Bandwidths
        topology = discovered_info["topology"]
        for i, block in enumerate(prod_model.blocks):
            key = f"block_{i}"
            if key in topology:
                block.register_buffer("centroids", topology[key]["centroids"].to(self.device))
                block.bandwidth = topology[key]["bandwidth"]

        # Clean Optimizer & Cosine Annealing Schedule
        optimizer = torch.optim.AdamW(prod_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prod_epochs, eta_min=1e-6)

        for epoch in range(1, prod_epochs + 1):
            prod_model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}

                optimizer.zero_grad()
                preds = prod_model(images)
                total_loss, _ = self.loss_module(preds, targets)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(prod_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += total_loss.item()

            scheduler.step()

            if epoch % max(1, prod_epochs // 5) == 0 or epoch == prod_epochs:
                val_metrics = evaluate_model(prod_model, val_loader, self.device)
                print(
                    f"  [Stage 2 Epoch {epoch:3d}/{prod_epochs:3d}] Loss: {epoch_loss/len(train_loader):.4f} | "
                    f"mIoU: {val_metrics['seg_miou']:.2f}% | Depth AbsRel: {val_metrics['depth_abs_rel']:.4f} | "
                    f"Normals Mean Angle: {val_metrics['normals_mean_angle']:.2f} deg | Multi-Task Gain Delta M: {val_metrics['delta_m']:+.2f}%"
                )

        final_metrics = evaluate_model(prod_model, val_loader, self.device)
        return prod_model, final_metrics


def main():
    parser = argparse.ArgumentParser(description="AS-ViT Training & Evaluation Pipeline")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--disc_epochs", type=int, default=6)
    parser.add_argument("--prod_epochs", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    trainer = TwoStageASViTTrainer(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    train_loader, val_loader = trainer.get_dataloaders(synthetic=args.synthetic)

    # 1. Stage 1: AMR Discovery
    disc_info = trainer.run_stage1_discovery(train_loader, val_loader, disc_epochs=args.disc_epochs)

    # 2. Stage 2: Clean Production Retraining
    final_model, metrics = trainer.run_stage2_production(disc_info, train_loader, val_loader, prod_epochs=args.prod_epochs)
    print("\n" + "=" * 75)
    print("  FINAL AS-ViT MULTI-TASK EVALUATION REPORT:")
    for k, v in metrics.items():
        print(f"    - {k:22s}: {v:8.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
