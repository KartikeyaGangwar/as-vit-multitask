"""
================================================================================
Autonomous Latent Subspace AMR Manager for AS-ViT.
Manages conflict centroid extraction, non-redundancy filtering, and subspace cleavage.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeyagangwar@proton.me
================================================================================
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.gram_clash_profiler import VectorizedGramClashProfiler


class LatentSubspaceAMRManager:
    r"""
    Autonomous Latent Subspace AMR Manager.
    
    Orchestrates:
      1. Stage 1 continuous inter-task gradient clash monitoring.
      2. Sensitivity-weighted latent token centroid extraction c_{N+1} \in R^D.
      3. Non-redundant spatial separation filtering (min ||c - c_j|| >= \delta * \sigma).
      4. Zero-disruption expert parameter cloning (\Theta_{N+1} = \Theta_k + \epsilon).
      5. Voronoi bandwidth recalibration.
    """
    def __init__(
        self,
        max_subspaces_per_block: int = 8,
        min_centroid_distance_factor: float = 0.35,
        conflict_threshold: float = 0.20,
        cooldown_epochs: int = 2,
        min_bandwidth: float = 0.20,
        initial_bandwidth: float = 0.50,
    ):
        self.max_subspaces = max_subspaces_per_block
        self.min_dist_factor = min_centroid_distance_factor
        self.conflict_threshold = conflict_threshold
        self.cooldown_epochs = cooldown_epochs
        self.min_bandwidth = min_bandwidth
        self.initial_bandwidth = initial_bandwidth
        self.last_cleavage_epoch: Dict[int, int] = {}
        self.cleavage_events: List[Dict] = []

    def extract_clashing_latent_centroid(
        self,
        latent_tokens: torch.Tensor,       # [B, M, D]
        task_losses: List[torch.Tensor],    # Length T
        clashing_task_indices: List[int],
    ) -> torch.Tensor:
        r"""
        Extracts conflict-weighted center of mass in token feature space R^D.
        
        c_{N+1} = \frac{\sum_m w_m z_m}{\sum_m w_m}, \quad w_m = \sum_{t \in T_clash} || \nabla_{z_m} L_t ||_2
        """
        B, M, D = latent_tokens.shape
        grad_weights = torch.zeros((B, M), device=latent_tokens.device, dtype=latent_tokens.dtype)
        
        for t_idx in clashing_task_indices:
            if t_idx < len(task_losses) and task_losses[t_idx].requires_grad:
                g_tokens = torch.autograd.grad(
                    task_losses[t_idx], latent_tokens, retain_graph=True, create_graph=False, allow_unused=True
                )[0]
                if g_tokens is not None:
                    token_norms = torch.norm(g_tokens, p=2, dim=-1) # [B, M]
                    grad_weights = grad_weights + token_norms

        if grad_weights.sum() < 1e-6:
            # Fallback 1: token variance weighted center
            token_vars = torch.var(latent_tokens, dim=-1) # [B, M]
            grad_weights = token_vars
            if grad_weights.sum() < 1e-6:
                # Fallback 2: direct spatial token mean across mini-batch
                return latent_tokens.mean(dim=(0, 1)).detach()

        grad_weights = grad_weights.clamp_min(1e-6)
        flat_tokens = latent_tokens.reshape(B * M, D)
        flat_weights = grad_weights.reshape(B * M, 1)
        
        # Conflict-Weighted Centroid
        centroid = (flat_tokens * flat_weights).sum(dim=0) / flat_weights.sum() # [D]
        return centroid

    def evaluate_and_cleave_block(
        self,
        block_idx: int,
        block: nn.Module,
        latent_tokens: torch.Tensor,
        task_losses: List[torch.Tensor],
        profiler: VectorizedGramClashProfiler,
        epoch: int,
        force: bool = False,
    ) -> bool:
        """
        Checks all active experts in block_idx and spawns a child expert if persistent conflict is detected.
        
        Returns:
            cleaved (bool): True if a new subspace was spawned.
        """
        if block.num_subspaces >= self.max_subspaces:
            return False

        if (not force) and (block_idx in self.last_cleavage_epoch):
            if (epoch - self.last_cleavage_epoch[block_idx]) < self.cooldown_epochs:
                return False

        for k in range(block.num_subspaces):
            expert = block.experts[k]
            should_cleave, metrics, task_clash_scores = profiler.profile_subspace_expert(
                block_idx, k, expert, task_losses
            )
            
            if should_cleave or force:
                clashing_tasks = torch.where(task_clash_scores > 0.0)[0].tolist()
                if not clashing_tasks:
                    clashing_tasks = list(range(len(task_losses)))
                    
                child_centroid = self.extract_clashing_latent_centroid(
                    latent_tokens, task_losses, clashing_tasks
                ).detach()
                
                # Spatial Non-Redundancy Filter
                existing_dists = torch.norm(block.centroids - child_centroid.unsqueeze(0), dim=-1)
                min_dist = torch.min(existing_dists).item()
                if (not force) and (min_dist < self.min_dist_factor * block.bandwidth):
                    continue # Suppress redundant duplicate spawn
                    
                # If forcing or if centroid happens to be identical to parent, perturb child centroid slightly
                if min_dist < 1e-4:
                    child_centroid = child_centroid + torch.randn_like(child_centroid) * 0.1 * block.bandwidth

                # Execute Cleavage
                new_idx = block.spawn_subspace(child_centroid, parent_idx=k)
                self.last_cleavage_epoch[block_idx] = epoch
                
                # Recalibrate Voronoi Bandwidth: sigma_k = 0.5 * min_{j != k} ||c_k - c_j||
                all_centroids = block.centroids # [N+1, D]
                if all_centroids.shape[0] > 1:
                    dists = torch.cdist(all_centroids, all_centroids)
                    dists.fill_diagonal_(float("inf"))
                    min_pairwise = torch.min(dists).item()
                    block.bandwidth = max(self.min_bandwidth, 0.50 * min_pairwise)
                else:
                    block.bandwidth = self.initial_bandwidth
                
                event = {
                    "epoch": epoch,
                    "block_idx": block_idx,
                    "parent_expert": k,
                    "child_expert": new_idx,
                    "min_eigenvalue": metrics.get("min_eigenvalue", 0.0),
                    "min_clash": metrics.get("min_clash", 0.0),
                    "mean_clash": metrics.get("mean_clash", 0.0),
                    "clash_ratio": metrics.get("clash_ratio", 0.0),
                    "centroid_norm": float(child_centroid.norm().item()),
                    "total_experts_in_block": block.num_subspaces,
                }
                self.cleavage_events.append(event)
                print(f"  [AMR Cleavage @ Epoch {epoch:3d}] Block {block_idx:2d} Expert {k} -> Spawned Expert {new_idx} (Total: {block.num_subspaces}, clash_ratio: {metrics.get('clash_ratio', 0.0):.2f}, min_clash: {metrics.get('min_clash', 0.0):.3f})")
                return True
                
        return False
