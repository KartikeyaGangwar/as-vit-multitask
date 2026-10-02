"""
================================================================================
Adaptive Subspace Vision Transformer (AS-ViT) Core Architecture.
Official PyTorch Implementation for IEEE TPAMI / CVPR.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeyagangwar@proton.me
================================================================================
"""
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subspace_expert import SubspaceExpertMLP
from models.vit_backbone import PatchEmbed, Attention
from models.task_heads import (
    SegmentationHead,
    DepthHead,
    SurfaceNormalHead,
    MultiTaskHeadDict,
)


class AdaptiveSubspaceBlock(nn.Module):
    r"""
    Adaptive Subspace Transformer Block with Feature-Space Partition of Unity (PoU).
    
    Mathematical Formulation:
      Z_{l} = Z_{l-1} + Attn(LN(Z_{l-1}))
      Z_{l} = Z_{l} + \sum_{k=1}^N \psi_k(Z_l) * \Phi_k( (LN(Z_l) - c_k) / \sigma_k )
      
    where \psi_k(z) = Softmax_k( - ||z - c_k||^2 / (2 \sigma_k^2) ).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        initial_subspaces: int = 1,
        mlp_ratio: float = 4.0,
        act_layer: str = "gelu",
        bandwidth: float = 1.0,
        initial_centroids: Optional[torch.Tensor] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer
        self.bandwidth = float(bandwidth)
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        # Adaptive Subspace MLP Branch
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Centroids Buffer: [N, embed_dim]
        if initial_centroids is not None:
            c_init = initial_centroids.clone().detach().to(dtype=torch.float32)
            N = c_init.shape[0]
        else:
            c_init = torch.zeros((initial_subspaces, embed_dim), dtype=torch.float32)
            N = initial_subspaces
            
        self.register_buffer("centroids", c_init)
        
        # Subspace Experts
        self.experts = nn.ModuleList([
            SubspaceExpertMLP(embed_dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop)
            for _ in range(N)
        ])

    @property
    def num_subspaces(self) -> int:
        return len(self.experts)

    def partition_of_unity(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Computes Symmetric Voronoi Partition of Unity gating weights \psi_k(z).
        Guarantees \sum_{k=1}^N \psi_k(z) = 1.0 everywhere.
        
        Args:
            z: [B, M, D] token features.
        Returns:
            psi: [B, M, N] gating weights.
        """
        # z: [B, M, D], centroids: [N, D]
        # diff: [B, M, N, D]
        diff = z.unsqueeze(2) - self.centroids.unsqueeze(0).unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1) # [B, M, N]
        
        # Dimension-scaled Voronoi potential
        dim_scale = math.sqrt(self.embed_dim)
        logits = -dist_sq / (2.0 * (self.bandwidth ** 2) * dim_scale + 1e-8)
        return torch.softmax(logits, dim=-1) # [B, M, N]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, M, D] token representations.
        Returns:
            (x_out, psi_weights)
        """
        # Self-Attention
        x = x + self.attn(self.norm1(x))
        
        # PoU Multi-Subspace MLP
        z_norm = self.norm2(x) # [B, M, D]
        psi = self.partition_of_unity(x) # [B, M, N]
        
        mlp_out = torch.zeros_like(x)
        for k in range(self.num_subspaces):
            # Local coordinate centering
            z_local = (z_norm - self.centroids[k]) / self.bandwidth
            expert_out = self.experts[k](z_local) # [B, M, D]
            mlp_out = mlp_out + psi[:, :, k:k+1] * expert_out
            
        x = x + mlp_out
        return x, psi

    def spawn_subspace(self, new_centroid: torch.Tensor, parent_idx: int = 0) -> int:
        r"""
        Spawns a child expert \Phi_{N+1} centered at `new_centroid`.
        Clones parent weights with Gaussian perturbation for zero-disruption cleavage.
        """
        device = self.centroids.device
        dtype = self.centroids.dtype
        new_c = new_centroid.to(device=device, dtype=dtype).reshape(1, self.embed_dim)
        
        # Append centroid buffer
        updated_centroids = torch.cat([self.centroids, new_c], dim=0)
        self.register_buffer("centroids", updated_centroids)
        
        # Instantiate child expert
        child_expert = SubspaceExpertMLP(
            self.embed_dim, mlp_ratio=self.mlp_ratio, act_layer=self.act_layer
        )
        
        # Clone parent parameters with epsilon perturbation
        parent_params = self.experts[parent_idx].state_dict()
        child_state = {}
        for k, v in parent_params.items():
            noise = torch.randn_like(v) * 1e-4 if "weight" in k else torch.zeros_like(v)
            child_state[k] = v.clone() + noise
        child_expert.load_state_dict(child_state)
        child_expert.to(device=device)
        
        self.experts.append(child_expert)
        return len(self.experts) - 1


class ASViT(nn.Module):
    """
    Adaptive Subspace Vision Transformer (AS-ViT) Multi-Task Model.
    
    Components:
      - PatchEmbed: Maps [B, 3, H, W] -> [B, M, D].
      - Positional Embedding: Learnable 1D position embeddings.
      - Encoder: Stack of L AdaptiveSubspaceBlock layers.
      - Multi-Task Heads: Segmentation, Depth, Surface Normals.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        act_layer: str = "gelu",
        bandwidth: float = 1.0,
        initial_subspaces_per_block: Union[int, List[int]] = 1,
        tasks: Optional[List[str]] = None,
        num_seg_classes: int = 13,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.tasks = tasks or ["segmentation", "depth", "surface_normals"]

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Blocks
        if isinstance(initial_subspaces_per_block, int):
            subspace_counts = [initial_subspaces_per_block] * depth
        else:
            subspace_counts = initial_subspaces_per_block

        self.blocks = nn.ModuleList([
            AdaptiveSubspaceBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                initial_subspaces=subspace_counts[i],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                bandwidth=bandwidth,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-Task Prediction Heads
        heads_dict = {}
        target_size = (img_size, img_size)
        if "segmentation" in self.tasks:
            heads_dict["segmentation"] = SegmentationHead(
                embed_dim=embed_dim,
                num_classes=num_seg_classes,
                patch_size=patch_size,
                target_size=target_size,
            )
        if "depth" in self.tasks:
            heads_dict["depth"] = DepthHead(
                embed_dim=embed_dim,
                patch_size=patch_size,
                target_size=target_size,
            )
        if "surface_normals" in self.tasks:
            heads_dict["surface_normals"] = SurfaceNormalHead(
                embed_dim=embed_dim,
                patch_size=patch_size,
                target_size=target_size,
            )
        self.heads = MultiTaskHeadDict(heads_dict)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through PatchEmbed and Transformer Encoder.
        Returns:
            (final_tokens, intermediate_tokens_per_block, pou_weights_per_block)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        intermediate_tokens = []
        pou_weights = []

        for block in self.blocks:
            intermediate_tokens.append(x)
            x, psi = block(x)
            pou_weights.append(psi)

        x = self.norm(x)
        return x, intermediate_tokens, pou_weights

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        """
        Full Forward Pass:
          Input Image [B, 3, H, W] -> Multi-Task Predictions Dict
        """
        tokens, inter_tokens, pou_weights = self.forward_features(x)
        predictions = self.heads(tokens, self.grid_size)

        if return_intermediates:
            return predictions, inter_tokens, pou_weights
        return predictions

    def get_topology(self) -> Dict:
        """Returns the discovered subspace topology across all transformer blocks."""
        topology = {}
        for i, block in enumerate(self.blocks):
            topology[f"block_{i}"] = {
                "num_subspaces": block.num_subspaces,
                "centroids": block.centroids.clone().detach(),
                "bandwidth": block.bandwidth,
            }
        return topology
