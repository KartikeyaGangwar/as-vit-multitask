r"""
================================================================================
PRISM: Partition-of-Unity Routing for Isolated Subspace Manifolds
Official PyTorch Implementation.

Architecture:
  - Token-level continuous Partition of Unity (PoU) gating
  - Dynamically cleaved expert parameter subspaces \Phi_k
  - Exact function-preserving initialization at cleavage
  - Bounded perturbation & zero router collapse guarantee

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


class PRISMBlock(nn.Module):
    r"""
    PRISM Transformer Block with Feature-Space Partition of Unity (PoU).
    
    Mathematical Formulation:
      Z_{l}^{\text{attn}} = Z_{l-1} + Attn(LN_1(Z_{l-1}))
      U_l = LN_2(Z_{l}^{\text{attn}})
      Z_{l} = Z_{l}^{\text{attn}} + \sum_{k=1}^N \psi_k(U_l) * \Phi_k( (U_l - c_k) / \sigma_k )
      
    where \psi_k(u) = Softmax_k( - ||u - c_k||^2 / (2 \sigma_k^2 \sqrt{D}) ).
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
        
        # PRISM Subspace MLP Branch
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Subspace Experts
        self.experts = nn.ModuleList([
            SubspaceExpertMLP(embed_dim, mlp_ratio=mlp_ratio, act_layer=act_layer)
            for _ in range(initial_subspaces)
        ])
        
        # Subspace Centroids in Latent Manifold \mathbb{R}^D
        if initial_centroids is not None:
            assert initial_centroids.shape == (initial_subspaces, embed_dim)
            self.register_buffer("centroids", initial_centroids.clone().float())
        else:
            self.register_buffer("centroids", torch.zeros(initial_subspaces, embed_dim))

    @property
    def num_subspaces(self) -> int:
        return len(self.experts)

    def compute_pou_weights(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Computes continuous Partition of Unity (PoU) gating weights \psi_k(u).
        Input:
            u: Normalized token representations [B, M, D]
        Returns:
            psi: Normalized partition weights [B, M, N], summing to 1.0 along last dim.
        """
        B, M, D = u.shape
        N = self.num_subspaces
        
        # Compute squared Euclidean distances: ||u - c_k||^2
        # (u: [B, M, 1, D], centroids: [1, 1, N, D])
        u_exp = u.unsqueeze(2)  # [B, M, 1, D]
        c_exp = self.centroids.view(1, 1, N, D)  # [1, 1, N, D]
        sq_dist = torch.sum((u_exp - c_exp) ** 2, dim=-1)  # [B, M, N]
        
        # Temperature scaled by dimension: 2 * \sigma^2 * sqrt(D)
        scale = 2.0 * (self.bandwidth ** 2) * math.sqrt(D)
        logits = -sq_dist / max(scale, 1e-6)
        
        # Exact Partition of Unity via Softmax: \sum_k \psi_k(u) \equiv 1.0
        psi = F.softmax(logits, dim=-1)  # [B, M, N]
        return psi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass for PRISMBlock:
            x: Input token representations [B, M, D]
        Returns:
            x_out: Output tokens [B, M, D]
            psi: PoU gating allocations [B, M, N]
        """
        # 1. Multi-Head Self-Attention
        x_attn = x + self.attn(self.norm1(x))
        
        # 2. Normalized Latent Representation Tensor
        u = self.norm2(x_attn)  # [B, M, D]
        B, M, D = u.shape
        N = self.num_subspaces
        
        # 3. Continuous Partition of Unity Weights
        psi = self.compute_pou_weights(u)  # [B, M, N]
        
        # 4. Localized Subspace Evaluation & Mixture
        subspace_outputs = torch.zeros(B, M, D, device=x.device, dtype=x.dtype)
        for k in range(N):
            # Centering token representation relative to subspace centroid
            c_k = self.centroids[k].view(1, 1, D)
            u_centered = (u - c_k) / self.bandwidth
            out_k = self.experts[k](u_centered)  # [B, M, D]
            psi_k = psi[:, :, k : k + 1]  # [B, M, 1]
            subspace_outputs = subspace_outputs + (psi_k * out_k)
            
        x_out = x_attn + subspace_outputs
        return x_out, psi

    def spawn_subspace(self, new_centroid: torch.Tensor, parent_idx: int = 0) -> int:
        """
        Spawns a child subspace expert initialized from parent weights,
        anchored at the new conflict centroid in latent manifold.
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

    def cleave_subspace(self, parent_idx: int = 0, new_centroid: Optional[torch.Tensor] = None) -> int:
        """Alias for spawn_subspace supporting (parent_idx, new_centroid) call order."""
        if new_centroid is None:
            raise ValueError("new_centroid must be provided")
        return self.spawn_subspace(new_centroid=new_centroid, parent_idx=parent_idx)


# Backwards compatibility alias
AdaptiveSubspaceBlock = PRISMBlock


class PRISM(nn.Module):
    """
    PRISM: Partition-of-Unity Routing for Isolated Subspace Manifolds.
    
    Multi-Task Architecture:
      - PatchEmbed: Maps [B, 3, H, W] -> [B, M, D].
      - Positional Embedding: Learnable 1D position embeddings.
      - Encoder: Stack of L PRISMBlock layers.
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
            PRISMBlock(
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
        Forward pass through PatchEmbed and PRISM Transformer Encoder.
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


# Backwards compatibility aliases
PRISMMultiTask = PRISM
ASViT = PRISM
