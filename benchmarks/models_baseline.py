"""
Baseline Model Implementations (Single-Task ViT, Monolithic MT-ViT, Static MoE-ViT).
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_backbone import PatchEmbed, Attention, StandardMLP, TransformerBlock
from models.task_heads import SegmentationHead, DepthHead, SurfaceNormalHead, MultiTaskHeadDict


class SingleTaskViT(nn.Module):
    """
    Independent Single-Task Vision Transformer Baseline.
    Specialized for a single task (Segmentation, Depth, or Surface Normals).
    """
    def __init__(
        self,
        task: str = "segmentation",
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_seg_classes: int = 13,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.task = task
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        target_size = (img_size, img_size)
        if task == "segmentation":
            self.head = SegmentationHead(embed_dim, num_classes=num_seg_classes, patch_size=patch_size, target_size=target_size)
        elif task == "depth":
            self.head = DepthHead(embed_dim, patch_size=patch_size, target_size=target_size)
        elif task == "surface_normals":
            self.head = SurfaceNormalHead(embed_dim, patch_size=patch_size, target_size=target_size)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        out = self.head(x, self.grid_size)
        return {self.task: out}


class MonolithicMTViT(nn.Module):
    """
    Standard Monolithic Multi-Task Vision Transformer.
    Shared backbone with monolithic MLP blocks backpropagated with summed multi-task losses.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_seg_classes: int = 13,
        tasks: Optional[List[str]] = None,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.tasks = tasks or ["segmentation", "depth", "surface_normals"]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        target_size = (img_size, img_size)
        heads_dict = {}
        if "segmentation" in self.tasks:
            heads_dict["segmentation"] = SegmentationHead(embed_dim, num_classes=num_seg_classes, patch_size=patch_size, target_size=target_size)
        if "depth" in self.tasks:
            heads_dict["depth"] = DepthHead(embed_dim, patch_size=patch_size, target_size=target_size)
        if "surface_normals" in self.tasks:
            heads_dict["surface_normals"] = SurfaceNormalHead(embed_dim, patch_size=patch_size, target_size=target_size)
        self.heads = MultiTaskHeadDict(heads_dict)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.heads(x, self.grid_size)


class StaticMoEBlock(nn.Module):
    """
    Static Mixture-of-Experts (MoE) Transformer Block with fixed E experts and Top-k / Softmax routing.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        top_k: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)

        # Router: Linear projection to expert logits
        self.router = nn.Linear(dim, num_experts)

        # Static list of experts
        self.experts = nn.ModuleList([
            StandardMLP(dim, int(dim * mlp_ratio), drop=drop)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self Attention
        x = x + self.attn(self.norm1(x))

        # MoE FFN
        z = self.norm2(x) # [B, M, D]
        router_logits = self.router(z) # [B, M, E]
        
        # Softmax routing weights
        weights = F.softmax(router_logits, dim=-1) # [B, M, E]

        # Top-k gating mask
        if self.top_k < self.num_experts:
            topk_weights, topk_indices = torch.topk(weights, k=self.top_k, dim=-1)
            # Re-normalize top-k
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            mask = torch.zeros_like(weights).scatter_(-1, topk_indices, topk_weights)
            weights = mask

        moe_out = torch.zeros_like(x)
        for i in range(self.num_experts):
            w_i = weights[:, :, i:i+1] # [B, M, 1]
            if (w_i > 0).any():
                expert_out = self.experts[i](z)
                moe_out = moe_out + w_i * expert_out

        x = x + moe_out
        return x


class StaticMoEViT(nn.Module):
    """
    Static MoE Vision Transformer Baseline with fixed E experts per block.
    """
    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 2,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_seg_classes: int = 13,
        tasks: Optional[List[str]] = None,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.tasks = tasks or ["segmentation", "depth", "surface_normals"]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            StaticMoEBlock(dim=embed_dim, num_heads=num_heads, num_experts=num_experts, mlp_ratio=mlp_ratio, top_k=top_k, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        target_size = (img_size, img_size)
        heads_dict = {}
        if "segmentation" in self.tasks:
            heads_dict["segmentation"] = SegmentationHead(embed_dim, num_classes=num_seg_classes, patch_size=patch_size, target_size=target_size)
        if "depth" in self.tasks:
            heads_dict["depth"] = DepthHead(embed_dim, patch_size=patch_size, target_size=target_size)
        if "surface_normals" in self.tasks:
            heads_dict["surface_normals"] = SurfaceNormalHead(embed_dim, patch_size=patch_size, target_size=target_size)
        self.heads = MultiTaskHeadDict(heads_dict)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.heads(x, self.grid_size)
