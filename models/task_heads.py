"""
Dense Prediction Task Heads for Multi-Task Vision (Segmentation, Depth, Surface Normals).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class DenseDecoderStage(nn.Module):
    """Convolutional decoder block with residual convolution and upsampling."""
    def __init__(self, in_chans: int, out_chans: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        res = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if res.shape[1] == x.shape[1]:
            x = x + res
        x = self.act(x)
        return x


class SegmentationHead(nn.Module):
    """
    Semantic Segmentation Head:
    Transforms [B, M, D] -> [B, num_classes, H, W].
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 13,
        patch_size: int = 16,
        target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.target_size = target_size

        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.stage1 = DenseDecoderStage(256, 128, scale_factor=2) # 14x14 -> 28x28
        self.stage2 = DenseDecoderStage(128, 64, scale_factor=2)  # 28x28 -> 56x56
        self.stage3 = DenseDecoderStage(64, 32, scale_factor=2)   # 56x56 -> 112x112
        self.stage4 = DenseDecoderStage(32, 32, scale_factor=2)   # 112x112 -> 224x224
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        B, M, D = tokens.shape
        gh, gw = grid_size
        x = tokens.permute(0, 2, 1).reshape(B, D, gh, gw)
        x = self.proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        logits = self.classifier(x)
        if logits.shape[-2:] != self.target_size:
            logits = F.interpolate(logits, size=self.target_size, mode="bilinear", align_corners=False)
        return logits


class DepthHead(nn.Module):
    """
    Monocular Metric Depth Head:
    Transforms [B, M, D] -> [B, 1, H, W] (Continuous metric depth regression in meters).
    """
    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        min_depth: float = 0.001,
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.stage1 = DenseDecoderStage(256, 128, scale_factor=2)
        self.stage2 = DenseDecoderStage(128, 64, scale_factor=2)
        self.stage3 = DenseDecoderStage(64, 32, scale_factor=2)
        self.stage4 = DenseDecoderStage(32, 32, scale_factor=2)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        B, M, D = tokens.shape
        gh, gw = grid_size
        x = tokens.permute(0, 2, 1).reshape(B, D, gh, gw)
        x = self.proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        raw_depth = self.out_conv(x)
        if raw_depth.shape[-2:] != self.target_size:
            raw_depth = F.interpolate(raw_depth, size=self.target_size, mode="bilinear", align_corners=False)
        depth = F.softplus(raw_depth) + self.min_depth
        return depth


class SurfaceNormalHead(nn.Module):
    """
    3D Surface Normal Orientation Head:
    Transforms [B, M, D] -> [B, 3, H, W], unit-normalized onto the unit sphere S^2.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 16,
        target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.target_size = target_size

        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.stage1 = DenseDecoderStage(256, 128, scale_factor=2)
        self.stage2 = DenseDecoderStage(128, 64, scale_factor=2)
        self.stage3 = DenseDecoderStage(64, 32, scale_factor=2)
        self.stage4 = DenseDecoderStage(32, 32, scale_factor=2)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        B, M, D = tokens.shape
        gh, gw = grid_size
        x = tokens.permute(0, 2, 1).reshape(B, D, gh, gw)
        x = self.proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        raw_normals = self.out_conv(x)
        if raw_normals.shape[-2:] != self.target_size:
            raw_normals = F.interpolate(raw_normals, size=self.target_size, mode="bilinear", align_corners=False)
        # S^2 Unit Normalization
        normals = F.normalize(raw_normals, p=2, dim=1, eps=1e-6)
        return normals


class MultiTaskHeadDict(nn.Module):
    """Unified container for all downstream multi-task heads."""
    def __init__(self, heads: Dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(tokens, grid_size)
        return outputs
