"""
Multi-Task Loss Functions for Semantic Segmentation, Monocular Depth, and Surface Normals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SegmentationLoss(nn.Module):
    """Cross Entropy Loss for Semantic Segmentation."""
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H, W], target: [B, H, W] (long)
        return self.loss_fn(pred, target.long())


class DepthLoss(nn.Module):
    """
    Scale-Invariant Logarithmic + L1 Loss for Monocular Depth Estimation.
    Computed only on valid depth pixels (depth > 0).
    """
    def __init__(self, eps: float = 1e-6, alpha: float = 0.5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 1, H, W], target: [B, 1, H, W]
        mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_val = pred[mask].clamp_min(self.eps)
        target_val = target[mask].clamp_min(self.eps)

        # L1 Error
        l1_loss = F.l1_loss(pred_val, target_val)

        # Scale Invariant Log Error
        d = torch.log(pred_val) - torch.log(target_val)
        silog = torch.mean(d ** 2) - self.alpha * (torch.mean(d) ** 2)

        return l1_loss + silog


class SurfaceNormalLoss(nn.Module):
    """
    Cosine Distance Loss for 3D Surface Normal Vectors.
    L = 1 - <pred, target> (normalized on S^2).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 3, H, W], target: [B, 3, H, W]
        # Valid mask: normal vectors with non-zero norm
        target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
        mask = (target_norm > 0.1) & (~torch.isnan(target_norm))
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Broadcast mask to 3 channels
        mask_3d = mask.expand_as(pred)
        p = pred[mask_3d].reshape(-1, 3)
        t = target[mask_3d].reshape(-1, 3)

        p = F.normalize(p, p=2, dim=-1, eps=self.eps)
        t = F.normalize(t, p=2, dim=-1, eps=self.eps)

        cos_sim = torch.sum(p * t, dim=-1).clamp(-1.0, 1.0)
        return torch.mean(1.0 - cos_sim)


class MultiTaskLossModule(nn.Module):
    """
    Unified Multi-Task Loss Module computing individual task losses and total weighted loss.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.seg_loss = SegmentationLoss()
        self.depth_loss = DepthLoss()
        self.normal_loss = SurfaceNormalLoss()
        self.weights = weights or {
            "segmentation": 1.0,
            "depth": 1.0,
            "surface_normals": 1.0,
        }

    def compute_individual_losses(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes separate scalar loss tensors per task."""
        losses = {}
        if "segmentation" in preds and "segmentation" in targets:
            losses["segmentation"] = self.seg_loss(preds["segmentation"], targets["segmentation"])
        if "depth" in preds and "depth" in targets:
            losses["depth"] = self.depth_loss(preds["depth"], targets["depth"])
        if "surface_normals" in preds and "surface_normals" in targets:
            losses["surface_normals"] = self.normal_loss(preds["surface_normals"], targets["surface_normals"])
        return losses

    def forward(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.compute_individual_losses(preds, targets)
        total_loss = torch.tensor(0.0, device=next(iter(preds.values())).device)
        for task_name, loss_val in losses.items():
            w = self.weights.get(task_name, 1.0)
            total_loss = total_loss + w * loss_val
        return total_loss, losses
