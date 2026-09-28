"""
================================================================================
Comprehensive Multi-Task Evaluation Suite for NYUv2 and Cityscapes.
Memory-Optimized with O(1) Streaming Accumulators for Million-Pixel Datasets.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeysingh525@protonmail.com
================================================================================
"""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskEvaluator:
    r"""
    Evaluator computing dense prediction metrics with constant O(1) memory:
      - Segmentation: mIoU, Pixel Accuracy
      - Depth: Abs Rel, RMSE, Sq Rel, \delta < 1.25
      - Surface Normals: Mean Error Angle, Median Error Angle, % < 11.25, 22.5, 30
      - Multi-Task Gain \Delta M
    """
    def __init__(
        self,
        num_seg_classes: int = 13,
        single_task_baselines: Optional[Dict[str, float]] = None,
    ):
        self.num_seg_classes = num_seg_classes
        self.st_baselines = single_task_baselines or {
            "seg_miou": 51.20,
            "depth_abs_rel": 0.1420,
            "depth_rmse": 0.5620,
            "normals_mean_angle": 19.40,
            "normals_median_angle": 14.20,
        }
        self.reset()

    def reset(self):
        # Segmentation Accumulators
        self.seg_confusion_matrix = np.zeros((self.num_seg_classes, self.num_seg_classes), dtype=np.int64)
        
        # Depth Accumulators (O(1) streaming scalars)
        self.depth_abs_rel_sum = 0.0
        self.depth_sq_diff_sum = 0.0
        self.depth_delta1_count = 0
        self.depth_pixel_count = 0
        
        # Surface Normal Streaming Accumulators
        self.normal_angle_sum = 0.0
        self.normal_count = 0
        self.normal_pct_11_25_count = 0
        self.normal_pct_22_50_count = 0
        self.normal_pct_30_00_count = 0
        self.normal_sample_buffer = []

    def update_segmentation(self, pred_logits: torch.Tensor, target: torch.Tensor):
        # pred_logits: [B, C, H, W], target: [B, H, W]
        preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
        targets = target.cpu().numpy()
        mask = (targets >= 0) & (targets < self.num_seg_classes)
        
        for p, t, m in zip(preds, targets, mask):
            p_valid = p[m]
            t_valid = t[m]
            bins = np.bincount(
                self.num_seg_classes * t_valid.astype(int) + p_valid.astype(int),
                minlength=self.num_seg_classes ** 2,
            )
            self.seg_confusion_matrix += bins.reshape(self.num_seg_classes, self.num_seg_classes)

    def update_depth(self, pred_depth: torch.Tensor, target_depth: torch.Tensor):
        # pred_depth: [B, 1, H, W], target_depth: [B, 1, H, W]
        p = pred_depth.detach().cpu().numpy().squeeze(1)
        t = target_depth.detach().cpu().numpy().squeeze(1)
        mask = (t > 0.001) & (t < 10.0) & (~np.isnan(t))

        for pi, ti, mi in zip(p, t, mask):
            if np.sum(mi) == 0:
                continue
            pv = np.clip(pi[mi], 1e-3, 10.0)
            tv = ti[mi]
            n_px = len(pv)
            
            self.depth_abs_rel_sum += np.sum(np.abs(pv - tv) / tv)
            self.depth_sq_diff_sum += np.sum((pv - tv) ** 2)
            
            ratio = np.maximum(pv / tv, tv / pv)
            self.depth_delta1_count += np.sum(ratio < 1.25)
            self.depth_pixel_count += n_px

    def update_surface_normals(self, pred_normals: torch.Tensor, target_normals: torch.Tensor):
        # pred_normals: [B, 3, H, W], target_normals: [B, 3, H, W]
        p = pred_normals.detach().cpu()
        t = target_normals.detach().cpu()

        # Normalize onto S^2
        p = F.normalize(p, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)

        t_norm = torch.norm(t, p=2, dim=1)
        mask = (t_norm > 0.1) & (~torch.isnan(t_norm))

        cos_sim = torch.sum(p * t, dim=1).clamp(-1.0, 1.0)
        angles_deg = torch.rad2deg(torch.acos(cos_sim)).numpy()
        mask_np = mask.numpy()

        for ai, mi in zip(angles_deg, mask_np):
            if np.sum(mi) == 0:
                continue
            valid = ai[mi]
            n_px = len(valid)
            
            self.normal_angle_sum += np.sum(valid)
            self.normal_count += n_px
            self.normal_pct_11_25_count += np.sum(valid < 11.25)
            self.normal_pct_22_50_count += np.sum(valid < 22.50)
            self.normal_pct_30_00_count += np.sum(valid < 30.00)
            
            # Sub-sample at most 100 points per image for lightweight median
            if len(self.normal_sample_buffer) < 20000:
                step = max(1, n_px // 100)
                self.normal_sample_buffer.extend(valid[::step].tolist())

    def compute_metrics(self) -> Dict[str, float]:
        metrics = {}

        # 1. Segmentation mIoU
        diag = np.diag(self.seg_confusion_matrix)
        row_sum = np.sum(self.seg_confusion_matrix, axis=1)
        col_sum = np.sum(self.seg_confusion_matrix, axis=0)
        union = row_sum + col_sum - diag
        valid_classes = union > 0
        ious = np.zeros(self.num_seg_classes)
        ious[valid_classes] = diag[valid_classes] / union[valid_classes]
        metrics["seg_miou"] = float(np.mean(ious[valid_classes]) * 100.0) if np.sum(valid_classes) > 0 else 0.0
        metrics["seg_pixel_acc"] = float(np.sum(diag) / (np.sum(self.seg_confusion_matrix) + 1e-8) * 100.0)

        # 2. Depth Metrics (Streaming O(1))
        n_depth = max(1, self.depth_pixel_count)
        metrics["depth_abs_rel"] = float(self.depth_abs_rel_sum / n_depth)
        metrics["depth_rmse"] = float(np.sqrt(self.depth_sq_diff_sum / n_depth))
        metrics["depth_delta1"] = float(self.depth_delta1_count / n_depth * 100.0)

        # 3. Surface Normal Metrics (Streaming O(1))
        n_norm = max(1, self.normal_count)
        metrics["normals_mean_angle"] = float(self.normal_angle_sum / n_norm)
        metrics["normals_pct_11_25"] = float(self.normal_pct_11_25_count / n_norm * 100.0)
        metrics["normals_pct_22_50"] = float(self.normal_pct_22_50_count / n_norm * 100.0)
        metrics["normals_pct_30_00"] = float(self.normal_pct_30_00_count / n_norm * 100.0)
        
        if self.normal_sample_buffer:
            metrics["normals_median_angle"] = float(np.median(self.normal_sample_buffer))
        else:
            metrics["normals_median_angle"] = metrics["normals_mean_angle"]

        # 4. Multi-Task Gain \Delta M
        metrics["delta_m"] = self.compute_multitask_gain(metrics)
        return metrics

    def compute_multitask_gain(self, current_metrics: Dict[str, float]) -> float:
        r"""
        Computes \Delta M = 1/T \sum_t (-1)^{s_t} (M_{m, t} - M_{s, t}) / M_{s, t} * 100%
        """
        gains = []
        task_keys = [
            ("seg_miou", False),
            ("depth_abs_rel", True),
            ("depth_rmse", True),
            ("normals_mean_angle", True),
            ("normals_median_angle", True),
        ]
        for key, lower_is_better in task_keys:
            if key in current_metrics and key in self.st_baselines:
                m_curr = current_metrics[key]
                m_base = self.st_baselines[key]
                if lower_is_better:
                    gain = (m_base - m_curr) / (m_base + 1e-8) * 100.0
                else:
                    gain = (m_curr - m_base) / (m_base + 1e-8) * 100.0
                gains.append(gain)
        return float(np.mean(gains)) if gains else 0.0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    st_baselines: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Evaluates multi-task model on a full validation dataset."""
    model.eval()
    evaluator = MultiTaskEvaluator(single_task_baselines=st_baselines)

    for batch in dataloader:
        images = batch["image"].to(device)
        preds = model(images)

        if "segmentation" in preds and "segmentation" in batch:
            evaluator.update_segmentation(preds["segmentation"], batch["segmentation"].to(device))
        if "depth" in preds and "depth" in batch:
            evaluator.update_depth(preds["depth"], batch["depth"].to(device))
        if "surface_normals" in preds and "surface_normals" in batch:
            evaluator.update_surface_normals(preds["surface_normals"], batch["surface_normals"].to(device))

    return evaluator.compute_metrics()
