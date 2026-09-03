"""
Multi-Task Joint Transformations for RGB, Segmentation, Depth, and Surface Normals.
"""
import random
from typing import Dict, Optional, Tuple
import torch
import torchvision.transforms.functional as TF


class JointMultiTaskTransform:
    """
    Joint spatial and photometric transformations for Multi-Task dense prediction.
    Correctly transforms geometric targets (e.g. flips normal x-axis upon horizontal flip).
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        hflip_prob: float = 0.5,
    ):
        self.target_size = target_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob

    def __call__(
        self,
        image: torch.Tensor,                    # [3, H, W]
        segmentation: Optional[torch.Tensor] = None, # [H, W]
        depth: Optional[torch.Tensor] = None,        # [1, H, W]
        normals: Optional[torch.Tensor] = None,      # [3, H, W]
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Resize to target size
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        if segmentation is not None:
            if segmentation.ndim == 2:
                segmentation = segmentation.unsqueeze(0)
            segmentation = TF.resize(segmentation, self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        if depth is not None:
            depth = TF.resize(depth, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        if normals is not None:
            normals = TF.resize(normals, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            # Re-normalize
            normals = TF.normalize(normals, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

        # 2. Horizontal Flip
        if self.is_train and (random.random() < self.hflip_prob):
            image = TF.hflip(image)
            if segmentation is not None:
                segmentation = TF.hflip(segmentation)
            if depth is not None:
                depth = TF.hflip(depth)
            if normals is not None:
                normals = TF.hflip(normals)
                # Geometric reflection: flip the x-component of surface normals
                normals[0, :, :] = -normals[0, :, :]

        # 3. Standard Image Normalization (ImageNet stats)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        sample = {"image": image}
        if segmentation is not None:
            sample["segmentation"] = segmentation
        if depth is not None:
            sample["depth"] = depth
        if normals is not None:
            sample["surface_normals"] = normals

        return sample
