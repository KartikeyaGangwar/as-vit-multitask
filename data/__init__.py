"""
AS-ViT Data Package (Synthetic & Real-World Pipelines).
"""
from data.transforms import JointMultiTaskTransform
from data.transforms import JointMultiTaskTransform
from data.nyuv2_dataset import NYUv2Dataset
from data.real_nyuv2_pipeline import RealNYUv2Dataset, compute_surface_normals_from_depth

__all__ = [
    "JointMultiTaskTransform",
    "NYUv2Dataset",
    "RealNYUv2Dataset",
    "compute_surface_normals_from_depth",
]

