"""
AS-ViT Benchmarks Package.
Exports baseline architectures and benchmark runners.
"""
from benchmarks.models_baseline import SingleTaskViT, MonolithicMTViT, StaticMoEViT

__all__ = [
    "SingleTaskViT",
    "MonolithicMTViT",
    "StaticMoEViT",
]
