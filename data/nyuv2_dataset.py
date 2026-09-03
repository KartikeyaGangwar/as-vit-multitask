"""
NYUv2 Multi-Task Dataset Loader with Built-in Synthetic Mode.
"""
import os
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import JointMultiTaskTransform


class NYUv2Dataset(Dataset):
    """
    NYUv2 Multi-Task Dataset (13-Class Semantic Segmentation, Monocular Depth, 3D Surface Normals).
    
    Supports:
      1. Real data loading from directory containing images and task labels.
      2. Automated synthetic mode (synthetic=True) for self-contained testing and benchmarking.
    """
    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        synthetic: bool = False,
        synthetic_samples: int = 64,
        img_size: Tuple[int, int] = (224, 224),
        tasks: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.synthetic = synthetic or (root is None) or (not os.path.exists(str(root)))
        self.synthetic_samples = synthetic_samples
        self.img_size = img_size
        self.tasks = tasks or ["segmentation", "depth", "surface_normals"]
        self.transform = transform or JointMultiTaskTransform(
            target_size=img_size, is_train=(split == "train")
        )
        self.seed = seed
        self._cache = {}

        if self.synthetic:
            self._generate_synthetic_metadata()
        else:
            self._load_dataset_files()

    def _generate_synthetic_metadata(self):
        """Generates deterministic synthetic scene parameters."""
        np.random.seed(self.seed + (0 if self.split == "train" else 1000))
        self.samples = []
        H, W = self.img_size
        for idx in range(self.synthetic_samples):
            # Synthetic indoor geometry parameters
            plane_slope = np.random.uniform(-0.5, 0.5, size=2)
            num_objects = np.random.randint(2, 6)
            self.samples.append({
                "id": idx,
                "plane_slope": plane_slope,
                "num_objects": num_objects,
            })

    def _load_dataset_files(self):
        """Loads real file list from root directory."""
        self.samples = []
        if self.root and os.path.exists(self.root):
            split_dir = os.path.join(self.root, self.split)
            if os.path.exists(split_dir):
                file_list = sorted([f for f in os.listdir(split_dir) if f.endswith(".npz") or f.endswith(".pkl")])
                self.samples = [os.path.join(split_dir, f) for f in file_list]
        if not self.samples:
            self.synthetic = True
            self._generate_synthetic_metadata()

    def _synthesize_sample(self, meta: Dict) -> Dict[str, torch.Tensor]:
        """Synthesizes a realistic indoor multi-task RGB-D-Normal frame using float32."""
        H, W = self.img_size
        # 1. Coordinate Grid (float32)
        ys, xs = np.meshgrid(
            np.linspace(-1.0, 1.0, H, dtype=np.float32),
            np.linspace(-1.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        
        # 2. Synthetic Depth: Base slanted floor/wall plane + spherical objects
        base_depth = (2.5 + meta["plane_slope"][0] * xs + meta["plane_slope"][1] * ys).astype(np.float32)
        depth_map = base_depth.copy()
        
        # 3. Synthetic Segmentation: Class 0 (Floor), Class 1 (Wall), Classes 2-12 (Furniture)
        seg_map = np.zeros((H, W), dtype=np.int64)
        seg_map[ys > 0.3] = 0 # Floor
        seg_map[ys <= 0.3] = 1 # Wall

        for obj_idx in range(meta["num_objects"]):
            cx, cy = np.random.uniform(-0.6, 0.6, size=2).astype(np.float32)
            r = np.float32(np.random.uniform(0.15, 0.35))
            dist_sq = (xs - cx)**2 + (ys - cy)**2
            mask = dist_sq < (r**2)
            cls_id = 2 + (obj_idx % 11)
            seg_map[mask] = cls_id
            obj_depth = (1.8 - 0.5 * np.sqrt(np.clip(r**2 - dist_sq[mask], 0.0, None))).astype(np.float32)
            depth_map[mask] = obj_depth

        # 4. Synthetic Surface Normals: Compute gradient of depth map (float32)
        dy, dx = np.gradient(depth_map)
        nz = np.ones_like(depth_map, dtype=np.float32)
        nx = (-dx).astype(np.float32)
        ny = (-dy).astype(np.float32)
        norm = np.sqrt(nx**2 + ny**2 + nz**2).astype(np.float32) + 1e-6
        normals = np.stack([nx / norm, ny / norm, nz / norm], axis=0).astype(np.float32)

        # 5. Synthetic RGB Image: Render shaded surface + texturing
        light_dir = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        shading = np.clip(np.sum(normals * light_dir[:, None, None], axis=0), 0.1, 1.0).astype(np.float32)
        
        rgb = np.stack([
            0.5 + 0.4 * np.sin(3 * xs + seg_map),
            0.5 + 0.4 * np.cos(3 * ys + seg_map),
            0.5 + 0.3 * np.sin(5 * depth_map),
        ], axis=0).astype(np.float32)
        rgb = np.clip(rgb * shading[None, :, :], 0.0, 1.0).astype(np.float32)

        # Convert to torch Tensors
        rgb_t = torch.from_numpy(rgb)
        seg_t = torch.from_numpy(seg_map)
        depth_t = torch.from_numpy(depth_map).unsqueeze(0)
        norm_t = torch.from_numpy(normals)

        return {
            "image": rgb_t,
            "segmentation": seg_t,
            "depth": depth_t,
            "surface_normals": norm_t,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.synthetic:
            if idx not in self._cache:
                self._cache[idx] = self._synthesize_sample(self.samples[idx])
            cached = self._cache[idx]
            raw_sample = {k: v.clone() for k, v in cached.items()}
        else:
            data = np.load(self.samples[idx])
            raw_sample = {
                "image": torch.tensor(data["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0,
                "segmentation": torch.tensor(data["segmentation"], dtype=torch.long),
                "depth": torch.tensor(data["depth"], dtype=torch.float32).unsqueeze(0),
                "surface_normals": torch.tensor(data["normals"], dtype=torch.float32).permute(2, 0, 1),
            }

        if self.transform is not None:
            sample = self.transform(
                image=raw_sample["image"],
                segmentation=raw_sample.get("segmentation"),
                depth=raw_sample.get("depth"),
                normals=raw_sample.get("surface_normals"),
            )
        else:
            sample = raw_sample

        return sample
