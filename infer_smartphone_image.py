"""
================================================================================
AS-ViT In-the-Wild Smartphone Inference & Visual Showcase
Run dense multi-task visual perception on ANY custom photo taken from your phone!

Outputs 5-panel visualization:
  1. Input Smartphone RGB Photo
  2. AS-ViT 13-Class Semantic Segmentation
  3. AS-ViT Monocular Metric Depth Map (Meters)
  4. AS-ViT 3D Surface Normal Vector Field (S^2 unit sphere colormap)
  5. AS-ViT Discovered Partition of Unity (PoU) Expert Allocation Map

Author: Kartikey Singh (Department of Mathematics, University of Delhi)
================================================================================
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from models.as_vit import ASViT


def colormap_normals(normals: np.ndarray) -> np.ndarray:
    """Converts surface normal vectors in [-1, 1] to RGB [0, 1]."""
    n = normals.transpose(1, 2, 0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    n = n / norm
    rgb = (n + 1.0) / 2.0
    return np.clip(rgb, 0.0, 1.0)


def colormap_segmentation(seg: np.ndarray) -> np.ndarray:
    """Standard color palette for NYUv2 13-class semantic categories."""
    palette = np.array([
        [0.10, 0.10, 0.10], # 0: unlabeled
        [0.85, 0.20, 0.20], # 1: bed
        [0.90, 0.60, 0.10], # 2: books
        [0.70, 0.70, 0.70], # 3: ceiling
        [0.20, 0.80, 0.40], # 4: chair
        [0.60, 0.35, 0.15], # 5: floor
        [0.20, 0.50, 0.80], # 6: furniture / people
        [0.80, 0.20, 0.80], # 7: objects
        [0.95, 0.85, 0.20], # 8: picture
        [0.15, 0.75, 0.75], # 9: sofa
        [0.40, 0.20, 0.60], # 10: table
        [0.30, 0.30, 0.30], # 11: tv
        [0.80, 0.80, 0.50], # 12: wall
        [0.50, 0.80, 0.90], # 13: window
    ], dtype=np.float32)
    H, W = seg.shape
    clipped = np.clip(seg, 0, len(palette) - 1)
    return palette[clipped]


def run_smartphone_inference(image_path: str, output_path: str = "smartphone_multitask_prediction.png"):
    print(f"[*] Loading smartphone image from: {image_path}...")
    if not os.path.exists(image_path):
        print(f"[-] Error: File not found: {image_path}")
        return

    # 1. Load and resize image
    raw_img = Image.open(image_path).convert("RGB")
    target_size = (224, 224)
    img_resized = raw_img.resize(target_size, Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # PyTorch Tensor: [1, 3, 224, 224]
    img_tensor = torch.tensor(img_np.transpose(2, 0, 1)).unsqueeze(0).float()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    # 2. Instantiate AS-ViT with discovered topology
    model = ASViT(
        img_size=224,
        embed_dim=192,
        depth=4,
        num_heads=4,
        initial_subspaces_per_block=[1, 2, 3, 2],
        bandwidth=0.5,
    ).to(device)
    model.eval()

    # 3. Model Inference
    with torch.no_grad():
        preds, _, pou_weights = model(img_tensor, return_intermediates=True)

    # Extract task outputs
    pred_seg = torch.argmax(preds["segmentation"], dim=1).squeeze().cpu().numpy() # [224, 224]
    pred_depth = preds["depth"].squeeze().cpu().numpy() # [224, 224]
    pred_normals = preds["surface_normals"].squeeze().cpu().numpy() # [3, 224, 224]

    # PoU Expert Allocation Map (from deepest multi-subspace block)
    # pou_weights[-2]: [1, 196, 3] -> reshape to [14, 14, 3] -> resize to [224, 224, 3]
    pou_last = pou_weights[-2].squeeze().cpu().numpy() # [196, N]
    N = pou_last.shape[-1]
    pou_grid = pou_last.reshape(14, 14, N)
    
    pou_rgb = np.zeros((14, 14, 3), dtype=np.float32)
    colors = [
        np.array([0.15, 0.75, 0.40]), # Subspace 1 (Walls)
        np.array([0.85, 0.45, 0.15]), # Subspace 2 (Floors)
        np.array([0.25, 0.40, 0.90]), # Subspace 3 (Foreground Objects/People)
    ]
    for k in range(min(N, 3)):
        pou_rgb += pou_grid[:, :, k:k+1] * colors[k]
        
    pou_map = np.array(Image.fromarray((pou_rgb * 255).astype(np.uint8)).resize(target_size, Image.BILINEAR)) / 255.0

    # 4. Plot 5-Panel Showcase
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=300)
    plt.subplots_adjust(wspace=0.04)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Smartphone Photo", fontsize=11, fontweight="bold", pad=8)
    
    axes[1].imshow(colormap_segmentation(pred_seg))
    axes[1].set_title("AS-ViT Segmentation", fontsize=11, fontweight="bold", pad=8)
    
    d_min, d_max = float(pred_depth.min()), float(pred_depth.max())
    axes[2].imshow(pred_depth, cmap="plasma", vmin=d_min, vmax=d_max)
    axes[2].set_title("AS-ViT Metric Depth (m)", fontsize=11, fontweight="bold", pad=8)
    
    axes[3].imshow(colormap_normals(pred_normals))
    axes[3].set_title("AS-ViT 3D Surface Normals", fontsize=11, fontweight="bold", pad=8)
    
    axes[4].imshow(pou_map)
    axes[4].set_title("Discovered PoU Allocation", fontsize=11, fontweight="bold", pad=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Multi-Task prediction on smartphone image saved to: {output_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AS-ViT In-the-Wild Smartphone Inference")
    parser.add_argument("--image", type=str, default="data/nyuv2_real_samples/00001.h5", help="Path to input smartphone image (.jpg/.png)")
    parser.add_argument("--output", type=str, default="smartphone_multitask_prediction.png", help="Path to save output visualization")
    args = parser.parse_args()
    
    if args.image.endswith(".h5"):
        # Demo mode using real RGB from h5
        import h5py
        with h5py.File(args.image, "r") as h:
            rgb = h["rgb"][:].transpose(1, 2, 0)
            demo_png = "demo_smartphone_input.png"
            Image.fromarray(rgb).save(demo_png)
            run_smartphone_inference(demo_png, args.output)
            if os.path.exists(demo_png):
                os.remove(demo_png)
    else:
        run_smartphone_inference(args.image, args.output)
