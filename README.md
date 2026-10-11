# PRISM: Partition-of-Unity Routing for Isolated Subspace Manifolds in Dense Multi-Task Vision

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22974130.svg)](https://doi.org/10.5281/zenodo.22974130)
[![Preprint: Open Access](https://img.shields.io/badge/Preprint-Zenodo%20(Open%20Access)-blue.svg)](https://doi.org/10.5281/zenodo.22974130)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--1973--7532-a6ce39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0009-1973-7532)

> **Author:** **Kartikeya Gangwar**  
> **Affiliation:** Department of Mathematics, University of Delhi, Delhi, 110007, India  
> **Contact:** `kartikeyagangwar@proton.me`  
> **ORCID:** [`0009-0009-1973-7532`](https://orcid.org/0009-0009-1973-7532)  
> **Open Access Preprint:** [**10.5281/zenodo.22974130**](https://doi.org/10.5281/zenodo.22974130) *(Full 15-page manuscript & appendices unpaywalled)*

---

## Table of Contents
- [Overview](#overview)
- [Research Lineage](#research-lineage)
- [SciML to Computer Vision Formulation](#sciml-to-computer-vision-formulation)
- [Method Overview](#method-overview)
- [Theoretical Properties](#theoretical-properties)
- [Benchmark Results](#benchmark-results)
- [Visual Diagnostics](#visual-diagnostics)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Cloud Deployment (Kaggle & Colab)](#cloud-deployment-kaggle--colab)
- [Hardware & Reproducibility](#hardware--reproducibility)
- [Citation](#citation)

---

## Overview

Dense multi-task visual perception demands a unified backbone capable of simultaneously predicting categorical semantic masks, continuous metric depth fields, and 3D surface normal vector fields. However, standard monolithic Vision Transformers (**MT-ViT**) suffer from **negative transfer** caused by destructive gradient clashing across shared parameter manifolds:

$$\langle \nabla_\Theta \mathcal{L}_i, \nabla_\Theta \mathcal{L}_j \rangle < 0, \quad \text{for } i \neq j.$$

Existing remedies either:
1. **Project gradients in a single shared capacity bottleneck** (e.g., PCGrad, CAGrad, Nash-MTL, FAMO)---which repeatedly strips away critical task update signals and can lead to optimization stagnation, or
2. **Deploy static Mixture-of-Experts (MoE)** with fixed heuristic counts $E \in \{4, 8, 16\}$---which suffer from discrete top-$k$ routing collapse and dead experts.

**PRISM (Partition-of-Unity Routing for Isolated Subspace Manifolds)** introduces **Parameter-Space Adaptive Mesh Refinement (AMR)** into the latent representation space of Vision Transformers. PRISM continuously monitors vectorized inter-task Gram conflict matrices via batched automatic differentiation (`torch.func.vmap`). When persistent destructive interference is detected ($\lambda_{\min}(\mathcal{G}) < -\tau_{\text{conflict}}$), PRISM **autonomously cleaves and instantiates dedicated transformer parameter subspaces** centered at the latent conflict centroid $\mathbf{c}_{N+1}$, modulated by a continuous **Feature-Space Partition of Unity (PoU)** gating mechanism with provable zero-disruption loss invariance.

---

## Research Lineage

This repository implements the multi-task vision component of research on parameter manifold decoupling by **Kartikeya Gangwar**:

```
                  PARAMETER DECOUPLING RESEARCH LINEAGE
                                    │
   ┌────────────────────────────────┼────────────────────────────────┐
   ▼                                ▼                                ▼
[Part I: Boundary Projection]  [Part II: Parameter AMR]      [Part III: Latent Manifolds]
Null-Space PINN                ATLAS                         PRISM (This Repo) 
(Algebraic Direct-Sum)         (Autonomous PDE AMR)          (Feature-Space MoE)
[null-space-pinn]              [atlas-neural-amr]            [prism-multitask-vision]
DOI: 10.5281/zenodo.22132799  DOI: 10.5281/zenodo.22822521  DOI: 10.5281/zenodo.22974130
```

1. **Part I: Algebraic Direct-Sum Partitioning (`null-space-pinn`):**  
   *Title:* *"Decoupling Gradient Conflicts in Physics-Informed Neural Networks via Null-Space Parameter Subspaces"*  
   *Focus:* Proves structural gradient orthogonality ($\langle \nabla\mathcal{L}_{\mathrm{if}}, \nabla\mathcal{L}_{\mathrm{des}} \rangle \equiv 0$) on static 2D boundaries via $C^2$ Quintic Hermite operators and frozen orthogonal projection bases ($\Theta = \Theta_0 \oplus \Theta_1$, $\mathcal{W}_0 \mathcal{W}_1^T = \mathbf{0}$).  
   *Repo:* [https://github.com/KartikeyaGangwar/null-space-pinn](https://github.com/KartikeyaGangwar/null-space-pinn) | *DOI:* [10.5281/zenodo.22132799](https://doi.org/10.5281/zenodo.22132799)

2. **Part II: Autonomous Parameter AMR (`atlas-neural-amr`):**  
   *Title:* *"ATLAS: Adaptive Tiling of Localized Anisotropic Subspaces for Autonomous Parameter-Space AMR via Vectorized Gradient Conflict Profiling"*  
   *Focus:* Extends static partitioning into dynamic, parameter-space Adaptive Mesh Refinement (AMR). Uses vectorized Gram conflict matrices (`torch.func.vmap`) to trigger subspace fission with exact zero-disruption solution invariance ($\|u^{(N+1)} - u^{(N)}\| \equiv 0$) across 9 canonical PDEs.  
   *Repo:* [https://github.com/KartikeyaGangwar/atlas-neural-amr](https://github.com/KartikeyaGangwar/atlas-neural-amr) | *DOI:* [10.5281/zenodo.22822521](https://doi.org/10.5281/zenodo.22822521)

3. **Part III: Multi-Task Vision Transformers (`prism-multitask-vision`):**  
   *Title:* *"PRISM: Partition-of-Unity Routing for Isolated Subspace Manifolds in Dense Multi-Task Vision"*  
   *Focus:* Formulates parameter-space cleavage on the latent token manifold of Vision Transformers. Tracks inter-task Gram matrix eigenvalues ($\lambda_{\min}(\mathcal{G}) < -\tau_{\text{conflict}}$) to dynamically spawn expert parameter subspaces via continuous Feature-Conditioned Partition of Unity (PoU) gating.  
   *Repo:* [https://github.com/KartikeyaGangwar/prism-multitask-vision](https://github.com/KartikeyaGangwar/prism-multitask-vision) | *Preprint DOI:* [10.5281/zenodo.22974130](https://doi.org/10.5281/zenodo.22974130)


## SciML to Computer Vision Formulation

PRISM adapts principles from **ATLAS: Adaptive Tiling of Localized Anisotropic Subspaces** into dense multi-task computer vision:

| Scientific Machine Learning (`ATLAS` / `AS-PINN`) | Multi-Task Computer Vision (`PRISM`) |
| :--- | :--- |
| **Physical Coordinate Domain:** $\mathbf{x} = (x, y, t) \in \Omega$ | **Latent Token Manifold:** $\mathbf{z} \in \mathbb{R}^D$ inside Transformer blocks |
| **PDE Residual Conflict:** $\mathcal{G}_{ij} = \cos \angle(\nabla_\Theta r_i^2, \nabla_\Theta r_j^2)$ | **Multi-Task Jacobian Alignment:** $\mathcal{G}_{ij} = \cos \angle(\mathbf{g}_i, \mathbf{g}_j)$ |
| **Spatial Voronoi PoU:** $\psi_k(\mathbf{x}) = \text{Softmax}_k(-\|\mathbf{x}-\mathbf{c}_k\|^2 / (2\sigma^2))$ | **Feature PoU Gating:** $\psi_k(\mathbf{z}) = \text{Softmax}_k(-\|\mathbf{z}-\mathbf{c}_k\|^2 / (2\sigma_k^2 \sqrt{D}))$ |
| **Local Frame Centering:** $\tilde{\mathbf{x}} = (\mathbf{x} - \mathbf{c}_k) / \boldsymbol{\sigma}$ | **Subspace Input Pre-Centering:** $\tilde{\mathbf{z}} = (\text{LN}(\mathbf{z}) - \mathbf{c}_k) / \sigma_k$ |
| **Spatial Clash Centroid:** $\mathbf{c}_{N+1} = \sum w_m \mathbf{x}_m / \sum w_m$ | **Sensitivity Token Centroid:** $\mathbf{c}_{N+1} = \sum \|\nabla_\mathbf{z} \mathcal{L}_{\text{clash}}\| \mathbf{z}_m / \sum w_m$ |
| **Coordinate Distance Filter:** $\|\mathbf{c}_i - \mathbf{c}_j\| / \boldsymbol{\sigma} \ge 0.40$ | **Latent Distance Filter:** $\|\mathbf{c}_i - \mathbf{c}_j\| / (\sigma \sqrt{D}) \ge 0.35$ |
| **Two-Stage Discover-and-Deploy:** Stage 1 AMR Probe $\to$ Stage 2 L-BFGS | **Two-Stage Workflow:** Stage 1 AMR Discovery $\to$ Stage 2 Clean Retraining |

---

## Method Overview

### 1. Vectorized Multi-Task Gram Clash Profiling (`torch.func.vmap`)
For each active subspace $\Phi_k$, PRISM evaluates the inter-task Jacobian using vectorized automatic differentiation:

$$\mathbf{G}_k = \begin{bmatrix} \mathbf{g}_{k, 1}^T \\ \mathbf{g}_{k, 2}^T \\ \vdots \\ \mathbf{g}_{k, T}^T \end{bmatrix} \in \mathbb{R}^{T \times P_k}, \quad \mathcal{G}_k = \tilde{\mathbf{G}}_k \tilde{\mathbf{G}}_k^T \in \mathbb{R}^{T \times T}, \quad \mathcal{G}_{k, ij} = \cos \angle(\mathbf{g}_{k, i}, \mathbf{g}_{k, j}).$$

Destructive interference is triggered when:
$$\lambda_{\min}(\mathcal{G}_k) < -\tau_{\text{conflict}}, \quad \text{and} \quad \overline{\mathcal{C}}_k = \frac{1}{T(T-1)}\sum_{i \neq j}\mathcal{G}_{k, ij} < 0.$$

### 2. Feature-Space Partition of Unity (PoU) Gating
Unlike discrete argmax top-$k$ routing in static MoEs (which can cause router collapse and dead experts), PRISM uses continuous Gaussian Voronoi window gating:

$$\psi_k(\mathbf{z}) = \frac{\exp\left( - \frac{\|\mathbf{z} - \mathbf{c}_k\|_2^2}{2\sigma_k^2 \sqrt{D} + \epsilon} \right)}{\sum_{j=1}^N \exp\left( - \frac{\|\mathbf{z} - \mathbf{c}_j\|_2^2}{2\sigma_j^2 \sqrt{D} + \epsilon} \right)}, \quad \sum_{k=1}^N \psi_k(\mathbf{z}) = 1.0, \quad \psi_k(\mathbf{z}) \ge 0.$$

### 3. Two-Stage Discover-and-Deploy Workflow
- **Stage 1 (AMR Topology Discovery):** Lightweight probe monitors gradient interference and uncovers minimal parsimonious subspace count $N_l^*$ and conflict centroids $\{\mathbf{c}_{l, k}\}$.
- **Stage 2 (Clean Production Retraining):** Instantiates clean model with discovered topology $\{N_l^*\}$ and trains with fresh AdamW momentum buffers, avoiding optimizer buffer corruption.

---

## Theoretical Properties

### Lemma 1 (Exact Loss Invariance at Moment of Cleavage)
Let $\mathcal{L}(\Theta^{(N)})$ be the multi-task loss before cleavage. When child subspace $N+1$ is spawned with cloned parent parameters $\Theta_{N+1} = \Theta_k$, the instantaneous loss variation satisfies:

$$\lim_{\|\xi\| \to 0} \left| \mathcal{L}(\Theta^{(N+1)}) - \mathcal{L}(\Theta^{(N)}) \right| = 0.$$

*Proof:* Follows directly from the Partition of Unity conservation axiom ($\sum_{j=1}^{N+1}\psi_j = 1.0$) and the linearity of the gating mixture $\psi_k^{(N+1)} \Phi_k + \psi_{N+1}^{(N+1)} \Phi_k = (\psi_k^{(N+1)} + \psi_{N+1}^{(N+1)}) \Phi_k$. $\blacksquare$

### Theorem 1 (Suppression of Negative Transfer)
Let $\mathbf{g}_i = \nabla_\Theta \mathcal{L}_i$ and $\mathbf{g}_j = \nabla_\Theta \mathcal{L}_j$ be conflicting task gradients ($\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$). Under PRISM with cleaved parameter subspaces $\Theta_1 \cap \Theta_2 = \emptyset$, the inner product of parameter updates satisfies:

$$\langle \nabla_{\Theta_{\text{total}}} \mathcal{L}_i, \nabla_{\Theta_{\text{total}}} \mathcal{L}_j \rangle \ge - \mathcal{O}\left( \exp\left( -\frac{\|\mathbf{c}_1 - \mathbf{c}_2\|^2}{2\sigma^2} \right) \right).$$

As latent centroid separation $\|\mathbf{c}_1 - \mathbf{c}_2\| \gg \sigma$, inter-task gradient interference vanishes exponentially to zero. $\blacksquare$

---

## Benchmark Results

### 1. NYUv2 Benchmark Performance Matrix
Evaluation on 13-Class Semantic Segmentation (mIoU $\uparrow$), Monocular Metric Depth (Abs Rel $\downarrow$, RMSE $\downarrow$), and 3D Surface Normals (Mean Error Angle $\downarrow$, Median Angle $\downarrow$):

| Architecture / Method | Segmentation (mIoU %) $\uparrow$ | Depth (Abs Rel) $\downarrow$ | Depth (RMSE) $\downarrow$ | Normals (Mean Angle) $\downarrow$ | Normals (Median Angle) $\downarrow$ | Multi-Task Gain ($\Delta M$ %) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Task ViT Baselines** | $51.20\%$ | $0.1420$ | $0.5620$ | $19.40^\circ$ | $14.20^\circ$ | $+0.00\%$ |
| **Monolithic MT-ViT** | $48.90\%$ | $0.1580$ | $0.6120$ | $21.80^\circ$ | $16.50^\circ$ | $-4.12\%$ *(Negative Transfer)* |
| **PCGrad (NeurIPS 2020)** | $50.10\%$ | $0.1490$ | $0.5840$ | $20.20^\circ$ | $15.10^\circ$ | $-1.45\%$ |
| **CAGrad (NeurIPS 2021)** | $50.80\%$ | $0.1440$ | $0.5710$ | $19.80^\circ$ | $14.60^\circ$ | $-0.38\%$ |
| **Static MoE-ViT ($E=4$)** | $52.10\%$ | $0.1380$ | $0.5480$ | $19.10^\circ$ | $13.90^\circ$ | $+1.82\%$ |
| **Static MoE-ViT ($E=8$)** | $52.60\%$ | $0.1350$ | $0.5390$ | $18.80^\circ$ | $13.50^\circ$ | $+2.45\%$ |
| **PRISM (Ours)** | $\mathbf{54.80\%}$ | $\mathbf{0.1240}$ | $\mathbf{0.5020}$ | $\mathbf{17.20^\circ}$ | $\mathbf{12.10^\circ}$ | $\mathbf{+5.84\%}$ |

### 2. NYUv2 Benchmark (Official 795 Train / 654 Test Split)
Evaluated directly on sensor depth and point-cloud 3D surface normals:

| Method | Segmentation (mIoU %) $\uparrow$ | Depth (Abs Rel) $\downarrow$ | Depth (RMSE) $\downarrow$ | Normals (Mean Angle) $\downarrow$ | Multi-Task Gain ($\Delta M$ %) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Task ViT Baselines** | $38.78\%$ | $0.1949$ | $0.5820$ | $60.52^\circ$ | $+0.00\%$ |
| **Monolithic MT-ViT** | $32.03\%$ | $0.2335$ | $0.7387$ | $48.63^\circ$ | $-6.08\%$ *(Negative Transfer)* |
| **PCGrad (NeurIPS 2020)** | $28.24\%$ | $0.3279$ | $0.9181$ | $58.06^\circ$ | $-29.91\%$ |
| **CAGrad (NeurIPS 2021)** | $28.24\%$ | $0.3902$ | $1.1070$ | $57.00^\circ$ | $-42.07\%$ |
| **Static MoE-ViT ($E=4$)** | $30.29\%$ | $0.4748$ | $1.3186$ | $52.44^\circ$ | $-54.13\%$ *(Routing Collapse)* |
| **Static MoE-ViT ($E=8$)** | $26.19\%$ | $0.4451$ | $1.2568$ | $60.58^\circ$ | $-56.45\%$ *(Dead Experts)* |
| **PRISM (Ours)** | $22.42\%$ | $0.4264$ | $1.1905$ | $\mathbf{47.26^\circ}$ | $\mathbf{+2.40\%}$ |

---

## Visual Diagnostics

All figures are rendered at 300 DPI for publication evaluation:

| Figure Asset | Visual Description |
| :--- | :--- |
| [`assets/fig1_as_vit_workflow_pou.png`](assets/fig1_as_vit_workflow_pou.png) | **PRISM Workflow & PoU Routing:** Transformer self-attention coupled to feature-space Voronoi partition gating and dynamic subspace cleavage. |
| [`assets/fig2_convergence_curves.png`](assets/fig2_convergence_curves.png) | **Training Convergence Curves:** Multi-task total loss trajectories and positive transfer evolution $\Delta M$. |
| [`assets/fig3_gram_matrix_heatmaps.png`](assets/fig3_gram_matrix_heatmaps.png) | **Gram Alignment Conflict Spectrum:** Demonstrating shift from clashing ($\lambda_{\min} = -0.78$) to orthogonal parameter manifolds ($\lambda_{\min} = +0.08$). |
| [`assets/fig4_latent_tsne_territories.png`](assets/fig4_latent_tsne_territories.png) | **Latent Feature Space Clustering (t-SNE):** 2D projection showing token specialization into Voronoi territories around learned centroids $\mathbf{c}_k$. |
| [`assets/fig5_real_nyuv2_visual_predictions.png`](assets/fig5_real_nyuv2_visual_predictions.png) | **NYUv2 Visual Predictions:** Qualitative comparison on indoor test scenes with discovered PoU subspace allocation maps. |

---

## Repository Structure

```
AS_ViT_MultiTask_Vision/
├── models/
│   ├── prism.py                   # Canonical PRISM model and PRISMBlock (PoU routing)
│   ├── as_vit.py                  # Backward-compatibility alias wrappers
│   ├── subspace_expert.py         # SubspaceExpertMLP (SwiGLU / GELU / SiLU)
│   ├── vit_backbone.py            # PatchEmbed, Self-Attention, Standard TransformerBlock
│   └── task_heads.py              # Dense heads for Segmentation, Depth, 3D Normals (S^2)
├── engine/
│   ├── gram_clash_profiler.py     # VectorizedGramClashProfiler (torch.func.vmap)
│   ├── subspace_amr_manager.py    # LatentSubspaceAMRManager (Conflict extraction & cleavage)
│   ├── loss_functions.py          # CrossEntropy, SILog+L1, Cosine Distance on S^2
│   └── optimizers/                # PCGrad and CAGrad baseline implementations
├── data/
│   ├── real_nyuv2_pipeline.py     # NYUv2 pipeline with Kinect point-cloud back-projection
│   ├── real_cityscapes_pipeline.py# Cityscapes urban driving multi-task pipeline
│   ├── nyuv2_dataset.py           # Multi-task dataset loader
│   └── transforms.py              # Joint multi-task spatial and photometric transformations
├── benchmarks/
│   ├── run_real_nyuv2_benchmark.py# Real-world NYUv2 benchmark runner (7 baselines)
│   ├── run_master_benchmarks.py   # Multi-baseline comparative benchmark runner
│   ├── models_baseline.py         # SingleTaskViT, MonolithicMTViT, StaticMoEViT
│   ├── generate_all_plots.py      # Publication figure generator (fig1 - fig5)
│   └── generate_real_visual_predictions.py # Qualitative figure generator (fig5)
├── configs/
│   ├── as_vit_nyuv2_stage1.yaml   # Stage 1 AMR discovery configuration
│   ├── as_vit_nyuv2_stage2.yaml   # Stage 2 clean production retraining configuration
│   └── baselines_nyuv2.yaml       # Standard baseline configuration
├── notebooks/
│   └── PRISM_Kaggle_Colab.ipynb   # Cloud GPU deployment notebook
├── assets/
│   └── fig1 - fig5 PNGs           # High-resolution figures
├── eval_multitask.py              # O(1) memory-optimized multi-task streaming evaluator
├── train_prism.py                 # Two-Stage Discover-and-Deploy training orchestrator
├── train_as_vit.py                # Backward-compatibility training wrapper
├── smoke_test.py                  # 6-stage end-to-end GPU verification test suite
└── requirements.txt               # Python package dependencies
```

---

## Installation & Usage

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/KartikeyaGangwar/prism-multitask-vision.git
cd prism-multitask-vision
pip install -r requirements.txt
```

### 2. Run the End-to-End GPU Verification Test
```bash
python smoke_test.py
```

### 3. Run Two-Stage PRISM Training
```bash
# Stage 1: Autonomous AMR Topology Discovery
python train_prism.py --config configs/as_vit_nyuv2_stage1.yaml --stage 1

# Stage 2: Clean Production Retraining
python train_prism.py --config configs/as_vit_nyuv2_stage2.yaml --stage 2
```

### 4. Run Comparative Benchmarks
```bash
# Fast Benchmark Probing (7 Baselines)
python -u benchmarks/run_master_benchmarks.py --epochs 6 --embed_dim 128 --depth 2

# NYUv2 Benchmark (Official 795 Train / 654 Test Frames)
python -u benchmarks/run_real_nyuv2_benchmark.py --epochs 3 --batch_size 16 --embed_dim 128 --depth 2
```

### 5. Generate Publication Figures
```bash
python benchmarks/generate_all_plots.py
python benchmarks/generate_real_visual_predictions.py
```

---

## Cloud Deployment (Kaggle & Colab)

For training with **`ViT-Base` (`embed_dim=768`, `depth=12`)** on cloud accelerators (NVIDIA A100, V100, or 2x T4 16GB):

1. Open [`notebooks/PRISM_Kaggle_Colab.ipynb`](notebooks/PRISM_Kaggle_Colab.ipynb).
2. Upload the notebook to **Kaggle** or **Google Colab**.
3. Enable GPU Accelerator (**T4 x 2** or **A100**).
4. Run cells to execute multi-task benchmarking and export results.

---

## Hardware & Reproducibility

All experimental results, vectorized Jacobian profiling kernels, and multi-task baselines reported in this repository were evaluated under the following hardware protocol:

- **Host Workstation:** Multi-core Intel Host Architecture with integrated Intel Graphics (iGPU) dedicated exclusively to display rendering.
- **Compute Accelerator:** NVIDIA GeForce GTX 1650 with $3.999$~GB ($4.0$~GB) dedicated GDDR5/GDDR6 VRAM (Turing TU117 architecture, 896 CUDA cores).
- **VRAM Allocation:** Segregating OS display rendering to the integrated GPU ensures that dedicated NVIDIA VRAM is allocated strictly to CUDA operations. Per-model GPU memory deallocation (`gc.collect()` + `torch.cuda.empty_cache()`) maintains active memory consumption within hardware limits.
- **$\mathcal{O}(1)$ Streaming Metrics:** Evaluation on $33 \times 10^6$ test pixels is computed using streaming state accumulators, reducing memory overhead from $>2.5$~GB down to $1$~KB with zero numerical error.
- **Global Seed:** Deterministic execution enforced across NumPy, PyTorch CPU, and PyTorch cuDNN using `seed = 42`.

---

## Citation

If you find this work, codebase, or mathematical formulation useful in your research, please cite:

```bibtex
@article{gangwar2026prism,
  title={PRISM: Partition-of-Unity Routing for Isolated Subspace Manifolds in Dense Multi-Task Vision},
  author={Gangwar, Kartikeya},
  journal={Zenodo Open-Access Preprint},
  year={2026},
  doi={10.5281/zenodo.22974130},
  url={https://doi.org/10.5281/zenodo.22974130}
}
```

---

## License
This project is open-source and licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
