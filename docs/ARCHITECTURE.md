# AS-ViT: Architecture & System Design Specification

This document provides the formal architectural blueprint and computational graph specification for the **Adaptive Subspace Vision Transformer (AS-ViT)** framework.

---

## 🏗️ 1. Multi-Task Vision Transformer Pipeline

Given an input RGB scene $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, AS-ViT tokenizes the image into non-overlapping spatial patches of resolution $P \times P$ (typically $16 \times 16$):

$$\mathbf{x}_p = \text{Flatten}(\mathbf{I}_{i, j}) \in \mathbb{R}^{3 P^2}, \quad p \in \{1, \dots, M\}, \quad M = \frac{HW}{P^2}.$$

A linear projection with learnable 1D position embeddings maps the patches into a sequence of latent tokens:

$$\mathbf{Z}_0 = [\mathbf{x}_1 \mathbf{W}_E; \mathbf{x}_2 \mathbf{W}_E; \dots; \mathbf{x}_M \mathbf{W}_E] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{M \times D},$$

where $D$ is the embedding dimension (e.g., $128, 192, 384, 768$).

---

## 🧩 2. The Adaptive Subspace Block (`AdaptiveSubspaceBlock`)

In each transformer layer $l \in \{1, \dots, L\}$, the traditional monolithic Multi-Layer Perceptron (MLP) is replaced by an **Adaptive $N$-Subspace Module** containing $N$ dynamically allocated localized expert networks $\{\Phi_k(\cdot; \Theta_k)\}_{k=1}^N$:

$$\mathbf{Z}_l' = \mathbf{Z}_{l-1} + \text{MultiHeadAttention}(\text{LN}(\mathbf{Z}_{l-1})),$$

$$\mathbf{Z}_l = \mathbf{Z}_l' + \sum_{k=1}^N \psi_k(\mathbf{Z}_l') \odot \Phi_k\left( \frac{\text{LN}(\mathbf{Z}_l') - \mathbf{c}_k}{\sigma_k}; \Theta_k \right),$$

where:
- $\text{LN}(\cdot)$ is Layer Normalization.
- $\mathbf{c}_k \in \mathbb{R}^D$ is the learned or discovered latent centroid of subspace $k$.
- $\sigma_k \in \mathbb{R}^+$ is the local Voronoi bandwidth.
- $\odot$ denotes element-wise token modulation across embedding dimensions.

---

## 🌐 3. Feature-Space Partition of Unity (PoU) Gating

Unlike discrete Mixture-of-Experts (MoE) routers which select experts via non-differentiable $\text{Top-}k(\text{Softmax}(\mathbf{z} \mathbf{W}_g))$ routing, AS-ViT enforces a continuous **Partition of Unity (PoU)** over the latent representation manifold:

$$\psi_k(\mathbf{z}) = \frac{\exp\left( - \frac{\|\mathbf{z} - \mathbf{c}_k\|_2^2}{2 \sigma_k^2 \sqrt{D} + \epsilon} \right)}{\sum_{j=1}^N \exp\left( - \frac{\|\mathbf{z} - \mathbf{c}_j\|_2^2}{2 \sigma_j^2 \sqrt{D} + \epsilon} \right)}.$$

### PoU Axioms Satisfied:
1. **Positivity:** $\psi_k(\mathbf{z}) \ge 0, \quad \forall \mathbf{z} \in \mathbb{R}^D, \forall k \in \{1, \dots, N\}$.
2. **Strict Normalization:** $\sum_{k=1}^N \psi_k(\mathbf{z}) \equiv 1.0, \quad \forall \mathbf{z} \in \mathbb{R}^D$.
3. **Smooth Differentiability:** $\psi_k \in C^\infty(\mathbb{R}^D)$.
4. **Dimension Scaling ($\sqrt{D}$):** In high-dimensional latent space ($\mathbb{R}^{128} \dots \mathbb{R}^{768}$), Euclidean distances $\|\mathbf{z}-\mathbf{c}\|^2$ scale with $D$. Scaling the variance denominator by $\sqrt{D}$ prevents softmax saturation and vanishing gradients.

---

## 🎯 4. Localized Coordinate Centering

Prior to feeding token $\mathbf{z}$ into expert $\Phi_k$, AS-ViT transforms $\mathbf{z}$ into the **local centered frame of that subspace**:

$$\tilde{\mathbf{z}}_k = \frac{\text{LN}(\mathbf{z}) - \mathbf{c}_k}{\sigma_k}.$$

### Mathematical Rationale:
- Centering $\mathbf{z}$ around $\mathbf{c}_k$ forces the expert neural network $\Phi_k$ to specialize on local perturbations within its semantic basin.
- This conditions the local Hessian matrix $\nabla_{\Theta_k}^2 \mathcal{L}$, avoiding ill-conditioned ellipsoidal level sets and accelerating gradient descent convergence.

---

## 🔬 5. Subspace Expert Architectures (`SubspaceExpertMLP`)

Each parameter subspace $\Phi_k$ is implemented with modern, high-throughput transformer FFN variants:

### A. SwiGLU Variant (Default for High Capacity)
$$\Phi_k(\mathbf{x}) = \left( \text{Swish}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}}) \right) \mathbf{W}_{\text{down}},$$
where $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{D \times \frac{8}{3}D}$ and $\mathbf{W}_{\text{down}} \in \mathbb{R}^{\frac{8}{3}D \times D}$.

### B. Standard GELU / SiLU MLP Variant
$$\Phi_k(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2,$$
where $\mathbf{W}_1 \in \mathbb{R}^{D \times 4D}$ and $\mathbf{W}_2 \in \mathbb{R}^{4D \times D}$.

---

## 👁️ 6. Task-Specific Dense Decoders (`task_heads.py`)

To evaluate diverse vision objectives, AS-ViT routes the final transformer representations $\mathbf{Z}_L \in \mathbb{R}^{B \times M \times D}$ to three decoupled task heads:

```
                  ┌────────────────────────────────────────┐
                  │  Final Transformer Layer Z_L           │
                  │  Shape: [B, M, D]                      │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│ 1. Segmentation   │       │ 2. Monocular Depth│       │ 3. Surface Normals│
│    Head           │       │    Head           │       │    Head           │
├───────────────────┤       ├───────────────────┤       ├───────────────────┤
│ ConvTranspose2d   │       │ ConvTranspose2d   │       │ ConvTranspose2d   │
│ Cross-Entropy     │       │ Softplus (Depth>0)│       │ S^2 Normalization │
│ Output: [B,13,H,W]│       │ Output: [B,1,H,W] │       │ Output: [B,3,H,W] │
└───────────────────┘       └───────────────────┘       └───────────────────┘
```

1. **Semantic Segmentation Head:** Decodes tokens into class logits via progressive transposed convolutions, trained with cross-entropy loss.
2. **Monocular Metric Depth Head:** Regresses metric distance $Z \in (0, 10]\text{ m}$, strictly enforcing non-negativity via $\text{Softplus}(x) + \epsilon$, trained with combined SILog and L1 loss.
3. **3D Surface Normal Head:** Regresses continuous unit normal fields, strictly projecting outputs onto the 2-sphere $\mathbb{S}^2$ via $\ell_2$-normalization $\mathbf{n} = \mathbf{u} / \|\mathbf{u}\|_2$, trained with cosine distance loss $\mathcal{L}_{\text{normal}} = 1 - \langle \mathbf{n}_{\text{pred}}, \mathbf{n}_{\text{gt}} \rangle$.
