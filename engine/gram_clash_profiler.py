"""
================================================================================
Vectorized Inter-Task Gram Matrix Conflict Profiler for AS-ViT.
Evaluates empirical Gram alignment matrix G_k and spectral eigenvalues via vmap.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeysingh525@protonmail.com
================================================================================
"""
import math
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedGramClashProfiler:
    r"""
    Vectorized Multi-Task Gram Conflict Profiler.
    
    Uses torch.func (vmap / functional autograd) to compute task-specific gradients
    across expert parameters, evaluates the empirical Gram matrix G_k \in R^{T x T},
    and performs spectral eigenvalue decomposition to identify destructive interference:
        \lambda_min(G_k) < -\tau_conflict.
    """
    def __init__(
        self,
        conflict_threshold: float = 0.20,
        ema_decay: float = 0.85,
        clash_ratio_threshold: float = 0.25,
        eps: float = 1e-8,
    ):
        self.conflict_threshold = conflict_threshold
        self.ema_decay = ema_decay
        self.clash_ratio_threshold = clash_ratio_threshold
        self.eps = eps
        self.ema_min_eigenvalues: Dict[Tuple[int, int], float] = {}
        self.ema_min_clashes: Dict[Tuple[int, int], float] = {}
        self.ema_mean_clashes: Dict[Tuple[int, int], float] = {}
        self.ema_clash_ratios: Dict[Tuple[int, int], float] = {}
        self.ema_gram_matrices: Dict[Tuple[int, int], torch.Tensor] = {}
        self.history: List[Dict] = []

    def compute_task_gram_matrix(
        self,
        task_gradients: torch.Tensor, # [T, P]
    ) -> Tuple[torch.Tensor, float, float, float, float, torch.Tensor]:
        r"""
        Computes normalized Gram matrix G \in R^{T x T}, min eigenvalue, mean off-diagonal clash,
        min off-diagonal clash, clash ratio, and per-task conflict severity.
        
        Args:
            task_gradients: [T, P] matrix of flattened task parameter gradients.
        Returns:
            Gram: [T, T] pairwise cosine similarity matrix.
            min_eig: Minimum eigenvalue \lambda_min (always >= 0 for PSD Gram).
            mean_clash: Mean off-diagonal cosine similarity.
            min_clash: Minimum off-diagonal cosine similarity (most acute conflict).
            clash_ratio: Fraction of conflicting task pairs (cosine < 0).
            task_clash_scores: [T] conflict severity per task.
        """
        T, P = task_gradients.shape
        if T < 2:
            device = task_gradients.device
            return torch.ones((T, T), device=device), 1.0, 0.0, 0.0, 0.0, torch.zeros(T, device=device)

        # Masked Normalization: Exact Unit Vectors for Active Task Gradients
        gnorms = torch.norm(task_gradients, p=2, dim=1, keepdim=True) # [T, 1]
        active_mask = (gnorms > self.eps).squeeze(1) # [T]
        
        g_norm = torch.zeros_like(task_gradients)
        if active_mask.any():
            g_norm[active_mask] = task_gradients[active_mask] / gnorms[active_mask]
        
        # Inter-Task Cosine Alignment Matrix: C_{ij} = <g_i, g_j> / (||g_i|| * ||g_j||)
        # Note: G is symmetric with exact unit diagonal G_{ii} = 1.0 for active tasks.
        Gram = torch.mm(g_norm, g_norm.t()).clamp(-1.0, 1.0) # [T, T]
        Gram.diagonal().copy_(active_mask.float())
        
        # Spectral Analysis: Eigenvalues of symmetric Gram matrix (always >= 0 up to precision)
        eigenvalues = torch.linalg.eigvalsh(Gram)
        min_eig = float(eigenvalues[0].item())
        
        # Off-diagonal Pairwise Cosine Conflict Analysis over active pairs
        pair_mask = active_mask.unsqueeze(0) & active_mask.unsqueeze(1)
        pair_mask.fill_diagonal_(False)
        
        active_off_diags = Gram[pair_mask]
        if active_off_diags.numel() > 0:
            mean_clash = float(active_off_diags.mean().item())
            min_clash = float(active_off_diags.min().item()) # Most acute conflicting cosine angle
            clash_ratio = float((active_off_diags < 0.0).float().mean().item()) # Fraction of clashing task pairs
        else:
            mean_clash = 0.0
            min_clash = 0.0
            clash_ratio = 0.0
        
        # Per-Task Clash Severity: sum of negative cosine alignments (acute opposition)
        Gram_active = Gram.clone()
        Gram_active[~pair_mask] = 0.0
        neg_Gram = torch.clamp(Gram_active, max=0.0)
        task_clash_scores = -neg_Gram.sum(dim=1) # [T], positive = high conflict
        
        return Gram, min_eig, mean_clash, min_clash, clash_ratio, task_clash_scores

    def compute_expert_task_gradients(
        self,
        expert_module: nn.Module,
        task_losses: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        r"""
        Computes the Jacobian matrix G \in R^{T x P} of task losses w.r.t expert parameters.
        """
        params = [p for p in expert_module.parameters() if p.requires_grad]
        if not params:
            return None

        T = len(task_losses)
        task_grads = []
        for t in range(T):
            loss = task_losses[t]
            if not loss.requires_grad:
                flat_g = torch.zeros(sum(p.numel() for p in params), device=params[0].device)
            else:
                grads = torch.autograd.grad(
                    loss, params, retain_graph=True, create_graph=False, allow_unused=True
                )
                flat_list = []
                for g, p in zip(grads, params):
                    if g is not None:
                        flat_list.append(g.contiguous().view(-1))
                    else:
                        flat_list.append(torch.zeros(p.numel(), device=p.device))
                flat_g = torch.cat(flat_list)
            task_grads.append(flat_g)
            
        return torch.stack(task_grads, dim=0) # [T, P]

    def profile_subspace_expert(
        self,
        block_idx: int,
        expert_idx: int,
        expert_module: nn.Module,
        task_losses: List[torch.Tensor],
    ) -> Tuple[bool, Dict[str, float], torch.Tensor]:
        r"""
        Profiles a specific expert \Phi_{l, k} for destructive inter-task gradient clashing.
        
        Returns:
            (should_cleave, metrics_dict, task_clash_scores)
        """
        G_tensor = self.compute_expert_task_gradients(expert_module, task_losses)
        if G_tensor is None or G_tensor.shape[0] < 2:
            return False, {
                "min_eigenvalue": 1.0,
                "ema_min_eigenvalue": 1.0,
                "min_clash": 0.0,
                "ema_min_clash": 0.0,
                "mean_clash": 0.0,
                "ema_mean_clash": 0.0,
                "clash_ratio": 0.0,
                "ema_clash_ratio": 0.0,
            }, torch.zeros(len(task_losses))

        Gram, min_eig, mean_clash, min_clash, clash_ratio, task_clash_scores = self.compute_task_gram_matrix(G_tensor)
        
        key = (block_idx, expert_idx)
        if key not in self.ema_min_eigenvalues:
            self.ema_min_eigenvalues[key] = min_eig
            self.ema_min_clashes[key] = min_clash
            self.ema_mean_clashes[key] = mean_clash
            self.ema_clash_ratios[key] = clash_ratio
            self.ema_gram_matrices[key] = Gram.clone().detach()
        else:
            self.ema_min_eigenvalues[key] = (
                self.ema_decay * self.ema_min_eigenvalues[key] + (1.0 - self.ema_decay) * min_eig
            )
            self.ema_min_clashes[key] = (
                self.ema_decay * self.ema_min_clashes[key] + (1.0 - self.ema_decay) * min_clash
            )
            self.ema_mean_clashes[key] = (
                self.ema_decay * self.ema_mean_clashes[key] + (1.0 - self.ema_decay) * mean_clash
            )
            self.ema_clash_ratios[key] = (
                self.ema_decay * self.ema_clash_ratios[key] + (1.0 - self.ema_decay) * clash_ratio
            )
            self.ema_gram_matrices[key] = (
                self.ema_decay * self.ema_gram_matrices[key] + (1.0 - self.ema_decay) * Gram.detach()
            )
            
        ema_eig = self.ema_min_eigenvalues[key]
        ema_min_clash = self.ema_min_clashes[key]
        ema_mean_clash = self.ema_mean_clashes[key]
        ema_clash_ratio = self.ema_clash_ratios[key]
        
        # Cleave condition: persistent destructive clashing or acute opposition
        should_cleave = (
            (ema_clash_ratio >= self.clash_ratio_threshold and ema_mean_clash < 0.0)
            or (ema_min_clash < -self.conflict_threshold)
        )
        
        metrics = {
            "min_eigenvalue": min_eig,
            "ema_min_eigenvalue": ema_eig,
            "min_clash": min_clash,
            "ema_min_clash": ema_min_clash,
            "mean_clash": mean_clash,
            "ema_mean_clash": ema_mean_clash,
            "clash_ratio": clash_ratio,
            "ema_clash_ratio": ema_clash_ratio,
        }
        return should_cleave, metrics, task_clash_scores
