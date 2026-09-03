"""
CAGrad: Conflict-Averse Gradient Descent for Multi-Task Learning (Liu et al., NeurIPS 2021).
"""
import numpy as np
from typing import List, Optional, Tuple
import torch
from scipy.optimize import minimize
from torch.optim.optimizer import Optimizer


class CAGrad:
    r"""
    CAGrad Optimizer Wrapper.
    Finds a conflict-averse gradient update vector g_0 within a dual cone parameterized by c \in [0, 1).
    """
    def __init__(self, optimizer: Optimizer, c: float = 0.5, eps: float = 1e-8):
        self.optimizer = optimizer
        self.c = c
        self.eps = eps

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def cagrad_backward(self, objectives: List[torch.Tensor]):
        """Computes CAGrad aggregated gradient direction."""
        grads, shapes, has_grads = self._pack_grad(objectives)
        GG = torch.stack(grads) # [T, P]
        
        g_cagrad = self._solve_cagrad(GG)
        g_unflat = self._unflatten_grad(g_cagrad, shapes[0])
        self._set_grad(g_unflat)

    def _solve_cagrad(self, GG: torch.Tensor) -> torch.Tensor:
        r"""Solves dual optimization problem for CAGrad."""
        # GG: [T, P]
        T = GG.shape[0]
        g0 = GG.mean(dim=0) # [P]
        
        # Gram matrix: M = GG @ GG.T
        M = torch.mm(GG, GG.t()).cpu().numpy().astype(np.float64)
        g0_norm = torch.norm(g0).item()
        
        if g0_norm < 1e-8:
            return g0

        # Solve dual: min w^T M w  s.t. \sum w_i = 1, w_i >= 0
        w_init = np.ones(T) / T
        bounds = [(0, 1) for _ in range(T)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            return 0.5 * np.dot(w, np.dot(M, w))

        res = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        w_opt = torch.tensor(res.x, dtype=GG.dtype, device=GG.device)
        
        # Dual-cone scaling
        gw = torch.sum(w_opt.unsqueeze(1) * GG, dim=0) # [P]
        gw_norm = torch.norm(gw).item()
        
        c = self.c
        if gw_norm > 1e-8:
            scale = (g0_norm / (gw_norm + 1e-8)) * c
            g_cagrad = g0 + scale * gw
        else:
            g_cagrad = g0
            
        return g_cagrad

    def _pack_grad(self, objectives: List[torch.Tensor]):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self.optimizer.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(has_grad)
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = torch.prod(torch.tensor(shape)).item()
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads: List[torch.Tensor], shapes: List[torch.Size]) -> torch.Tensor:
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p))
                    has_grad.append(False)
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(True)
        return grad, shape, has_grad

    def _set_grad(self, grads: List[torch.Tensor]):
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1
