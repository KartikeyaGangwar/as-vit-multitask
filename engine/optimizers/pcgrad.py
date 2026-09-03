"""
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning (Yu et al., NeurIPS 2020).
"""
import copy
import random
from typing import List, Optional
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class PCGrad:
    """
    PCGrad Optimizer Wrapper.
    
    Given T task losses L_1, ..., L_T, computes task gradients g_1, ..., g_T w.r.t shared parameters.
    If g_i . g_j < 0, projects g_i onto the normal plane of g_j:
      g_i = g_i - (g_i . g_j / ||g_j||^2) g_j.
    """
    def __init__(self, optimizer: Optimizer, reduction: str = "mean"):
        self.optimizer = optimizer
        self.reduction = reduction

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def pc_backward(self, objectives: List[torch.Tensor]):
        """
        Calculates gradient of each objective, projects conflicting components,
        and aggregates into parameter .grad attributes.
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads: List[torch.Tensor], has_grads: List[List[bool]]) -> torch.Tensor:
        shared = torch.stack(grads)
        num_tasks = len(grads)
        task_order = list(range(num_tasks))

        for i in range(num_tasks):
            task_i = task_order[i]
            random.shuffle(task_order)
            for j in task_order:
                task_j = j
                if task_i == task_j:
                    continue
                g_i = shared[task_i]
                g_j = shared[task_j]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    shared[task_i] -= (g_i_g_j / (g_j.norm() ** 2 + 1e-8)) * g_j

        if self.reduction == "mean":
            return shared.mean(dim=0)
        else:
            return shared.sum(dim=0)

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
