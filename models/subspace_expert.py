"""
Individual Subspace Expert Module for AS-ViT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SubspaceExpertMLP(nn.Module):
    r"""
    Subspace Multi-Layer Perceptron (MLP) Expert:
      \Phi_k( \tilde{z}_k; \Theta_k )
    
    Receives locally-centered and normalized token representations.
    Supports GELU, SiLU, and SwiGLU activation variants.
    """
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        act_layer: str = "gelu",
        drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.act_layer = act_layer.lower()
        self.drop_rate = drop

        if self.act_layer == "swiglu":
            # SwiGLU requires 2 * hidden_dim in projection
            self.fc1 = nn.Linear(embed_dim, 2 * self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        else:
            self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
            if self.act_layer == "silu":
                self.act = nn.SiLU()
            elif self.act_layer == "tanh":
                self.act = nn.Tanh()
            else:
                self.act = nn.GELU()

        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_layer == "swiglu":
            x_proj = self.fc1(x)
            x_gate, x_val = x_proj.chunk(2, dim=-1)
            x = F.silu(x_gate) * x_val
        else:
            x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
