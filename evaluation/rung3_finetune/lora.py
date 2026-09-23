"""LoRA (Hu et al. 2022) for the ``nn.Linear`` layers of any encoder under the HALO contract.

Generic by design: the same wrapper is applied to HALO's temporal/cross-channel attention and
feed-forward projections and to the released trunks' linear layers, so "LoRA" means one thing
across the comparison. The base weight is frozen; ``B`` starts at zero so the wrapped module is
exactly the original at step 0.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype,
                                               device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + update

    def merged(self) -> nn.Linear:
        """An ordinary Linear with the adapter folded into the weight (for export / checks)."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scaling)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device)


def apply_lora(module: nn.Module, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               include: Callable[[str, nn.Linear], bool] | None = None) -> list[str]:
    """Freeze ``module`` and wrap every eligible ``nn.Linear`` in place; return the wrapped names."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    wrapped: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and (include is None or include(qualified, child)):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                wrapped.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.lora_a, m.lora_b)]


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
