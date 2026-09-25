"""LoRA (Hu et al. 2022) for the ``nn.Linear`` and ``nn.Conv1d``/``nn.Conv2d`` layers of any
encoder under the HALO contract.

Generic by design: the same wrapper is applied to HALO's attention and feed-forward projections
and to every released trunk, so "LoRA" means one thing across the comparison. Convolutions are
included because two of the released trunks (HARNet's ResNet, UniMTS's ST-GCN) are convolutional:
wrapping linear layers alone would adapt only their projection head and leave 100 % of the
pretrained trunk frozen (found in the 2026-09-25 debug sweep). A convolution's update is the usual
factorisation for convolutional LoRA: ``A`` is a rank-r convolution with the base kernel, stride,
padding and dilation, ``B`` a zero-initialised 1x1 convolution back to the output channels.
Grouped convolutions are not wrapped. The base weight is frozen; ``B`` starts at zero so the
wrapped module is exactly the original at step 0.
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


class LoRAConv(nn.Module):
    """LoRA for an ungrouped ``nn.Conv1d`` / ``nn.Conv2d``: ``base(x) + s * B(A(x))``."""

    def __init__(self, base: nn.Conv1d | nn.Conv2d, *, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        if base.groups != 1:
            raise ValueError("grouped convolutions are not wrapped")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        conv = type(base)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = conv(base.in_channels, self.rank, base.kernel_size, stride=base.stride,
                           padding=base.padding, dilation=base.dilation, bias=False,
                           padding_mode=base.padding_mode, **factory)
        self.lora_b = conv(self.rank, base.out_channels, 1, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


def apply_lora(module: nn.Module, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               include: Callable[[str, nn.Module], bool] | None = None) -> list[str]:
    """Freeze ``module`` and wrap every eligible ``nn.Linear`` and ungrouped ``nn.Conv1d``/``nn.Conv2d``
    in place; return the wrapped names."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    wrapped: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (LoRALinear, LoRAConv)):
                continue
            eligible = include is None or include(qualified, child)
            if isinstance(child, nn.Linear) and eligible:
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                wrapped.append(qualified)
            elif isinstance(child, (nn.Conv1d, nn.Conv2d)) and child.groups == 1 and eligible:
                setattr(parent, name, LoRAConv(child, rank=rank, alpha=alpha))
                wrapped.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            out.extend((m.lora_a, m.lora_b))
        elif isinstance(m, LoRAConv):
            out.extend((m.lora_a.weight, m.lora_b.weight))
    return out


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
