"""Small PyTorch helpers shared by optional research notebooks."""

from __future__ import annotations

import torch


def gradient_calculator(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    """Differentiate ``output`` with respect to ``input_`` while retaining the graph."""

    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]

