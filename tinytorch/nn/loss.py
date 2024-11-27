from __future__ import annotations

from .modules import Module
from ..tensor import Tensor
from .. import funcs


class MSELoss(Module):
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return funcs.MSE(target)(x)
    

class CrossEntropyLoss(Module):
    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        return funcs.CrossEntropy(labels)(x)
