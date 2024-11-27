from __future__ import annotations

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class CrossEntropyLoss(Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return funcs.CrossEntropy(y)(x)
