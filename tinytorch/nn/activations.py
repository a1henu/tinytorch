from __future__ import annotations

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return funcs.ReLU()(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return funcs.Sigmoid()(x)