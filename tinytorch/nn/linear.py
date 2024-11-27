from __future__ import annotations
from math import sqrt

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier initialization
        limit = sqrt(6 / (in_features + out_features))
        self.weight = limit * Tensor.randn([in_features, out_features], requires_grad=True)
        self.bias = Tensor.zeros([1, out_features], requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return funcs.FC()(x, self.weight, self.bias)