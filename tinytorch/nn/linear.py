from __future__ import annotations

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Tensor.randn([in_features, out_features], requires_grad=True)
        self.bias = Tensor.randn([1, out_features], requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return funcs.FC()(x, self.weight, self.bias)