from __future__ import annotations

from .modules import Module
from ..tensor import Tensor

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.modules = args

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x
    