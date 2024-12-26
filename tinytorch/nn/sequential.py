from __future__ import annotations

from .modules import Module
from ..tensor import Tensor

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self._modules_list = []
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            self._modules_list.append(module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules_list:
            x = module(x)
        return x