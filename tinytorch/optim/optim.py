from __future__ import annotations
from typing import List

from ..tensor import Tensor

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = Tensor.full(param.shape, 0)