from __future__ import annotations
from typing import List

from .optim import Optimizer
from ..tensor import Tensor

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [Tensor.zeros(param.shape, param.device, param.requires_grad) for param in params]

    def step(self):
        for param, velocity in zip(self.params, self.velocities):
            if param.grad is not None:
                velocity.data = self.momentum * velocity.data + self.lr * param.grad.data
                param.data -= velocity.data