from __future__ import annotations
from typing import List, Tuple

from .optim import Optimizer
from ..tensor import Tensor

class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [Tensor.zeros(param.shape, param.device, param.requires_grad) for param in params]
        self.v = [Tensor.zeros(param.shape, param.device, param.requires_grad) for param in params]

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for param, m, v in zip(self.params, self.m, self.v):
            if param.grad is not None:
                grad = param.grad.data

                m.data = beta1 * m.data + (1 - beta1) * grad
                v.data = beta2 * v.data + (1 - beta2) * grad**2

                m_hat = m.data / (1 - beta1**self.t)
                v_hat = v.data / (1 - beta2**self.t)

                param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)