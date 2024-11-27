from __future__ import annotations
from typing import Tuple
from math import sqrt

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class Conv2d(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        limit = sqrt(6 / (in_channels * kernel_size[0] * kernel_size[1] + out_channels))
        self.weight = limit * Tensor.randn([out_channels, in_channels, kernel_size[0], kernel_size[1]], requires_grad=True)
        self.bias = Tensor.zeros([out_channels], requires_grad=True)
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return funcs.Conv2d(self.padding, self.stride)(x, self.weight, self.bias)