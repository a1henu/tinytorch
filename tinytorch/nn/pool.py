from __future__ import annotations
from typing import Tuple

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class MaxPool2d(Module):
    def __init__(
        self, 
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        return funcs.MaxPool2d(self.kernel_size, self.padding, self.stride)(x)
    