from __future__ import annotations
from typing import Tuple

from .modules import Module
from ..tensor import Tensor
from .. import funcs

class MaxPool2d(Module):
    def __init__(
        self, 
        kernel_size: Tuple[int, int] | int,
        stride: Tuple[int, int] | int | None = None,
        padding: Tuple[int, int] | int = (0, 0),
    ) -> None:
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding
        
        self.kernel_size = kernel_size
        if stride is None: 
            self.stride = kernel_size
        else:
            self.stride = stride
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        return funcs.MaxPool2d(self.kernel_size, self.padding, self.stride)(x)
    