from __future__ import annotations

from .modules import Module
from ..tensor import Tensor

class Flatten(Module):
    def __init__(self, start_idx = 1, end_idx = -1) -> None:
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_idx, self.end_idx)