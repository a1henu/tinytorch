from __future__ import annotations

from .modules import Module
from ..tensor import Tensor
from .. import funcs


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return funcs.Softmax()(x)