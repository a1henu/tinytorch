from __future__ import annotations

from .modules import Module
from .linear import Linear
from .relu import ReLU
from .conv import Conv2d
from .pool import MaxPool2d

__all__ = ['Module', 'Linear', 'ReLU', 'Conv2d', 'MaxPool2d']
# TODO: Implement Sigmoid and Softmax functions
# TODO: Implement Softmax backward calculation