from __future__ import annotations
from typing import List, overload

import numpy as np 
from numpy.typing import NDArray

from ._libtensor import DeviceType as _DeviceType, Tensor as _Tensor

class DeviceType(_DeviceType):
    ...
    
class Tensor(_Tensor):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType, data: List[float]) -> None: ...
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def cpu(self) -> Tensor:
        return super().cpu()
    
    def gpu(self) -> Tensor:
        return super().gpu()
    
    def to_cpu(self) -> None:
        super().to_cpu()
    
    def to_gpu(self) -> None:
        super().to_gpu()
    
    def in_cpu(self) -> bool:
        return super().in_cpu()
    
    def in_gpu(self) -> bool:
        return super().in_gpu()
    
    def dim(self) -> int:
        return super().dim()
    
    def shape(self) -> List[int]:
        return super().shape()
    
    def reshape(self, shape: List[int]) -> Tensor:
        return super().reshape(shape)
    
    def transpose(self) -> Tensor:
        return super().transpose()
    
    def size(self) -> int:
        return super().size()
    
    def __add__(self, other: Tensor) -> Tensor:
        return super().__add__(other)
    
    def __sub__(self, other: Tensor) -> Tensor:
        return super().__sub__(other)
    
    def __matmul__(self, other: Tensor) -> Tensor:
        return super().__matmul__(other)
    
    def __eq__(self, other: Tensor) -> bool:
        if not isinstance(other, Tensor):
            return False
        return super().__eq__(other)
    
    def __getitem__(self, *args) -> Tensor:
        return super().__getitem__(list(args))
    
    def __repr__(self) -> str:
        return super().__repr__()
    
    def __str__(self) -> str:
        return super().__str__()
    
    @staticmethod
    def from_numpy(array: NDArray) -> Tensor:
        array = array.astype(np.float64, order="C")
        return _Tensor.from_numpy(array)
    
    def to_numpy(self) -> NDArray:
        return super().to_numpy()
    
    @staticmethod
    def ones(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor:
        return _Tensor.ones(shape, device)
    
    @staticmethod
    def randn(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor:
        return _Tensor.from_numpy(np.random.randn(*shape), device)