from __future__ import annotations
from typing import List, overload

import numpy as np 
from numpy.typing import NDArray

from ._libtensor import DeviceType as _DeviceType, Tensor as _Tensor

class DeviceType(_DeviceType):
    """
    A device type class.
    
    Attributes:
        CPU (DeviceType): The CPU device.
        GPU (DeviceType): The GPU device.
    """
    CPU = _DeviceType.CPU
    GPU = _DeviceType.GPU
    
    
class Tensor(_Tensor):
    """
    A tensor class.
    
    Attributes:
        shape (List[int]): The shape of the tensor.
        device (DeviceType): The device of the tensor.
        data (List[float]): The data of the tensor.
        
    Methods:
        cpu(): Move the tensor to CPU and return a new tensor.
        gpu(): Move the tensor to GPU and return a new tensor.
        to_cpu(): Move the tensor to CPU.
        to_gpu(): Move the tensor to GPU.
        in_cpu(): Check if the tensor is on CPU.
        in_gpu(): Check if the tensor is on GPU.
        
        dim(): Get the dimension of the tensor.
        shape(): Get the shape of the tensor.
        reshape(shape: List[int]) -> Tensor: Reshape the tensor to the given shape.
        transpose(): Transpose the tensor.
        size(): Get the total size of the tensor.
        
        __add__(other: Tensor) -> Tensor: Add another tensor to the current tensor.
        __sub__(other: Tensor) -> Tensor: Subtract another tensor from the current tensor.
        __matmul__(other: Tensor) -> Tensor: Matrix multiplication of two tensors.
        __eq__(other: Tensor) -> bool: Check if the current tensor is equal to another tensor.
        __getitem__(args) -> float: Get item from the tensor.
        __repr__(): Get the string representation of the tensor.
        __str__(): Get the string representation of the tensor.
        
        from_numpy(array: NDArray) -> Tensor: Create a tensor from a numpy array.
        to_numpy() -> NDArray: Convert the tensor to a numpy array.
        ones(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor: Create a tensor with all elements set to 1.
        randn(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor: Create a tensor with elements drawn from a normal distribution.
        
    Examples:
        >>> t = Tensor([2, 3], DeviceType.CPU)
        >>> t_cpu = t.cpu()
        >>> t_gpu = t.gpu()
        >>> t.to_cpu()
        >>> t.to_gpu()
        
        >>> a = Tensor.randn([2, 3], DeviceType.CPU)
        >>> b = Tensor.randn([2, 3], DeviceType.CPU)
        >>> c = a + b
        >>> d = a * b
        >>> e = a @ b.transpose()
    """
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType, data: List[float]) -> None: ...
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def cpu(self) -> Tensor:
        """
        Move the tensor to CPU and return a new tensor.
        
        Returns:
            Tensor: The tensor on CPU.
        """
        return super().cpu()
    
    def gpu(self) -> Tensor:
        """
        Move the tensor to GPU and return a new tensor.
        
        Returns:
            Tensor: The tensor on GPU.
        """
        return super().gpu()
    
    def to_cpu(self) -> None:
        """
        Move the tensor to CPU.
        
        Returns:
            None
        """
        super().to_cpu()
    
    def to_gpu(self) -> None:
        """
        Move the tensor to GPU.
        
        Returns:
            None
        """
        super().to_gpu()
    
    def in_cpu(self) -> bool:
        """
        Check if the tensor is on CPU.
        
        Returns:
            bool: True if the tensor is on CPU, False otherwise.
        """
        return super().in_cpu()
    
    def in_gpu(self) -> bool:
        """
        Check if the tensor is on GPU.
        
        Returns:
            bool: True if the tensor is on GPU, False otherwise.
        """
        return super().in_gpu()
    
    def dim(self) -> int:
        """
        Get the dimension of the tensor.
        
        Returns:
            int: The dimension of the tensor.
        """
        return super().dim()
    
    def shape(self) -> List[int]:
        """
        Get the shape of the tensor.
        
        Returns:
            List[int]: The shape of the tensor.
        """
        return super().shape()
    
    def reshape(self, shape: List[int]) -> Tensor:
        """
        Reshape the tensor to the given shape.
        
        Parameters:
            shape (List[int]): The shape to reshape the tensor to.
        
        Returns:
            Tensor: The reshaped tensor.
        """
        return super().reshape(shape)
    
    def transpose(self) -> Tensor:
        """
        Transpose the tensor.
        
        Returns:
            Tensor: The transposed tensor.
        """
        return super().transpose()
    
    def size(self) -> int:
        """
        Get the total size of the tensor.
        
        Returns:
            int: The size of the tensor.
        """
        return super().size()
    
    def __add__(self, other: Tensor) -> Tensor:
        """
        Add another tensor to the current tensor.
        
        Returns:
            Tensor: The result of addition.
        """
        return super().__add__(other)
    
    def __sub__(self, other: Tensor) -> Tensor:
        """
        Subtract another tensor from the current tensor.
        
        Returns:
            Tensor: The result of subtraction.
        """
        return super().__sub__(other)
    
    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Matrix multiplication of two tensors.
        
        Returns:
            Tensor: The result of matrix multiplication.
        """
        return super().__matmul__(other)
    
    def __eq__(self, other: Tensor) -> bool:
        """
        Check if the current tensor is equal to another tensor.
        
        Returns:
            bool: True if the tensors are equal, False otherwise.
        """
        if not isinstance(other, Tensor):
            return False
        return super().__eq__(other)
    
    def __getitem__(self, *args) -> float:
        """
        Get item from the tensor.
        
        Parameters:
            args: The indices of the item.
        
        Returns:
            float: The item.
        """
        return super().__getitem__(list(args))
    
    def __repr__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return super().__repr__()
    
    def __str__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return super().__str__()
    
    @staticmethod
    def from_numpy(array: NDArray) -> Tensor:
        """
        Create a tensor from a numpy array.
        
        Returns:
            Tensor: The tensor.
        """
        array = array.astype(np.float64, order="C")
        return _Tensor.from_numpy(array)
    
    def to_numpy(self) -> NDArray:
        """
        Convert the tensor to a numpy array.
        
        Returns:
            NDArray: The numpy array.
        """
        return super().to_numpy()
    
    @staticmethod
    def ones(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor:
        """
        Create a tensor with all elements set to 1.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device of the tensor.
            
        Returns:
            Tensor: The tensor with all elements set to 1.
        """
        return _Tensor.ones(shape, device)
    
    @staticmethod
    def randn(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor:
        """
        Create a tensor with elements drawn from a normal distribution.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device of the tensor.
        
        Returns:
            Tensor: The tensor with elements drawn from a normal distribution.
        """
        return _Tensor.from_numpy(np.random.randn(*shape), device)
