from __future__ import annotations
from typing import List, Dict, overload

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
    
    
class TensorBase(_Tensor):
    """
    A class for storing and manipulating tensors.
    
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
        device(): Get the device of the tensor.
        
        dim(): Get the dimension of the tensor.
        shape(): Get the shape of the tensor.
        reshape(shape: List[int]) -> Tensor: Reshape the tensor to the given shape.
        transpose(): Transpose the tensor.
        size(): Get the total size of the tensor.
        __len__() -> int: Get the length of the tensor.
        
        __add__(other: Tensor) -> Tensor: Add another tensor to the current tensor.
        __sub__(other: Tensor) -> Tensor: Subtract another tensor from the current tensor.
        __matmul__(other: Tensor) -> Tensor: Matrix multiplication of two tensors.
        __mul__(other: float) -> Tensor: Scalar multiplication of the tensor.
        __rmul__(other: float) -> Tensor: Scalar multiplication of the tensor.
        __eq__(other: Tensor) -> bool: Check if the current tensor is equal to another tensor.
        __getitem__(args) -> float: Get item from the tensor.
        __repr__(): Get the string representation of the tensor.
        __str__(): Get the string representation of the tensor.
        
        to_numpy() -> NDArray: Convert the tensor to a numpy array.
        from_numpy(array: NDArray) -> Tensor: Create a tensor from a numpy array.
        save(filename: str, tensor: Tensor) -> None: Save the tensor to a numpy file.
        savez(filename: str, **kwargs) -> None: Save tensors to a compressed numpy file.
        load(filename: str) -> Tensor: Load the tensor from a numpy file.
        loadz(filename: str, *args) -> Dict[Tensor]: Load tensors from a compressed numpy file.
        zeros(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor: Create a tensor with all elements set to 0.
        ones(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor: Create a tensor with all elements set to 1.
        randn(shape: List[int], device: DeviceType = DeviceType.CPU) -> Tensor: Create a tensor with elements drawn from a normal distribution.
        
    Examples:
        >>> t = TensorBase([2, 3], DeviceType.CPU)
        >>> t_cpu = t.cpu()
        >>> t_gpu = t.gpu()
        >>> t.to_cpu()
        >>> t.to_gpu()
        
        >>> a = TensorBase.randn([2, 3], DeviceType.CPU)
        >>> b = TensorBase.randn([2, 3], DeviceType.CPU)
        >>> c = a + b
        >>> d = a * b
        >>> e = a @ b.transpose()
    """
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, _tensor: _Tensor) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType, data: List[float]) -> None: ...
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def __deepcopy__(self, memo) -> TensorBase:
        """
        Deepcopy the tensor.
        
        Returns:
            TensorBase: The copied tensor.
        """
        return TensorBase(super().__deepcopy__(memo))
        
    def cpu(self) -> TensorBase:
        """
        Move the tensor to CPU and return a new tensor.
        
        Returns:
            TensorBase: The tensor on CPU.
        """
        return TensorBase(super().cpu())
    
    def gpu(self) -> TensorBase:
        """
        Move the tensor to GPU and return a new tensor.
        
        Returns:
            TensorBase: The tensor on GPU.
        """
        return TensorBase(super().gpu())
    
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
    
    @property
    def device(self) -> DeviceType:
        """
        Get the device of the tensor.
        
        Returns:
            DeviceType: The device of the tensor.
        """
        return super().device()
    
    @property
    def dim(self) -> int:
        """
        Get the dimension of the tensor.
        
        Returns:
            int: The dimension of the tensor.
        """
        return super().dim()
    
    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the tensor.
        
        Returns:
            List[int]: The shape of the tensor.
        """
        return super().shape()
    
    def reshape(self, shape: List[int]) -> TensorBase:
        """
        Reshape the tensor to the given shape.
        
        Parameters:
            shape (List[int]): The shape to reshape the tensor to.
        
        Returns:
            TensorBase: The reshaped tensor.
        """
        return TensorBase(super().reshape(shape))
    
    def transpose(self) -> TensorBase:
        """
        Transpose the tensor.
        
        Returns:
            TensorBase: The transposed tensor.
        """
        return TensorBase(super().transpose())
    
    @property
    def size(self) -> int:
        """
        Get the total size of the tensor.
        
        Returns:
            int: The size of the tensor.
        """
        return super().size()
    
    def __len__(self) -> int:
        """
        Get the length of the tensor.
        
        Returns:
            int: The length of the tensor.
        """
        return super().__len__()
    
    def __add__(self, other) -> TensorBase:
        """
        Add another tensor to the current tensor.
        
        Returns:
            TensorBase: The result of addition.
        """
        if isinstance(other, np.ndarray):
            other = TensorBase.from_numpy(other, self.device)
        elif isinstance(other, int) or isinstance(other, float):
            other = TensorBase.full(self.shape, other, self.device)
        elif not isinstance(other, TensorBase):
            raise TypeError(f"Unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")
        return TensorBase(super().__add__(other))
    
    def __sub__(self, other) -> TensorBase:
        """
        Subtract another tensor from the current tensor.
        
        Returns:
            TensorBase: The result of subtraction.
        """
        if isinstance(other, np.ndarray):
            other = TensorBase.from_numpy(other, self.device)
        elif isinstance(other, int) or isinstance(other, float):
            other = TensorBase.full(self.shape, other, self.device)
        elif not isinstance(other, TensorBase):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        return TensorBase(super().__sub__(other))
    
    def __matmul__(self, other) -> TensorBase:
        """
        Matrix multiplication of two tensors.
        
        Returns:
            TensorBase: The result of matrix multiplication.
        """
        if isinstance(other, np.ndarray):
            other = TensorBase.from_numpy(other, self.device)
        elif not isinstance(other, TensorBase):
            raise TypeError(f"Unsupported operand type(s) for @: '{type(self).__name__}' and '{type(other).__name__}'")
        return TensorBase(super().__matmul__(other))
    
    def __mul__(self, other) -> TensorBase:
        """
        Scalar multiplication of the tensor.
        
        Returns:
            TensorBase: The result of scalar multiplication.
        """
        if isinstance(other, int) or isinstance(other, float):
            return TensorBase(super().__mul__(float(other)))
        elif isinstance(other, TensorBase):
            return TensorBase(super().ewise_mul(other))
    
    def __rmul__(self, other) -> TensorBase:
        """
        Scalar multiplication of the tensor.
        
        Returns:
            TensorBase: The result of scalar multiplication.
        """
        return TensorBase(super().__rmul__(other))
    
    def __truediv__(self, other) -> TensorBase:
        """
        Element-wise division of the tensor.
        
        Returns:
            TensorBase: The result of element-wise division.
        """
        if isinstance(other, int) or isinstance(other, float):
            return TensorBase(super().__mul__(float(1 / other)))
        elif isinstance(other, TensorBase):
            return TensorBase(super().ewise_mul(other**(-1)))
    
    def __pow__(self, scalar) -> TensorBase:
        """
        Element-wise power of the tensor.
        
        Returns:
            TensorBase: The result of element-wise power.
        """
        return TensorBase(super().power(scalar))
    
    def __eq__(self, other) -> bool:
        """
        Check if the current tensor is equal to another tensor.
        
        Returns:
            bool: True if the tensors are equal, False otherwise.
        """
        if isinstance(other, np.ndarray):
            other = TensorBase.from_numpy(other, self.device)
        elif not isinstance(other, TensorBase):
            return False
        return super().__eq__(other)
    
    def __assign__(self, other) -> TensorBase:
        """
        Assign another tensor to the current tensor.
        
        Returns:
            TensorBase: The current tensor.
        """
        if isinstance(other, np.ndarray):
            other = TensorBase.from_numpy(other, self.device)
        elif not isinstance(other, TensorBase):
            raise TypeError(f"Expected Tensor, got {type(other).__name__}")
        super().__assign__(other)
        return self
    
    def __getitem__(self, *args) -> float:
        """
        Get item from the tensor.
        
        Parameters:
            args: The indices of the item.
        
        Returns:
            float: The item.
        """
        if len(args) == 1 and isinstance(args[0], tuple):
            indices = list(args[0])  # flatten the tuple
        else:
            indices = list(args)
        
        return super().__getitem__(indices)
    
    def __repr__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return super().__repr__()[:100] + '...'
    
    def __str__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return super().__str__()
    
    def to_numpy(self) -> NDArray:
        """
        Convert the tensor to a numpy array.
        
        Returns:
            NDArray: The numpy array.
        """
        return super().to_numpy()
    
    @staticmethod
    def from_numpy(array: NDArray, device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Create a tensor from a numpy array.
        
        Parameters:
            array (NDArray): Input numpy array
            device (DeviceType): Target device (default: CPU)
            
        Returns:
            Tensor: New tensor on specified device
        """
        array = array.astype(np.float64, order="C")
        _t = _Tensor.from_numpy(array)
        if device == DeviceType.GPU: _t.to_gpu()
        return TensorBase(_t)
    
    @staticmethod
    def save(filename: str, tensor: TensorBase) -> None:
        """
        Save the tensor to a numpy file.
        
        Parameters:
            tensor (Tensor): The tensor to save.
            path (str): The path to the file.
        
        Returns:
            None
        """
        if not isinstance(tensor, TensorBase):
            raise TypeError(f"Expected Tensor, got {type(tensor).__name__}")
        np.save(filename, tensor.to_numpy())
    
    @staticmethod
    def savez(filename: str, **kwargs) -> None:
        """
        Save tensors to a compressed numpy file.
        
        Parameters:
            filename (str): The path to the file.
            kwargs: The tensors to save.
        
        Returns:
            None
        """
        for key, value in kwargs.items():
            if not isinstance(value, TensorBase):
                raise TypeError(f"Expected Tensor, got {type(value).__name__}")
            kwargs[key] = value.to_numpy()
        np.savez(filename, **kwargs)
    
    @staticmethod
    def load(filename: str, device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Load the tensor from a numpy file.
        
        Parameters:
            filename (str): The path to the file.
            device (DeviceType): The device of the tensor.
        
        Returns:
            Tensor: The loaded tensor.
        """
        array = np.load(filename)
        return TensorBase.from_numpy(array, device)
    
    @staticmethod
    def loadz(filename: str, device: DeviceType = DeviceType.CPU, *args) -> Dict[TensorBase]:
        """
        Load tensors from a compressed numpy file.
        
        Parameters:
            filename (str): The path to the file.
            device (DeviceType): The device of the tensors.
            args: The keys of the tensors to load.
        
        Returns:
            Dict[Tensor]: The loaded tensors.
        """
        tensors = {}
        with np.load(filename) as data:
            for key in args:
                tensors[key] = TensorBase.from_numpy(data[key], device)
        return tensors
    
    @staticmethod
    def full(shape: List[int], fill_value: float, device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Create a tensor with all elements set to a fill value.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            fill_value (float): The fill value.
            device (DeviceType): The device of the tensor.
            
        Returns:
            Tensor: The tensor with all elements set to the fill value.
        """
        return TensorBase.from_numpy(np.full(shape, fill_value), device)
    
    @staticmethod
    def zeros(shape: List[int], device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Create a tensor with all elements set to 0.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device of the tensor.
            
        Returns:
            Tensor: The tensor with all elements set to 0.
        """
        _t = _Tensor.zeros(shape, device)
        return TensorBase(_t)
    
    @staticmethod
    def ones(shape: List[int], device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Create a tensor with all elements set to 1.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device of the tensor.
            
        Returns:
            Tensor: The tensor with all elements set to 1.
        """
        _t = _Tensor.ones(shape, device)
        return TensorBase(_t)
    
    @staticmethod
    def randn(shape: List[int], device: DeviceType = DeviceType.CPU) -> TensorBase:
        """
        Create a tensor with elements drawn from a normal distribution.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device of the tensor.
        
        Returns:
            Tensor: The tensor with elements drawn from a normal distribution.
        """
        return TensorBase.from_numpy(np.random.randn(*shape), device)
