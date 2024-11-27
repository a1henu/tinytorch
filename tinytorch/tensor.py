from __future__ import annotations
from typing import Tuple, List, Callable, Optional, overload
from numpy.typing import NDArray

from .tensorbase import DeviceType, TensorBase
from .operators import Ops
from .node import Node
from .autodiff import compute_gradient_of_variables

class Tensor(Node):
    """
    A tensor class that supports autograd
    """
    @overload
    def __init__(self, data: TensorBase) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType, requires_grad: bool = False) -> None: ...
    @overload
    def __init__(self, shape: List[int], device: DeviceType, data: List[float], requires_grad: bool = False) -> None: ...
    
    def __init__(self, *args, requires_grad: bool = False, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], TensorBase):
            self.requires_grad = False
            self._data = args[0]
        else:
            self.requires_grad = requires_grad
            self._data = TensorBase(*args, **kwargs)
        
    def __hash__(self) -> int:
        return id(self)
    
    def __TensorBase__(self) -> TensorBase:
        return self.get_cached_data()
    
    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        """
        Set the tensor to require gradient computation.
        
        Parameters:
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor with the updated requirement.
        """
        self.requires_grad = requires_grad
        return self
    
    def cpu(self) -> Tensor:
        """
        Move the tensor to CPU and return a new tensor.
        
        Returns:
            TensorBase: The tensor on CPU.
        """
        return Tensor(self.get_cached_data().cpu())
    
    def gpu(self) -> Tensor:
        """
        Move the tensor to GPU and return a new tensor.
        
        Returns:
            TensorBase: The tensor on GPU.
        """
        return Tensor(self.get_cached_data().gpu())
    
    def to_cpu(self) -> None:
        """
        Move the tensor to CPU.
        
        Returns:
            None
        """
        self.get_cached_data().to_cpu()
    
    def to_gpu(self) -> None:
        """
        Move the tensor to GPU.
        
        Returns:
            None
        """
        self.get_cached_data().to_gpu()
    
    def in_cpu(self) -> bool:
        """
        Check if the tensor is on CPU.
        
        Returns:
            bool: True if the tensor is on CPU, False otherwise.
        """
        return self.get_cached_data().in_cpu()
    
    def in_gpu(self) -> bool:
        """
        Check if the tensor is on GPU.
        
        Returns:
            bool: True if the tensor is on GPU, False otherwise.
        """
        return self.get_cached_data().in_gpu()
    
    @property
    def device(self) -> DeviceType:
        """
        Get the device of the tensor.
        
        Returns:
            DeviceType: The device of the tensor.
        """
        return self.get_cached_data().device
    
    @property
    def dim(self) -> int:
        """
        Get the dimension of the tensor.
        
        Returns:
            int: The dimension of the tensor.
        """
        return self.get_cached_data().dim
    
    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the tensor.
        
        Returns:
            List[int]: The shape of the tensor.
        """
        return self.get_cached_data().shape
    
    def reshape(self, shape: List[int]) -> Tensor:
        """
        Reshape the tensor to the given shape.
        
        Parameters:
            shape (List[int]): The shape to reshape the tensor to.
        
        Returns:
            TensorBase: The reshaped tensor.
        """
        return Tensor(self.get_cached_data().reshape(shape))
    
    def transpose(self) -> Tensor:
        """
        Transpose the tensor.
        
        Returns:
            TensorBase: The transposed tensor.
        """
        return Tensor(self.get_cached_data().transpose())
    
    @property
    def size(self) -> int:
        """
        Get the total size of the tensor.
        
        Returns:
            int: The size of the tensor.
        """
        return self.get_cached_data().size
    
    def __len__(self) -> int:
        """
        Get the length of the tensor.
        
        Returns:
            int: The length of the tensor.
        """
        return self.get_cached_data().__len__()
    
    def __add__(self, other) -> Tensor:
        """
        Add the tensor to another tensor.
        
        Returns:
            Tensor: The sum of the tensors.
        """
        return AddOp()(self, other)
    
    def __sub__(self, other) -> Tensor:
        """
        Subtract the tensor by another tensor.
        
        Returns:
            Tensor: The difference of the tensors.
        """
        return SubOp()(self, other)
    
    def __pow__(self, scalar) -> Tensor:
        """
        Raise the tensor to a scalar power.
        
        Returns:
            Tensor: The tensor raised to the power.
        """
        return PowOp(scalar)(self)
    
    def __mul__(self, other) -> Tensor:
        """
        Multiply the tensor by a scalar.
        
        Returns:
            Tensor: The product of the tensors.
        """
        if isinstance(other, int) or isinstance(other, float):
            return MulOp(float(other))(self)
        elif isinstance(other, Tensor):
            return EWiseMulOp()(self, other)
    
    def __truediv__(self, other) -> Tensor:
        """
        Divide the tensor by a scalar.
        
        Returns:
            Tensor: The quotient of the tensors.
        """
        if isinstance(other, int) or isinstance(other, float):
            return MulOp(float(1 / other))(self)
        elif isinstance(other, Tensor):
            return EWiseMulOp()(self, other**(-1))
    
    def __matmul__(self, other) -> Tensor:
        """
        Multiply the tensor by another tensor.
        
        Returns:
            Tensor: The product of the tensors.
        """
        return MatMulOp()(self, other)
    
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    
    def __eq__(self, other) -> bool:
        """
        Check if the current tensor is equal to another tensor.
        
        Returns:
            bool: True if the tensors are equal, False otherwise.
        """
        return self.get_cached_data() == other.get_cached_data()
    
    def __getitem__(self, *args) -> float:
        """
        Get item from the tensor.
        
        Parameters:
            args: The indices of the item.
        
        Returns:
            float: The item.
        """
        return self.get_cached_data().__getitem__(*args)
    
    def __repr__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return self.get_cached_data().__repr__()
    
    def __str__(self) -> str:
        """
        Get the string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return self.get_cached_data().__str__()
    
    def to_numpy(self) -> NDArray:
        """
        Convert the tensor to a numpy array.
        
        Returns:
            NDArray: The numpy array.
        """
        return self.get_cached_data().to_numpy()
    
    def backward(self, out_grad=None):
        if out_grad is None:
            out_grad = Tensor.ones(self.shape, self.device, requires_grad=True)
        compute_gradient_of_variables(self, out_grad)
    
    @staticmethod
    def from_numpy(array: NDArray, device: DeviceType = DeviceType.CPU, requires_grad: bool = False) -> Tensor:
        """
        Make a tensor from a numpy array.
        
        Parameters:
            data (NDArray): The numpy array to make the tensor from.
            device (DeviceType): The device to load the tensor to.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor from the numpy array.
        """
        data = TensorBase.from_numpy(array, device)
        tensor = Tensor(data)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    @staticmethod
    def make_from_op(op: Ops, inputs: List[Tensor], requires_grad: bool = None) -> Tensor:
        """
        Make a tensor from an operation and its inputs.
        
        Parameters:
            op (Ops): The operation to make the tensor from.
            inputs (List[Tensor]): The input tensors to the operation.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor from the operation.
        """
        if requires_grad is None:
            requires_grad = any(t.requires_grad for t in inputs)
        output = Tensor.__new__(Tensor)
        output.init_op(op, inputs, None, requires_grad)
        output.get_cached_data()
        return output
    
    @staticmethod
    def full(shape: List[int], fill_value: float, device: DeviceType = DeviceType.CPU, requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with a scalar value.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            fill_value (float): The scalar value to fill the tensor with.
            device (DeviceType): The device to load the tensor to.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor filled with the scalar value.
        """
        data = TensorBase.full(shape, fill_value, device)
        tensor = Tensor(data)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    @staticmethod
    def zeros(shape: List[int], device: DeviceType = DeviceType.CPU, requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with zeros.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device to load the tensor to.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor filled with zeros.
        """
        data = TensorBase.zeros(shape, device)
        tensor = Tensor(data)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    @staticmethod
    def ones(shape: List[int], device: DeviceType = DeviceType.CPU, requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with ones.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device to load the tensor to.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor filled with ones.
        """
        data = TensorBase.ones(shape, device)
        tensor = Tensor(data)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    @staticmethod
    def randn(shape: List[int], device: DeviceType = DeviceType.CPU, requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with random values.
        
        Parameters:
            shape (List[int]): The shape of the tensor.
            device (DeviceType): The device to load the tensor to.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Tensor: The tensor filled with random values.
        """
        data = TensorBase.randn(shape, device)
        tensor = Tensor(data)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    
class TensorOp(Ops):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    

class AddOp(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a + b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
    
    
class SubOp(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a - b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, -out_grad
    
    
class PowOp(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
        
    def compute(self, a: TensorBase):
        return a ** self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
    
    
class MulOp(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
        
    def compute(self, a: TensorBase):
        if isinstance(self.scalar, int) or isinstance(self.scalar, float):
            return self.scalar * a
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return self.scalar * out_grad


class EWiseMulOp(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a * b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * node.inputs[1], out_grad * node.inputs[0]


class MatMulOp(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a @ b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        return grad_a, grad_b