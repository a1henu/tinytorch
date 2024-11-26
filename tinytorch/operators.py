from __future__ import annotations
from typing import List
from .tensorbase import TensorBase

class Ops:
    def __call__(self):
        raise NotImplementedError()
    
    def compute(self, *args):
        raise NotImplementedError()
    
    def gradient(self, out_grad, node):
        raise NotImplementedError()
    
    def gradient_as_tuple(self, out_grad, node):
        grads = self.gradient(out_grad, node)
        if isinstance(grads, tuple):
            return grads
        elif isinstance(output, list):
            return tuple(grads)
        else:
            return (grads,)
        
        
class Node:
    data: TensorBase
    requires_grad: bool
    op: Ops = None
    inputs: List[Node] = None
    grad: Node = None
    
    def init_op(
        self,
        op: Ops,
        inputs: List[Node],
        data: TensorBase = None,
        requires_grad: bool = None
    ) -> None:
        """
        Initialize the operation of the tensor.
        
        Parameters:
            op (Ops): The operation to initialize.
            inputs (List[Tensor]): The input tensors to the operation.
            data (TensorBase): The data of the tensor.
            requires_grad (bool): Whether the tensor requires gradient computation.
        """
        if requires_grad is None:
            requires_grad = any(t.requires_grad for t in inputs)
        self.op = op
        self.inputs = inputs
        self.data = data
        self.requires_grad = requires_grad

    def get_cached_data(self) -> TensorBase:
        """
        Get the cached data of the tensor to avoid recomputation.
        
        Returns:
            TensorBase: The cached data of the tensor.
        """
        if self.data is not None:
            return self.data
        self.data = self.op.compute(
            *[t.get_cached_data() for t in self.inputs]
        )
        return self.data
    
    def is_leaf(self) -> bool:
        return self.op is None
    
    @staticmethod
    def make_from_op(
        cls, 
        op: Ops, 
        inputs: List[Node], 
        requires_grad: bool = None
    ) -> Node:
        """
        Make a Node from an operation and its inputs.
        
        Parameters:
            cls (Type[Node]): The class to make the node from.
            op (Ops): The operation to make the tensor from.
            inputs (List[Node]): The input node to the operation.
            requires_grad (bool): Whether the tensor requires gradient computation.
        
        Returns:
            Node: The node from the operation.
        """
        if requires_grad is None:
            requires_grad = any(t.requires_grad for t in inputs)
        output = cls.__new__(cls)
        output.init_op(op, inputs, None, requires_grad)
        output.get_cached_data()
        return output
    