import torch
from torch.functional import F
import numpy as np

from tinytorch import Tensor
from tinytorch.funcs import ReLU

def op_auto_diff():
    print('Auto Diff with Op')
    t1 = Tensor.randn([2, 3], requires_grad=True)
    t2 = Tensor.randn([2, 3], requires_grad=True)
    t3 = t1 + 2 * t2
    out_grad = Tensor.ones([2, 3], requires_grad=True)

    params = [t1, t2, t3]

    t3.backward(out_grad)

    for param in params:
        print(param.grad)
    
def relu_auto_diff():
    print('Auto Diff with ReLU')
    t1 = Tensor.randn([2, 3], requires_grad=True)
    t2 = ReLU()(t1)
    out_grad = Tensor.ones([2, 3], requires_grad=True)

    t2.backward(out_grad)

    tensor1 = torch.tensor(t1.to_numpy(), requires_grad=True)
    tensor2 = F.relu(tensor1)
    tensor2.backward(torch.ones_like(tensor2))
    
    print('TinyTorch')
    print(t1.grad)
    print('PyTorch')
    print(tensor1.grad)
    
if __name__ == '__main__':
    op_auto_diff()
    relu_auto_diff()