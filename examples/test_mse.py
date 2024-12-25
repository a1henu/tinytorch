import numpy as np
from numpy.testing import assert_allclose
import torch

from tinytorch import DeviceType, Tensor, TensorBase
from tinytorch.funcs import mse_forward
from tinytorch.nn import MSELoss

def test_mse(l, x, t, gpu):
    # Tinytorch
    if gpu:
        l.to_gpu()
    else:
        l.to_cpu()
    d = DeviceType.GPU if gpu else DeviceType.CPU
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=d, requires_grad=True)
    tinytorch_target = Tensor.from_numpy(t, device=d, requires_grad=True)
    tinytorch_output = l(tinytorch_x, tinytorch_target)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_target = torch.tensor(t, requires_grad=True)
    torch_mse_loss = torch.nn.MSELoss()
    torch_output = torch_mse_loss(torch_x, torch_target)
    
    # Compare outputs
    # assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    print('=== loss ===')
    print('--- tinytorch ---')
    print(tinytorch_output.to_numpy())
    print('--- torch ---')
    print(torch_output.detach().numpy())
    print('--- diff ---')
    print(torch_output.detach().numpy() / tinytorch_output.to_numpy()[0])
    print('--- numpy ---')
    l_n = (x - t) ** 2
    print(l_n.sum() / (x.shape[0] * x.shape[1]))
    
    # Backward pass
    tinytorch_output.backward()
    torch_output.backward()
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    # assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    print('=== grad ===')
    print('--- tinytorch ---')
    print(tinytorch_x.grad.to_numpy())
    print('--- torch ---')
    print(torch_x.grad.numpy())

if __name__ == "__main__":
    l = MSELoss()
    x = np.random.randn(3, 5).astype(np.float32)
    t = np.random.randn(3, 5).astype(np.float32)
    
    for gpu in [False, True]:
        print(f"Running on {'GPU' if gpu else 'CPU'}")
        test_mse(l, x, t, gpu)
