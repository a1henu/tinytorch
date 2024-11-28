from __future__ import annotations
from typing import Tuple

import pytest
import numpy as np
from numpy.testing import assert_allclose

import torch
import torch.nn as nn
import torch.optim as optim
from tinytorch import Tensor, DeviceType
from tinytorch.nn import Conv2d, Linear, ReLU, Sigmoid, MaxPool2d, Softmax, MSELoss, CrossEntropyLoss

from utils import skip_if_no_cuda

def test_conv2d():
    # Initialize tensors
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Tinytorch
    tinytorch_conv = Conv2d(3, 6, (5, 5))
    weight = tinytorch_conv.weight.to_numpy().astype(np.float32)
    bias = tinytorch_conv.bias.to_numpy().astype(np.float32)

    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_y = tinytorch_conv(tinytorch_x)
    tinytorch_output = tinytorch_y.to_numpy()

    # PyTorch
    torch_conv = nn.Conv2d(3, 6, (5, 5))
    torch_conv.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
    torch_conv.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    torch_x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    torch_y = torch_conv(torch_x)
    torch_output = torch_y.detach().numpy()

    # Compare outputs
    assert_allclose(tinytorch_output, torch_output, atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_y.backward(Tensor.from_numpy(grad))
    torch_y.backward(torch.tensor(grad, dtype=torch.float32))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    
    assert tinytorch_conv.weight.grad is not None, "tinytorch_weight.grad is None"
    assert torch_conv.weight.grad is not None, "torch_weight.grad is None"
    assert_allclose(tinytorch_conv.weight.grad.to_numpy(), torch_conv.weight.grad.numpy(), atol=1e-3)
    
    assert tinytorch_conv.bias.grad is not None, "tinytorch_bias.grad is None"
    assert torch_conv.bias.grad is not None, "torch_bias.grad is None"
    assert_allclose(tinytorch_conv.bias.grad.to_numpy(), torch_conv.bias.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_conv2d_gpu():
    # Initialize tensors
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Tinytorch
    tinytorch_conv = Conv2d(3, 6, (5, 5))
    tinytorch_conv.to_gpu()
    weight = tinytorch_conv.weight.to_numpy().astype(np.float32)
    bias = tinytorch_conv.bias.to_numpy().astype(np.float32)

    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_y = tinytorch_conv(tinytorch_x)
    tinytorch_output = tinytorch_y.to_numpy()

    # PyTorch
    torch_conv = nn.Conv2d(3, 6, (5, 5))
    torch_conv.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
    torch_conv.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    torch_x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    torch_y = torch_conv(torch_x)
    torch_output = torch_y.detach().numpy()

    # Compare outputs
    assert_allclose(tinytorch_output, torch_output, atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_y.backward(Tensor.from_numpy(grad, device=DeviceType.GPU))
    torch_y.backward(torch.tensor(grad, dtype=torch.float32))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    
    assert tinytorch_conv.weight.grad is not None, "tinytorch_weight.grad is None"
    assert torch_conv.weight.grad is not None, "torch_weight.grad is None"
    assert_allclose(tinytorch_conv.weight.grad.to_numpy(), torch_conv.weight.grad.numpy(), atol=1e-3)
    
    assert tinytorch_conv.bias.grad is not None, "tinytorch_bias.grad is None"
    assert torch_conv.bias.grad is not None, "torch_bias.grad is None"
    assert_allclose(tinytorch_conv.bias.grad.to_numpy(), torch_conv.bias.grad.numpy(), atol=1e-3)

def test_linear():
    # Initialize tensors
    x = np.random.randn(5, 10).astype(np.float32)
    
    # Tinytorch
    tinytorch_fc = Linear(10, 20)
    weight = tinytorch_fc.weight.to_numpy().astype(np.float32)
    bias = tinytorch_fc.bias.to_numpy().astype(np.float32)

    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_y = tinytorch_fc(tinytorch_x)
    tinytorch_output = tinytorch_y.to_numpy()

    # PyTorch
    torch_fc = nn.Linear(10, 20)
    torch_fc.weight = nn.Parameter(torch.tensor(weight.T, dtype=torch.float32))
    torch_fc.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    torch_x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    torch_y = torch_fc(torch_x)
    torch_output = torch_y.detach().numpy()
    
    # Compare outputs
    assert_allclose(tinytorch_output, torch_output, atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_y.backward(Tensor.from_numpy(grad))
    torch_y.backward(torch.tensor(grad, dtype=torch.float32))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    
    assert tinytorch_fc.weight.grad is not None, "tinytorch_weight.grad is None"
    assert torch_fc.weight.grad is not None, "torch_weight.grad is None"
    assert_allclose(tinytorch_fc.weight.grad.to_numpy(), torch_fc.weight.grad.numpy().T, atol=1e-3)
    
    assert tinytorch_fc.bias.grad is not None, "tinytorch_bias.grad is None"
    assert torch_fc.bias.grad is not None, "torch_bias.grad is None"
    assert_allclose(tinytorch_fc.bias.grad.to_numpy(), torch_fc.bias.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_linear_gpu():
    # Initialize tensors
    x = np.random.randn(5, 10).astype(np.float32)
    
    # Tinytorch
    tinytorch_fc = Linear(10, 20)
    tinytorch_fc.to_gpu()
    weight = tinytorch_fc.weight.to_numpy().astype(np.float32)
    bias = tinytorch_fc.bias.to_numpy().astype(np.float32)

    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_y = tinytorch_fc(tinytorch_x)
    tinytorch_output = tinytorch_y.to_numpy()

    # PyTorch
    torch_fc = nn.Linear(10, 20)
    torch_fc.weight = nn.Parameter(torch.tensor(weight.T, dtype=torch.float32))
    torch_fc.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    torch_x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    torch_y = torch_fc(torch_x)
    torch_output = torch_y.detach().numpy()
    
    # Compare outputs
    assert_allclose(tinytorch_output, torch_output, atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_y.backward(Tensor.from_numpy(grad, device=DeviceType.GPU))
    torch_y.backward(torch.tensor(grad, dtype=torch.float32))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    
    assert tinytorch_fc.weight.grad is not None, "tinytorch_weight.grad is None"
    assert torch_fc.weight.grad is not None, "torch_weight.grad is None"
    assert_allclose(tinytorch_fc.weight.grad.to_numpy(), torch_fc.weight.grad.numpy().T, atol=1e-3)
    
    assert tinytorch_fc.bias.grad is not None, "tinytorch_bias.grad is None"
    assert torch_fc.bias.grad is not None, "torch_bias.grad is None"
    assert_allclose(tinytorch_fc.bias.grad.to_numpy(), torch_fc.bias.grad.numpy(), atol=1e-3)

def test_relu():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_relu = ReLU()
    tinytorch_output = tinytorch_relu(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_relu = nn.ReLU()
    torch_output = torch_relu(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_relu_gpu():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_relu = ReLU()
    tinytorch_output = tinytorch_relu(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_relu = nn.ReLU()
    torch_output = torch_relu(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad, device=DeviceType.GPU))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

def test_sigmoid():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_sigmoid = Sigmoid()
    tinytorch_output = tinytorch_sigmoid(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_sigmoid = nn.Sigmoid()
    torch_output = torch_sigmoid(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_sigmoid_gpu():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_sigmoid = Sigmoid()
    tinytorch_output = tinytorch_sigmoid(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_sigmoid = nn.Sigmoid()
    torch_output = torch_sigmoid(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad, device=DeviceType.GPU))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

def test_maxpool2d():
    # Initialize tensors
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_maxpool = MaxPool2d((2, 2), (2, 2))
    tinytorch_output = tinytorch_maxpool(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_maxpool = nn.MaxPool2d((2, 2), (2, 2))
    torch_output = torch_maxpool(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_maxpool2d_gpu():
    # Initialize tensors
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_maxpool = MaxPool2d((2, 2), (2, 2))
    tinytorch_output = tinytorch_maxpool(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_maxpool = nn.MaxPool2d((2, 2), (2, 2))
    torch_output = torch_maxpool(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)
    
    # Backward pass
    grad = np.random.randn(*tinytorch_output.shape).astype(np.float32)
    tinytorch_output.backward(Tensor.from_numpy(grad, device=DeviceType.GPU))
    torch_output.backward(torch.tensor(grad))
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

def test_softmax():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_softmax = Softmax()
    tinytorch_output = tinytorch_softmax(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_softmax = nn.Softmax(dim=1)
    torch_output = torch_softmax(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)

@skip_if_no_cuda
def test_softmax_gpu():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_softmax = Softmax()
    tinytorch_output = tinytorch_softmax(tinytorch_x)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_softmax = nn.Softmax(dim=1)
    torch_output = torch_softmax(torch_x)
    
    # Compare outputs
    assert_allclose(tinytorch_output.to_numpy(), torch_output.detach().numpy(), atol=1e-3)

def test_mse_loss():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    target = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_target = Tensor.from_numpy(target, requires_grad=True)
    tinytorch_mse_loss = MSELoss()
    tinytorch_output = tinytorch_mse_loss(tinytorch_x, tinytorch_target)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_target = torch.tensor(target, requires_grad=True)
    torch_mse_loss = nn.MSELoss()
    torch_output = torch_mse_loss(torch_x, torch_target)
    
    # Backward pass
    tinytorch_output.backward()
    torch_output.backward()
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)
    
@skip_if_no_cuda
def test_mse_loss_gpu():
    # Initialize tensors
    x = np.random.randn(10, 20).astype(np.float32)
    target = np.random.randn(10, 20).astype(np.float32)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_target = Tensor.from_numpy(target, device=DeviceType.GPU, requires_grad=True)
    tinytorch_mse_loss = MSELoss()
    tinytorch_output = tinytorch_mse_loss(tinytorch_x, tinytorch_target)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_target = torch.tensor(target, requires_grad=True)
    torch_mse_loss = nn.MSELoss()
    torch_output = torch_mse_loss(torch_x, torch_target)
    
    # Backward pass
    tinytorch_output.backward()
    torch_output.backward()
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

def test_cross_entropy_loss():
    # Initialize tensors
    x = np.random.randn(10, 5).astype(np.float32)
    labels = np.random.randint(0, 5, size=(10,)).astype(np.int64)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, requires_grad=True)
    tinytorch_labels = Tensor.from_numpy(labels, requires_grad=False)
    tinytorch_ce_loss = CrossEntropyLoss()
    tinytorch_output = tinytorch_ce_loss(tinytorch_x, tinytorch_labels)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_labels = torch.tensor(labels, dtype=torch.long)
    torch_ce_loss = nn.CrossEntropyLoss()
    torch_output = torch_ce_loss(torch_x, torch_labels)
    
    # Backward pass
    tinytorch_output.backward()
    torch_output.backward()
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

@skip_if_no_cuda
def test_cross_entropy_loss_gpu():
    # Initialize tensors
    x = np.random.randn(10, 5).astype(np.float32)
    labels = np.random.randint(0, 5, size=(10,)).astype(np.int64)
    
    # Tinytorch
    tinytorch_x = Tensor.from_numpy(x, device=DeviceType.GPU, requires_grad=True)
    tinytorch_labels = Tensor.from_numpy(labels, device=DeviceType.GPU, requires_grad=False)
    tinytorch_ce_loss = CrossEntropyLoss()
    tinytorch_output = tinytorch_ce_loss(tinytorch_x, tinytorch_labels)
    
    # PyTorch
    torch_x = torch.tensor(x, requires_grad=True)
    torch_labels = torch.tensor(labels, dtype=torch.long)
    torch_ce_loss = nn.CrossEntropyLoss()
    torch_output = torch_ce_loss(torch_x, torch_labels)
    
    # Backward pass
    tinytorch_output.backward()
    torch_output.backward()
    
    # Compare gradients
    assert tinytorch_x.grad is not None, "tinytorch_x.grad is None"
    assert torch_x.grad is not None, "torch_x.grad is None"
    assert_allclose(tinytorch_x.grad.to_numpy(), torch_x.grad.numpy(), atol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])