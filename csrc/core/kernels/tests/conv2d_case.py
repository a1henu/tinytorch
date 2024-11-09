import torch
import numpy as np

torch.manual_seed(42)

batch_size = 2
in_channels = 2
height = 4
width = 4
x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)

out_channels = 3
kernel_size = 3
stride = 1
padding = 1

weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
bias = torch.randn(out_channels, requires_grad=True)

output = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)

grad_output = torch.randn_like(output)

grad_input, grad_weight, grad_bias = torch.autograd.grad(
    output, [x, weight, bias], grad_output
)

print("Input tensor:")
print(x.detach().numpy().flatten().tolist())
print("\nWeight tensor:")
print(weight.detach().numpy().flatten().tolist())
print("\nBias tensor:")
print(bias.detach().numpy().tolist())
print("\nOutput tensor:")
print(output.detach().numpy().flatten().tolist())
print("\nGrad output tensor:")
print(grad_output.numpy().flatten().tolist())
print("\nGrad input tensor:")
print(grad_input.numpy().flatten().tolist())
print("\nGrad weight tensor:")
print(grad_weight.numpy().flatten().tolist())
print("\nGrad bias tensor:")
print(grad_bias.numpy().tolist())