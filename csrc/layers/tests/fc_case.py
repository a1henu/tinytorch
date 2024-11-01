import torch
import torch.nn as nn

def format_as_c_array(array):
    c_array_str = "{"
    flat_array = array.flatten(order='C')
    for i, x in enumerate(flat_array):
        c_array_str += f"{x:.6f}, "
    c_array_str = c_array_str.rstrip(", ")  
    c_array_str += "};"
    return c_array_str

batch_size = 2
in_features = 3
out_features = 4

# generate random input, weight and bias
input = torch.randn(batch_size, in_features, requires_grad=True)
weight = torch.randn(out_features, in_features, requires_grad=True)
bias = torch.randn(out_features, requires_grad=True)

# generate random gradient
gradient = torch.randn(batch_size, out_features, requires_grad=False)

fc = nn.Linear(in_features, out_features)
fc.weight = nn.Parameter(weight)
fc.bias = nn.Parameter(bias)

output = fc(input)

output.backward(gradient)

w = weight.T
dw = fc.weight.grad.T

# print X, W, b, Y, dL/dY, dL/dX, dL/dW, dL/db
print("X:\n", format_as_c_array(input.detach().numpy()))
print()
print("W:\n", format_as_c_array(w.detach().numpy()))
print()
print("b:\n", format_as_c_array(bias.detach().numpy()))
print()
print("Y:\n", format_as_c_array(output.detach().numpy()))
print()
print("dL/dY:\n", format_as_c_array(gradient.detach().numpy()))
print()
print("dL/dX:\n", format_as_c_array(input.grad.detach().numpy()))
print()
print("dL/dW:\n", format_as_c_array(dw.detach().numpy()))
print()
print("dL/db:\n", format_as_c_array(fc.bias.grad.detach().numpy()))
print()