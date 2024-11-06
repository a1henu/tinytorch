import torch
import torch.nn.functional as F

def format_as_c_array(array):
    c_array_str = ""
    for x in array.flatten():
        c_array_str += f"{x:.6f}, "
    return c_array_str[:-2]

# generate random gradient
gradient = torch.randn(30, requires_grad=False)

# ReLU activation test cases
v = torch.randn(30, requires_grad=True)

v_relu = F.relu(v)
v_relu.backward(gradient)
v_relu_grad = v.grad

print("v:\n", format_as_c_array(v.detach().numpy()))
print()
print("gradient:\n", format_as_c_array(gradient.detach().numpy()))
print()
print("v_relu:\n", format_as_c_array(v_relu.detach().numpy()))
print()
print("v_relu_backward:\n", format_as_c_array(v_relu_grad.detach().numpy()))
print()

# Sigmoid activation test cases
x = torch.randn(30, requires_grad=True)

x_sigmoid = torch.sigmoid(x)
x_sigmoid.backward(gradient)
x_sigmoid_grad = x.grad

print("x:\n", format_as_c_array(x.detach().numpy()))
print()
print("gradient:\n", format_as_c_array(gradient.detach().numpy()))
print()
print("x_sigmoid:\n", format_as_c_array(x_sigmoid.detach().numpy()))
print()
print("x_sigmoid_backward:\n", format_as_c_array(x_sigmoid_grad.detach().numpy()))
print()

# shape: (barch_size, num_classes)
y = torch.randn((5, 10), requires_grad=True)
y_softmax = F.softmax(y, dim=1)
y_flatten = y.view(-1)
y_softmax_flatten = y_softmax.view(-1)
print("y:\n", format_as_c_array(y.detach().numpy()))
print("y_softmax:\n", format_as_c_array(y_softmax_flatten.detach().numpy()))
