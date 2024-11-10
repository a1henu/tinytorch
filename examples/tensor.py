import numpy as np

from tinytorch import DeviceType, Tensor

t = Tensor.randn([2, 3], DeviceType.CPU)
print(t)

y = Tensor.ones([2, 3], DeviceType.CPU)
print(y)

print(y.shape())

print(t + y)
