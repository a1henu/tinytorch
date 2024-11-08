import math

import torch
import torch.nn.functional as F

z = torch.tensor([
    [-1.028684, 0.856440, 1.369762, -1.437391, 1.551560, 1.139737, -1.240337, -0.648702, -0.400014, 1.586942], 
    [2.365771, 2.535360, -0.772002, 0.039393, -1.142135, 1.507503, 0.550930, 0.630071, -0.746441, 0.497415], 
    [-0.382562, -1.579024, 1.228670, -0.061057, -0.585326, -1.225693, -0.035275, 0.099546, 0.465645, 0.714231], 
    [-0.739603, 0.209539, 0.564118, 0.357420, -0.649761, 1.078385, -0.351789, -1.801129, -0.612122, -0.219620], 
    [0.764168, -1.062313, 0.094680, -0.484254, -1.003578, 0.560764, -0.030785, 0.453219, 0.187955, 0.185473]
], requires_grad=True) # batch_size = 5, num_classes = 10

# the number represents the correct class of each sample
t = torch.tensor([3, 7, 8, 2, 9])

# calculate the cross entropy loss
loss = F.cross_entropy(z, t)
print(loss)

loss.backward()
print(z.grad)

# calculate the cross entropy loss = -\sum p_i * log(q_i)
z_s = F.softmax(z, dim=1)
sum = 0
for i in range(len(t)):
    sum -= math.log(z_s[i][t[i]])
    
print(sum / len(t))

# calculate the gradient of the cross entropy loss
for i in range(len(t)):
    z_s[i][t[i]] -= 1
    
print(z_s / len(t))