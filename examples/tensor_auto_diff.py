from tinytorch import Tensor

t1 = Tensor.randn([2, 3], requires_grad=True)
t2 = Tensor.randn([2, 3], requires_grad=True)
t3 = t1 + t2
out_grad = Tensor.ones([2, 3], requires_grad=True)

params = [t1, t2, t3]

# t3.backward(out_grad)



for param in params:
    print(param.grad)