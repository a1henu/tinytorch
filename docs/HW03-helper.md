# 人工智能中的编程——第三次作业

## 问题描述

- 使用`Pybind11`将第一次作业实现的`Tensor`类封装进`Python`中，并能在`Python`侧实例化你的`Tensor`
- 将前两次作业中实现的7种卷积神经网络中常用的`Module`(`Sigmoid`, `ReLU`, `Fully Connected Layer`, `Convolution Layer`, `Pooling Layer`, `Cross Entropy Loss`, `SoftMax`)封装进`Python`中，并支持在`Python`中传入你的`Tensor`并实现输入输出
- 在`Python`中，自己选择合适的方法读取`MNIST`数据集，然后转换为`numpy`数组，并使用`pybind11/numpy.h`中的`pybind11::array_t`转换为自己的`Tensor`，以便之后使用自己的`Tensor`训练MNIST数据集
- 在`Python`中对封装好的7个算子进行测试，并与`Pytorch`中`torch.nn.functional`提供的标准函数的计算结果进行`UnitTest`(单元测试)，以尽可能减少在后续实现模块拼接和计算图构建时进行不必要的繁杂的debug

## 实现

我使用`pybind11`+`scikit-build-core`构建，[官方示例](https://github.com/pybind/scikit_build_example)，[`scikit-build-core`文档](https://scikit-build-core.readthedocs.io/)。

核心是通过`pyproject.toml`文件配置`pybind11`的`cmake`构建，然后使用`scikit-build`进行构建。

对于`Python`绑定，本次作业进行了部分封装，主要是对`Tensor`类和`Layer`命名空间中的函数进行了封装，之后我书写了`Python`前端，以支持Type Hint和自动补全，相关代码为：
- `csrc/tensor/tensor_binding.cpp`
- `csrc/layers/funcs_binding.cpp`
- `tinytorch`文件夹内的前端代码

我将代码编译出的动态链接库链接到`pybind11`编译出的`Python`模块中，通过前端进行一层封装，并且在封装中检查输入维度等信息，以保证`Python`端的输入正确性。

具体可参见`tinytorch/tensor.py`和`tinytorch/funcs/funcs.py`中的代码

为了方便`MNIST`数据集的读取，我还编写了`tinytorch/data/dataset.py`、`tinytorch/data/dataloader.py`和`tinytorch/data/mnist.py`，以方便读取`MNIST`数据集，并且进一步训练神经网络。

其中，`tinytorch/data/dataset.py`和`tinytorch/data/dataloader.py`是通用的数据集和数据加载器的实现，`tinytorch/data/mnist.py`是`MNIST`数据集的实现。

## 安装

请进入**项目根目录**，然后根据需求选择以下两种安装方式：

CPU版本：
```bash
CUDA_BUILD=OFF pip install -v .         # 安装CPU版本，不安装测试依赖
CUDA_BUILD=OFF pip install -v .[test]   # 安装CPU版本，安装测试依赖
```

CUDA版本：
```bash
CUDA_BUILD=ON pip install -v .          # 安装CUDA版本，不安装测试依赖
CUDA_BUILD=ON pip install -v .[test]    # 安装CUDA版本，安装测试依赖
```

随后即可通过`import tinytorch`导入`Python`模块。

模块目前由下面几个部分组成：
- `tinytorch.DeviceType`：`DeviceType`枚举类
- `tinytorch.Tensor`：`Tensor`类的封装
- `tinytorch.funcs`：`funcs`子模块
    - `relu_forward`：`ReLU`前向传播
    - `relu_backward`：`ReLU`反向传播
    - `sigmoid_forward`：`Sigmoid`前向传播
    - `sigmoid_backward`：`Sigmoid`反向传播
    - `fc_forward`：`FullyConnected`前向传播
    - `fc_backward`：`FullyConnected`反向传播
    - `conv2d_forward`：`Convolution`前向传播
    - `conv2d_backward`：`Convolution`反向传播
    - `max_pool2d_forward`：`MaxPooling`前向传播
    - `softmax_forward`：`SoftMax`前向传播
    - `cross_entropy_forward`：`CrossEntropyLoss`前向传播（内含`softmax`）
    - `cross_entropy_backward`：`CrossEntropyLoss`反向传播（内含`softmax`）