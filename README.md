# tinytorch

This is a project for the course "[Programming in AI](https://pkuprogramminginai.github.io/Labs-Documentation/#/)" in Peking University.

We purpose to implement a tiny deep learning framework, and we will use it to train a simple neural network.

## Installation

**TODO**: Describe the installation process

## Usage

**TODO**: Write usage instructions

## Unit Test

tinytorch is a C++ project binded with [pybind11](https://github.com/pybind/pybind11). For C++ source code, tinytorch uses [GoogleTest](https://github.com/google/googletest) to implement unit tests. For Python source code, tinytorch uses ...

### C++/CUDA Part

To test the C++/CUDA code, you need to install [GoogleTest](https://github.com/google/googletest) first. This module is included in `third_party` folder. You can install it by running the following commands:

```bash
git submodule update --init --recursive
```

**Testing C++ Code**

You can run the following commands to test the C++ code:

```bash
mkdir build
cd build
cmake -DTEST=ON -DCUDA=OFF ..
make
ctest --verbose --output-on-failure -C Debug -T test
```

**Testing CUDA Code**

You can run the following commands to test the CUDA code:

```bash
mkdir build
cd build
cmake -DTEST=ON -DCUDA=ON ..
make
ctest --verbose --output-on-failure -C Debug -T test
```