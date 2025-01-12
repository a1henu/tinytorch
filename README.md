# tinytorch

This is my project for the course "[Programming in AI](https://pkuprogramminginai.github.io/Labs-Documentation/#/)" in 2024 Fall in Peking University.

We purpose to implement a tiny deep learning framework, and we will use it to train a simple neural network.

**IMPORTANT: PLEASE DO NOT COPY MY CODE DIRECTLY FOR YOUR ASSIGNMENT.**

<!-- toc -->

- [Installation](#installation)
- [Usage](#usage)
- [Unit Test](#unit-test)

<!-- tocstop -->

## Installation

Firstly, make sure you have installed `openblas`. If you haven't installed it, you can run the following command to install it:

```bash
apt-get install libopenblas-dev 
# or if you are using brew
brew install openblas
```

And export the following environment variables:

```bash
export CPLUS_INCLUDE_PATH=/opt/homebrew/opt/openblas/include
export LIBRARY_PATH=/opt/homebrew/opt/openblas/lib
export LD_LIBRARY_PATH=/opt/homebrew/opt/openblas/lib
```

Then, you can install `tinytorch` by running the following command:

```bash
# if you want to build the CPU version and you have installed CUDA
CUDA_BUILD=ON pip install -v .[test] 
# if you want to build the CPU version
CUDA_BUILD=OFF pip install -v .[test]
```

## Usage

The usage of `tinytorch` is very simple, nearly the same as `PyTorch`. You can run the code in `examples` folder to see how to use it.

## Unit Test

`tinytorch` is a C++ project binded with [pybind11](https://github.com/pybind/pybind11). For C++ source code, tinytorch uses [GoogleTest](https://github.com/google/googletest) to implement unit tests. For Python source code, tinytorch uses ...

### C++/CUDA Part

To test the C++/CUDA code, you need to install [GoogleTest](https://github.com/google/googletest) first. This module is included in `third_party` folder. You can install it by running the following commands:

```bash
git submodule update --init --recursive
```

#### Building C++ Code

To build the C++ code independently, you can run the script `build.sh`. The script will create a folder `build` and compile the source code in it.

```bash
bash build.sh --cpu # build the CPU Version
bash build.sh --gpu # build the GPU Version
```

If you want to clean the build folder, you can run the following script:

```bash
bash clean.sh
```

You can also run the following commands to do the same thing:

```bash
mkdir build
cd build
cmake -DCUDA=OFF -DTEST=ON .. # build the CPU Version
cmake -DCUDA=ON -DTEST=ON .. # build the GPU Version
make
```

#### Testing C++ Code

You can run the script `test.sh` to test the C++ code. The script will create a folder `build` and compile the source code in it.

```bash
bash test.sh --cpu # test the CPU Version
bash test.sh --gpu # test the GPU Version
```

You can also run the following commands to do the same thing:

```bash
mkdir build
cd build
cmake -DTEST=ON -DCUDA=OFF .. # test the CPU Version
cmake -DTEST=ON -DCUDA=ON ..  # test the GPU Version
make
ctest --verbose --output-on-failure -C Debug -T test
```

### Python Part

If you want to run the unit test for python part, you can run the following command:

```bash
cd tests
pytest -v
```
